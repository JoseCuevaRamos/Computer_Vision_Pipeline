from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.config import ExperimentConfig
from src.dataset import ImageRecord, load_manifest, resolve_dataset_path
from src.inference import process_image, sequential_map
from src.metrics import aggregate_runs, run_metrics
from src.model import WEIGHTS, load_model
from src.plotting import generate_figures
from src.preprocessing import preprocess_image
from src.reporting import system_info, write_csv, write_json, write_text
from src.resource_monitor import ResourceMonitor


def balanced_subset(records: list[ImageRecord], count: int) -> list[ImageRecord]:
    if count >= len(records):
        return list(records)
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_wnid].append(record)
    selected: list[ImageRecord] = []
    offsets = {class_name: 0 for class_name in sorted(by_class)}
    while len(selected) < count:
        progressed = False
        for class_name in sorted(by_class):
            offset = offsets[class_name]
            if offset < len(by_class[class_name]) and len(selected) < count:
                selected.append(by_class[class_name][offset])
                offsets[class_name] += 1
                progressed = True
        if not progressed:
            break
    return selected


def prepare_warmup_batches(records: list[ImageRecord], data_root: Path, count: int) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    if count == 0:
        return batches
    for index in range(count):
        record = records[index % len(records)]
        path = resolve_dataset_path(data_root, record.relative_path)
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
        batches.append(preprocess_image(image))
    return batches


def warmup_model(model: torch.nn.Module, batches: list[torch.Tensor], device: str) -> None:
    with torch.inference_mode():
        for batch in batches:
            output = model(batch.to(device))
            if tuple(output.shape) != (1, 1000):
                raise RuntimeError(f"Expected 1000 logits, got {tuple(output.shape)}")


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    exported = frame.copy()
    for column in ("top5_ids", "top5_names", "top5_probabilities"):
        exported[column] = exported[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    return exported


def technical_summary(summary: pd.DataFrame, first_metrics: dict, config: ExperimentConfig) -> str:
    classification = first_metrics["classification"]
    return f"""# Resumen técnico — Experimento 1

- Modelo: EfficientNet-B0 (`IMAGENET1K_V1`), sin entrenamiento.
- Dispositivo: CPU; batch size 1; pipeline secuencial.
- Imágenes por ejecución: {config.num_images}.
- Ejecuciones: {config.runs}.
- Tiempo total promedio: {summary['total_seconds'].mean():.4f} s (desviación estándar {summary['total_seconds'].std(ddof=0):.4f} s).
- Throughput promedio: {summary['throughput_images_per_second'].mean():.4f} imágenes/s.
- Latencia media entre ejecuciones: {summary['latency_mean_ms'].mean():.4f} ms/imagen.
- Latencia p95 promedio: {summary['latency_p95_ms'].mean():.4f} ms/imagen.
- Accuracy Top-1: {classification['accuracy_top1']:.4%}.
- Accuracy Top-5: {classification['accuracy_top5']:.4%}.
- Predicciones externas a las diez clases: {classification['outside_imagenette_predictions']}.

La calidad corresponde al mismo subconjunto de imágenes repetido en cada ejecución. No se consideran las inferencias repetidas como observaciones independientes. Imagenette deriva de ImageNet y estos valores no sustituyen la evaluación de referencia en ImageNet-1K.
"""


def execute(config: ExperimentConfig) -> None:
    config.validate()
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(
        config.manifest_path,
        config.data_root,
        verify_hashes=config.verify_hashes,
        require_complete=True,
    )
    selected = balanced_subset(records, config.num_images)
    if len(selected) != config.num_images:
        raise RuntimeError(f"Requested {config.num_images} images, selected {len(selected)}")

    existing_runs = [
        config.output_dir / f"run_{run_number:02d}" / "predictions.csv"
        for run_number in range(1, config.runs + 1)
    ]
    if not config.overwrite and any(path.exists() for path in existing_runs):
        raise FileExistsError("Run results already exist. Use --overwrite or a different --output-subdir.")

    write_json(config.output_dir / "config.json", config.as_json())
    model, model_load_ms = load_model(
        config.device,
        torch_threads=config.torch_threads,
        require_cached=True,
    )
    categories = list(WEIGHTS.meta["categories"])
    info = system_info()
    info["model_load_ms"] = model_load_ms
    write_json(config.output_dir / "system_info.json", info)
    warmup_batches = prepare_warmup_batches(selected, config.data_root, config.warmup)

    predictions_by_run: list[pd.DataFrame] = []
    resources_by_run: list[pd.DataFrame] = []
    metrics_by_run: list[dict] = []
    summary_rows: list[dict[str, float | str]] = []

    for run_number in range(1, config.runs + 1):
        run_id = f"run_{run_number:02d}"
        run_dir = config.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        warmup_model(model, warmup_batches, config.device)

        monitor = ResourceMonitor(config.resource_interval)
        monitor.start()
        benchmark_start = time.perf_counter_ns()
        try:
            def process_record(record: ImageRecord) -> dict:
                image_start = time.perf_counter_ns()
                result = process_image(
                    record,
                    resolve_dataset_path(config.data_root, record.relative_path),
                    model,
                    categories,
                    config.device,
                )
                result["total_latency_ms"] = (time.perf_counter_ns() - image_start) / 1_000_000
                result["unattributed_overhead_ms"] = (
                    result["total_latency_ms"] - result["stage_sum_ms"]
                )
                result["run_id"] = run_id
                return result

            results = sequential_map(tqdm(selected, desc=run_id, unit="image"), process_record)
        finally:
            benchmark_end = time.perf_counter_ns()
            resource_summary = monitor.stop()
        total_seconds = (benchmark_end - benchmark_start) / 1_000_000_000
        if len(results) != len(selected):
            raise RuntimeError(f"{run_id} stopped after {len(results)} of {len(selected)} images")

        predictions = pd.DataFrame(results)
        resources = pd.DataFrame(monitor.samples)
        metrics = run_metrics(predictions, total_seconds, resource_summary, config.as_json())
        write_csv(run_dir / "predictions.csv", _csv_ready(predictions))
        write_csv(run_dir / "resources.csv", resources)
        write_json(run_dir / "metrics.json", metrics)

        latency = metrics["latency_statistics"]["total"]
        summary_rows.append(
            {
                "run_id": run_id,
                "total_seconds": total_seconds,
                "throughput_images_per_second": metrics["throughput_images_per_second"],
                "latency_mean_ms": latency["mean_ms"],
                "latency_p95_ms": latency["p95_ms"],
                "latency_p99_ms": latency["p99_ms"],
                "inference_mean_ms": metrics["latency_statistics"]["inference"]["mean_ms"],
                "cpu_mean_percent": resource_summary["process_cpu_percent_mean"],
                "ram_max_mb": resource_summary["process_rss_mb_max"],
                "accuracy_top1": metrics["classification"]["accuracy_top1"],
                "accuracy_top5": metrics["classification"]["accuracy_top5"],
            }
        )
        predictions_by_run.append(predictions)
        resources_by_run.append(resources)
        metrics_by_run.append(metrics)

    summary = pd.DataFrame(summary_rows)
    summary_dir = config.output_dir / "summary"
    write_csv(summary_dir / "summary_runs.csv", summary)
    write_json(summary_dir / "summary_metrics.json", aggregate_runs(summary))
    write_text(summary_dir / "technical_summary.md", technical_summary(summary, metrics_by_run[0], config))
    generate_figures(
        predictions_by_run,
        resources_by_run,
        summary,
        metrics_by_run[0],
        config.output_dir / "figures",
    )
    print(f"Experiment complete. Results: {config.output_dir}")


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Experiment 1: local sequential EfficientNet-B0 inference")
    parser.add_argument("--data-root", type=Path, default=Path("/app/data/imagenette"))
    parser.add_argument("--results-root", type=Path, default=Path("/app/results"))
    parser.add_argument("--manifest", type=Path, default=Path("/app/results/experiment1/dataset_manifest.csv"))
    parser.add_argument("--output-subdir", default="experiment1")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--resource-interval", type=float, default=0.2)
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return ExperimentConfig(
        data_root=args.data_root,
        results_root=args.results_root,
        manifest_path=args.manifest,
        output_subdir=args.output_subdir,
        num_images=args.num_images,
        runs=args.runs,
        warmup=args.warmup,
        seed=args.seed,
        device=args.device,
        torch_threads=args.torch_threads,
        resource_interval=args.resource_interval,
        verify_hashes=not args.skip_hash_verification,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    execute(parse_args())
