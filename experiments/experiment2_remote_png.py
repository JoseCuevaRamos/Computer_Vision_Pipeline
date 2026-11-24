from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.client.http_client import InferenceHTTPClient
from src.client.pipeline import run_pipeline
from src.dataset import load_manifest, resolve_dataset_path
from src.metrics import descriptive_stats, quality_metrics
from src.resource_monitor import ResourceMonitor


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("top5_ids", "top5_probabilities"):
        if column in result:
            result[column] = result[column].map(lambda value: json.dumps(value) if isinstance(value, list) else value)
    return result


def calculate_metrics(
    frame: pd.DataFrame,
    total_seconds: float,
    resources: dict[str, Any],
    pipeline_metrics: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    successes = frame[frame["status"] == "ok"].copy()
    if successes.empty:
        raise RuntimeError("No image was processed successfully")
    return {
        "attempted_images": len(frame), "successful_images": len(successes),
        "failed_images": int((frame["status"] != "ok").sum()),
        "total_seconds": total_seconds,
        "throughput_images_per_second": len(successes) / total_seconds,
        "latency": descriptive_stats(successes["end_to_end_ms"]),
        "http_latency": descriptive_stats(successes["http_total_ms"]),
        "total_png_payload_bytes": int(successes["png_bytes"].sum()),
        "total_response_bytes": int(successes["response_bytes"].sum()),
        "total_retries": int(successes["retries"].sum()),
        "classification": quality_metrics(successes),
        "client_resources": resources,
        "pipeline": pipeline_metrics,
        "configuration": configuration,
        "communication_note": "HTTP duration includes proxy, communication, server queueing, processing and serialization.",
    }


async def execute(args: argparse.Namespace) -> None:
    records = load_manifest(args.manifest, args.data_root, verify_hashes=not args.skip_hash_verification)
    records = select_records(records, args.num_images)
    paths = {record.image_id: resolve_dataset_path(args.data_root, record.relative_path) for record in records}
    args.results_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_resources: list[pd.DataFrame] = []

    for concurrency in args.concurrencies:
        for run_number in range(1, args.runs + 1):
            run_id = f"run_{run_number:02d}"
            run_dir = args.results_root / f"run_c{concurrency}" / run_id
            if (run_dir / "predictions.csv").exists() and not args.overwrite:
                raise FileExistsError(f"Results already exist at {run_dir}; use --overwrite")
            async with InferenceHTTPClient(
                args.server_url, args.api_key, timeout_seconds=args.timeout,
                max_retries=args.max_retries, require_https=not args.allow_http,
            ) as client:
                await client.health()
                warmup_records = [records[index % len(records)] for index in range(args.warmup)]
                if warmup_records:
                    await run_pipeline(warmup_records, paths, client, concurrency=concurrency)

                monitor = ResourceMonitor(args.resource_interval)
                monitor.start()
                start = time.perf_counter_ns()
                try:
                    values, pipeline_metrics = await run_pipeline(
                        records, paths, client, concurrency=concurrency, queue_capacity=args.queue_capacity
                    )
                finally:
                    end = time.perf_counter_ns()
                    resource_summary = monitor.stop()
                total_seconds = (end - start) / 1_000_000_000

            frame = pd.DataFrame(values)
            frame.insert(0, "run_id", run_id)
            frame.insert(1, "concurrency", concurrency)
            resources = pd.DataFrame(monitor.samples)
            configuration = {
                "server_url_host": args.server_url.split("//", 1)[-1].split("/", 1)[0],
                "num_images": args.num_images, "runs": args.runs, "concurrency": concurrency,
                "warmup": args.warmup, "timeout_seconds": args.timeout,
                "max_retries": args.max_retries, "network_conditions": args.network_conditions,
            }
            metrics = calculate_metrics(frame, total_seconds, resource_summary, pipeline_metrics, configuration)
            write_csv(run_dir / "predictions.csv", csv_ready(frame))
            write_csv(run_dir / "resources.csv", resources)
            write_json(run_dir / "metrics.json", metrics)
            latency = metrics["latency"]
            summary_rows.append({
                "configuration": f"C{concurrency}", "concurrency": concurrency, "run_id": run_id,
                "total_seconds": total_seconds,
                "throughput_images_per_second": metrics["throughput_images_per_second"],
                "latency_mean_ms": latency["mean_ms"], "latency_p95_ms": latency["p95_ms"],
                "latency_p99_ms": latency["p99_ms"],
                "cpu_mean_percent": resource_summary["process_cpu_percent_mean"],
                "ram_max_mb": resource_summary["process_rss_mb_max"],
                "png_payload_bytes": metrics["total_png_payload_bytes"],
                "retries": metrics["total_retries"], "errors": metrics["failed_images"],
                "accuracy_top1": metrics["classification"]["accuracy_top1"],
                "accuracy_top5": metrics["classification"]["accuracy_top5"],
                "overlap_count": pipeline_metrics["preparation_http_overlap_count"],
                "max_in_flight": pipeline_metrics["max_in_flight_observed"],
            })
            all_predictions.append(frame)
            all_resources.append(resources.assign(configuration=f"C{concurrency}", run_id=run_id))
            tqdm.write(f"C{concurrency} {run_id}: {len(frame)} images in {total_seconds:.3f}s")

    summary = pd.DataFrame(summary_rows)
    summary_dir = args.results_root / "summary"
    write_csv(summary_dir / "summary_runs.csv", summary)
    aggregate = summary.groupby("configuration").agg(
        total_seconds_mean=("total_seconds", "mean"), total_seconds_std=("total_seconds", "std"),
        throughput_mean=("throughput_images_per_second", "mean"),
        latency_mean_ms=("latency_mean_ms", "mean"), latency_p95_ms=("latency_p95_ms", "mean"),
        accuracy_top1=("accuracy_top1", "mean"), accuracy_top5=("accuracy_top5", "mean"),
    ).fillna(0).reset_index()
    write_json(summary_dir / "summary_metrics.json", {"configurations": aggregate.to_dict(orient="records")})
    comparison = compare_baseline(aggregate, args.baseline_summary)
    write_csv(summary_dir / "comparison_with_experiment1.csv", comparison)
    consistency = compare_local_predictions(all_predictions[0], args.baseline_predictions)
    write_csv(summary_dir / "prediction_consistency.csv", consistency["rows"])
    write_json(summary_dir / "prediction_consistency.json", consistency["summary"])
    if consistency["summary"]["status"] == "available":
        tqdm.write(
            "Consistencia con E1: "
            f"{consistency['summary']['top1_matches']}/{consistency['summary']['compared_images']} top-1"
        )
    generate_figures(summary, pd.concat(all_predictions), pd.concat(all_resources), aggregate, comparison, args.results_root / "figures")


def select_records(records: list, count: int) -> list:
    """Use all 500 in manifest order, or a balanced prefix for smoke tests."""
    if count == len(records):
        return records
    by_class: dict[str, list] = {}
    class_order: list[str] = []
    for record in records:
        if record.class_wnid not in by_class:
            by_class[record.class_wnid] = []
            class_order.append(record.class_wnid)
        by_class[record.class_wnid].append(record)
    selected: list = []
    position = 0
    while len(selected) < count:
        added = False
        for class_wnid in class_order:
            values = by_class[class_wnid]
            if position < len(values) and len(selected) < count:
                selected.append(values[position])
                added = True
        if not added:
            break
        position += 1
    if len(selected) != count:
        raise ValueError(f"Could not select {count} unique manifest records")
    return selected


def compare_baseline(remote: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    result = remote.copy()
    if baseline_path.is_file():
        baseline = pd.read_csv(baseline_path)
        baseline_seconds = float(baseline["total_seconds"].mean())
        result["experiment1_total_seconds"] = baseline_seconds
        result["speedup_relative"] = baseline_seconds / result["total_seconds_mean"]
        result["baseline_status"] = "available"
    else:
        result["experiment1_total_seconds"] = float("nan")
        result["speedup_relative"] = float("nan")
        result["baseline_status"] = "baseline file unavailable"
    return result


def compare_local_predictions(remote: pd.DataFrame, baseline_path: Path) -> dict[str, Any]:
    columns = [
        "image_id", "predicted_class_id_remote", "predicted_class_id_local",
        "top1_match", "confidence_top1_remote", "confidence_top1_local",
        "confidence_absolute_difference",
    ]
    if not baseline_path.is_file():
        return {
            "rows": pd.DataFrame(columns=columns),
            "summary": {"status": "baseline predictions unavailable", "compared_images": 0},
        }
    local = pd.read_csv(baseline_path)
    required = {"image_id", "predicted_class_id", "confidence_top1"}
    if not required.issubset(local.columns):
        raise ValueError(f"Baseline predictions are missing columns: {sorted(required - set(local.columns))}")
    remote_ok = remote[remote["status"] == "ok"][
        ["image_id", "predicted_class_id", "confidence_top1"]
    ].drop_duplicates("image_id")
    local = local[["image_id", "predicted_class_id", "confidence_top1"]].drop_duplicates("image_id")
    merged = remote_ok.merge(local, on="image_id", suffixes=("_remote", "_local"), validate="one_to_one")
    merged["top1_match"] = merged["predicted_class_id_remote"] == merged["predicted_class_id_local"]
    merged["confidence_absolute_difference"] = (
        merged["confidence_top1_remote"] - merged["confidence_top1_local"]
    ).abs()
    merged = merged[columns]
    return {
        "rows": merged,
        "summary": {
            "status": "available",
            "compared_images": int(len(merged)),
            "top1_matches": int(merged["top1_match"].sum()),
            "top1_match_rate": float(merged["top1_match"].mean()) if len(merged) else 0.0,
            "max_confidence_absolute_difference":
                float(merged["confidence_absolute_difference"].max()) if len(merged) else None,
            "note": "Small confidence differences can occur across hardware; class IDs are compared exactly.",
        },
    }


def generate_figures(summary: pd.DataFrame, predictions: pd.DataFrame, resources: pd.DataFrame, aggregate: pd.DataFrame, comparison: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = aggregate["configuration"]
    plots = [
        ("01_average_latency.png", aggregate["latency_mean_ms"], "Latencia promedio", "ms"),
        ("02_p95_latency.png", aggregate["latency_p95_ms"], "Latencia p95", "ms"),
        ("03_throughput.png", aggregate["throughput_mean"], "Throughput", "imágenes/s"),
        ("04_total_time.png", aggregate["total_seconds_mean"], "Tiempo total", "s"),
    ]
    for filename, values, title, ylabel in plots:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(labels, values); ax.set(title=title, xlabel="Configuración", ylabel=ylabel); _save(fig, output / filename)
    ok = predictions[predictions["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(8, 4));
    for name, group in ok.groupby("concurrency"): ax.hist(group["http_total_ms"], bins=25, alpha=.5, label=f"C{name}")
    ax.set(title="Distribución de latencia HTTP", xlabel="ms", ylabel="Frecuencia"); ax.legend(); _save(fig, output / "05_http_latency_distribution.png")
    stage_columns = ["read_ms", "encode_ms", "client_queue_wait_ms", "server_decode_ms", "server_preprocess_ms", "server_inference_ms", "server_postprocess_ms"]
    fig, ax = plt.subplots(figsize=(10, 4)); ok[stage_columns].mean().plot.bar(ax=ax); ax.set(title="Tiempo promedio por etapa", ylabel="ms"); _save(fig, output / "06_pipeline_stages.png")
    fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(ok["png_bytes"] / 1024, bins=30); ax.set(title="Tamaño de payload PNG", xlabel="KiB", ylabel="Frecuencia"); _save(fig, output / "07_png_sizes.png")
    fig, ax = plt.subplots(figsize=(8, 4));
    for name, group in resources.groupby("configuration"): ax.plot(group["elapsed_seconds"], group["process_cpu_percent"], label=name)
    ax.set(title="CPU del cliente", xlabel="s", ylabel="%"); ax.legend(); _save(fig, output / "08_client_cpu.png")
    fig, ax = plt.subplots(figsize=(8, 4));
    for name, group in resources.groupby("configuration"): ax.plot(group["elapsed_seconds"], group["process_rss_mb"], label=name)
    ax.set(title="RAM del cliente", xlabel="s", ylabel="MB"); ax.legend(); _save(fig, output / "09_client_ram.png")
    if comparison["experiment1_total_seconds"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(["E1 local", *labels], [comparison["experiment1_total_seconds"].iloc[0], *aggregate["total_seconds_mean"]]); ax.set(title="Comparación con experimento 1", ylabel="s"); _save(fig, output / "11_comparison_experiment1.png")
        fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(labels, comparison["speedup_relative"]); ax.axhline(1, color="black"); ax.set(title="Speedup relativo", ylabel="E1 / E2"); _save(fig, output / "12_relative_speedup.png")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2: distributed PNG pipeline")
    parser.add_argument("--server-url", default=os.environ.get("SERVER_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--data-root", type=Path, default=Path("/app/data/imagenette"))
    parser.add_argument("--manifest", type=Path, default=Path("/app/config/dataset_manifest.csv"))
    parser.add_argument("--results-root", type=Path, default=Path("/app/results/client"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("/app/baseline/summary/summary_runs.csv"))
    parser.add_argument("--baseline-predictions", type=Path, default=Path("/app/baseline/run_01/predictions.csv"))
    parser.add_argument("--num-images", type=int, default=int(os.environ.get("EXPERIMENT2_IMAGES", "500")))
    parser.add_argument("--runs", type=int, default=int(os.environ.get("EXPERIMENT2_RUNS", "5")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("EXPERIMENT2_WARMUP", "20")))
    default_concurrency = os.environ.get("EXPERIMENT2_CONCURRENCIES", os.environ.get("EXPERIMENT2_CONCURRENCY", "1,2,4"))
    parser.add_argument("--concurrencies", type=lambda value: [int(item) for item in value.split(",")], default=[int(item) for item in default_concurrency.split(",")])
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("HTTP_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("HTTP_MAX_RETRIES", "2")))
    parser.add_argument("--queue-capacity", type=int)
    parser.add_argument("--resource-interval", type=float, default=.2)
    parser.add_argument("--network-conditions", default=os.environ.get("NETWORK_CONDITIONS", "not documented"))
    parser.add_argument("--allow-http", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.server_url or not args.api_key: parser.error("SERVER_URL and API_KEY are required")
    if not 1 <= args.num_images <= 500 or args.runs < 1 or args.warmup < 0: parser.error("Invalid image/run/warmup values")
    if any(value not in {1, 2, 4} for value in args.concurrencies): parser.error("Concurrencies must be 1, 2 or 4")
    return args


if __name__ == "__main__":
    asyncio.run(execute(parse_args()))
