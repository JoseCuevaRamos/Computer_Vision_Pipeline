"""Experiment 3: client-side spatial preprocessing, lossless PNG transport."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from experiments.experiment2_remote_png import execute, parse_args, write_csv, write_json, _save

import matplotlib.pyplot as plt
import pandas as pd


def compare_experiment2(results_root: Path, experiment2_root: Path) -> pd.DataFrame:
    """Compare matching complete workloads; report missing/incompatible baselines."""
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for metrics_path in sorted(results_root.glob("run_c*/run_*/metrics.json")):
        relative = metrics_path.parent.relative_to(results_root)
        row: dict[str, Any] = {"configuration": relative.parts[0], "run_id": relative.parts[1]}
        baseline_dir = experiment2_root / relative
        if not (baseline_dir / "metrics.json").is_file() or not (baseline_dir / "predictions.csv").is_file():
            rows.append({**row, "status": "experiment2 results unavailable"})
            continue
        current = json.loads(metrics_path.read_text())
        baseline = json.loads((baseline_dir / "metrics.json").read_text())
        config3, config2 = current["configuration"], baseline["configuration"]
        compared_fields = (
            "num_images", "concurrency", "warmup", "timeout_seconds", "max_retries",
            "network_conditions", "server_url_host",
        )
        different = [key for key in compared_fields if config3.get(key) != config2.get(key)]
        for key in ("dataset_fingerprint", "model", "weights"):
            if key in config2 and config3.get(key) != config2[key]:
                different.append(key)
        if config2.get("input_representation", "full_png") != "full_png":
            different.append("input_representation")
        if different:
            rows.append({**row, "status": "incompatible: " + ", ".join(different)})
            continue
        reduced = pd.read_csv(metrics_path.parent / "predictions.csv")
        full = pd.read_csv(baseline_dir / "predictions.csv")
        if (full["image_id"].tolist() != reduced["image_id"].tolist()
                or full["image_id"].duplicated().any()
                or len(full) != config3["num_images"]):
            rows.append({**row, "status": "incompatible image IDs or image count"})
            continue
        if not full["status"].eq("ok").all() or not reduced["status"].eq("ok").all():
            rows.append({**row, "status": "incomplete: failed image requests"})
            continue
        merged = full[["image_id", "predicted_class_id", "png_bytes"]].merge(
            reduced[["image_id", "predicted_class_id", "png_bytes"]],
            on="image_id", suffixes=("_e2", "_e3"), validate="one_to_one",
        )
        merged["top1_match"] = merged["predicted_class_id_e2"] == merged["predicted_class_id_e3"]
        merged["configuration"] = row["configuration"]
        merged["run_id"] = row["run_id"]
        details.append(merged)
        bytes2, bytes3 = int(merged["png_bytes_e2"].sum()), int(merged["png_bytes_e3"].sum())
        rows.append({
            **row, "status": "available", "compared_images": len(merged),
            "e2_total_seconds": baseline["total_seconds"], "e3_total_seconds": current["total_seconds"],
            "speedup_e2_over_e3": baseline["total_seconds"] / current["total_seconds"],
            "e2_png_bytes": bytes2, "e3_png_bytes": bytes3,
            "png_reduction_percent": 100.0 * (1.0 - bytes3 / bytes2),
            "e2_latency_p95_ms": baseline["latency"]["p95_ms"],
            "e3_latency_p95_ms": current["latency"]["p95_ms"],
            "accuracy_top1_difference": current["classification"]["accuracy_top1"] - baseline["classification"]["accuracy_top1"],
            "top1_matches": int(merged["top1_match"].sum()),
            "top1_match_rate": float(merged["top1_match"].mean()),
            "dataset_fingerprint_verified": "dataset_fingerprint" in config2,
        })
    comparison = pd.DataFrame(rows, columns=None if rows else ["configuration", "run_id", "status"])
    summary = results_root / "summary"
    write_csv(summary / "comparison_with_experiment2.csv", comparison)
    write_json(summary / "comparison_with_experiment2.json", {
        "runs": json.loads(comparison.to_json(orient="records")),
        "note": "Compare on the same client/server hardware and network. PNG bytes are application payload, not total network traffic. Legacy E2 results lack a dataset fingerprint; re-run E2 for stronger verification.",
    })
    write_csv(summary / "prediction_consistency_with_experiment2.csv", pd.concat(details, ignore_index=True) if details else pd.DataFrame(columns=[
        "image_id", "predicted_class_id_e2", "predicted_class_id_e3", "png_bytes_e2", "png_bytes_e3", "top1_match", "configuration", "run_id",
    ]))
    available = comparison[comparison["status"] == "available"]
    if not available.empty:
        means = available.groupby("configuration")[["speedup_e2_over_e3", "png_reduction_percent"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        means["speedup_e2_over_e3"].plot.bar(ax=axes[0], rot=0)
        axes[0].axhline(1, color="black")
        axes[0].set(title="Speedup E2 / E3", ylabel="Mayor que 1: E3 más rápido")
        means["png_reduction_percent"].plot.bar(ax=axes[1], rot=0)
        axes[1].set(title="Reducción de payload PNG", ylabel="%")
        output = results_root / "figures"
        output.mkdir(parents=True, exist_ok=True)
        _save(fig, output / "13_comparison_experiment2.png")
    return comparison


async def main() -> None:
    args = parse_args(experiment=3)
    await execute(args)
    comparison = compare_experiment2(args.results_root, args.experiment2_results)
    print("Comparación E2/E3:", comparison["status"].value_counts().to_dict())


if __name__ == "__main__":
    asyncio.run(main())
