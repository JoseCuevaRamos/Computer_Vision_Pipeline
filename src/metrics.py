from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.config import IMAGENETTE_CLASSES


STAGE_COLUMNS = ["read_ms", "preprocess_ms", "inference_ms", "postprocess_ms"]


def descriptive_stats(values: pd.Series | np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot calculate statistics for an empty collection")
    return {
        "mean_ms": float(np.mean(array)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
        "median_ms": float(np.median(array)),
        "std_ms": float(np.std(array, ddof=0)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def quality_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    true_ids = predictions["true_class_id"].astype(int).to_numpy()
    predicted_ids = predictions["predicted_class_id"].astype(int).to_numpy()
    labels = [int(value["imagenet_class_id"]) for value in IMAGENETTE_CLASSES.values()]
    names = [str(value["class_name"]) for value in IMAGENETTE_CLASSES.values()]

    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids, predicted_ids, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_ids, predicted_ids, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        true_ids, predicted_ids, labels=labels, average="weighted", zero_division=0
    )

    id_to_column = {class_id: index for index, class_id in enumerate(labels)}
    confusion = np.zeros((10, 11), dtype=int)
    for true_id, predicted_id in zip(true_ids, predicted_ids, strict=True):
        row = id_to_column[int(true_id)]
        column = id_to_column.get(int(predicted_id), 10)
        confusion[row, column] += 1

    per_class: dict[str, Any] = {}
    for index, name in enumerate(names):
        per_class[name] = {
            "imagenet_class_id": labels[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "accuracy": float(recall[index]),
        }

    correct_top1 = int(predictions["correct_top1"].sum())
    correct_top5 = int(predictions["correct_top5"].sum())
    return {
        "accuracy_top1": correct_top1 / len(predictions),
        "accuracy_top5": correct_top5 / len(predictions),
        "correct_top1": correct_top1,
        "correct_top5": correct_top5,
        "errors_top1": int(len(predictions) - correct_top1),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "outside_imagenette_predictions": int((~predictions["predicted_class_id"].isin(labels)).sum()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_row_labels": names,
        "confusion_column_labels": names + ["other ImageNet class"],
    }


def run_metrics(
    predictions: pd.DataFrame,
    total_seconds: float,
    resource_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    latency = {"total": descriptive_stats(predictions["total_latency_ms"])}
    for column in STAGE_COLUMNS:
        latency[column.removesuffix("_ms")] = descriptive_stats(predictions[column])

    stage_totals = {column.removesuffix("_ms"): float(predictions[column].sum()) for column in STAGE_COLUMNS}
    direct_total_ms = total_seconds * 1000
    accounted_ms = sum(stage_totals.values())
    distribution = {
        name: 100.0 * value / direct_total_ms for name, value in stage_totals.items()
    }
    distribution["unattributed_overhead"] = 100.0 * (direct_total_ms - accounted_ms) / direct_total_ms

    return {
        "num_images": int(len(predictions)),
        "failed_images": 0,
        "total_seconds": float(total_seconds),
        "throughput_images_per_second": float(len(predictions) / total_seconds),
        "latency_statistics": latency,
        "stage_total_ms": stage_totals,
        "time_distribution_percent": distribution,
        "classification": quality_metrics(predictions),
        "resources": resource_summary,
        "configuration": config,
    }


def aggregate_runs(summary: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [column for column in summary.columns if column != "run_id"]
    return {
        "run_count": int(len(summary)),
        "metrics": {
            column: {
                "mean": float(summary[column].mean()),
                "std": float(summary[column].std(ddof=0)),
            }
            for column in numeric_columns
        },
        "note": "Accuracy describes the same 500-image sample repeated across runs, not independent observations.",
    }

