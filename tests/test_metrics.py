from __future__ import annotations

import pandas as pd
import pytest

from src.metrics import descriptive_stats, run_metrics


def test_percentiles_are_calculated_correctly() -> None:
    stats = descriptive_stats([1, 2, 3, 4, 5])
    assert stats["mean_ms"] == pytest.approx(3.0)
    assert stats["p90_ms"] == pytest.approx(4.6)
    assert stats["p95_ms"] == pytest.approx(4.8)
    assert stats["p99_ms"] == pytest.approx(4.96)


def test_throughput_and_classification_metrics() -> None:
    frame = pd.DataFrame(
        {
            "true_class_id": [0, 217],
            "predicted_class_id": [0, 999],
            "correct_top1": [True, False],
            "correct_top5": [True, True],
            "read_ms": [1.0, 1.0],
            "preprocess_ms": [2.0, 2.0],
            "inference_ms": [3.0, 3.0],
            "postprocess_ms": [1.0, 1.0],
            "total_latency_ms": [7.5, 7.5],
        }
    )
    metrics = run_metrics(frame, 1.0, {}, {})
    assert metrics["throughput_images_per_second"] == 2.0
    assert metrics["classification"]["accuracy_top1"] == 0.5
    assert metrics["classification"]["accuracy_top5"] == 1.0
    assert metrics["classification"]["outside_imagenette_predictions"] == 1

