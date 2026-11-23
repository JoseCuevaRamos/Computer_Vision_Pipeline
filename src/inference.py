from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import torch
from PIL import Image

from src.dataset import ImageRecord
from src.preprocessing import preprocess_image


T = TypeVar("T")
R = TypeVar("R")


def sequential_map(items: Iterable[T], processor: Callable[[T], R]) -> list[R]:
    """Process one item completely before starting the next one."""
    results: list[R] = []
    for item in items:
        results.append(processor(item))
    return results


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000


def process_image(
    record: ImageRecord,
    image_path: Path,
    model: torch.nn.Module,
    categories: list[str],
    device: str = "cpu",
) -> dict[str, Any]:
    total_start = time.perf_counter_ns()
    stage_start = time.perf_counter_ns()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.load()
    read_ms = _elapsed_ms(stage_start)

    stage_start = time.perf_counter_ns()
    batch = preprocess_image(image).to(device)
    if batch.shape[0] != 1:
        raise RuntimeError("Batch size must remain equal to one")
    preprocess_ms = _elapsed_ms(stage_start)

    stage_start = time.perf_counter_ns()
    with torch.inference_mode():
        logits = model(batch)
    inference_ms = _elapsed_ms(stage_start)

    stage_start = time.perf_counter_ns()
    probabilities = logits[0].softmax(dim=0)
    top_probabilities, top_ids = probabilities.topk(5)
    ids = [int(value) for value in top_ids.tolist()]
    probs = [float(value) for value in top_probabilities.tolist()]
    predicted_id = ids[0]
    postprocess_ms = _elapsed_ms(stage_start)

    total_latency_ms = _elapsed_ms(total_start)
    stage_sum_ms = read_ms + preprocess_ms + inference_ms + postprocess_ms
    return {
        "image_id": record.image_id, "relative_path": record.relative_path,
        "true_class_id": record.imagenet_class_id, "true_class_name": record.class_name,
        "predicted_class_id": predicted_id, "predicted_class_name": categories[predicted_id],
        "confidence_top1": probs[0], "top5_ids": ids,
        "top5_names": [categories[index] for index in ids], "top5_probabilities": probs,
        "correct_top1": predicted_id == record.imagenet_class_id,
        "correct_top5": record.imagenet_class_id in ids,
        "read_ms": read_ms, "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms, "postprocess_ms": postprocess_ms,
        "total_latency_ms": total_latency_ms, "stage_sum_ms": stage_sum_ms,
        "unattributed_overhead_ms": total_latency_ms - stage_sum_ms,
    }
