from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.client.http_client import InferenceHTTPClient, RemoteRequestError
from src.client.image_encoder import EncodedImage, encode_image
from src.dataset import ImageRecord


@dataclass
class PipelineResult:
    index: int
    values: dict[str, Any]


class InFlightTracker:
    def __init__(self) -> None:
        self.current = 0
        self.maximum = 0
        self.lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self.lock:
            self.current += 1
            self.maximum = max(self.maximum, self.current)

    async def exit(self) -> None:
        async with self.lock:
            self.current -= 1


async def run_pipeline(
    records: list[ImageRecord],
    paths: dict[str, Path],
    client: InferenceHTTPClient,
    *,
    concurrency: int,
    queue_capacity: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    queue: asyncio.Queue[tuple[int, ImageRecord, EncodedImage, int] | None] = asyncio.Queue(
        maxsize=queue_capacity or max(2, concurrency * 2)
    )
    output: list[PipelineResult] = []
    tracker = InFlightTracker()
    prepare_intervals: list[tuple[int, int]] = []
    http_intervals: list[tuple[int, int]] = []

    async def producer() -> None:
        for index, record in enumerate(records):
            preparation_start = time.perf_counter_ns()
            encoded = await asyncio.to_thread(encode_image, paths[record.image_id], record.image_id)
            preparation_end = time.perf_counter_ns()
            prepare_intervals.append((preparation_start, preparation_end))
            await queue.put((index, record, encoded, preparation_start))
        for _ in range(concurrency):
            await queue.put(None)

    async def worker() -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            index, record, encoded, end_to_end_start = item
            queue_wait_ms = (time.perf_counter_ns() - encoded.prepared_ns) / 1_000_000
            http_start = time.perf_counter_ns()
            await tracker.enter()
            try:
                response = await client.predict(record.image_id, encoded.png_bytes)
                body = response.body
                metrics = body["server_metrics"]
                values = {
                    "image_id": record.image_id, "relative_path": record.relative_path,
                    "true_class_id": record.imagenet_class_id, "true_class_name": record.class_name,
                    "predicted_class_id": int(body["predicted_class_id"]),
                    "predicted_class_name": body["predicted_class_name"],
                    "confidence_top1": float(body["confidence_top1"]),
                    "top5_ids": body["top5_ids"], "top5_probabilities": body["top5_probabilities"],
                    "correct_top1": int(body["predicted_class_id"]) == record.imagenet_class_id,
                    "correct_top5": record.imagenet_class_id in body["top5_ids"],
                    "read_ms": encoded.read_ms, "encode_ms": encoded.encode_ms,
                    "client_queue_wait_ms": queue_wait_ms, "http_total_ms": response.http_total_ms,
                    "end_to_end_ms": (time.perf_counter_ns() - end_to_end_start) / 1_000_000,
                    "png_bytes": len(encoded.png_bytes), "response_bytes": response.response_bytes,
                    "retries": response.retries, "status": "ok", "error": "",
                    **{f"server_{key}": value for key, value in metrics.items()},
                }
            except RemoteRequestError as error:
                values = {
                    "image_id": record.image_id, "relative_path": record.relative_path,
                    "true_class_id": record.imagenet_class_id, "true_class_name": record.class_name,
                    "read_ms": encoded.read_ms, "encode_ms": encoded.encode_ms,
                    "client_queue_wait_ms": queue_wait_ms,
                    "end_to_end_ms": (time.perf_counter_ns() - end_to_end_start) / 1_000_000,
                    "png_bytes": len(encoded.png_bytes), "status": "error", "error": str(error),
                    "retries": error.retries,
                }
            finally:
                await tracker.exit()
                http_intervals.append((http_start, time.perf_counter_ns()))
                queue.task_done()
            output.append(PipelineResult(index, values))

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(producer(), *tasks)
    output.sort(key=lambda item: item.index)
    overlap_count = sum(
        1 for prepare_start, prepare_end in prepare_intervals
        if any(max(prepare_start, http_start) < min(prepare_end, http_end) for http_start, http_end in http_intervals)
    )
    return [item.values for item in output], {
        "max_in_flight_observed": tracker.maximum,
        "preparation_http_overlap_count": overlap_count,
        "pipeline_overlap_demonstrated": overlap_count > 0,
        "queue_capacity": queue.maxsize,
    }
