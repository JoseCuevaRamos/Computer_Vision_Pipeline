from __future__ import annotations

import asyncio
import io
import time
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from src.model import WEIGHTS, load_model
from src.preprocessing import preprocess_image


class InvalidImageError(ValueError):
    pass


class PredictionService:
    model_name = "efficientnet_b0"

    def __init__(self, *, torch_threads: int | None = None) -> None:
        self.model, self.model_load_ms = load_model("cpu", torch_threads=torch_threads, require_cached=True)
        self.categories = list(WEIGHTS.meta["categories"])
        self.inference_lock = asyncio.Lock()
        self.active_inferences = 0
        self.max_active_inferences = 0

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(batch)

    async def predict_png(self, payload: bytes, min_dimension: int = 32, max_dimension: int = 4096) -> dict[str, Any]:
        total_start = time.perf_counter_ns()
        decode_start = time.perf_counter_ns()
        try:
            with Image.open(io.BytesIO(payload)) as source:
                if source.format != "PNG":
                    raise InvalidImageError("Only decoded PNG images are accepted")
                width, height = source.size
                if not (min_dimension <= width <= max_dimension and min_dimension <= height <= max_dimension):
                    raise InvalidImageError(f"Image dimensions {width}x{height} are outside allowed limits")
                image = source.convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError) as error:
            raise InvalidImageError("The uploaded file is not a decodable PNG image") from error
        decode_ms = (time.perf_counter_ns() - decode_start) / 1_000_000

        preprocess_start = time.perf_counter_ns()
        batch = preprocess_image(image)
        preprocess_ms = (time.perf_counter_ns() - preprocess_start) / 1_000_000

        queue_start = time.perf_counter_ns()
        async with self.inference_lock:
            queue_wait_ms = (time.perf_counter_ns() - queue_start) / 1_000_000
            self.active_inferences += 1
            self.max_active_inferences = max(self.max_active_inferences, self.active_inferences)
            inference_start = time.perf_counter_ns()
            try:
                # Keep the FastAPI event loop responsive while the CPU-bound
                # forward pass runs. The explicit lock still guarantees that
                # only one model inference is active at any instant.
                logits = await asyncio.to_thread(self._forward, batch)
            finally:
                self.active_inferences -= 1
            inference_ms = (time.perf_counter_ns() - inference_start) / 1_000_000

        postprocess_start = time.perf_counter_ns()
        probabilities = logits[0].softmax(dim=0)
        top_probabilities, top_ids_tensor = probabilities.topk(5)
        top5_ids = [int(value) for value in top_ids_tensor.tolist()]
        top5_probabilities = [float(value) for value in top_probabilities.tolist()]
        postprocess_ms = (time.perf_counter_ns() - postprocess_start) / 1_000_000
        total_internal_ms = (time.perf_counter_ns() - total_start) / 1_000_000
        return {
            "model": self.model_name,
            "predicted_class_id": top5_ids[0],
            "predicted_class_name": self.categories[top5_ids[0]],
            "confidence_top1": top5_probabilities[0],
            "top5_ids": top5_ids,
            "top5_names": [self.categories[index] for index in top5_ids],
            "top5_probabilities": top5_probabilities,
            "server_metrics": {
                "queue_wait_ms": queue_wait_ms,
                "decode_ms": decode_ms,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "total_internal_ms": total_internal_ms,
            },
        }

