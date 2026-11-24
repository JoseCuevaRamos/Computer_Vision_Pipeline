import asyncio
import time
from pathlib import Path

import pytest

pytest.importorskip("httpx")

from src.client.http_client import HTTPResult
from src.client.image_encoder import EncodedImage
from src.client.pipeline import run_pipeline
from src.dataset import ImageRecord


def _record(index: int) -> ImageRecord:
    return ImageRecord(
        image_id=f"class/image-{index}", filename=f"image-{index}.JPEG",
        relative_path=f"imagenette2-320/val/class/image-{index}.JPEG",
        class_wnid="class", class_name="test", imagenet_class_id=0,
        width=64, height=64, original_size_bytes=1, sha256="x",
    )


class SlowFakeClient:
    max_retries = 2

    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def predict(self, image_id: str, png_bytes: bytes) -> HTTPResult:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.03)
        self.active -= 1
        return HTTPResult(
            body={
                "image_id": image_id, "model": "efficientnet_b0",
                "predicted_class_id": 0, "predicted_class_name": "test",
                "confidence_top1": 1.0, "top5_ids": [0, 1, 2, 3, 4],
                "top5_probabilities": [1.0, 0.0, 0.0, 0.0, 0.0],
                "server_metrics": {
                    "queue_wait_ms": 0.1, "decode_ms": 0.2, "preprocess_ms": 0.3,
                    "inference_ms": 1.0, "postprocess_ms": 0.1, "total_internal_ms": 1.7,
                },
            },
            http_total_ms=30.0, response_bytes=200, retries=0,
        )


def test_pipeline_respects_concurrency_and_demonstrates_overlap(monkeypatch) -> None:
    def slow_encode(path: Path, image_id: str) -> EncodedImage:
        time.sleep(0.01)
        return EncodedImage(
            image_id=image_id, png_bytes=b"png", read_ms=1.0, encode_ms=9.0,
            width=64, height=64, prepared_ns=time.perf_counter_ns(),
        )

    monkeypatch.setattr("src.client.pipeline.encode_image", slow_encode)
    records = [_record(index) for index in range(8)]
    paths = {record.image_id: Path(record.filename) for record in records}
    client = SlowFakeClient()
    values, metrics = asyncio.run(
        run_pipeline(records, paths, client, concurrency=2, queue_capacity=2)
    )
    assert [row["image_id"] for row in values] == [record.image_id for record in records]
    assert client.maximum == 2
    assert metrics["max_in_flight_observed"] == 2
    assert metrics["pipeline_overlap_demonstrated"] is True
    assert metrics["preparation_http_overlap_count"] > 0

