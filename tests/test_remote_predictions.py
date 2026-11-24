import asyncio
import io
import time

import torch
from PIL import Image

from src.client.image_encoder import encode_image
from src.server.prediction_service import PredictionService


class FixedModel:
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        time.sleep(0.02)
        output = torch.zeros((1, 1000), dtype=torch.float32)
        output[0, 42] = 10.0
        return output


def _png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 80, 150)).save(stream, format="PNG")
    return stream.getvalue()


def _service_without_loading_weights() -> PredictionService:
    service = PredictionService.__new__(PredictionService)
    service.model = FixedModel()
    service.categories = [f"class-{index}" for index in range(1000)]
    service.inference_lock = asyncio.Lock()
    service.active_inferences = 0
    service.max_active_inferences = 0
    return service


def test_png_encoding_preserves_decoded_rgb_pixels(tmp_path) -> None:
    original = tmp_path / "source.jpg"
    Image.new("RGB", (80, 60), (25, 100, 175)).save(original, format="JPEG", quality=91)
    expected = Image.open(original).convert("RGB")
    encoded = encode_image(original, "img-1")
    actual = Image.open(io.BytesIO(encoded.png_bytes)).convert("RGB")
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


def test_prediction_is_deterministic_and_server_serializes_forward_passes() -> None:
    async def scenario() -> tuple[list[dict], PredictionService]:
        service = _service_without_loading_weights()
        results = await asyncio.gather(
            service.predict_png(_png()), service.predict_png(_png()), service.predict_png(_png())
        )
        return results, service

    results, service = asyncio.run(scenario())
    assert {result["predicted_class_id"] for result in results} == {42}
    assert {result["predicted_class_name"] for result in results} == {"class-42"}
    assert service.max_active_inferences == 1
    assert max(result["server_metrics"]["queue_wait_ms"] for result in results) > 0

