import asyncio
import io
from pathlib import Path
import os

import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from PIL import Image

from src.client.image_encoder import encode_image
from src.image_geometry import PREPROCESSING_ID
from src.model import WEIGHTS, cached_weights_path
from src.preprocessing import preprocess_image
from src.server.api import create_app
from src.server.prediction_service import InvalidImageError, PredictionService


def _image(size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    pixels = np.random.default_rng(42).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(pixels).convert(mode)


def _bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    stream = io.BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


@pytest.mark.parametrize("size", [(321, 480), (480, 321), (320, 320), (256, 257), (257, 256), (35, 39), (224, 224), (40, 1200)])
@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_client_png_produces_exactly_the_official_input_tensor(tmp_path, size, mode) -> None:
    source = tmp_path / "original.png"
    source.write_bytes(_bytes(_image(size, mode)))
    original = source.read_bytes()
    encoded = encode_image(source, "img-1", preprocess_on_client=True)
    with Image.open(source) as full, Image.open(io.BytesIO(encoded.png_bytes)) as reduced:
        expected = preprocess_image(full.convert("RGB"))
        actual = preprocess_image(reduced, spatial_preprocessed=True)
        assert reduced.format == "PNG" and reduced.mode == "RGB"
        assert reduced.size == (224, 224)
    assert torch.equal(actual, expected)
    assert source.read_bytes() == original
    assert encoded.source_width == size[0] and encoded.source_height == size[1]


class RecordingModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, batch):
        self.inputs.append(batch.clone())
        logits = torch.zeros((1, 1000))
        logits[0, 42] = 10
        return logits


def _service() -> PredictionService:
    service = PredictionService.__new__(PredictionService)
    service.model = RecordingModel()
    service.model_load_ms = 0.0
    service.categories = list(WEIGHTS.meta["categories"])
    service.inference_lock = asyncio.Lock()
    service.active_inferences = service.max_active_inferences = 0
    return service


def test_api_separates_routes_and_persists_experiment3(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "test-api-key-123456")
    monkeypatch.setenv("SERVER_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("EXPERIMENT3_SERVER_RESULTS_DIR", raising=False)
    original = tmp_path / "original.png"
    original.write_bytes(_bytes(_image((321, 480))))
    prepared = encode_image(original, "img-1", preprocess_on_client=True)
    service = _service()
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/health").json()["preprocessing_id"] == PREPROCESSING_ID
        for route, payload in [("/predict", original.read_bytes()), ("/predict/preprocessed", prepared.png_bytes)]:
            response = client.post(route, data={"image_id": "img-1"},
                                   files={"image": ("image.png", payload, "image/png")},
                                   headers={"X-API-Key": "test-api-key-123456"})
            assert response.status_code == 200, response.text
        assert response.json()["preprocessing_id"] == PREPROCESSING_ID
        assert torch.equal(*service.model.inputs)
    assert (tmp_path / "requests.csv").is_file()
    for filename in ("requests.csv", "resources.csv", "resource_summary.json", "system_info.json"):
        assert (tmp_path / "experiment3" / filename).is_file()


@pytest.mark.parametrize("size,mode,fmt,mime,status", [
    ((223, 224), "RGB", "PNG", "image/png", 422),
    ((224, 224), "L", "PNG", "image/png", 422),
    ((224, 224), "RGBA", "PNG", "image/png", 422),
    ((224, 224), "RGB", "BMP", "image/png", 422),
    ((224, 224), "RGB", "PNG", "image/bmp", 415),
])
def test_preprocessed_route_rejects_incompatible_inputs(tmp_path, monkeypatch, size, mode, fmt, mime, status) -> None:
    monkeypatch.setenv("API_KEY", "test-api-key-123456")
    monkeypatch.setenv("SERVER_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("EXPERIMENT3_SERVER_RESULTS_DIR", raising=False)
    service = _service()
    with TestClient(create_app(lambda: service)) as client:
        response = client.post("/predict/preprocessed", data={"image_id": "img-1"},
                               files={"image": ("image.png", _bytes(_image(size, mode), fmt), mime)},
                               headers={"X-API-Key": "test-api-key-123456"})
        assert response.status_code == status, response.text
        assert not service.model.inputs


def test_preprocessed_route_checks_authentication_and_size(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "test-api-key-123456")
    monkeypatch.setenv("SERVER_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_IMAGE_BYTES", "1")
    with TestClient(create_app(_service)) as client:
        for headers, expected in [({}, 401), ({"X-API-Key": "test-api-key-123456"}, 413)]:
            response = client.post("/predict/preprocessed", data={"image_id": "img-1"},
                                   files={"image": ("image.png", b"too-large", "image/png")}, headers=headers)
            assert response.status_code == expected


def test_cached_model_returns_the_same_predictions_for_both_paths(tmp_path) -> None:
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    if not cached_weights_path(torch_home).is_file():
        pytest.skip("Run prepare_model to cache model weights")
    original = tmp_path / "original.png"
    original.write_bytes(_bytes(_image((479, 321))))
    prepared = encode_image(original, "img-1", preprocess_on_client=True)

    async def scenario():
        service = PredictionService(torch_threads=2)
        full = await service.predict_png(original.read_bytes())
        reduced = await service.predict_png(prepared.png_bytes, spatial_preprocessed=True)
        return full, reduced

    full, reduced = asyncio.run(scenario())
    assert full["top5_ids"] == reduced["top5_ids"]
    assert full["top5_probabilities"] == reduced["top5_probabilities"]
