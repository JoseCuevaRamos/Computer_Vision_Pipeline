import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from PIL import Image

from src.server.api import create_app
from src.server.prediction_service import InvalidImageError
from src.image_geometry import PREPROCESSING_ID


class FakePredictionService:
    model_load_ms = 1.25

    async def predict_png(self, payload: bytes) -> dict:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "PNG":
                    raise InvalidImageError("Only PNG is accepted")
                image.load()
        except (OSError, ValueError) as error:
            raise InvalidImageError("The uploaded file is not a decodable PNG image") from error
        return {
            "model": "efficientnet_b0",
            "predicted_class_id": 0,
            "predicted_class_name": "tench",
            "confidence_top1": 0.9,
            "top5_ids": [0, 1, 2, 3, 4],
            "top5_names": ["tench", "goldfish", "shark", "tiger shark", "hammerhead"],
            "top5_probabilities": [0.9, 0.04, 0.03, 0.02, 0.01],
            "server_metrics": {
                "queue_wait_ms": 0.1, "decode_ms": 0.2, "preprocess_ms": 0.3,
                "inference_ms": 1.0, "postprocess_ms": 0.1, "total_internal_ms": 1.7,
            },
        }


def _png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), (20, 40, 60)).save(stream, format="PNG")
    return stream.getvalue()


def test_health_authentication_prediction_and_persistence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "test-api-key-123456")
    monkeypatch.setenv("SERVER_RESULTS_DIR", str(tmp_path))
    app = create_app(lambda: FakePredictionService())
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok", "model_loaded": True,
            "model": "efficientnet_b0", "weights": "IMAGENET1K_V1",
            "preprocessing_id": PREPROCESSING_ID,
        }
        unauthorized = client.post(
            "/predict", data={"image_id": "img-1"},
            files={"image": ("x.png", _png(), "image/png")},
        )
        assert unauthorized.status_code == 401
        response = client.post(
            "/predict", data={"image_id": "img-1"},
            files={"image": ("x.png", _png(), "image/png")},
            headers={"X-API-Key": "test-api-key-123456"},
        )
        assert response.status_code == 200
        assert response.json()["image_id"] == "img-1"
    assert (tmp_path / "requests.csv").is_file()
    assert (tmp_path / "resources.csv").is_file()
    assert (tmp_path / "system_info.json").is_file()


def test_api_rejects_invalid_content_and_invalid_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "test-api-key-123456")
    monkeypatch.setenv("SERVER_RESULTS_DIR", str(tmp_path))
    app = create_app(lambda: FakePredictionService())
    headers = {"X-API-Key": "test-api-key-123456"}
    with TestClient(app) as client:
        wrong_type = client.post(
            "/predict", data={"image_id": "img-1"},
            files={"image": ("x.txt", b"hello", "text/plain")}, headers=headers,
        )
        assert wrong_type.status_code == 415
        invalid_png = client.post(
            "/predict", data={"image_id": "img-1"},
            files={"image": ("x.png", b"not-a-png", "image/png")}, headers=headers,
        )
        assert invalid_png.status_code == 422

