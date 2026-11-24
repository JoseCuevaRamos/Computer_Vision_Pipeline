import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from src.client.http_client import InferenceHTTPClient, RemoteRequestError
from src.image_geometry import PREPROCESSING_ID


def _payload(image_id: str) -> dict:
    return {
        "image_id": image_id,
        "model": "efficientnet_b0",
        "predicted_class_id": 0,
        "predicted_class_name": "tench",
        "confidence_top1": 0.8,
        "top5_ids": [0, 1, 2, 3, 4],
        "top5_probabilities": [0.8, 0.1, 0.05, 0.03, 0.02],
        "server_metrics": {"inference_ms": 10.0},
    }


def test_client_retries_transient_error_and_reuses_contract() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["X-API-Key"] == "secret-value-123"
        assert request.headers["ngrok-skip-browser-warning"] == "true"
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, request=request, json=_payload("img-1"))

    async def scenario() -> None:
        async with InferenceHTTPClient(
            "https://example.test", "secret-value-123", max_retries=1,
            transport=httpx.MockTransport(handler),
        ) as client:
            result = await client.predict("img-1", b"png")
            assert result.retries == 1
            assert result.body["image_id"] == "img-1"
            assert result.response_bytes > 0

    asyncio.run(scenario())
    assert calls == 2


def test_client_rejects_mismatched_image_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=_payload("other-id"))

    async def scenario() -> None:
        async with InferenceHTTPClient(
            "https://example.test", "secret-value-123",
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(RemoteRequestError, match="image_id"):
                await client.predict("expected-id", b"png")

    asyncio.run(scenario())


def test_client_requires_https_by_default() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        InferenceHTTPClient("http://localhost:8000", "secret-value-123")


def test_experiment3_health_and_prediction_route() -> None:
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model_loaded": True, "preprocessing_id": PREPROCESSING_ID})
        payload = {**_payload("img-1"), "preprocessing_id": PREPROCESSING_ID}
        assert b"image/png" in request.content
        return httpx.Response(200, json=payload)

    async def scenario():
        async with InferenceHTTPClient("https://example.test", "secret-value-123",
                                       preprocess_on_client=True, transport=httpx.MockTransport(handler)) as client:
            await client.health()
            await client.predict("img-1", b"png")
    asyncio.run(scenario())
    assert paths == ["/health", "/predict/preprocessed"]


def test_experiment3_rejects_an_old_server_before_processing_images() -> None:
    async def scenario():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok", "model_loaded": True}))
        async with InferenceHTTPClient("https://example.test", "secret-value-123",
                                       preprocess_on_client=True, transport=transport) as client:
            with pytest.raises(RemoteRequestError, match="rebuild"):
                await client.health()
    asyncio.run(scenario())

