import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from src.client.http_client import InferenceHTTPClient, RemoteRequestError


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

