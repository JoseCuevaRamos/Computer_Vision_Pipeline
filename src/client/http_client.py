from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from src.image_geometry import PREPROCESSING_ID


class RemoteRequestError(RuntimeError):
    def __init__(self, message: str, *, retries: int = 0) -> None:
        super().__init__(message)
        self.retries = retries


@dataclass(frozen=True)
class HTTPResult:
    body: dict[str, Any]
    http_total_ms: float
    response_bytes: int
    retries: int


class InferenceHTTPClient:
    def __init__(
        self,
        server_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        require_https: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        preprocess_on_client: bool = False,
    ) -> None:
        base_url = server_url.rstrip("/")
        parsed = urlparse(base_url)
        if require_https and parsed.scheme != "https":
            raise ValueError("SERVER_URL must use HTTPS")
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("SERVER_URL is invalid")
        if not api_key:
            raise ValueError("API_KEY cannot be empty")
        self.max_retries = max_retries
        self.preprocess_on_client = preprocess_on_client
        self.prediction_path = "/predict/preprocessed" if preprocess_on_client else "/predict"
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"X-API-Key": api_key, "ngrok-skip-browser-warning": "true"},
            transport=transport,
        )

    async def __aenter__(self) -> "InferenceHTTPClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.client.aclose()

    async def health(self) -> dict[str, Any]:
        response = await self.client.get("/health")
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok" or not payload.get("model_loaded"):
            raise RemoteRequestError(f"Server is not ready: {payload}")
        if self.preprocess_on_client and payload.get("preprocessing_id") != PREPROCESSING_ID:
            raise RemoteRequestError("Server does not support experiment 3; rebuild and restart the server")
        return payload

    async def predict(self, image_id: str, png_bytes: bytes) -> HTTPResult:
        start = time.perf_counter_ns()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(
                    self.prediction_path,
                    data={"image_id": image_id},
                    files={"image": (f"{image_id}.png", png_bytes, "image/png")},
                )
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                response.raise_for_status()
                payload = response.json()
                self._validate_response(payload, image_id)
                if self.preprocess_on_client and payload.get("preprocessing_id") != PREPROCESSING_ID:
                    raise ValueError("Server did not confirm the experiment 3 preprocessing contract")
                return HTTPResult(
                    body=payload,
                    http_total_ms=(time.perf_counter_ns() - start) / 1_000_000,
                    response_bytes=len(response.content),
                    retries=attempt,
                )
            except (httpx.TransportError, httpx.TimeoutException) as error:
                last_error = error
                if attempt < self.max_retries:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                break
            except (httpx.HTTPStatusError, ValueError, KeyError) as error:
                raise RemoteRequestError(
                    f"Request for {image_id} failed: {error}", retries=attempt
                ) from error
        raise RemoteRequestError(
            f"Request for {image_id} failed after retries: {last_error}", retries=self.max_retries
        )

    @staticmethod
    def _validate_response(payload: dict[str, Any], expected_image_id: str) -> None:
        required = {
            "image_id", "model", "predicted_class_id", "predicted_class_name",
            "confidence_top1", "top5_ids", "top5_probabilities", "server_metrics",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"Response is missing fields: {sorted(missing)}")
        if payload["image_id"] != expected_image_id:
            raise ValueError("Response image_id does not match request")
        if len(payload["top5_ids"]) != 5 or len(payload["top5_probabilities"]) != 5:
            raise ValueError("Response must contain five predictions")

