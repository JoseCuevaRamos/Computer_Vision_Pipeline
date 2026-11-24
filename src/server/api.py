from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status

from src.reporting import system_info, write_csv, write_json
from src.resource_monitor import ResourceMonitor
from src.server.prediction_service import InvalidImageError, PredictionService


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
LOGGER = logging.getLogger("experiment2.server")


class RequestRecorder:
    FIELDNAMES = [
        "timestamp", "image_id", "payload_bytes", "predicted_class_id",
        "queue_wait_ms", "decode_ms", "preprocess_ms", "inference_ms",
        "postprocess_ms", "total_internal_ms", "request_handler_ms", "status", "error",
    ]

    def __init__(self, path: Path, flush_size: int = 25) -> None:
        self.path = path
        self.flush_size = flush_size
        self.pending: list[dict] = []
        self.lock = threading.Lock()

    def add(self, row: dict) -> None:
        with self.lock:
            self.pending.append(row)
            if len(self.pending) >= self.flush_size:
                self._flush_locked()

    def flush(self) -> None:
        with self.lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self.pending:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDNAMES, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(self.pending)
        self.pending.clear()


def create_app(service_factory: Callable[[], PredictionService] | None = None) -> FastAPI:
    results_dir = Path(os.environ.get("SERVER_RESULTS_DIR", "/app/results/server"))
    recorder = RequestRecorder(results_dir / "requests.csv")
    monitor = ResourceMonitor(float(os.environ.get("RESOURCE_INTERVAL", "0.2")))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        api_key = os.environ.get("API_KEY", "")
        if len(api_key) < 12:
            raise RuntimeError("API_KEY must contain at least 12 characters")
        factory = service_factory or (lambda: PredictionService(torch_threads=_optional_int("TORCH_THREADS")))
        application.state.prediction_service = factory()
        application.state.api_key = api_key
        application.state.request_recorder = recorder
        monitor.start()
        info = system_info()
        info["model_load_ms"] = application.state.prediction_service.model_load_ms
        write_json(results_dir / "system_info.json", info)
        LOGGER.info(json.dumps({"event": "server_ready", "model": "efficientnet_b0"}))
        try:
            yield
        finally:
            recorder.flush()
            summary = monitor.stop()
            write_csv(results_dir / "resources.csv", pd.DataFrame(monitor.samples))
            write_json(results_dir / "resource_summary.json", summary)

    application = FastAPI(title="Experiment 2 PNG inference API", lifespan=lifespan)

    @application.get("/health")
    async def health(request: Request) -> dict:
        service = getattr(request.app.state, "prediction_service", None)
        return {
            "status": "ok",
            "model_loaded": service is not None,
            "model": "efficientnet_b0",
            "weights": "IMAGENET1K_V1",
        }

    @application.post("/predict")
    async def predict(
        request: Request,
        image_id: str = Form(..., min_length=1, max_length=200),
        image: UploadFile = File(...),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict:
        if not x_api_key or not _constant_time_equal(x_api_key, request.app.state.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        if image.content_type != "image/png":
            raise HTTPException(status_code=415, detail="Only image/png is accepted")
        max_bytes = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
        payload = await image.read(max_bytes + 1)
        if not payload or len(payload) > max_bytes:
            raise HTTPException(status_code=413, detail="Image payload is empty or too large")
        request_start = time.perf_counter_ns()
        try:
            result = await request.app.state.prediction_service.predict_png(payload)
        except InvalidImageError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            recorder.add({
                "timestamp": time.time(), "image_id": image_id, "payload_bytes": len(payload),
                "predicted_class_id": "", "request_handler_ms":
                    (time.perf_counter_ns() - request_start) / 1_000_000,
                "status": "error", "error": type(error).__name__,
            })
            LOGGER.exception(json.dumps({"event": "prediction", "image_id": image_id, "status": "error"}))
            raise HTTPException(status_code=500, detail="Internal inference error") from error
        result["image_id"] = image_id
        metrics = result["server_metrics"]
        recorder.add({
            "timestamp": time.time(), "image_id": image_id, "payload_bytes": len(payload),
            "predicted_class_id": result["predicted_class_id"],
            **metrics, "request_handler_ms": (time.perf_counter_ns() - request_start) / 1_000_000,
            "status": "ok", "error": "",
        })
        LOGGER.info(json.dumps({"event": "prediction", "image_id": image_id, "status": "ok"}))
        return result

    return application


def _constant_time_equal(value: str, expected: str) -> bool:
    import hmac
    return hmac.compare_digest(value.encode(), expected.encode())


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


app = create_app()
