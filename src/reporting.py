from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import torch
import torchvision

from src.model import WEIGHTS
from src.resource_monitor import _read_number


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _limit_mb(paths: list[Path]) -> float | None:
    value = _read_number(paths)
    if value is None or value >= 2**60:
        return None
    return value / (1024 * 1024)


def system_info() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    cpu_quota = _read_number([Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")])
    cpu_period = _read_number([Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")])
    cpu_max = None
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return {
        "processor": platform.processor() or "unavailable from container",
        "architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_mb": memory.total / (1024 * 1024),
        "host_os": os.environ.get("HOST_OS", "unavailable from container"),
        "container_os": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "model": "EfficientNet-B0",
        "weights": "IMAGENET1K_V1",
        "weights_url": WEIGHTS.url,
        "docker_image_id": os.environ.get("DOCKER_IMAGE_ID", "unavailable from container"),
        "docker_version": os.environ.get("DOCKER_VERSION", "unavailable from container"),
        "compose_version": os.environ.get("COMPOSE_VERSION", "unavailable from container"),
        "cgroup": {
            "cpu_max_v2": cpu_max,
            "cpu_quota_v1": cpu_quota,
            "cpu_period_v1": cpu_period,
            "effective_cpu_quota_v1": (cpu_quota / cpu_period) if cpu_quota and cpu_period and cpu_quota > 0 else None,
            "memory_limit_mb": _limit_mb([Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")]),
        },
    }
