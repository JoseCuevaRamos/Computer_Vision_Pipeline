from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import psutil


def _read_number(paths: list[Path]) -> int | None:
    for path in paths:
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value != "max":
                return int(value)
        except (OSError, ValueError):
            continue
    return None


def container_memory_mb() -> float | None:
    value = _read_number([Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")])
    return None if value is None else value / (1024 * 1024)


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_perf = 0.0
        self._cpu_start = 0.0
        self.available_memory_before_mb = psutil.virtual_memory().available / (1024 * 1024)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Resource monitor has already been started")
        self._start_perf = time.perf_counter()
        times = self.process.cpu_times()
        self._cpu_start = times.user + times.system
        self.process.cpu_percent(interval=None)
        self._take_sample()
        self._thread = threading.Thread(target=self._sample_loop, name="resource-monitor", daemon=True)
        self._thread.start()

    def _take_sample(self) -> None:
        self.samples.append(
            {
                "timestamp": time.time(),
                "elapsed_seconds": time.perf_counter() - self._start_perf,
                "process_cpu_percent": self.process.cpu_percent(interval=None),
                "process_rss_mb": self.process.memory_info().rss / (1024 * 1024),
                "container_memory_mb": container_memory_mb(),
            }
        )

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._take_sample()
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 3))
        times = self.process.cpu_times()
        cpu_seconds = times.user + times.system - self._cpu_start
        cpu_values = [float(sample["process_cpu_percent"]) for sample in self.samples]
        rss_values = [float(sample["process_rss_mb"]) for sample in self.samples]
        return {
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "process_cpu_percent_mean": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            "process_cpu_percent_max": max(cpu_values, default=0.0),
            "process_cpu_time_seconds": cpu_seconds,
            "process_rss_mb_mean": sum(rss_values) / len(rss_values) if rss_values else 0.0,
            "process_rss_mb_max": max(rss_values, default=0.0),
            "available_memory_before_mb": self.available_memory_before_mb,
        }

