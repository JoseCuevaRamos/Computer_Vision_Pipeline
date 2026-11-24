from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1


def cached_weights_path(torch_home: Path) -> Path:
    return torch_home / "hub" / "checkpoints" / Path(urlparse(WEIGHTS.url).path).name


def load_model(
    device: str = "cpu",
    *,
    torch_threads: int | None = None,
    require_cached: bool = True,
) -> tuple[torch.nn.Module, float]:
    if device != "cpu":
        raise ValueError("This project configuration is CPU-only")
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    if require_cached:
        torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        cache_path = cached_weights_path(torch_home)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Pretrained weights are not cached at {cache_path}. Run the prepare service first."
            )
    start = time.perf_counter_ns()
    model = efficientnet_b0(weights=WEIGHTS)
    model.eval().to(device)
    load_ms = (time.perf_counter_ns() - start) / 1_000_000
    return model, load_ms


def validate_model_output(model: torch.nn.Module) -> None:
    sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    with torch.inference_mode():
        output = model(sample)
    if tuple(output.shape) != (1, 1000):
        raise RuntimeError(f"Expected output shape (1, 1000), got {tuple(output.shape)}")

