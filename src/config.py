from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path = Path("/app/data/imagenette")
    results_root: Path = Path("/app/results")
    manifest_path: Path = Path("/app/results/experiment1/dataset_manifest.csv")
    output_subdir: str = "experiment1"
    num_images: int = 500
    runs: int = 5
    warmup: int = 20
    seed: int = 42
    device: str = "cpu"
    torch_threads: int | None = None
    resource_interval: float = 0.2
    verify_hashes: bool = True
    overwrite: bool = False

    @property
    def output_dir(self) -> Path:
        return self.results_root / self.output_subdir

    def validate(self) -> None:
        if self.device != "cpu":
            raise ValueError("Experiment 1 only supports --device cpu")
        if not 1 <= self.num_images <= 500:
            raise ValueError("--num-images must be between 1 and 500")
        if self.runs < 1:
            raise ValueError("--runs must be at least 1")
        if self.warmup < 0:
            raise ValueError("--warmup cannot be negative")
        if self.torch_threads is not None and self.torch_threads < 1:
            raise ValueError("--torch-threads must be positive")
        if self.resource_interval <= 0:
            raise ValueError("--resource-interval must be positive")
        if not self.output_subdir or Path(self.output_subdir).is_absolute():
            raise ValueError("--output-subdir must be a relative directory name")

    def as_json(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = value.as_posix()
        return values


IMAGENETTE_CLASSES: dict[str, dict[str, Any]] = {
    "n01440764": {"class_name": "tench", "imagenet_class_id": 0},
    "n02102040": {"class_name": "English springer", "imagenet_class_id": 217},
    "n02979186": {"class_name": "cassette player", "imagenet_class_id": 482},
    "n03000684": {"class_name": "chain saw", "imagenet_class_id": 491},
    "n03028079": {"class_name": "church", "imagenet_class_id": 497},
    "n03394916": {"class_name": "French horn", "imagenet_class_id": 566},
    "n03417042": {"class_name": "garbage truck", "imagenet_class_id": 569},
    "n03425413": {"class_name": "gas pump", "imagenet_class_id": 571},
    "n03445777": {"class_name": "golf ball", "imagenet_class_id": 574},
    "n03888257": {"class_name": "parachute", "imagenet_class_id": 701},
}

