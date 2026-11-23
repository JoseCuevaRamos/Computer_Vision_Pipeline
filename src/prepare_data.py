from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision.datasets import Imagenette
from torchvision.models import efficientnet_b0

from src.config import IMAGENETTE_CLASSES
from src.dataset import MANIFEST_COLUMNS, load_manifest, sha256_file
from src.model import WEIGHTS, cached_weights_path, validate_model_output


def build_manifest(data_root: Path, manifest_path: Path, seed: int = 42) -> None:
    if manifest_path.exists():
        load_manifest(manifest_path, data_root, verify_hashes=True, require_complete=True)
        print(f"Existing manifest verified: {manifest_path}")
        return

    validation_root = data_root / "imagenette2-320" / "val"
    if not validation_root.is_dir():
        raise FileNotFoundError(f"Imagenette validation split not found at {validation_root}")
    class_dirs = {path.name: path for path in validation_root.iterdir() if path.is_dir()}
    if set(class_dirs) != set(IMAGENETTE_CLASSES):
        raise ValueError(f"Unexpected Imagenette class directories: {sorted(class_dirs)}")

    rng = random.Random(seed)
    selected: list[tuple[str, Path]] = []
    for wnid in sorted(IMAGENETTE_CLASSES):
        images = sorted(class_dirs[wnid].glob("*.JPEG"), key=lambda path: path.name)
        if len(images) < 50:
            raise ValueError(f"Class {wnid} has only {len(images)} images")
        selected.extend((wnid, path) for path in sorted(rng.sample(images, 50)))

    rows: list[dict[str, object]] = []
    for index, (wnid, path) in enumerate(selected, start=1):
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        metadata = IMAGENETTE_CLASSES[wnid]
        rows.append(
            {
                "image_id": f"img_{index:06d}",
                "filename": path.name,
                "relative_path": path.relative_to(data_root).as_posix(),
                "class_wnid": wnid,
                "class_name": metadata["class_name"],
                "imagenet_class_id": metadata["imagenet_class_id"],
                "width": width,
                "height": height,
                "original_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if len(frame) != 500 or frame["relative_path"].duplicated().any():
        raise RuntimeError("The deterministic selection did not produce 500 unique images")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(manifest_path)
    load_manifest(manifest_path, data_root, verify_hashes=True, require_complete=True)
    print(f"Created and verified manifest: {manifest_path}")


def prepare(data_root: Path, manifest_path: Path, seed: int = 42) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    print("Checking Imagenette 320 px...")
    dataset = Imagenette(root=data_root, split="val", size="320px", download=True)
    if len(dataset) < 500:
        raise RuntimeError(f"Imagenette validation split has only {len(dataset)} images")

    torch_home = Path(os.environ.get("TORCH_HOME", "/app/model_cache"))
    torch_home.mkdir(parents=True, exist_ok=True)
    print(f"Checking EfficientNet-B0 weights in {torch_home}...")
    model = efficientnet_b0(weights=WEIGHTS)
    model.eval()
    validate_model_output(model)
    cache_path = cached_weights_path(torch_home)
    if not cache_path.is_file():
        raise RuntimeError(f"Weights were loaded but not found in expected cache: {cache_path}")

    categories = list(WEIGHTS.meta["categories"])
    for wnid, metadata in IMAGENETTE_CLASSES.items():
        class_id = int(metadata["imagenet_class_id"])
        expected_name = str(metadata["class_name"])
        if categories[class_id] != expected_name:
            raise RuntimeError(
                f"ImageNet mapping mismatch for {wnid}: {class_id} is {categories[class_id]!r}, "
                f"expected {expected_name!r}"
            )
    build_manifest(data_root, manifest_path, seed)
    print("Preparation complete: dataset, weights, mappings and manifest are ready.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Imagenette and EfficientNet-B0 weights")
    parser.add_argument("--data-root", type=Path, default=Path("/app/data/imagenette"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/app/results/experiment1/dataset_manifest.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare(arguments.data_root, arguments.manifest, arguments.seed)

