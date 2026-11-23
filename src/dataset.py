from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

from src.config import IMAGENETTE_CLASSES


MANIFEST_COLUMNS = [
    "image_id", "filename", "relative_path", "class_wnid", "class_name",
    "imagenet_class_id", "width", "height", "original_size_bytes", "sha256",
]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    filename: str
    relative_path: str
    class_wnid: str
    class_name: str
    imagenet_class_id: int
    width: int
    height: int
    original_size_bytes: int
    sha256: str

    @property
    def true_class_id(self) -> int:
        return self.imagenet_class_id


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_path(data_root: Path, relative_path: str) -> Path:
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe relative path in manifest: {relative_path}")
    candidate = data_root.joinpath(*posix_path.parts).resolve()
    root = data_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes dataset root: {relative_path}")
    return candidate


def load_manifest(
    manifest_path: Path,
    data_root: Path,
    *,
    verify_hashes: bool = True,
    require_complete: bool = True,
) -> list[ImageRecord]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run the prepare service first."
        )
    frame = pd.read_csv(manifest_path, dtype={"image_id": str, "imagenet_class_id": int})
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if require_complete and len(frame) != 500:
        raise ValueError(f"Manifest must contain 500 rows, found {len(frame)}")
    if frame["image_id"].duplicated().any() or frame["relative_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate image IDs or paths")

    counts = frame.groupby("class_wnid").size().to_dict()
    if require_complete and (
        set(counts) != set(IMAGENETTE_CLASSES) or any(count != 50 for count in counts.values())
    ):
        raise ValueError(f"Manifest must contain 50 images for every class; found {counts}")

    records: list[ImageRecord] = []
    for row in frame[MANIFEST_COLUMNS].to_dict(orient="records"):
        record = ImageRecord(
            image_id=str(row["image_id"]), filename=str(row["filename"]),
            relative_path=str(row["relative_path"]), class_wnid=str(row["class_wnid"]),
            class_name=str(row["class_name"]), imagenet_class_id=int(row["imagenet_class_id"]),
            width=int(row["width"]), height=int(row["height"]),
            original_size_bytes=int(row["original_size_bytes"]), sha256=str(row["sha256"]),
        )
        expected = IMAGENETTE_CLASSES.get(record.class_wnid)
        if expected is None or expected["class_name"] != record.class_name:
            raise ValueError(f"Invalid Imagenette class metadata for {record.image_id}")
        if expected["imagenet_class_id"] != record.imagenet_class_id:
            raise ValueError(f"Invalid ImageNet ID for {record.image_id}")
        path = resolve_dataset_path(data_root, record.relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image is missing: {path}")
        if path.stat().st_size != record.original_size_bytes:
            raise ValueError(f"File size changed for {record.relative_path}")
        if verify_hashes and sha256_file(path) != record.sha256:
            raise ValueError(f"SHA-256 mismatch for {record.relative_path}")
        records.append(record)
    return records

