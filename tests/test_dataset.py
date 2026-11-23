from __future__ import annotations

from pathlib import Path

import pytest

from experiments.experiment1_local import balanced_subset
from src.dataset import ImageRecord, resolve_dataset_path


def make_record(index: int, wnid: str) -> ImageRecord:
    return ImageRecord(
        image_id=f"img_{index:06d}", filename=f"{index}.JPEG",
        relative_path=f"imagenette2-320/val/{wnid}/{index}.JPEG",
        class_wnid=wnid, class_name=wnid, imagenet_class_id=index,
        width=320, height=320, original_size_bytes=1, sha256="0" * 64,
    )


def test_quick_subset_is_balanced_across_ten_classes() -> None:
    records = [make_record(class_index * 50 + item, f"class_{class_index}") for class_index in range(10) for item in range(50)]
    selected = balanced_subset(records, 10)
    assert len(selected) == 10
    assert len({record.class_wnid for record in selected}) == 10


def test_path_resolution_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_dataset_path(tmp_path, "../outside.JPEG")

