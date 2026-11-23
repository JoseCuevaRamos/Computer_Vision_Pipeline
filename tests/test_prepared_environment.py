from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.config import IMAGENETTE_CLASSES
from src.dataset import load_manifest


DATA_ROOT = Path("/app/data/imagenette")
MANIFEST = Path("/app/results/experiment1/dataset_manifest.csv")


def test_prepared_dataset_has_500_verified_images() -> None:
    records = load_manifest(MANIFEST, DATA_ROOT, verify_hashes=True, require_complete=True)
    assert len(records) == 500
    assert len({record.relative_path for record in records}) == 500
    assert Counter(record.class_wnid for record in records) == Counter(
        {wnid: 50 for wnid in IMAGENETTE_CLASSES}
    )
