from __future__ import annotations

import torch

from src.config import IMAGENETTE_CLASSES
from src.model import WEIGHTS, load_model


def test_efficientnet_output_has_1000_logits_and_no_training() -> None:
    model, _ = load_model(require_cached=True)
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 224, 224)))
    assert output.shape == (1, 1000)
    assert not model.training


def test_imagenette_mapping_matches_weight_categories() -> None:
    categories = WEIGHTS.meta["categories"]
    for metadata in IMAGENETTE_CLASSES.values():
        assert categories[metadata["imagenet_class_id"]] == metadata["class_name"]

