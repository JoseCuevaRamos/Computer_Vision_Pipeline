from __future__ import annotations

from PIL import Image
from torch import Tensor

from src.model import WEIGHTS


PREPROCESS = WEIGHTS.transforms()


def preprocess_image(image: Image.Image) -> Tensor:
    return PREPROCESS(image).unsqueeze(0)

