from __future__ import annotations

from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import functional as F

from src.model import WEIGHTS


PREPROCESS = WEIGHTS.transforms()


def preprocess_image(image: Image.Image, *, spatial_preprocessed: bool = False) -> Tensor:
    if spatial_preprocessed:
        tensor = F.convert_image_dtype(F.pil_to_tensor(image), torch.float)
        return F.normalize(tensor, mean=PREPROCESS.mean, std=PREPROCESS.std).unsqueeze(0)
    return PREPROCESS(image).unsqueeze(0)

