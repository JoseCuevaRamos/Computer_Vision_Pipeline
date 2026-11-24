"""Pillow-only spatial transforms matching EfficientNet-B0 IMAGENET1K_V1.

Keep this module free of torch imports: the client runs without PyTorch.
"""
from __future__ import annotations

from PIL import Image


RESIZE_SIZE = 256
CROP_SIZE = 224
PREPROCESSING_ID = "efficientnet_b0_imagenet1k_v1_resize256_centercrop224"


def prepare_spatial_image(image: Image.Image) -> Image.Image:
    """Match torchvision's PIL Resize(256, BICUBIC) and CenterCrop(224)."""
    width, height = image.size
    if width <= height:
        target = (RESIZE_SIZE, int(RESIZE_SIZE * height / width))
    else:
        target = (int(RESIZE_SIZE * width / height), RESIZE_SIZE)
    if image.size != target:
        image = image.resize(target, resample=Image.Resampling.BICUBIC)
    left = int(round((image.width - CROP_SIZE) / 2.0))
    top = int(round((image.height - CROP_SIZE) / 2.0))
    return image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
