from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.image_geometry import prepare_spatial_image


@dataclass(frozen=True)
class EncodedImage:
    image_id: str
    png_bytes: bytes
    read_ms: float
    encode_ms: float
    width: int
    height: int
    prepared_ns: int
    decode_ms: float = 0.0
    preprocess_ms: float = 0.0
    source_width: int = 0
    source_height: int = 0


def encode_image(
    path: Path, image_id: str, *, image_format: str = "PNG", jpeg_quality: int = 90,
    preprocess_on_client: bool = False,
) -> EncodedImage:
    if preprocess_on_client and image_format != "PNG":
        raise ValueError("Client preprocessing requires PNG")
    read_start = time.perf_counter_ns()
    source_bytes = path.read_bytes()
    read_ms = (time.perf_counter_ns() - read_start) / 1_000_000
    decode_start = time.perf_counter_ns()
    with Image.open(io.BytesIO(source_bytes)) as source:
        image = source.convert("RGB")
        image.load()
    decode_ms = (time.perf_counter_ns() - decode_start) / 1_000_000
    source_width, source_height = image.size
    preprocess_ms = 0.0
    if preprocess_on_client:
        preprocess_start = time.perf_counter_ns()
        image = prepare_spatial_image(image)
        preprocess_ms = (time.perf_counter_ns() - preprocess_start) / 1_000_000
    encode_start = time.perf_counter_ns()
    output = io.BytesIO()
    if image_format == "PNG":
        image.save(output, format="PNG", optimize=False)
    elif image_format == "JPEG":
        image.save(output, format="JPEG", quality=jpeg_quality, optimize=False)
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    encode_ms = (time.perf_counter_ns() - encode_start) / 1_000_000
    return EncodedImage(
        image_id=image_id, png_bytes=output.getvalue(), read_ms=read_ms, encode_ms=encode_ms,
        width=image.width, height=image.height, prepared_ns=time.perf_counter_ns(),
        decode_ms=decode_ms, preprocess_ms=preprocess_ms,
        source_width=source_width, source_height=source_height,
    )

