from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class EncodedImage:
    image_id: str
    png_bytes: bytes
    read_ms: float
    encode_ms: float
    width: int
    height: int
    prepared_ns: int


def encode_image(path: Path, image_id: str, *, image_format: str = "PNG", jpeg_quality: int = 90) -> EncodedImage:
    read_start = time.perf_counter_ns()
    source_bytes = path.read_bytes()
    read_ms = (time.perf_counter_ns() - read_start) / 1_000_000
    encode_start = time.perf_counter_ns()
    with Image.open(io.BytesIO(source_bytes)) as source:
        image = source.convert("RGB")
        image.load()
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
    )

