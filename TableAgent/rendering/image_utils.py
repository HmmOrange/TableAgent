from __future__ import annotations

from pathlib import Path
from typing import Any


def compute_viewport_and_scale(
    estimated_width: int,
    estimated_height: int,
    image_scale: float,
    max_viewport_width: int,
    max_viewport_height: int,
    max_image_dimension: int | None,
    max_image_pixels: int | None,
) -> tuple[int, int, float]:
    viewport_width = min(max(320, estimated_width), max_viewport_width)
    viewport_height = min(max(240, estimated_height), max_viewport_height)
    scale = image_scale

    if max_image_dimension is not None and max(viewport_width, viewport_height) * scale > max_image_dimension:
        scale = max_image_dimension / max(viewport_width, viewport_height)
    if max_image_pixels is not None:
        base_pixels = viewport_width * viewport_height
        if base_pixels * (scale ** 2) > max_image_pixels:
            scale = (max_image_pixels / base_pixels) ** 0.5

    return viewport_width, viewport_height, max(0.1, scale)


def _generate_image_tiles(
    image_path: Path,
    tile_size: int,
    overlap: int,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        if not image_path.is_file():
            return []
        with Image.open(image_path) as image:
            width, height = image.size
            if width <= tile_size and height <= tile_size:
                return []
            step = max(1, tile_size - overlap)
            x_coords = _tile_starts(width, tile_size, step)
            y_coords = _tile_starts(height, tile_size, step)
            tiles: list[dict[str, Any]] = []
            for row_index, y_start in enumerate(y_coords):
                for col_index, x_start in enumerate(x_coords):
                    x_end = min(width, x_start + tile_size)
                    y_end = min(height, y_start + tile_size)
                    filename = f"table_tile_{row_index}_{col_index}.png"
                    image.crop((x_start, y_start, x_end, y_end)).save(image_path.parent / filename)
                    tiles.append({
                        "filename": filename,
                        "x_start": x_start,
                        "y_start": y_start,
                        "width": x_end - x_start,
                        "height": y_end - y_start,
                    })
            return tiles
    except Exception as exc:
        if logger:
            logger.error(f"Failed to generate image tiles for {image_path}: {exc}")
        return []


def _resize_image_file_to_fit(
    image_path: Path,
    max_dim: int | None = None,
    max_pixels: int | None = None,
    logger: Any | None = None,
) -> None:
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        if not image_path.is_file():
            return
        with Image.open(image_path) as image:
            width, height = image.size
            ratio = 1.0
            if max_dim is not None and max(width, height) > max_dim:
                ratio = min(ratio, max_dim / max(width, height))
            if max_pixels is not None and width * height * (ratio ** 2) > max_pixels:
                ratio = min(ratio, (max_pixels / (width * height)) ** 0.5)
            if ratio >= 1.0:
                return
            new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            resampling = getattr(getattr(Image, "Resampling", None), "LANCZOS", getattr(Image, "LANCZOS", 1))
            image.resize(new_size, resample=resampling).save(image_path)
    except Exception as exc:
        if logger:
            logger.error(f"Failed to resize image {image_path}: {exc}")


def _tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    starts = []
    current = 0
    while current < length:
        starts.append(current)
        if current + tile_size >= length:
            break
        current += step
    return starts
