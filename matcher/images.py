"""Image loading and the crop ensemble that makes cropped screenshots work.

Customers rarely send a clean product photo. They send an Instagram screenshot
with the app's top and bottom bars attached, a reel still, a photo of a phone
screen, or an arbitrary crop. Rather than trying to detect and classify that
chrome (brittle, and it changes whenever Meta reskins the app), we strip
uniform borders and then match a small ensemble of crops, keeping the best
score. It is cheap and it degrades gracefully on inputs we have never seen.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MAX_EDGE = 1024  # downscale ceiling; matching never needs more


@dataclass(frozen=True)
class Crop:
    name: str
    image: Image.Image


def load(source: str | Path | bytes) -> Image.Image:
    """Load, honour EXIF orientation, drop alpha, bound the size."""
    if isinstance(source, bytes):
        img = Image.open(BytesIO(source))
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
    return img


def strip_uniform_borders(
    img: Image.Image, std_tol: float = 6.0, max_fraction: float = 0.45
) -> Image.Image:
    """Remove flat letterbox/padding bands around the edges.

    Only strips bands that are genuinely near-constant, and never more than
    `max_fraction` of each dimension, so a legitimately plain background does
    not get eaten.
    """
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    row_std = arr.reshape(h, -1).std(axis=1)
    col_std = arr.transpose(1, 0, 2).reshape(w, -1).std(axis=1)

    def advance(values: np.ndarray, limit: int) -> int:
        i = 0
        while i < limit and values[i] < std_tol:
            i += 1
        return i

    top = advance(row_std, int(h * max_fraction))
    bottom = advance(row_std[::-1], int(h * max_fraction))
    left = advance(col_std, int(w * max_fraction))
    right = advance(col_std[::-1], int(w * max_fraction))

    if top + bottom >= h or left + right >= w:
        return img
    if (top or bottom or left or right):
        return img.crop((left, top, w - right, h - bottom))
    return img


def _fractional(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    l, t, r, b = box
    return img.crop((int(w * l), int(h * t), int(w * r), int(h * b)))


def crop_ensemble(img: Image.Image) -> list[Crop]:
    """The views we match against the catalogue, best score wins.

    `full` and `trimmed` carry an uncropped screenshot. The band crops cover
    Instagram/browser chrome at the top and bottom. The body crops cover the
    common case of a customer cropping to just the garment, or to just the
    skirt or just the yoke.
    """
    trimmed = strip_uniform_borders(img)
    crops = [
        Crop("full", img),
        Crop("trimmed", trimmed),
        Crop("centre", _fractional(trimmed, (0.08, 0.08, 0.92, 0.92))),
        # Drop a phone status bar / app header and a bottom action bar.
        Crop("chrome_stripped", _fractional(img, (0.0, 0.10, 1.0, 0.90))),
        Crop("upper", _fractional(trimmed, (0.0, 0.0, 1.0, 0.62))),
        Crop("lower", _fractional(trimmed, (0.0, 0.38, 1.0, 1.0))),
    ]
    # Deduplicate identical geometry so we do not pay to embed the same pixels.
    seen: set[tuple[int, int, bytes]] = set()
    unique: list[Crop] = []
    for crop in crops:
        if min(crop.image.size) < 32:
            continue
        key = (
            crop.image.size[0],
            crop.image.size[1],
            crop.image.resize((16, 16)).tobytes(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(crop)
    return unique


def to_png_bytes(img: Image.Image, max_edge: int = 768) -> bytes:
    """Encode for sending to the verification model."""
    out = img.copy()
    if max(out.size) > max_edge:
        out.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
