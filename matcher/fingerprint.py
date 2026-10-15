"""Perceptual hashes and classical descriptors.

These need no model weights, which is why the near-duplicate stage works on
day one, before any SigLIP weights are installed on the VPS.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

HASH_BITS = 64


def _grey(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.LANCZOS), dtype=np.float32)


def dhash(img: Image.Image, side: int = 8) -> int:
    """Gradient hash: compares neighbouring pixels, so it survives rescaling
    and JPEG recompression well. 64 bits at side=8."""
    grid = _grey(img, (side + 1, side))
    bits = grid[:, 1:] > grid[:, :-1]
    return int("".join("1" if b else "0" for b in bits.flatten()), 2)


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= np.sqrt(1 / n)
    m[1:] *= np.sqrt(2 / n)
    return m


_DCT32 = _dct_matrix(32)


def phash(img: Image.Image, keep: int = 8) -> int:
    """DCT hash: keeps low-frequency structure, so it tolerates blur and the
    moire of a photo-of-a-screen better than dhash."""
    grid = _grey(img, (32, 32))
    freq = _DCT32 @ grid @ _DCT32.T
    block = freq[:keep, :keep].flatten()
    # Skip the DC term when choosing the threshold: it only encodes brightness.
    median = np.median(block[1:])
    bits = block > median
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def dup_distance(a_dhash: int, a_phash: int, b_dhash: int, b_phash: int) -> int:
    """Combined near-duplicate distance: the more optimistic of the two hashes.

    They fail on different inputs - dhash on blur, phash on tight crops - so
    taking the minimum makes the stage robust without lowering precision much.
    """
    return min(hamming(a_dhash, b_dhash), hamming(a_phash, b_phash))


def dup_score(distance: int) -> float:
    """Map hamming distance to 0..1. 0 bits different -> 1.0, >=24 -> 0.0."""
    return float(max(0.0, 1.0 - distance / 24.0))


# --- classical descriptors (the no-weights embedder, and colour family) ---

_HUE_BINS = 12
_SAT_BINS = 3
_VAL_BINS = 3


def hsv_histogram(img: Image.Image) -> np.ndarray:
    hsv = np.asarray(img.convert("HSV").resize((96, 96), Image.LANCZOS), dtype=np.float32)
    h = (hsv[..., 0] / 256.0 * _HUE_BINS).astype(int).clip(0, _HUE_BINS - 1)
    s = (hsv[..., 1] / 256.0 * _SAT_BINS).astype(int).clip(0, _SAT_BINS - 1)
    v = (hsv[..., 2] / 256.0 * _VAL_BINS).astype(int).clip(0, _VAL_BINS - 1)
    idx = (h * _SAT_BINS + s) * _VAL_BINS + v
    hist = np.bincount(idx.flatten(), minlength=_HUE_BINS * _SAT_BINS * _VAL_BINS)
    hist = hist.astype(np.float32)
    total = hist.sum()
    return hist / total if total else hist


def edge_orientation_histogram(img: Image.Image, bins: int = 12) -> np.ndarray:
    """Coarse texture/silhouette signal: separates a straight kurta from a
    flared lehenga, and plain fabric from heavy embroidery."""
    grid = _grey(img, (96, 96))
    gx = np.zeros_like(grid)
    gy = np.zeros_like(grid)
    gx[:, 1:-1] = grid[:, 2:] - grid[:, :-2]
    gy[1:-1, :] = grid[2:, :] - grid[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # 0..1
    idx = (ang * bins).astype(int).clip(0, bins - 1)
    hist = np.bincount(idx.flatten(), weights=mag.flatten(), minlength=bins).astype(np.float32)
    total = hist.sum()
    return hist / total if total else hist


def luma_grid(img: Image.Image, side: int = 8) -> np.ndarray:
    grid = _grey(img, (side, side)).flatten()
    rng = grid.max() - grid.min()
    return (grid - grid.min()) / rng if rng else np.zeros_like(grid)


_COLOUR_FAMILIES: tuple[tuple[str, float, float], ...] = (
    ("red", 0.0, 0.04),
    ("orange", 0.04, 0.09),
    ("gold", 0.09, 0.14),
    ("yellow", 0.14, 0.19),
    ("green", 0.19, 0.42),
    ("teal", 0.42, 0.52),
    ("blue", 0.52, 0.70),
    ("purple", 0.70, 0.80),
    ("pink", 0.80, 0.93),
    ("red", 0.93, 1.01),
)


def colour_family(img: Image.Image) -> str:
    """Dominant colour family, stored per product at sync time.

    Needed to answer "do you have this in another colour?" - which for Panache
    is a retrieval question, not a variant lookup, since each product is a
    single colourway.
    """
    hsv = np.asarray(img.convert("HSV").resize((96, 96), Image.LANCZOS), dtype=np.float32)
    h, s, v = hsv[..., 0] / 256.0, hsv[..., 1] / 256.0, hsv[..., 2] / 256.0
    # Neutrals first: hue is meaningless at low saturation.
    coloured = s > 0.22
    if coloured.mean() < 0.18:
        mean_v = float(v.mean())
        if mean_v < 0.22:
            return "black"
        if mean_v > 0.80:
            return "white"
        return "grey"
    hues = h[coloured]
    weights = (s * v)[coloured]
    # Circular mean so reds either side of the wrap do not average to cyan.
    angles = hues * 2 * np.pi
    x = float(np.sum(np.cos(angles) * weights))
    y = float(np.sum(np.sin(angles) * weights))
    mean_hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    mean_sat = float(np.average(s[coloured], weights=weights)) if weights.sum() else 0.0
    mean_val = float(np.average(v[coloured], weights=weights)) if weights.sum() else 0.0
    for name, lo, hi in _COLOUR_FAMILIES:
        if lo <= mean_hue < hi:
            if name == "red" and mean_val < 0.45:
                return "maroon"
            if name == "blue" and mean_val < 0.40:
                return "navy"
            if name == "gold" and mean_sat < 0.45:
                return "beige"
            return name
    return "multi"
