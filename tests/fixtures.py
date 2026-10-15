"""Synthetic catalogue and simulated customer enquiry images.

This exists so the pipeline can be measured before Panache's real catalogue
and real enquiry images are available. It is deliberately adversarial: it
includes near-identical garments that differ only in trim, so the margin and
confidence logic is exercised rather than flattered.

Numbers measured here are NOT predictions of production accuracy. They
validate the mechanism. Real figures require the labelled image set.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter

from matcher.catalogue import Catalogue, Product, ProductImage, Variant
from matcher.config import CANONICAL_SIZES

CANVAS = (360, 480)
BACKDROP = (246, 243, 239)

PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (186, 32, 46),
    "maroon": (104, 24, 38),
    "navy": (24, 38, 84),
    "blue": (38, 84, 176),
    "teal": (18, 118, 118),
    "green": (36, 110, 62),
    "gold": (198, 150, 44),
    "pink": (216, 104, 148),
    "purple": (104, 52, 140),
    "black": (26, 26, 28),
}

SILHOUETTES = ("lehenga", "anarkali", "straight")
TRIMS = ("plain", "border", "dots", "stripes")


def _draw_garment(
    colour: tuple[int, int, int], silhouette: str, trim: str, seed: int
) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", CANVAS, BACKDROP)
    draw = ImageDraw.Draw(img)
    w, h = CANVAS

    # Simple bodice + skirt shapes, distinct enough per silhouette that a
    # texture/shape descriptor can tell them apart.
    if silhouette == "lehenga":
        body = [(w * 0.42, h * 0.08), (w * 0.58, h * 0.08), (w * 0.62, h * 0.40),
                (w * 0.88, h * 0.92), (w * 0.12, h * 0.92), (w * 0.38, h * 0.40)]
    elif silhouette == "anarkali":
        body = [(w * 0.40, h * 0.08), (w * 0.60, h * 0.08), (w * 0.72, h * 0.45),
                (w * 0.78, h * 0.92), (w * 0.22, h * 0.92), (w * 0.28, h * 0.45)]
    else:  # straight
        body = [(w * 0.38, h * 0.08), (w * 0.62, h * 0.08), (w * 0.66, h * 0.92),
                (w * 0.34, h * 0.92)]
    draw.polygon(body, fill=colour)

    accent = tuple(min(255, int(c * 0.55 + 110)) for c in colour)
    if trim == "border":
        draw.rectangle([w * 0.12, h * 0.84, w * 0.88, h * 0.92], fill=accent)
    elif trim == "dots":
        for _ in range(70):
            x = rng.uniform(w * 0.30, w * 0.70)
            y = rng.uniform(h * 0.15, h * 0.88)
            draw.ellipse([x, y, x + 6, y + 6], fill=accent)
    elif trim == "stripes":
        for y in range(int(h * 0.15), int(h * 0.90), 26):
            draw.line([(w * 0.20, y), (w * 0.80, y)], fill=accent, width=5)

    # A second photo angle is simulated by a slight shear and lighting shift.
    return img


def build_catalogue(n_products: int = 24, seed: int = 7) -> tuple[Catalogue, dict[str, Image.Image]]:
    """Returns the catalogue plus an image_id -> PIL.Image store.

    Includes deliberate near-twins: the same colour and silhouette differing
    only in trim, which is the realistic hard case for Panache.
    """
    rng = random.Random(seed)
    colours = list(PALETTE)
    products: list[Product] = []
    store: dict[str, Image.Image] = {}

    combos: list[tuple[str, str, str]] = []
    for i in range(n_products):
        colour = colours[i % len(colours)]
        silhouette = SILHOUETTES[i % len(SILHOUETTES)]
        trim = TRIMS[i % len(TRIMS)]
        combos.append((colour, silhouette, trim))
    # Force four near-twin pairs: same colour + silhouette, different trim.
    for i in range(0, 8, 2):
        colour, silhouette, _ = combos[i]
        combos[i + 1] = (colour, silhouette, TRIMS[(TRIMS.index(combos[i][2]) + 1) % len(TRIMS)])

    for i, (colour, silhouette, trim) in enumerate(combos):
        pid = f"p{i:03d}"
        handle = f"{colour}-{silhouette}-{trim}-{i:03d}"
        price = rng.choice([14900, 18900, 24900, 29900, 34900, 44900])

        variants = []
        for size in CANONICAL_SIZES:
            # Realistic stock: popular middle sizes more often available.
            stock = rng.choice([0, 0, 1, 2, 3, 5]) if size in ("38", "46") else rng.choice([0, 1, 2, 4, 6])
            variants.append(Variant(f"{pid}-{size}", size, stock))
        # Guarantee a handful of fully sold-out products to exercise that path.
        if i % 7 == 6:
            variants = [Variant(f"{pid}-{s}", s, 0) for s in CANONICAL_SIZES]

        images = []
        for shot in range(2):
            image_id = f"{pid}-img{shot}"
            img = _draw_garment(PALETTE[colour], silhouette, trim, seed=i * 10 + shot)
            if shot == 1:
                img = img.transform(
                    CANVAS, Image.AFFINE, (1, 0.07, -12, 0, 1, 0), fillcolor=BACKDROP
                ).point(lambda v: min(255, int(v * 1.06)))
            store[image_id] = img
            images.append(ProductImage(image_id=image_id, path=f"memory://{image_id}", position=shot))

        products.append(
            Product(
                product_id=pid,
                handle=handle,
                title=f"{colour.title()} {silhouette.title()} ({trim})",
                price_cents=price,
                variants=variants,
                images=images,
                tags=[colour, silhouette, trim, "ethnic"],
            )
        )

    return Catalogue(products=products), store


# --- customer image simulation ---

@dataclass(frozen=True)
class Distortion:
    name: str
    description: str


DISTORTIONS = (
    Distortion("clean", "exact catalogue photo, as if forwarded"),
    Distortion("screenshot", "Instagram-style chrome bars added, then JPEG"),
    Distortion("crop", "tight crop to the garment only"),
    Distortion("photo_of_screen", "blur + noise, as if photographing a monitor"),
    Distortion("rotated_small", "slight rotation and rescale"),
)


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def simulate_customer_image(
    source: Image.Image, distortion: str, seed: int = 0
) -> Image.Image:
    """Turn a catalogue photo into something resembling what arrives on WhatsApp."""
    rng = random.Random(seed)
    img = source.copy()

    if distortion == "clean":
        return _jpeg(img, 88)

    if distortion == "screenshot":
        w, h = img.size
        top, bottom = int(h * 0.13), int(h * 0.11)
        canvas = Image.new("RGB", (w, h + top + bottom), (255, 255, 255))
        canvas.paste(img, (0, top))
        draw = ImageDraw.Draw(canvas)
        draw.ellipse([10, top // 2 - 12, 34, top // 2 + 12], fill=(200, 200, 205))
        draw.text((44, top // 2 - 6), "panachebydivya", fill=(20, 20, 20))
        for i, x in enumerate(range(16, w - 16, max(1, (w - 32) // 5))):
            draw.rectangle([x, h + top + bottom // 2 - 9, x + 18, h + top + bottom // 2 + 9],
                           outline=(120, 120, 125), width=2)
        return _jpeg(canvas, 62)

    if distortion == "crop":
        w, h = img.size
        left = rng.uniform(0.10, 0.22)
        top_f = rng.uniform(0.04, 0.14)
        return _jpeg(img.crop((int(w * left), int(h * top_f),
                               int(w * (1 - rng.uniform(0.10, 0.22))),
                               int(h * (1 - rng.uniform(0.04, 0.14))))), 72)

    if distortion == "photo_of_screen":
        out = img.filter(ImageFilter.GaussianBlur(1.1))
        out = out.point(lambda v: min(255, int(v * 0.92 + 14)))
        noise = Image.effect_noise(out.size, 11).convert("L")
        out = Image.blend(out, Image.merge("RGB", (noise, noise, noise)), 0.07)
        return _jpeg(out, 58)

    if distortion == "rotated_small":
        out = img.rotate(rng.uniform(-6, 6), expand=True, fillcolor=BACKDROP)
        scale = rng.uniform(0.62, 0.85)
        out = out.resize((int(out.width * scale), int(out.height * scale)), Image.LANCZOS)
        return _jpeg(out, 70)

    raise ValueError(f"unknown distortion: {distortion}")


def foreign_garment(seed: int = 99) -> Image.Image:
    """A garment that is NOT in the catalogue - a competitor's product.

    The correct outcome for these is escalation, not a best-effort guess, so
    the eval needs them.
    """
    rng = random.Random(seed)
    colour = (rng.randint(60, 240), rng.randint(60, 240), rng.randint(60, 240))
    img = Image.new("RGB", CANVAS, (238, 236, 240))
    draw = ImageDraw.Draw(img)
    w, h = CANVAS
    draw.ellipse([w * 0.22, h * 0.10, w * 0.78, h * 0.94], fill=colour)
    for _ in range(40):
        x, y = rng.uniform(w * 0.25, w * 0.75), rng.uniform(h * 0.15, h * 0.90)
        draw.line([(x, y), (x + 14, y + 14)], fill=(250, 250, 250), width=3)
    return _jpeg(img, 76)


def memory_loader(store: dict[str, Image.Image]):
    def loader(product, product_image):
        return store.get(product_image.image_id)
    return loader
