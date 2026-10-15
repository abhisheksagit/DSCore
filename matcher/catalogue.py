"""Catalogue model: what we know about a product, independent of Shopify.

Shopify is the source of truth, but the matcher reads this normalised shape so
it can be tested without network access and so a future catalogue source does
not ripple through the pipeline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import CANONICAL_SIZES, CURRENCY, STORE_BASE_URL

_SIZE_RE = re.compile(r"(\d{2})")


def normalise_size(raw: str) -> str | None:
    """Map a Shopify variant option value onto a canonical Panache size.

    Shopify stores whatever the merchant typed, so '42', 'Size 42' and
    '42 (M)' all have to land on '42'. Anything outside the known size set
    (free size, unstitched) returns None and is surfaced rather than guessed.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text in CANONICAL_SIZES:
        return text
    match = _SIZE_RE.search(text)
    if match and match.group(1) in CANONICAL_SIZES:
        return match.group(1)
    return None


@dataclass
class Variant:
    """One size of one garment."""

    variant_id: str
    size_raw: str
    inventory: int
    price_cents: int | None = None  # falls back to the product price

    @property
    def size(self) -> str | None:
        return normalise_size(self.size_raw)

    @property
    def in_stock(self) -> bool:
        return self.inventory > 0


@dataclass
class ProductImage:
    image_id: str
    path: str  # local cache path or URL
    position: int = 0


@dataclass
class Product:
    product_id: str
    handle: str
    title: str
    price_cents: int
    variants: list[Variant] = field(default_factory=list)
    images: list[ProductImage] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    # Derived at sync time, not per enquiry: needed to answer
    # "do you have this in another colour?" without re-reading every image.
    colour_family: str | None = None

    @property
    def url(self) -> str:
        return f"{STORE_BASE_URL}/products/{self.handle}"

    @property
    def price_display(self) -> str:
        return f"${self.price_cents / 100:,.2f} {CURRENCY}"

    def in_stock_sizes(self) -> list[str]:
        """Canonical sizes with stock, in catalogue order (38 -> 46)."""
        have = {v.size for v in self.variants if v.in_stock and v.size}
        return [s for s in CANONICAL_SIZES if s in have]

    def oos_sizes(self) -> list[str]:
        known = {v.size for v in self.variants if v.size}
        have = set(self.in_stock_sizes())
        return [s for s in CANONICAL_SIZES if s in known and s not in have]

    def unmapped_variants(self) -> list[Variant]:
        """Variants whose size we could not parse - a data-quality signal."""
        return [v for v in self.variants if v.size is None]

    def size_available(self, requested: str) -> bool | None:
        """True / False, or None when we do not stock that size at all."""
        size = normalise_size(requested)
        if size is None:
            return None
        known = {v.size for v in self.variants if v.size}
        if size not in known:
            return None
        return size in set(self.in_stock_sizes())

    @property
    def is_sellable(self) -> bool:
        return self.status == "active" and bool(self.in_stock_sizes())


@dataclass
class Catalogue:
    products: list[Product] = field(default_factory=list)

    def by_id(self, product_id: str) -> Product | None:
        return next((p for p in self.products if p.product_id == product_id), None)

    def by_handle(self, handle: str) -> Product | None:
        return next((p for p in self.products if p.handle == handle), None)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(p) for p in self.products], indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Catalogue":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        products = []
        for item in raw:
            variants = [Variant(**v) for v in item.pop("variants", [])]
            images = [ProductImage(**i) for i in item.pop("images", [])]
            products.append(Product(**item, variants=variants, images=images))
        return cls(products=products)
