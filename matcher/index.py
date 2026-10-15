"""Product index and hybrid retrieval.

Three independent signals are fused, because any one of them has a failure
mode the others cover:

* near-duplicate hashes catch screenshots of our own photography (the single
  most common enquiry) with very high precision;
* embeddings generalise to a different angle, crop or lighting;
* text (OCR'd from the screenshot, plus whatever the customer typed) resolves
  cases where two garments genuinely look alike - an Instagram screenshot
  usually carries the caption, a website screenshot the product name.

Scored per product, taking the best over (query crop x product image), because
a product has several photos and we only need one of them to match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import fingerprint as fp
from . import images as im
from .catalogue import Catalogue, Product
from .config import Config, DEFAULT
from .embed import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or of in for with this that is are do you have it its size price "
    "please pls pp cost how much rs aud available avail stock any".split()
)


def tokenise(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


@dataclass
class IndexedImage:
    product_id: str
    image_id: str
    vector: np.ndarray
    dhash: int
    phash: int
    colour_family: str
    # Which view of the catalogue photo this entry represents. We index a
    # small crop ensemble per photo, mirroring what we do to the query, so a
    # customer's tight crop is compared against a comparably tight catalogue
    # crop instead of against a full frame dominated by studio backdrop.
    view: str = "full"


@dataclass
class ScoreBreakdown:
    """Kept on every candidate so the audit log can explain any decision."""

    fused: float
    near_duplicate: float
    embedding: float
    text: float
    dup_distance: int
    best_image_id: str
    best_crop: str
    best_view: str

    def as_dict(self) -> dict:
        return {
            "fused": round(self.fused, 4),
            "near_duplicate": round(self.near_duplicate, 4),
            "embedding": round(self.embedding, 4),
            "text": round(self.text, 4),
            "dup_distance": self.dup_distance,
            "best_image_id": self.best_image_id,
            "best_crop": self.best_crop,
            "best_view": self.best_view,
        }


@dataclass
class Candidate:
    product: Product
    score: ScoreBreakdown


@dataclass
class QueryFeatures:
    crops: list[im.Crop]
    vectors: np.ndarray  # (n_crops, dim)
    hashes: list[tuple[int, int]]
    tokens: set[str] = field(default_factory=set)
    colour_family: str = "unknown"


class ProductIndex:
    """In-memory index. Brute force is correct here: a boutique catalogue is
    hundreds to low thousands of images, where a scan is sub-millisecond and
    an approximate index would only add a failure mode. The pgvector schema in
    schema/001_init.sql is the durable store and takes over at scale.
    """

    def __init__(self, catalogue: Catalogue, embedder: Embedder, config: Config = DEFAULT) -> None:
        self.catalogue = catalogue
        self.embedder = embedder
        self.config = config
        self.entries: list[IndexedImage] = []
        self._matrix: np.ndarray | None = None
        self._product_tokens: dict[str, set[str]] = {}

    # --- building ---

    # Catalogue-side views. Deliberately fewer than the query ensemble: these
    # multiply index size, and the query side already covers chrome stripping.
    CATALOGUE_VIEWS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
        ("full", (0.0, 0.0, 1.0, 1.0)),
        ("centre", (0.14, 0.06, 0.86, 0.96)),
        ("upper", (0.10, 0.02, 0.90, 0.62)),
        ("lower", (0.06, 0.38, 0.94, 0.98)),
    )

    def add_product_image(self, product: Product, image_id: str, image: Image.Image) -> None:
        width, height = image.size
        for view, (l, t_, r, b) in self.CATALOGUE_VIEWS:
            view_img = (
                image
                if view == "full"
                else image.crop((int(width * l), int(height * t_), int(width * r), int(height * b)))
            )
            if min(view_img.size) < 32:
                continue
            self.entries.append(
                IndexedImage(
                    product_id=product.product_id,
                    image_id=image_id,
                    vector=self.embedder.embed(view_img),
                    dhash=fp.dhash(view_img),
                    phash=fp.phash(view_img),
                    colour_family=fp.colour_family(view_img),
                    view=view,
                )
            )
        self._matrix = None

    def build(self, loader) -> "ProductIndex":
        """`loader(product, product_image) -> PIL.Image | None`.

        A product whose images all fail to load is indexed on text alone and
        reported by `coverage()`, never silently dropped.
        """
        for product in self.catalogue.products:
            for product_image in product.images:
                image = loader(product, product_image)
                if image is None:
                    continue
                self.add_product_image(product, product_image.image_id, image)
            self._product_tokens[product.product_id] = tokenise(
                " ".join([product.title, *product.tags, product.handle.replace("-", " ")])
            )
            if product.colour_family is None:
                families = [
                    e.colour_family
                    for e in self.entries
                    if e.product_id == product.product_id and e.view == "full"
                ]
                if families:
                    product.colour_family = max(set(families), key=families.count)
        return self

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (
                np.vstack([e.vector for e in self.entries])
                if self.entries
                else np.zeros((0, self.embedder.dim), dtype=np.float32)
            )
        return self._matrix

    def coverage(self) -> dict:
        indexed = {e.product_id for e in self.entries}
        missing = [p.handle for p in self.catalogue.products if p.product_id not in indexed]
        return {
            "products": len(self.catalogue.products),
            "indexed_images": len({(e.product_id, e.image_id) for e in self.entries}),
            "index_entries": len(self.entries),
            "products_without_images": missing,
            "embedder": self.embedder.name,
        }

    # --- querying ---

    def features(self, image: Image.Image, message_text: str | None = None,
                 ocr_text: str | None = None) -> QueryFeatures:
        crops = im.crop_ensemble(image)
        vectors = np.vstack([self.embedder.embed(c.image) for c in crops])
        hashes = [(fp.dhash(c.image), fp.phash(c.image)) for c in crops]
        return QueryFeatures(
            crops=crops,
            vectors=vectors,
            hashes=hashes,
            tokens=tokenise(f"{message_text or ''} {ocr_text or ''}"),
            colour_family=fp.colour_family(crops[1].image if len(crops) > 1 else image),
        )

    def search(self, query: QueryFeatures, limit: int = 8) -> list[Candidate]:
        if not self.entries:
            return []

        # (n_crops, n_images) cosine similarities in one matmul.
        sims = query.vectors @ self.matrix.T

        best: dict[str, ScoreBreakdown] = {}
        weights = self.config.weights

        for col, entry in enumerate(self.entries):
            emb_scores = sims[:, col]
            crop_idx = int(np.argmax(emb_scores))
            emb = float(emb_scores[crop_idx])

            distance = min(
                fp.dup_distance(qd, qp, entry.dhash, entry.phash)
                for qd, qp in query.hashes
            )
            dup = fp.dup_score(distance)

            product_tokens = self._product_tokens.get(entry.product_id, set())
            if query.tokens and product_tokens:
                overlap = len(query.tokens & product_tokens)
                text = overlap / len(product_tokens) if overlap else 0.0
                text = min(1.0, text)
            else:
                text = 0.0

            # Re-normalise over present signals: with no text available, text
            # weight is redistributed rather than quietly capping the score.
            active = [(weights.near_duplicate, dup), (weights.embedding, emb)]
            if query.tokens and product_tokens:
                active.append((weights.text, text))
            total_weight = sum(w for w, _ in active)
            fused = sum(w * s for w, s in active) / total_weight

            current = best.get(entry.product_id)
            if current is None or fused > current.fused:
                best[entry.product_id] = ScoreBreakdown(
                    fused=fused,
                    near_duplicate=dup,
                    embedding=emb,
                    text=text,
                    dup_distance=distance,
                    best_image_id=entry.image_id,
                    best_crop=query.crops[crop_idx].name,
                    best_view=entry.view,
                )

        candidates = [
            Candidate(product=self.catalogue.by_id(pid), score=score)
            for pid, score in best.items()
            if self.catalogue.by_id(pid) is not None
        ]
        candidates.sort(key=lambda c: c.score.fused, reverse=True)
        return candidates[:limit]

    def similar_in_other_colours(
        self, product: Product, limit: int = 3, in_stock_only: bool = True
    ) -> list[Product]:
        """Answers "do you have this in another colour?".

        Panache stocks one colourway per product, so there is no variant to
        look up: the honest answer is a visually similar garment in a
        different colour family.
        """
        anchor = [
            e for e in self.entries
            if e.product_id == product.product_id and e.view == "full"
        ]
        if not anchor:
            return []
        anchor_matrix = np.vstack([e.vector for e in anchor])

        scored: dict[str, float] = {}
        for col, entry in enumerate(self.entries):
            if entry.product_id == product.product_id:
                continue
            other = self.catalogue.by_id(entry.product_id)
            if other is None:
                continue
            if other.colour_family == product.colour_family:
                continue
            if in_stock_only and not other.is_sellable:
                continue
            similarity = float(np.max(anchor_matrix @ entry.vector))
            if similarity > scored.get(entry.product_id, -1.0):
                scored[entry.product_id] = similarity

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return [self.catalogue.by_id(pid) for pid, _ in ranked[:limit]]
