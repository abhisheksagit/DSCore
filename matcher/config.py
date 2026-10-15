"""Tunable constants for the Panache product matcher.

Every threshold that decides whether a customer gets an automatic answer lives
here, so it can be re-tuned from measured data (see eval/run_eval.py) without
touching pipeline logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Panache carries one colourway per product; sizes are numeric and fixed.
CANONICAL_SIZES: tuple[str, ...] = ("38", "40", "42", "44", "46")

STORE_BASE_URL = "https://panachebydivya.com"
CURRENCY = "AUD"


@dataclass(frozen=True)
class RetrievalWeights:
    """Relative trust in each retrieval signal.

    Re-normalised at scoring time over whichever signals are actually present,
    so a catalogue without OCR text does not silently lose score headroom.
    """

    near_duplicate: float = 0.45
    embedding: float = 0.40
    text: float = 0.15


@dataclass(frozen=True)
class Thresholds:
    # --- near-duplicate (perceptual hash) stage ---
    # Hamming distance over a 64-bit hash. A screenshot of our own photo
    # typically lands under 8 even after cropping and JPEG recompression.
    dup_strong_max: int = 8
    dup_plausible_max: int = 16
    # A near-duplicate is only strong evidence if it is UNAMBIGUOUS. Measured
    # on fixtures: every confident-wrong answer came from a hash match whose
    # runner-up was near-tied, so proximity alone is not identity. The
    # runner-up product must be at least this many bits worse before the hash
    # is allowed to answer without verification.
    dup_unique_margin: int = 6

    # --- fused retrieval ---
    # Minimum fused score for a candidate to be worth showing to verification.
    candidate_floor: float = 0.35
    # Gap between the best and second-best *product*, as a fraction of the top
    # score, required to call a match uncontested. A thin margin means two
    # different garments look alike. Expressed as a ratio because absolute
    # score ranges differ sharply between embedders.
    margin_clear_ratio: float = 0.10
    # How many candidates go to the verification stage.
    verify_top_k: int = 4

    # --- semantic embedding ---
    embedding_auto_floor: float = 0.82


@dataclass(frozen=True)
class VerificationConfig:
    # Chosen in the approved proposal: accuracy first, step down to
    # claude-haiku-5-5 on easy cases once the audit log justifies it.
    # Raise to claude-opus-5-5 if measured accuracy needs it.
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 2000
    # Server-side refusal fallback: recommended for claude-sonnet-5-5 on the
    # Claude API, and routes by refusal category so we maintain no model list.
    enable_refusal_fallback: bool = True


@dataclass(frozen=True)
class Config:
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    # Never auto-answer a customer on retrieval scores alone. If the
    # verification stage is unavailable (no API key, API down), the best a
    # match can achieve is a confirmation question. Exception: an unambiguous
    # near-duplicate of our own catalogue photo, which is a far stronger
    # signal than any model verdict.
    require_verification_for_auto: bool = True
    # Block an automatic answer when the query's dominant colour family
    # contradicts the chosen product's. With one colourway per product, colour
    # is a reliable signal, and disagreement means something is wrong even
    # when structural hashes agree.
    require_colour_agreement: bool = True


DEFAULT = Config()
