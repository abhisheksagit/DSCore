"""The confidence gate: decides whether a customer gets an answer, a question,
or a person.

This is the component that implements "never guess when uncertain". Every
other part of the pipeline produces evidence; this part alone decides what the
customer sees, and it is deliberately the smallest and most readable module in
the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import Config, DEFAULT
from .index import Candidate


class Action(str, Enum):
    ANSWER = "answer"          # quote price, sizes and link outright
    CONFIRM = "confirm"        # send the photo back: "is this the one?"
    ESCALATE = "escalate"      # hand to a human, warmly
    NO_CANDIDATES = "no_candidates"  # nothing in the catalogue is close


@dataclass
class Verdict:
    """Outcome of the verification stage. `product_id` None means the model
    actively rejected every candidate, which is a useful signal, not a
    failure."""

    product_id: str | None
    confidence: str  # "high" | "medium" | "low"
    evidence: str = ""
    available: bool = True


@dataclass
class Decision:
    action: Action
    product_id: str | None
    reasons: list[str] = field(default_factory=list)
    margin_ratio: float = 0.0
    top_score: float = 0.0
    dup_distance: int = 64
    verification: Verdict | None = None

    def as_dict(self) -> dict:
        return {
            "action": self.action.value,
            "product_id": self.product_id,
            "reasons": self.reasons,
            "margin_ratio": round(self.margin_ratio, 4),
            "top_score": round(self.top_score, 4),
            "dup_distance": self.dup_distance,
            "verification": (
                None
                if self.verification is None
                else {
                    "product_id": self.verification.product_id,
                    "confidence": self.verification.confidence,
                    "available": self.verification.available,
                }
            ),
        }


def margin_ratio(candidates: list[Candidate]) -> float:
    """Relative gap between the top two products. 1.0 when only one candidate."""
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return 1.0
    top = candidates[0].score.fused
    if top <= 0:
        return 0.0
    return (top - candidates[1].score.fused) / top


# Colour families that are genuinely easy to confuse in a compressed
# screenshot, and so are not treated as a contradiction.
_COLOUR_NEIGHBOURS: dict[str, set[str]] = {
    "red": {"maroon", "pink", "orange"},
    "maroon": {"red", "purple", "brown"},
    "pink": {"red", "purple"},
    "orange": {"red", "gold", "yellow"},
    "gold": {"yellow", "beige", "orange"},
    "yellow": {"gold", "beige"},
    "beige": {"gold", "white", "grey"},
    "green": {"teal"},
    "teal": {"green", "blue"},
    "blue": {"navy", "teal", "purple"},
    "navy": {"blue", "black", "purple"},
    "purple": {"navy", "blue", "pink", "maroon"},
    "black": {"navy", "grey"},
    "grey": {"black", "white", "beige"},
    "white": {"grey", "beige"},
}


def colour_conflict(query_family: str | None, product_family: str | None) -> bool:
    """True when the two colour families are incompatible.

    Unknown on either side is not a conflict - we only act on positive
    disagreement, never on absence of information.
    """
    if not query_family or not product_family:
        return False
    if query_family in ("unknown", "multi") or product_family in ("unknown", "multi"):
        return False
    if query_family == product_family:
        return False
    return product_family not in _COLOUR_NEIGHBOURS.get(query_family, set())


def unambiguous_near_duplicate(
    candidates: list[Candidate], config: Config = DEFAULT
) -> bool:
    """Is the top hit a near-duplicate that no other product comes close to?

    The single definition, used both by the gate (to decide it may answer
    without a model verdict) and by the pipeline (to decide it may skip the
    verification call). These two MUST agree: when they drifted apart, a
    contested near-duplicate had its verification skipped by the pipeline and
    then rejected by the gate, so almost every enquiry fell through to a
    confirmation with no verdict available.
    """
    if not candidates:
        return False
    thresholds = config.thresholds
    top_dup = candidates[0].score.dup_distance
    if top_dup > thresholds.dup_strong_max:
        return False
    runner_up = candidates[1].score.dup_distance if len(candidates) > 1 else 64
    return (runner_up - top_dup) >= thresholds.dup_unique_margin


def decide(
    candidates: list[Candidate],
    verdict: Verdict | None = None,
    config: Config = DEFAULT,
    query_colour_family: str | None = None,
) -> Decision:
    thresholds = config.thresholds
    reasons: list[str] = []

    if not candidates:
        return Decision(Action.NO_CANDIDATES, None, ["index returned nothing"])

    top = candidates[0]
    ratio = margin_ratio(candidates)
    base = Decision(
        action=Action.ESCALATE,
        product_id=top.product.product_id,
        margin_ratio=ratio,
        top_score=top.score.fused,
        dup_distance=top.score.dup_distance,
        verification=verdict,
    )

    # 1. A near-duplicate of our own photography is very strong evidence - the
    #    customer is looking at this product's image - but only when it is
    #    UNAMBIGUOUS. If another product's photo hashes almost as close, the
    #    hash cannot tell them apart and must not answer on its own.
    runner_up_dup = candidates[1].score.dup_distance if len(candidates) > 1 else 64
    dup_gap = runner_up_dup - top.score.dup_distance

    if unambiguous_near_duplicate(candidates, config):
        if config.require_colour_agreement and colour_conflict(
            query_colour_family, top.product.colour_family
        ):
            # Structural hashes can collide across colours; colour disagreement
            # means one of the two signals is wrong, so we do not guess.
            base.action = Action.CONFIRM
            base.reasons = [
                f"near-duplicate (hamming {top.score.dup_distance}) but colour "
                f"disagrees: query {query_colour_family} vs "
                f"{top.product.colour_family}"
            ]
            return base
        base.action = Action.ANSWER
        base.reasons = [
            f"unambiguous near-duplicate of catalogue image "
            f"(hamming {top.score.dup_distance} <= {thresholds.dup_strong_max}, "
            f"runner-up {runner_up_dup}, gap {dup_gap} >= "
            f"{thresholds.dup_unique_margin})"
        ]
        if verdict and verdict.available and verdict.product_id not in (None, top.product.product_id):
            base.action = Action.CONFIRM
            base.product_id = verdict.product_id
            base.reasons.append("verification named a different product; confirming instead")
        return base

    if top.score.dup_distance <= thresholds.dup_strong_max:
        reasons.append(
            f"near-duplicate contested: runner-up within {dup_gap} bits "
            f"(need {thresholds.dup_unique_margin}); deferring to verification"
        )

    # 2. Nothing plausible in the catalogue.
    if top.score.fused < thresholds.candidate_floor:
        base.action = Action.ESCALATE
        base.reasons = [
            f"best score {top.score.fused:.3f} below floor {thresholds.candidate_floor}"
        ]
        return base

    # 3. Without verification we never answer outright - a retrieval score
    #    alone has no notion of "these are different garments that look alike".
    if verdict is None or not verdict.available:
        base.action = Action.CONFIRM
        base.reasons = ["verification unavailable; asking the customer to confirm"]
        return base

    # 4. The model rejected every candidate. Trust it: this is how competitor
    #    products and items we never stocked get routed to a human.
    if verdict.product_id is None:
        base.action = Action.ESCALATE
        base.reasons = ["verification rejected all candidates"]
        return base

    # The model may prefer candidate 2 over candidate 1 - that is the whole
    # point of a verification stage, so its choice wins.
    chosen = next(
        (c for c in candidates if c.product.product_id == verdict.product_id), None
    )
    if chosen is None:
        base.action = Action.ESCALATE
        base.reasons = ["verification named a product that was not a candidate"]
        return base

    base.product_id = chosen.product.product_id

    # When the verifier picks something other than retrieval's top hit, two
    # independent signals disagree. Measured on fixtures, this is where
    # confident-wrong answers concentrate: a verifier that is wrong AND
    # confident produces an error the gate cannot out-vote. So an override is
    # allowed to inform the answer but never to trigger one on its own - it
    # goes to the customer as a confirmation instead.
    if chosen is not candidates[0]:
        base.action = Action.CONFIRM
        base.reasons = reasons + [
            f"verification chose {chosen.product.handle} over retrieval's "
            f"{candidates[0].product.handle}; signals disagree, so confirming"
        ]
        return base

    if verdict.confidence == "high" and ratio >= thresholds.margin_clear_ratio:
        if config.require_colour_agreement and colour_conflict(
            query_colour_family, chosen.product.colour_family
        ):
            base.action = Action.CONFIRM
            base.reasons = reasons + [
                f"verification high but colour disagrees: query "
                f"{query_colour_family} vs {chosen.product.colour_family}"
            ]
            return base
        base.action = Action.ANSWER
        base.reasons = reasons + [
            f"verification high, margin ratio {ratio:.3f} >= {thresholds.margin_clear_ratio}"
        ]
    elif verdict.confidence == "high":
        base.action = Action.CONFIRM
        base.reasons = reasons + [
            f"verification high but contested (margin ratio {ratio:.3f}"
            f" < {thresholds.margin_clear_ratio})"
        ]
    elif verdict.confidence == "medium":
        base.action = Action.CONFIRM
        base.reasons = reasons + ["verification medium confidence"]
    else:
        base.action = Action.ESCALATE
        base.reasons = reasons + ["verification low confidence"]
    return base
