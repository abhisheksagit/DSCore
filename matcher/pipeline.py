"""The pipeline: a customer image (and optional message) in, a decision and a
reply out.

Order matters and is deliberate:

1. intent, because a refund question needs a person regardless of any image;
2. retrieval, three signals fused;
3. verification, skipped when the near-duplicate evidence already settles it
   (which is both the most accurate and the cheapest path);
4. the gate, the only place that decides what the customer sees;
5. composition, from catalogue facts.

Every run returns an audit record. That record is how thresholds get tuned,
how accuracy is reported, and how any individual answer can be explained
after the fact.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image

from . import compose, ocr
from .catalogue import Product
from .config import Config, DEFAULT
from .gate import Action, Decision, Verdict, decide, unambiguous_near_duplicate
from .index import Candidate, ProductIndex
from .intent import Intent, detect
from .verify import VerificationUnavailable, verify


@dataclass
class Outcome:
    decision: Decision
    reply: compose.Reply
    intent: Intent
    requested_size: str | None
    product: Product | None
    candidates: list[Candidate] = field(default_factory=list)
    audit: dict = field(default_factory=dict)

    @property
    def action(self) -> Action:
        return self.decision.action


class Matcher:
    def __init__(
        self,
        index: ProductIndex,
        catalogue_images: dict[str, Image.Image],
        config: Config = DEFAULT,
        client=None,
        verification_enabled: bool = True,
        verifier=None,
    ) -> None:
        self.index = index
        self.catalogue_images = catalogue_images
        self.config = config
        self.client = client
        self.verification_enabled = verification_enabled
        # Injectable so tests and the eval harness can substitute a stub
        # without reaching the network. Signature matches verify.verify.
        self._verify = verifier or verify

    def match(
        self,
        image: Image.Image,
        message_text: str | None = None,
        first_contact: bool = False,
        use_ocr: bool = True,
    ) -> Outcome:
        started = time.perf_counter()
        intent, requested_size = detect(message_text)

        # Sensitive enquiries go to a person whatever the image shows. Matching
        # them first would risk answering a refund question with a price.
        if intent is Intent.HUMAN:
            return Outcome(
                decision=Decision(Action.ESCALATE, None, ["intent requires a person"]),
                reply=compose.escalation(first_contact=first_contact),
                intent=intent,
                requested_size=requested_size,
                product=None,
                audit={
                    "intent": intent.value,
                    "escalated_before_matching": True,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )

        ocr_text = ocr.extract(image) if use_ocr else None
        features = self.index.features(image, message_text=message_text, ocr_text=ocr_text)
        candidates = self.index.search(features, limit=8)

        verdict: Verdict | None = None
        verification_note = "not attempted"
        shortlist = candidates[: self.config.thresholds.verify_top_k]

        if candidates:
            if unambiguous_near_duplicate(candidates, self.config):
                # The hash already settles it, and the gate will answer on that
                # basis, so a verification call here buys nothing and costs
                # money on the most common enquiry shape. This condition must
                # stay identical to the gate's - hence the shared helper.
                verification_note = "skipped: unambiguous near-duplicate"
            elif not self.verification_enabled:
                verification_note = "disabled by configuration"
                verdict = Verdict(None, "low", available=False)
            else:
                try:
                    result = self._verify(
                        image,
                        shortlist,
                        self.catalogue_images,
                        config=self.config,
                        client=self.client,
                    )
                    verdict = result.verdict
                    verification_note = f"ran on {result.model}"
                except VerificationUnavailable as exc:
                    verdict = Verdict(None, "low", available=False)
                    verification_note = f"unavailable: {exc}"

        decision = decide(
            candidates,
            verdict,
            self.config,
            query_colour_family=features.colour_family,
        )
        product = (
            self.index.catalogue.by_id(decision.product_id) if decision.product_id else None
        )
        reply = self._compose(decision, product, intent, requested_size, first_contact)

        audit = {
            "intent": intent.value,
            "requested_size": requested_size,
            "ocr_used": ocr_text is not None,
            "ocr_text": ocr_text,
            "embedder": self.index.embedder.name,
            "query_colour_family": features.colour_family,
            "verification": verification_note,
            "decision": decision.as_dict(),
            "candidates": [
                {"handle": c.product.handle, **c.score.as_dict()} for c in candidates[:5]
            ],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        return Outcome(
            decision=decision,
            reply=reply,
            intent=intent,
            requested_size=requested_size,
            product=product,
            candidates=candidates,
            audit=audit,
        )

    def _compose(
        self,
        decision: Decision,
        product: Product | None,
        intent: Intent,
        requested_size: str | None,
        first_contact: bool,
    ) -> compose.Reply:
        if decision.action is Action.ANSWER and product is not None:
            if intent is Intent.COLOUR:
                alternatives = [
                    alt
                    for alt in self.index.similar_in_other_colours(product, limit=3)
                    if alt.product_id != product.product_id
                ]
                return compose.colour_answer(product, alternatives, first_contact=first_contact)
            alternatives = (
                self.index.similar_in_other_colours(product, limit=2, in_stock_only=True)
                if not product.in_stock_sizes()
                else None
            )
            return compose.answer(
                product,
                intent=intent,
                requested_size=requested_size,
                alternatives=alternatives,
                first_contact=first_contact,
            )

        if decision.action is Action.CONFIRM and product is not None:
            image_id = (
                product.images[0].image_id if product.images else None
            )
            return compose.confirmation(product, image_id, first_contact=first_contact)

        return compose.escalation(
            first_contact=first_contact,
            found_nothing=decision.action is Action.NO_CANDIDATES
            or "below floor" in " ".join(decision.reasons)
            or "rejected all candidates" in " ".join(decision.reasons),
        )
