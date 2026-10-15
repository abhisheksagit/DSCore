"""Verification stage: Claude compares the customer's image against the
shortlist and is actively encouraged to refuse.

Retrieval answers "which catalogue images look most like this?". That question
has an answer even when the garment is not ours at all. This stage asks the
different, harder question - "is this the same garment, and how do you know?"
- and can answer "none of these", which is what keeps the assistant honest.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

from PIL import Image

from . import images as im
from .config import Config, DEFAULT
from .gate import Verdict
from .index import Candidate


class VerificationUnavailable(RuntimeError):
    """No credential, or the API could not be reached."""


_SYSTEM = """You verify product identity for an Indian ethnic-wear boutique in Melbourne.

You are shown a customer's photo, then numbered catalogue products. Decide whether the \
customer's photo shows ONE of those exact products.

Be sceptical. The boutique stocks one colourway per product, so garments that differ only \
in trim, embroidery density, border pattern or neckline are DIFFERENT products, and calling \
them the same is the costliest error you can make - it leads to a customer being quoted the \
wrong price.

Rules:
- Cite the specific visual features that identify the match, and say what rules out the \
runners-up. Vague agreement is not evidence.
- The photo may be a cropped screenshot, a reel still, a photo of a screen, or worn by a \
customer. Judge the garment, not the image quality.
- Answer "high" only when the distinguishing details are actually legible and they agree.
- If the photo is too cropped or blurred to tell two candidates apart, that is "low" or \
"medium" - not a coin flip.
- If the garment is not among the candidates, set product_index to null. A competitor's \
product or an item we never stocked must return null rather than the nearest thing."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "product_index": {
            "type": ["integer", "null"],
            "description": "1-based index of the matching candidate, or null if none match.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "distinguishing_evidence": {
            "type": "string",
            "description": "Specific features that identify the match, or why none matched.",
        },
        "ruled_out": {
            "type": "string",
            "description": "Why the other candidates were rejected.",
        },
    },
    "required": ["product_index", "confidence", "distinguishing_evidence", "ruled_out"],
    "additionalProperties": False,
}


@dataclass
class VerificationResult:
    verdict: Verdict
    raw: dict
    model: str
    usage: dict | None = None


def _image_block(data: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


def build_content(
    query_image: Image.Image,
    candidates: list[Candidate],
    catalogue_images: dict[str, Image.Image],
) -> list[dict]:
    """Message content: the customer photo, then one photo per candidate.

    Kept as a separate function so it can be unit-tested, and inspected in the
    audit log, without spending an API call.
    """
    content: list[dict] = [
        {"type": "text", "text": "Customer's photo:"},
        _image_block(im.to_png_bytes(query_image)),
    ]
    for position, candidate in enumerate(candidates, start=1):
        product = candidate.product
        content.append(
            {
                "type": "text",
                "text": (
                    f"Candidate {position}: {product.title}"
                    f" (retrieval score {candidate.score.fused:.3f},"
                    f" colour family {product.colour_family})"
                ),
            }
        )
        image = catalogue_images.get(candidate.score.best_image_id)
        if image is not None:
            content.append(_image_block(im.to_png_bytes(image)))
    content.append(
        {
            "type": "text",
            "text": (
                "Which candidate, if any, is the garment in the customer's photo? "
                "Use 1-based indexing, or null if none of them is this garment."
            ),
        }
    )
    return content


def verify(
    query_image: Image.Image,
    candidates: list[Candidate],
    catalogue_images: dict[str, Image.Image],
    config: Config = DEFAULT,
    client=None,
) -> VerificationResult:
    if not candidates:
        raise ValueError("verify() needs at least one candidate")

    if client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise VerificationUnavailable(
                "No ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in the environment."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VerificationUnavailable("the `anthropic` package is not installed") from exc
        client = anthropic.Anthropic()

    settings = config.verification
    request = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": _SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": build_content(query_image, candidates, catalogue_images),
            }
        ],
        "output_config": {"format": {"type": "json_schema", "schema": _SCHEMA}},
    }
    if settings.enable_refusal_fallback:
        # Routes by refusal category, so no model list to maintain.
        request["betas"] = ["server-side-fallback-2026-07-01"]
        request["fallbacks"] = "default"

    try:
        response = client.messages.create(**request)
    except Exception as exc:  # network, auth, rate limit - all mean "no verdict"
        raise VerificationUnavailable(str(exc)) from exc

    if getattr(response, "stop_reason", None) == "refusal":
        raise VerificationUnavailable("model declined the request")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise VerificationUnavailable("no text block in response")
    payload = json.loads(text)

    index = payload.get("product_index")
    product_id = None
    if isinstance(index, int) and 1 <= index <= len(candidates):
        product_id = candidates[index - 1].product.product_id
    elif index is not None:
        # Out-of-range index is a malformed verdict, not a match.
        payload["_warning"] = f"product_index {index} out of range"

    usage = getattr(response, "usage", None)
    return VerificationResult(
        verdict=Verdict(
            product_id=product_id,
            confidence=payload.get("confidence", "low"),
            evidence=payload.get("distinguishing_evidence", ""),
            available=True,
        ),
        raw=payload,
        model=getattr(response, "model", settings.model),
        usage=(
            {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }
            if usage
            else None
        ),
    )
