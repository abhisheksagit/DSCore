"""Reply composition.

Every fact a customer receives - price, sizes, URL - is read from the
catalogue and formatted here. No model writes these numbers. That is a
structural guarantee, not a prompt instruction: the assistant cannot
hallucinate a price because no model is ever asked for one.

The wording lives in VOICE so it can be replaced wholesale once Panache's real
conversation samples arrive (proposal section 8.6). The current strings are a
reasonable warm-professional default, not an imitation of Divya's voice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalogue import Product
from .config import CANONICAL_SIZES
from .intent import Intent

VOICE = {
    "greeting": "Hi! Thanks for reaching out to Panache by Divya ✨",
    "sign_off": "Let me know if you'd like any more details!",
    "handover": (
        "Let me check this with our team and come straight back to you — "
        "someone will follow up here shortly."
    ),
    "not_found": (
        "I couldn't match this to anything in our collection just yet. "
        "Let me pass it to our team and we'll come back to you shortly!"
    ),
    "confirm": "Just to make sure I get this right — is this the one you meant?",
    "sold_out": "This one's sold out in every size at the moment",
    "one_colourway": (
        "Each of our pieces is made in the one colourway, so this design doesn't "
        "come in other shades"
    ),
    "disclosure": (
        "You're chatting with Panache by Divya's assistant — just ask for a person "
        "any time and one of our team will jump in."
    ),
}


@dataclass
class Reply:
    """What the customer receives. `attach_image_ids` are catalogue images the
    sender should attach, resolved to real media by the WhatsApp layer."""

    text: str
    attach_image_ids: list[str] = field(default_factory=list)
    requires_human: bool = False
    include_disclosure: bool = False

    def render(self) -> str:
        parts = [self.text]
        if self.include_disclosure:
            parts.append(VOICE["disclosure"])
        return "\n\n".join(parts)


def _sizes_sentence(product: Product) -> str:
    in_stock = product.in_stock_sizes()
    if not in_stock:
        return VOICE["sold_out"] + "."
    if len(in_stock) == 1:
        return f"Currently available in size {in_stock[0]} only."
    return f"Currently available in sizes {', '.join(in_stock)}."


def _product_header(product: Product) -> str:
    return f"*{product.title}*\n{product.price_display}"


def answer(
    product: Product,
    intent: Intent = Intent.PRICE,
    requested_size: str | None = None,
    alternatives: list[Product] | None = None,
    first_contact: bool = False,
) -> Reply:
    """The full answer: what it is, what it costs, what sizes, where to buy."""
    lines: list[str] = []
    if first_contact:
        lines.append(VOICE["greeting"])

    if intent is Intent.SIZE and requested_size:
        availability = product.size_available(requested_size)
        lines.append(_product_header(product))
        if availability is True:
            lines.append(f"Yes — size {requested_size} is in stock. ✨")
            others = [s for s in product.in_stock_sizes() if s != requested_size]
            if others:
                lines.append(f"We also have {', '.join(others)} in this one.")
        elif availability is False:
            lines.append(f"Size {requested_size} is sold out in this piece.")
            lines.append(_sizes_sentence(product))
        else:
            # A size we simply do not carry - say so plainly.
            lines.append(
                f"We don't carry size {requested_size} — our range is "
                f"{CANONICAL_SIZES[0]}–{CANONICAL_SIZES[-1]}."
            )
            lines.append(_sizes_sentence(product))
    elif intent is Intent.SIZE:
        lines.append(_product_header(product))
        lines.append(_sizes_sentence(product))
    elif intent is Intent.LINK:
        lines.append(_product_header(product))
        lines.append(_sizes_sentence(product))
    else:  # PRICE, GREETING, OTHER resolved onto a product
        lines.append(_product_header(product))
        lines.append(_sizes_sentence(product))

    lines.append(f"You can see it here: {product.url}")

    if not product.in_stock_sizes() and alternatives:
        lines.append(_alternatives_block(alternatives, "Here are a couple of similar pieces"))

    lines.append(VOICE["sign_off"])
    return Reply(
        text="\n\n".join(lines),
        attach_image_ids=[],
        include_disclosure=first_contact,
    )


def _alternatives_block(alternatives: list[Product], lead: str) -> str:
    rows = [
        f"• {alt.title} — {alt.price_display}\n  {alt.url}" for alt in alternatives
    ]
    return f"{lead}:\n" + "\n".join(rows)


def colour_answer(
    product: Product, alternatives: list[Product], first_contact: bool = False
) -> Reply:
    """"Do you have this in another colour?"

    Panache stocks one colourway per product, so there is no variant to look
    up. Answering honestly and immediately offering visually similar pieces in
    other colours is the difference between a lost sale and a redirected one.
    """
    lines: list[str] = []
    if first_contact:
        lines.append(VOICE["greeting"])
    lines.append(_product_header(product))
    if alternatives:
        lines.append(VOICE["one_colourway"] + ", but these are similar in other shades:")
        lines.append(
            "\n".join(
                f"• {alt.title} ({alt.colour_family}) — {alt.price_display}\n  {alt.url}"
                for alt in alternatives
            )
        )
    else:
        lines.append(VOICE["one_colourway"] + ".")
        lines.append(
            "I'll check with our team whether anything similar is coming in — "
            "someone will follow up shortly."
        )
    lines.append(VOICE["sign_off"])
    return Reply(
        text="\n\n".join(lines),
        requires_human=not alternatives,
        include_disclosure=first_contact,
    )


def confirmation(product: Product, image_id: str | None = None,
                 first_contact: bool = False) -> Reply:
    """The middle tier of the confidence gate.

    Sending the candidate back for confirmation converts a likely error into a
    normal, friendly exchange. It reads as attentiveness rather than failure,
    which is why it is preferred over guessing.
    """
    lines: list[str] = []
    if first_contact:
        lines.append(VOICE["greeting"])
    lines.append(VOICE["confirm"])
    lines.append(_product_header(product))
    lines.append(product.url)
    lines.append("If that's the one, I'll send the available sizes straight away.")
    return Reply(
        text="\n\n".join(lines),
        attach_image_ids=[image_id] if image_id else [],
        include_disclosure=first_contact,
    )


def escalation(first_contact: bool = False, found_nothing: bool = False) -> Reply:
    lines: list[str] = []
    if first_contact:
        lines.append(VOICE["greeting"])
    lines.append(VOICE["not_found"] if found_nothing else VOICE["handover"])
    return Reply(
        text="\n\n".join(lines),
        requires_human=True,
        include_disclosure=first_contact,
    )
