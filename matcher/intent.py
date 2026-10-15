"""What is the customer actually asking?

Deliberately rule-based rather than model-based. These patterns are the
boutique's real vocabulary, they are cheap, they are testable, and they never
drift. Anything these rules do not recognise falls through to OTHER, which the
pipeline treats as a reason to involve a person rather than to improvise.
"""
from __future__ import annotations

import re
from enum import Enum

from .config import CANONICAL_SIZES


class Intent(str, Enum):
    PRICE = "price"
    SIZE = "size"
    COLOUR = "colour"
    LINK = "link"
    GREETING = "greeting"
    HUMAN = "human"       # must reach a person: money, logistics, complaints
    OTHER = "other"


# Ordered: the first match wins, and HUMAN is checked before everything so a
# refund request inside a price question still reaches a person.
_HUMAN = re.compile(
    r"\b(refund|return|returns|exchange|alter|alteration|tailor|stitch(ing)?|"
    r"complain\w*|damaged|wrong (item|order|size)|deliver\w*|ship(ping|ped|ment)?|"
    r"courier|track(ing|ed)?|wholesale|bulk|reseller|discount\w*|best price|"
    r"bargain\w*|cheaper|negotiat\w*|cod|cash on delivery|invoice|gst|urgent)\b",
    re.I,
)
_PRICE = re.compile(
    r"(\bpp\b|\bprice\b|\bpricing\b|\bprize\b|\bcost\b|\brate\b|how much|"
    r"kitna|kitne|\bqty price\b|\bmrp\b|\$)",
    re.I,
)
_SIZE = re.compile(r"\b(size[sd]?|sizing|fit|fits|avail\w*|do you have|in stock|stock)\b", re.I)
_COLOUR = re.compile(
    r"\b(colour|color|colours|colors|shade|shades|other colour|another colour|"
    r"different colour|same design)\b",
    re.I,
)
_LINK = re.compile(r"\b(link|website|site|url|online|order online|buy|purchase|checkout)\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|namaste|hii+|good (morning|afternoon|evening))\b", re.I)

_SIZE_TOKEN = re.compile(r"\b(3[0-9]|4[0-9]|5[0-9])\b")


def extract_size(text: str | None) -> str | None:
    """Pull a requested size out of free text.

    Only returns a value for sizes Panache actually uses, so a number that is
    plainly something else - a price, a quantity - is not mistaken for a size.
    """
    if not text:
        return None
    for token in _SIZE_TOKEN.findall(text):
        if token in CANONICAL_SIZES:
            return token
    return None


def detect(text: str | None) -> tuple[Intent, str | None]:
    """Returns (intent, requested_size)."""
    if not text or not text.strip():
        # An image with no words is almost always "what is this and what does
        # it cost?" - the dominant enquiry shape for this business.
        return Intent.PRICE, None

    size = extract_size(text)

    if _HUMAN.search(text):
        return Intent.HUMAN, size
    if _COLOUR.search(text):
        return Intent.COLOUR, size
    # A bare size question ("42?", "do you have 44") outranks a price pattern
    # only when a size was actually named.
    if size and (_SIZE.search(text) or len(text.strip()) <= 12):
        return Intent.SIZE, size
    if _PRICE.search(text):
        return Intent.PRICE, size
    if _SIZE.search(text):
        return Intent.SIZE, size
    if _LINK.search(text):
        return Intent.LINK, size
    if _GREETING.search(text):
        return Intent.GREETING, size
    return Intent.OTHER, size
