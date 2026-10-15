"""Customer vocabulary. These are the real phrasings from the brief, plus the
sensitive intents that must always reach a person."""
import pytest

from matcher.intent import Intent, detect, extract_size


@pytest.mark.parametrize("text", ["pp", "PP", "price please", "Price?", "how much", "kitna hai", "cost?"])
def test_price_phrasings(text):
    assert detect(text)[0] is Intent.PRICE


@pytest.mark.parametrize("text,size", [
    ("Available in size 42?", "42"), ("42?", "42"), ("do you have 44", "44"),
    ("price for size 46", "46"), ("is this in stock", None),
])
def test_size_phrasings(text, size):
    intent, got = detect(text)
    assert intent is Intent.SIZE and got == size


def test_image_with_no_message_is_treated_as_a_price_enquiry():
    assert detect(None)[0] is Intent.PRICE
    assert detect("")[0] is Intent.PRICE


@pytest.mark.parametrize("text", [
    "Do you have this in another colour?", "any other shades", "same design different color",
])
def test_colour_enquiry(text):
    assert detect(text)[0] is Intent.COLOUR


@pytest.mark.parametrize("text", [
    "can I return this", "exchange possible?", "any discount?", "best price for 10 pieces",
    "when will it be delivered", "do you ship to Perth", "needs alteration",
    "the item arrived damaged", "wholesale rates?",
])
def test_sensitive_enquiries_reach_a_person(text):
    assert detect(text)[0] is Intent.HUMAN


def test_sensitive_intent_wins_over_a_price_question_in_the_same_message():
    assert detect("price please and can I return it if it doesn't fit")[0] is Intent.HUMAN


def test_only_sizes_we_carry_are_extracted():
    assert extract_size("size 42") == "42"
    assert extract_size("size 36") is None        # outside 38-46
    assert extract_size("$249") is None           # a price, not a size
    assert extract_size("2 pieces") is None
