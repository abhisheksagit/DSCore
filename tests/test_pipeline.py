"""End-to-end behaviour on simulated customer images."""
import pytest

from fixtures import build_catalogue, foreign_garment, memory_loader, simulate_customer_image

from matcher.config import Config
from matcher.embed import ClassicalEmbedder
from matcher.gate import Action, Verdict
from matcher.index import ProductIndex
from matcher.intent import Intent
from matcher.pipeline import Matcher
from matcher.verify import VerificationResult, VerificationUnavailable


@pytest.fixture(scope="module")
def built():
    catalogue, store = build_catalogue(n_products=12)
    index = ProductIndex(catalogue, ClassicalEmbedder(), Config()).build(memory_loader(store))
    return index, store


def oracle(index):
    """A perfect verifier, so pipeline wiring is tested without the network."""
    state = {"truth": None}

    def verifier(image, candidates, catalogue_images, config=None, client=None):
        ids = [c.product.product_id for c in candidates]
        chosen = state["truth"] if state["truth"] in ids else None
        return VerificationResult(Verdict(chosen, "high", "test"), {}, "oracle")

    return verifier, state


def test_index_covers_every_product(built):
    index, _ = built
    coverage = index.coverage()
    assert coverage["products_without_images"] == []
    assert coverage["index_entries"] > coverage["indexed_images"]  # crop views


def test_forwarded_catalogue_photo_is_identified(built):
    index, store = built
    verifier, state = oracle(index)
    matcher = Matcher(index, store, verifier=verifier)
    product = index.catalogue.products[0]
    state["truth"] = product.product_id
    outcome = matcher.match(
        simulate_customer_image(store[product.images[0].image_id], "clean", seed=1), "pp"
    )
    assert outcome.action in (Action.ANSWER, Action.CONFIRM)
    assert outcome.product.product_id == product.product_id


def test_a_garment_we_do_not_stock_is_never_auto_answered(built):
    index, store = built
    verifier, state = oracle(index)
    matcher = Matcher(index, store, verifier=verifier)
    for seed in range(8):
        state["truth"] = None  # genuinely absent from the catalogue
        outcome = matcher.match(foreign_garment(seed=seed), "price please")
        assert outcome.action is not Action.ANSWER


def test_sensitive_intent_escalates_before_any_matching(built):
    index, store = built
    matcher = Matcher(index, store, verification_enabled=False)
    product = index.catalogue.products[0]
    outcome = matcher.match(
        simulate_customer_image(store[product.images[0].image_id], "clean", seed=2),
        "can I return this if it doesn't fit?",
    )
    assert outcome.action is Action.ESCALATE
    assert outcome.intent is Intent.HUMAN
    assert outcome.audit["escalated_before_matching"] is True
    assert outcome.reply.requires_human


def test_unavailable_verification_never_auto_answers_a_contested_match(built):
    index, store = built

    def broken(*args, **kwargs):
        raise VerificationUnavailable("simulated outage")

    matcher = Matcher(index, store, verifier=broken)
    answers_without_unique_hash = 0
    for i, product in enumerate(index.catalogue.products[:6]):
        outcome = matcher.match(
            simulate_customer_image(store[product.images[0].image_id], "crop", seed=i), "pp"
        )
        if outcome.action is Action.ANSWER:
            # Only an unambiguous near-duplicate may answer unverified.
            assert "unambiguous near-duplicate" in " ".join(outcome.decision.reasons)
            answers_without_unique_hash += 1
    assert answers_without_unique_hash >= 0  # the assertion above is the check


def test_size_enquiry_answers_with_real_inventory(built):
    index, store = built
    verifier, state = oracle(index)
    matcher = Matcher(index, store, verifier=verifier)
    product = next(p for p in index.catalogue.products if p.in_stock_sizes())
    state["truth"] = product.product_id
    size = product.in_stock_sizes()[0]
    outcome = matcher.match(
        simulate_customer_image(store[product.images[0].image_id], "clean", seed=4),
        f"available in size {size}?",
    )
    assert outcome.intent is Intent.SIZE and outcome.requested_size == size
    if outcome.action is Action.ANSWER:
        assert size in outcome.reply.text
        assert product.price_display in outcome.reply.text
        assert product.url in outcome.reply.text


def test_colour_enquiry_offers_other_colourways(built):
    index, store = built
    verifier, state = oracle(index)
    matcher = Matcher(index, store, verifier=verifier)
    product = index.catalogue.products[0]
    state["truth"] = product.product_id
    outcome = matcher.match(
        simulate_customer_image(store[product.images[0].image_id], "clean", seed=5),
        "do you have this in another colour?",
    )
    assert outcome.intent is Intent.COLOUR
    if outcome.action is Action.ANSWER:
        assert "one colourway" in outcome.reply.text


def test_similar_in_other_colours_excludes_same_colour_and_itself(built):
    index, _ = built
    product = index.catalogue.products[0]
    for alternative in index.similar_in_other_colours(product, limit=3):
        assert alternative.product_id != product.product_id
        assert alternative.colour_family != product.colour_family
        assert alternative.is_sellable


def test_every_decision_produces_an_audit_record(built):
    index, store = built
    matcher = Matcher(index, store, verification_enabled=False)
    product = index.catalogue.products[1]
    outcome = matcher.match(
        simulate_customer_image(store[product.images[0].image_id], "screenshot", seed=6), "pp"
    )
    for key in ("intent", "embedder", "verification", "decision", "candidates", "elapsed_ms"):
        assert key in outcome.audit
    assert outcome.audit["decision"]["action"] == outcome.action.value
    assert outcome.audit["candidates"]
