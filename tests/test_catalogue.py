"""Catalogue facts: sizes, stock and prices must be exact - a customer is
quoted these verbatim."""
import pytest

from matcher.catalogue import Product, Variant, normalise_size


@pytest.mark.parametrize(
    "raw,expected",
    [("42", "42"), ("Size 42", "42"), ("42 (M)", "42"), (" 38 ", "38"),
     ("46", "46"), ("Free Size", None), ("36", None), ("48", None), ("", None)],
)
def test_normalise_size(raw, expected):
    assert normalise_size(raw) == expected


def product(**stock):
    return Product(
        product_id="p1", handle="red-lehenga", title="Red Lehenga", price_cents=24900,
        variants=[Variant(f"v{s}", s, stock.get(f"s{s}", 0)) for s in ("38","40","42","44","46")],
    )


def test_in_stock_sizes_are_ordered_small_to_large():
    assert product(s46=3, s38=1, s42=2).in_stock_sizes() == ["38", "42", "46"]


def test_size_available_distinguishes_sold_out_from_not_carried():
    p = product(s42=2)
    assert p.size_available("42") is True
    assert p.size_available("44") is False     # we carry it, no stock
    assert p.size_available("36") is None      # we never carry it
    assert p.size_available("Size 42") is True


def test_price_display_is_aud_with_cents():
    assert product(s42=1).price_display == "$249.00 AUD"


def test_url_uses_the_store_domain_and_handle():
    assert product().url == "https://panachebydivya.com/products/red-lehenga"


def test_unparseable_variants_are_surfaced_not_dropped():
    p = Product("p", "h", "T", 1000, variants=[Variant("v", "Free Size", 5)])
    assert [v.size_raw for v in p.unmapped_variants()] == ["Free Size"]
    assert p.in_stock_sizes() == []
    assert p.is_sellable is False
