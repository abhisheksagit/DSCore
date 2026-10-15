"""Panache by Divya - product matcher.

Given a customer's photo (and optionally their message), identify the product
in the Shopify catalogue and compose a reply with price, available sizes and a
purchase link - or decline to answer and hand over to a person.
"""
from .catalogue import Catalogue, Product, Variant
from .config import Config, DEFAULT
from .gate import Action, Decision
from .index import ProductIndex
from .pipeline import Matcher, Outcome

__all__ = [
    "Action", "Catalogue", "Config", "DEFAULT", "Decision",
    "Matcher", "Outcome", "Product", "ProductIndex", "Variant",
]
