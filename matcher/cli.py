"""Command line entry point.

    python -m matcher.cli demo
    python -m matcher.cli match path/to/customer-photo.jpg --text "pp"
    python -m matcher.cli match photo.jpg --catalogue catalogue.json --images ./images

`demo` runs against the synthetic fixtures so the pipeline can be exercised
with no catalogue and no credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

from .catalogue import Catalogue
from .config import Config
from .embed import ClassicalEmbedder, SiglipOnnxEmbedder
from .index import ProductIndex
from .pipeline import Matcher


def _build(catalogue_path: str | None, images_dir: str | None, onnx: str | None):
    if catalogue_path:
        catalogue = Catalogue.load(catalogue_path)
        root = Path(images_dir or ".")
        store: dict[str, Image.Image] = {}

        def loader(product, product_image):
            candidate = root / Path(product_image.path).name
            if not candidate.exists():
                return None
            image = Image.open(candidate).convert("RGB")
            store[product_image.image_id] = image
            return image
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from fixtures import build_catalogue, memory_loader  # noqa: PLC0415

        catalogue, store = build_catalogue()
        loader = memory_loader(store)

    embedder = SiglipOnnxEmbedder(onnx) if onnx else ClassicalEmbedder()
    index = ProductIndex(catalogue, embedder, Config()).build(loader)
    return index, store


def _report(outcome) -> None:
    print("=" * 72)
    print(f"ACTION: {outcome.action.value.upper()}   intent={outcome.intent.value}")
    if outcome.product:
        print(f"PRODUCT: {outcome.product.title}")
        print(f"         {outcome.product.price_display}")
        print(f"         sizes in stock: {', '.join(outcome.product.in_stock_sizes()) or 'none'}")
        print(f"         {outcome.product.url}")
    print("REASONS:")
    for reason in outcome.decision.reasons:
        print(f"  - {reason}")
    print("-" * 72)
    print("REPLY TO CUSTOMER:")
    print()
    print(outcome.reply.render())
    print("-" * 72)
    print("TOP CANDIDATES:")
    for candidate in outcome.candidates[:4]:
        score = candidate.score
        print(
            f"  {candidate.product.handle:<34} fused={score.fused:.3f}"
            f" dup={score.dup_distance:<3} emb={score.embedding:.3f}"
            f" text={score.text:.2f}"
        )
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="matcher")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("match", "demo"):
        cmd = sub.add_parser(name)
        if name == "match":
            cmd.add_argument("image")
        cmd.add_argument("--text", default=None, help="the customer's message, if any")
        cmd.add_argument("--catalogue", default=None, help="catalogue JSON")
        cmd.add_argument("--images", default=None, help="directory of product images")
        cmd.add_argument("--onnx", default=os.environ.get("SIGLIP_ONNX_PATH"),
                         help="SigLIP ONNX weights; omit to use the classical embedder")
        cmd.add_argument("--no-verification", action="store_true",
                         help="skip the model verification stage")
        cmd.add_argument("--audit", action="store_true", help="print the audit record")

    args = parser.parse_args(argv)
    index, store = _build(args.catalogue, args.images, args.onnx)
    matcher = Matcher(
        index, store, config=Config(), verification_enabled=not args.no_verification
    )

    if args.command == "demo":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from fixtures import simulate_customer_image  # noqa: PLC0415

        product = index.catalogue.products[0]
        query = simulate_customer_image(
            store[product.images[0].image_id], "screenshot", seed=3
        )
        print(f"(demo: simulated Instagram screenshot of '{product.title}')\n")
        outcome = matcher.match(query, args.text or "pp", first_contact=True)
    else:
        outcome = matcher.match(Image.open(args.image), args.text, first_contact=True)

    _report(outcome)
    if args.audit:
        print("AUDIT RECORD:")
        print(json.dumps(outcome.audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
