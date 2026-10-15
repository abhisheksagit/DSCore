"""Accuracy harness.

Reports the numbers that decide whether this is allowed to talk to customers:

* auto-answer rate - how often it resolves an enquiry unaided;
* ERROR RATE among auto-answers - how often it answers CONFIDENTLY and WRONG.
  This is the number that matters. A low auto-answer rate is a cost problem; a
  non-zero error rate is a reputation problem.
* escalation rate on garments that are not ours - these must not be answered.

Run against synthetic fixtures today; point `--catalogue` and `--labelled` at
Panache's real catalogue and labelled enquiry images when they exist. The
synthetic numbers validate the mechanism, they do not predict production.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fixtures import (  # noqa: E402
    DISTORTIONS,
    build_catalogue,
    foreign_garment,
    memory_loader,
    simulate_customer_image,
)

from matcher.config import Config, RetrievalWeights, Thresholds  # noqa: E402
from matcher.embed import ClassicalEmbedder  # noqa: E402
from matcher.gate import Action  # noqa: E402
from matcher.index import ProductIndex  # noqa: E402
from matcher.gate import Verdict  # noqa: E402
from matcher.pipeline import Matcher  # noqa: E402
from matcher.verify import VerificationResult  # noqa: E402


class SimulatedVerifier:
    """Stand-in for the Claude verification stage, at a STATED accuracy.

    Sensitivity analysis, not a measurement: it answers "given a verifier that
    is right P of the time, what does the gate turn that into?" - which is how
    you check that the gate dampens verifier error instead of amplifying it.
    Real figures need a real key and Panache's labelled images.

    The harness sets `truth_product_id` before each query.
    """

    def __init__(
        self,
        accuracy: float = 0.90,
        high_conf_rate: float = 0.85,
        foreign_reject_rate: float = 0.90,
        seed: int = 11,
    ) -> None:
        self.accuracy = accuracy
        self.high_conf_rate = high_conf_rate
        self.foreign_reject_rate = foreign_reject_rate
        self.rng = random.Random(seed)
        self.truth_product_id: str | None = None
        self.calls = 0

    def __call__(self, image, candidates, catalogue_images, config=None, client=None):
        self.calls += 1
        ids = [c.product.product_id for c in candidates]
        truth_index = (
            ids.index(self.truth_product_id) if self.truth_product_id in ids else None
        )

        if truth_index is None:
            # Not in the shortlist: a good verifier returns "none of these".
            chosen = None if self.rng.random() < self.foreign_reject_rate else self.rng.choice(ids)
        elif self.rng.random() < self.accuracy:
            chosen = ids[truth_index]
        else:
            others = [i for i in ids if i != self.truth_product_id]
            chosen = self.rng.choice(others) if others else self.truth_product_id

        roll = self.rng.random()
        confidence = "high" if roll < self.high_conf_rate else ("medium" if roll < 0.95 else "low")
        return VerificationResult(
            verdict=Verdict(chosen, confidence, "simulated", available=True),
            raw={"product_index": None if chosen is None else ids.index(chosen) + 1,
                 "confidence": confidence},
            model="simulated-verifier",
        )


@dataclass
class Tally:
    total: int = 0
    answered_right: int = 0
    answered_wrong: int = 0
    confirmed_right: int = 0
    confirmed_wrong: int = 0
    escalated: int = 0
    margins: list[float] = field(default_factory=list)

    @property
    def answered(self) -> int:
        return self.answered_right + self.answered_wrong

    @property
    def confirmed(self) -> int:
        return self.confirmed_right + self.confirmed_wrong

    def row(self, label: str) -> str:
        pct = lambda n: f"{100 * n / self.total:5.1f}%" if self.total else "    -"
        err = (
            f"{100 * self.answered_wrong / self.answered:5.1f}%"
            if self.answered
            else "    -"
        )
        return (
            f"{label:<18} n={self.total:<4} "
            f"auto={pct(self.answered)} (wrong {err})  "
            f"confirm={pct(self.confirmed)}  escalate={pct(self.escalated)}"
        )


def run(
    config: Config,
    n_products: int,
    n_foreign: int,
    verification: bool,
    verifier: SimulatedVerifier | None = None,
) -> dict:
    catalogue, store = build_catalogue(n_products=n_products)
    index = ProductIndex(catalogue, ClassicalEmbedder(), config).build(memory_loader(store))
    matcher = Matcher(
        index,
        store,
        config=config,
        verification_enabled=verification or verifier is not None,
        verifier=verifier,
    )

    by_distortion: dict[str, Tally] = {d.name: Tally() for d in DISTORTIONS}
    overall = Tally()

    for distortion in DISTORTIONS:
        tally = by_distortion[distortion.name]
        for i, product in enumerate(catalogue.products):
            for shot, product_image in enumerate(product.images):
                query = simulate_customer_image(
                    store[product_image.image_id], distortion.name, seed=i * 7 + shot
                )
                if verifier is not None:
                    verifier.truth_product_id = product.product_id
                outcome = matcher.match(query, "price please")
                correct = (
                    outcome.product is not None
                    and outcome.product.product_id == product.product_id
                )
                for target in (tally, overall):
                    target.total += 1
                    target.margins.append(outcome.decision.margin_ratio)
                    if outcome.action is Action.ANSWER:
                        if correct:
                            target.answered_right += 1
                        else:
                            target.answered_wrong += 1
                    elif outcome.action is Action.CONFIRM:
                        if correct:
                            target.confirmed_right += 1
                        else:
                            target.confirmed_wrong += 1
                    else:
                        target.escalated += 1

    # Garments that are not ours. The only acceptable outcomes are escalate,
    # or at worst a confirmation the customer will reject. An auto-answer here
    # is a hard failure.
    foreign = Tally()
    foreign_actions: Counter[str] = Counter()
    for seed in range(n_foreign):
        if verifier is not None:
            verifier.truth_product_id = None  # genuinely not in the catalogue
        outcome = matcher.match(foreign_garment(seed=seed), "price please")
        foreign.total += 1
        foreign_actions[outcome.action.value] += 1
        if outcome.action is Action.ANSWER:
            foreign.answered_wrong += 1
        elif outcome.action is Action.CONFIRM:
            foreign.confirmed_wrong += 1
        else:
            foreign.escalated += 1

    return {
        "config": {
            "candidate_floor": config.thresholds.candidate_floor,
            "margin_clear_ratio": config.thresholds.margin_clear_ratio,
            "dup_strong_max": config.thresholds.dup_strong_max,
            "verification_enabled": verification or verifier is not None,
            "verifier": "simulated" if verifier is not None else ("real" if verification else "none"),
            "embedder": index.embedder.name,
        },
        "coverage": index.coverage(),
        "by_distortion": by_distortion,
        "overall": overall,
        "foreign": foreign,
        "foreign_actions": dict(foreign_actions),
    }


def sweep_floor(n_products: int, n_foreign: int) -> list[dict]:
    """Find the candidate floor that stops foreign garments being offered.

    This is the calibration step. The floor should sit above where garments we
    do not stock score, and below where genuine matches score.
    """
    rows = []
    for floor in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        config = Config(
            weights=RetrievalWeights(),
            thresholds=Thresholds(candidate_floor=floor),
        )
        result = run(config, n_products, n_foreign, verification=False)
        overall, foreign = result["overall"], result["foreign"]
        rows.append(
            {
                "floor": floor,
                "genuine_auto_pct": 100 * overall.answered / overall.total,
                "genuine_auto_wrong": overall.answered_wrong,
                "genuine_escalate_pct": 100 * overall.escalated / overall.total,
                "foreign_offered": foreign.answered_wrong + foreign.confirmed_wrong,
                "foreign_escalate_pct": 100 * foreign.escalated / foreign.total,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Panache matcher accuracy harness")
    parser.add_argument("--products", type=int, default=24)
    parser.add_argument("--foreign", type=int, default=24)
    parser.add_argument("--verification", action="store_true",
                        help="call the real verification model (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--simulate-verification", type=float, default=None,
                        metavar="ACCURACY",
                        help="simulate the verification stage at this accuracy "
                             "(0-1) instead of calling the real model")
    parser.add_argument("--sweep", action="store_true", help="calibrate the candidate floor")
    parser.add_argument("--json", type=str, default=None, help="write raw results here")
    args = parser.parse_args()

    print("=" * 78)
    print("PANACHE MATCHER - ACCURACY HARNESS (SYNTHETIC FIXTURES)")
    print("These numbers validate the mechanism. They are NOT a prediction of")
    print("production accuracy, which requires the labelled real-image set.")
    print("=" * 78)

    verifier = (
        SimulatedVerifier(accuracy=args.simulate_verification)
        if args.simulate_verification is not None
        else None
    )
    if verifier is not None:
        print()
        print(f"!! verification SIMULATED at accuracy={verifier.accuracy:.2f}, "
              f"high-confidence rate={verifier.high_conf_rate:.2f}. "
              "Sensitivity analysis only.")
    result = run(Config(), args.products, args.foreign, args.verification, verifier)
    print()
    print("config:", json.dumps(result["config"]))
    print("coverage:", json.dumps(result["coverage"]))
    print()
    print("Genuine enquiries (image IS in the catalogue)")
    print("-" * 78)
    for name, tally in result["by_distortion"].items():
        print(" ", tally.row(name))
    print(" ", result["overall"].row("ALL"))
    print(f"  median margin ratio: {statistics.median(result['overall'].margins):.3f}")
    print()
    print("Garments we do NOT stock (correct outcome: escalate)")
    print("-" * 78)
    print(" ", result["foreign"].row("foreign"))
    print("  actions:", json.dumps(result["foreign_actions"]))

    errors = result["overall"].answered_wrong + result["foreign"].answered_wrong
    print()
    print("=" * 78)
    print(f"CONFIDENT-AND-WRONG ANSWERS: {errors}")
    print("(the go-live gate: this must be at or near zero on REAL labelled data)")
    print("=" * 78)

    if args.sweep:
        print()
        print("Candidate-floor calibration")
        print("-" * 78)
        print(f"{'floor':>6} {'genuine auto':>13} {'wrong':>6} {'genuine esc':>12}"
              f" {'foreign offered':>16} {'foreign esc':>12}")
        for row in sweep_floor(args.products, args.foreign):
            print(f"{row['floor']:>6.2f} {row['genuine_auto_pct']:>12.1f}%"
                  f" {row['genuine_auto_wrong']:>6} {row['genuine_escalate_pct']:>11.1f}%"
                  f" {row['foreign_offered']:>16} {row['foreign_escalate_pct']:>11.1f}%")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "config": result["config"],
                    "coverage": result["coverage"],
                    "overall": vars(result["overall"]),
                    "foreign": vars(result["foreign"]),
                    "foreign_actions": result["foreign_actions"],
                },
                indent=2,
                default=list,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
