"""The gate is the only component that decides what a customer sees, so its
rules are pinned individually."""
import pytest

from matcher.catalogue import Product, Variant
from matcher.config import Config, Thresholds
from matcher.gate import Action, Verdict, colour_conflict, decide, unambiguous_near_duplicate
from matcher.index import Candidate, ScoreBreakdown


def prod(pid, colour="red"):
    p = Product(pid, f"h-{pid}", f"T {pid}", 19900,
                variants=[Variant(f"{pid}-42", "42", 3)])
    p.colour_family = colour
    return p


def cand(pid, fused, dup, colour="red"):
    return Candidate(
        prod(pid, colour),
        ScoreBreakdown(fused, 0.0, fused, 0.0, dup, f"{pid}-img", "full", "full"),
    )


CFG = Config()


def test_no_candidates_is_its_own_action():
    assert decide([]).action is Action.NO_CANDIDATES


def test_unique_near_duplicate_answers_without_verification():
    cands = [cand("a", 0.95, 1), cand("b", 0.70, 20)]
    assert unambiguous_near_duplicate(cands, CFG)
    d = decide(cands, None, CFG, query_colour_family="red")
    assert d.action is Action.ANSWER and d.product_id == "a"


def test_contested_near_duplicate_does_not_answer_unverified():
    """The bug the eval caught: proximity is not identity. Two products within
    a few bits means the hash cannot tell them apart."""
    cands = [cand("a", 0.95, 1), cand("b", 0.94, 3)]
    assert not unambiguous_near_duplicate(cands, CFG)
    assert decide(cands, None, CFG, query_colour_family="red").action is Action.CONFIRM


def test_near_duplicate_blocked_when_colour_contradicts():
    cands = [cand("a", 0.95, 0, colour="black"), cand("b", 0.60, 30)]
    d = decide(cands, None, CFG, query_colour_family="red")
    assert d.action is Action.CONFIRM
    assert "colour" in " ".join(d.reasons)


def test_neighbouring_colours_are_not_a_conflict():
    assert colour_conflict("red", "maroon") is False
    assert colour_conflict("navy", "blue") is False
    assert colour_conflict("red", "green") is True
    assert colour_conflict(None, "red") is False       # absence is not disagreement
    assert colour_conflict("multi", "red") is False


def test_below_floor_escalates():
    cands = [cand("a", 0.10, 40), cand("b", 0.05, 44)]
    d = decide(cands, Verdict("a", "high"), CFG)
    assert d.action is Action.ESCALATE and "floor" in " ".join(d.reasons)


def test_unavailable_verification_caps_at_confirm():
    cands = [cand("a", 0.80, 30), cand("b", 0.40, 40)]
    d = decide(cands, Verdict(None, "low", available=False), CFG, query_colour_family="red")
    assert d.action is Action.CONFIRM


def test_verifier_rejecting_everything_escalates():
    """How a competitor's garment reaches a human."""
    cands = [cand("a", 0.80, 30), cand("b", 0.40, 40)]
    d = decide(cands, Verdict(None, "high"), CFG)
    assert d.action is Action.ESCALATE


def test_high_confidence_with_clear_margin_answers():
    cands = [cand("a", 0.90, 30), cand("b", 0.50, 40)]
    d = decide(cands, Verdict("a", "high"), CFG, query_colour_family="red")
    assert d.action is Action.ANSWER


def test_high_confidence_but_thin_margin_confirms():
    cands = [cand("a", 0.90, 30), cand("b", 0.895, 40)]
    d = decide(cands, Verdict("a", "high"), CFG, query_colour_family="red")
    assert d.action is Action.CONFIRM and "contested" in " ".join(d.reasons)


def test_medium_confidence_always_confirms():
    cands = [cand("a", 0.90, 30), cand("b", 0.20, 40)]
    assert decide(cands, Verdict("a", "medium"), CFG).action is Action.CONFIRM


def test_low_confidence_escalates():
    cands = [cand("a", 0.90, 30), cand("b", 0.20, 40)]
    assert decide(cands, Verdict("a", "low"), CFG).action is Action.ESCALATE


def test_verifier_override_confirms_rather_than_answering():
    """Signals disagreeing is where confident-wrong answers concentrated."""
    cands = [cand("a", 0.90, 30), cand("b", 0.60, 40)]
    d = decide(cands, Verdict("b", "high"), CFG, query_colour_family="red")
    assert d.action is Action.CONFIRM and d.product_id == "b"
    assert "disagree" in " ".join(d.reasons)


def test_verifier_naming_a_non_candidate_escalates():
    cands = [cand("a", 0.90, 30)]
    d = decide(cands, Verdict("zzz", "high"), CFG)
    assert d.action is Action.ESCALATE


def test_verification_disagreeing_with_a_unique_hash_downgrades_to_confirm():
    cands = [cand("a", 0.95, 1), cand("b", 0.70, 25)]
    d = decide(cands, Verdict("b", "high"), CFG, query_colour_family="red")
    assert d.action is Action.CONFIRM and d.product_id == "b"


def test_thresholds_are_configurable():
    cands = [cand("a", 0.95, 1), cand("b", 0.94, 3)]
    loose = Config(thresholds=Thresholds(dup_unique_margin=1))
    assert decide(cands, None, loose, query_colour_family="red").action is Action.ANSWER
