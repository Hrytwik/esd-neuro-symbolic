"""
Tests for HypothesisCertaintyPropagator — softmax normalisation and
contradiction dampening.

Validates certainty distribution properties, certainty gap calculation,
Shannon entropy, dampening behaviour, and stability classification.
"""

import math
import pytest

from src.reasoning.certainty_propagator import (
    CertaintyDistribution,
    HypothesisCertaintyPropagator,
)


# ── Certainty distribution properties ────────────────────────────────────────

class TestCertaintyDistribution:
    def test_certainties_sum_to_one(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        total = sum(h.certainty for h in dist.hypotheses)
        assert total == pytest.approx(1.0, rel=1e-5)

    def test_leading_disease_has_highest_certainty(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        leading_cert = dist.max_certainty
        for h in dist.hypotheses:
            assert leading_cert >= h.certainty

    def test_leading_disease_is_psoriasis(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        assert dist.leading_disease == "psoriasis"

    def test_six_hypotheses_always_present(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        assert len(dist.hypotheses) == 6

    def test_hypotheses_sorted_by_rank(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        ranks = [h.rank for h in dist.hypotheses]
        assert ranks == sorted(ranks)


# ── Certainty gap ─────────────────────────────────────────────────────────────

class TestCertaintyGap:
    def test_certainty_gap_is_top1_minus_top2(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        top1 = dist.hypotheses[0].certainty
        top2 = dist.hypotheses[1].certainty
        assert dist.certainty_gap == pytest.approx(top1 - top2, rel=1e-5)

    def test_certainty_gap_non_negative(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        assert dist.certainty_gap >= 0.0


# ── Shannon entropy ───────────────────────────────────────────────────────────

class TestShannonEntropy:
    def test_entropy_is_non_negative(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        assert dist.ambiguity_index >= 0.0

    def test_entropy_higher_for_uniform_distribution(self, certainty_propagator):
        """Uniform distribution has maximum entropy; peaked has low entropy."""
        assert HypothesisCertaintyPropagator._shannon_entropy(
            {"a": 1/6, "b": 1/6, "c": 1/6, "d": 1/6, "e": 1/6, "f": 1/6}
        ) == pytest.approx(math.log2(6), rel=1e-4)

    def test_entropy_zero_for_certain_distribution(self):
        # All mass on one disease
        h = HypothesisCertaintyPropagator._shannon_entropy(
            {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0, "f": 0.0}
        )
        assert h == pytest.approx(0.0, abs=1e-9)


# ── Contradiction dampening ───────────────────────────────────────────────────

class TestContradictionDampening:
    def test_dampening_activates_at_threshold(
        self, certainty_propagator, psoriasis_evidence_result, moderate_conflict_result
    ):
        # moderate_conflict_result has load=0.30 > 0.20 damping threshold
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, moderate_conflict_result
        )
        assert dist.contradiction_dampened

    def test_dampening_not_applied_below_threshold(
        self, certainty_propagator, psoriasis_evidence_result, no_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        assert not dist.contradiction_dampened

    def test_dampening_reduces_leading_certainty(
        self, certainty_propagator, psoriasis_evidence_result,
        no_conflict_result, moderate_conflict_result
    ):
        undamped = certainty_propagator.propagate(
            psoriasis_evidence_result, no_conflict_result
        )
        damped = certainty_propagator.propagate(
            psoriasis_evidence_result, moderate_conflict_result
        )
        assert damped.max_certainty <= undamped.max_certainty

    def test_certainties_still_sum_to_one_after_dampening(
        self, certainty_propagator, psoriasis_evidence_result, moderate_conflict_result
    ):
        dist = certainty_propagator.propagate(
            psoriasis_evidence_result, moderate_conflict_result
        )
        total = sum(h.certainty for h in dist.hypotheses)
        assert total == pytest.approx(1.0, rel=1e-5)


# ── Stability classification ──────────────────────────────────────────────────

class TestStabilityClassification:
    def test_stable_certainty_is_stable(self, stable_certainty):
        assert stable_certainty.is_stable

    def test_ambiguous_certainty_is_ambiguous(self, ambiguous_certainty):
        assert ambiguous_certainty.is_ambiguous
        assert not ambiguous_certainty.is_stable

    def test_high_certainty_is_highly_certain(self, high_certainty_dist):
        assert high_certainty_dist.is_highly_certain

    def test_certainty_for_query(self, stable_certainty):
        cert = stable_certainty.certainty_for("psoriasis")
        assert cert == pytest.approx(0.72)

    def test_certainty_for_unknown_returns_zero(self, stable_certainty):
        assert stable_certainty.certainty_for("unknown_disease") == 0.0

    def test_top_n_returns_correct_count(self, stable_certainty):
        top3 = stable_certainty.top_n(3)
        assert len(top3) == 3
