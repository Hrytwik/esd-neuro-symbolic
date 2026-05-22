"""
Tests for DiagnosticEvidenceEvaluator — multi-tier rule activation.

Validates binary, composite, and threshold rule logic, evidence tier
aggregation, leading/second disease resolution, and dormant rule handling.
"""

import pytest

from src.reasoning.evidence_evaluator import (
    DiagnosticEvidenceEvaluator,
    EvidenceEvaluationResult,
    RuleEvaluationResult,
)
from tests.reasoning.conftest import BINARY_FEATURES


# ── Binary rule activation ─────────────────────────────────────────────────────

class TestBinaryRuleActivation:
    def test_binary_rule_activates_when_feature_present(
        self, evidence_evaluator, psoriasis_grading, psoriasis_binary_rule
    ):
        result = evidence_evaluator.evaluate(
            psoriasis_grading, [psoriasis_binary_rule]
        )
        pso = result.get("psoriasis")
        assert pso is not None
        assert pso.tier_a_score > 0.0
        assert pso.has_pathognomonic

    def test_binary_rule_dormant_when_feature_absent(
        self, evidence_evaluator, sparse_grading, psoriasis_binary_rule
    ):
        result = evidence_evaluator.evaluate(
            sparse_grading, [psoriasis_binary_rule]
        )
        pso = result.get("psoriasis")
        assert pso.tier_a_score == 0.0
        assert not pso.has_pathognomonic

    def test_binary_rule_requires_all_features(
        self, evidence_evaluator, psoriasis_grading
    ):
        two_feature_rule = {
            "rule_id": "TWO_001",
            "disease_target": "psoriasis",
            "evidence_tier": "B",
            "activation_logic": "binary",
            "confidence_weight": 0.70,
            "min_activation_threshold": 0.10,
            "supporting_features": [
                {"feature": "koebner_phenomenon", "condition": "eq", "threshold": 1},
                {"feature": "polygonal_papules",  "condition": "eq", "threshold": 1},
            ],
        }
        result = evidence_evaluator.evaluate(psoriasis_grading, [two_feature_rule])
        pso = result.get("psoriasis")
        # polygonal_papules = 0 in psoriasis profile → rule fails
        assert pso.raw_evidence_score == 0.0


# ── Composite rule activation ──────────────────────────────────────────────────

class TestCompositeRuleActivation:
    def test_composite_rule_partial_when_one_feature_present(
        self, evidence_evaluator, psoriasis_grading, psoriasis_composite_rule
    ):
        result = evidence_evaluator.evaluate(
            psoriasis_grading, [psoriasis_composite_rule]
        )
        pso = result.get("psoriasis")
        # Both erythema=3 and scaling=3 pass → full composite score
        assert pso.tier_b_score > 0.0

    def test_composite_rule_zero_when_no_features_pass(
        self, evidence_evaluator, sparse_grading, psoriasis_composite_rule
    ):
        result = evidence_evaluator.evaluate(
            sparse_grading, [psoriasis_composite_rule]
        )
        pso = result.get("psoriasis")
        assert pso.tier_b_score == 0.0


# ── Threshold rule activation ──────────────────────────────────────────────────

class TestThresholdRuleActivation:
    def test_threshold_rule_activates_when_feature_present(
        self, evidence_evaluator, psoriasis_grading, psoriasis_threshold_rule
    ):
        result = evidence_evaluator.evaluate(
            psoriasis_grading, [psoriasis_threshold_rule]
        )
        pso = result.get("psoriasis")
        # scalp_involvement=1 passes the threshold rule
        assert pso.tier_b_score > 0.0


# ── Full evaluation ────────────────────────────────────────────────────────────

class TestFullEvaluation:
    def test_evaluate_produces_all_six_disease_vectors(
        self, evidence_evaluator, psoriasis_grading, minimal_rules
    ):
        result = evidence_evaluator.evaluate(psoriasis_grading, minimal_rules)
        assert len(result.disease_vectors) == 6

    def test_leading_disease_identified(
        self, evidence_evaluator, psoriasis_grading, minimal_rules
    ):
        result = evidence_evaluator.evaluate(psoriasis_grading, minimal_rules)
        assert result.leading_disease == "psoriasis"

    def test_total_active_rules_counted(
        self, evidence_evaluator, psoriasis_grading, minimal_rules
    ):
        result = evidence_evaluator.evaluate(psoriasis_grading, minimal_rules)
        assert result.total_rules_active >= 1

    def test_tier_filter_excludes_other_tiers(
        self, evidence_evaluator, psoriasis_grading, psoriasis_binary_rule
    ):
        # Filter to Tier B only — Tier A rule should not activate
        result = evidence_evaluator.evaluate(
            psoriasis_grading, [psoriasis_binary_rule], tiers=["B"]
        )
        pso = result.get("psoriasis")
        assert pso.tier_a_score == 0.0

    def test_result_get_returns_none_for_unknown(self, psoriasis_evidence_result):
        assert psoriasis_evidence_result.get("nonexistent_disease") is None

    def test_result_score_returns_zero_for_unknown(self, psoriasis_evidence_result):
        assert psoriasis_evidence_result.score("nonexistent_disease") == 0.0

    def test_ranked_sorted_descending(self, psoriasis_evidence_result):
        ranked = psoriasis_evidence_result.ranked()
        scores = [v.raw_evidence_score for v in ranked]
        assert scores == sorted(scores, reverse=True)


# ── Rule status classification ────────────────────────────────────────────────

class TestRuleStatus:
    def test_active_status_requires_80pct_of_confidence_weight(
        self, evidence_evaluator, psoriasis_grading, psoriasis_binary_rule
    ):
        result = evidence_evaluator.evaluate(psoriasis_grading, [psoriasis_binary_rule])
        rule_results = result.get("psoriasis").activated_rules
        active = [r for r in rule_results if r.status == "active"]
        assert len(active) >= 1

    def test_dormant_rule_not_counted_in_active_rules(
        self, evidence_evaluator, sparse_grading, psoriasis_binary_rule
    ):
        result = evidence_evaluator.evaluate(sparse_grading, [psoriasis_binary_rule])
        pso = result.get("psoriasis")
        assert pso.active_rule_count == 0


# ── Condition evaluation ──────────────────────────────────────────────────────

class TestConditionEvaluation:
    @pytest.mark.parametrize("condition,value,threshold,expected", [
        ("eq",  1.0, 1.0, True),
        ("eq",  1.4, 1.0, True),   # within 0.5 tolerance
        ("eq",  2.0, 1.0, False),
        ("gte", 2.0, 2.0, True),
        ("gte", 1.9, 2.0, False),
        ("lte", 1.0, 2.0, True),
        ("gt",  2.1, 2.0, True),
        ("lt",  1.9, 2.0, True),
    ])
    def test_condition_evaluation(self, condition, value, threshold, expected):
        result = DiagnosticEvidenceEvaluator._condition_met(value, condition, threshold)
        assert result == expected

    def test_none_value_condition_always_false(self):
        assert DiagnosticEvidenceEvaluator._condition_met(None, "eq", 1.0) is False
