"""
Tests for ClinicalSafetyGate — safety invariants and conditional gates.

Validates all 3 invariants (I1–I3) and 5 gates (G1–G5), the effective
cap selection, certainty penalty application, and the escalation-only
semantics (cap cannot reduce severity).
"""

import pytest

from src.reasoning.safety_gate import (
    ClinicalSafetyGate,
    SafetyGateReport,
    TriageRecommendation,
)


# ── TriageRecommendation severity ranks ───────────────────────────────────────

class TestTriageRecommendationRanks:
    def test_severity_ranks_ordered(self):
        assert TriageRecommendation.SAFE_NON_INVASIVE_TRIAGE.severity_rank == 0
        assert TriageRecommendation.MODERATE_CERTAINTY.severity_rank == 1
        assert TriageRecommendation.AMBIGUOUS_PRESENTATION.severity_rank == 2
        assert TriageRecommendation.BIOPSY_RECOMMENDED.severity_rank == 3
        assert TriageRecommendation.HIGH_RISK_CONTRADICTION.severity_rank == 4


# ── Invariant I1 — Contradiction Safety Ceiling ───────────────────────────────

class TestI1ContradictionCeiling:
    def test_i1_triggers_on_high_load(
        self, safety_gate, stable_certainty, high_conflict_result, psoriasis_evidence_result
    ):
        report = safety_gate.evaluate(stable_certainty, high_conflict_result, psoriasis_evidence_result)
        i1 = next(r for r in report.invariant_results if r.gate_id == "I1")
        assert i1.triggered
        assert i1.cap == TriageRecommendation.BIOPSY_RECOMMENDED

    def test_i1_does_not_trigger_on_safe_load(
        self, safety_gate, stable_certainty, no_conflict_result, psoriasis_evidence_result
    ):
        report = safety_gate.evaluate(stable_certainty, no_conflict_result, psoriasis_evidence_result)
        i1 = next(r for r in report.invariant_results if r.gate_id == "I1")
        assert not i1.triggered


# ── Invariant I2 — Evidence Sufficiency Floor ─────────────────────────────────

class TestI2EvidenceSufficiency:
    def test_i2_triggers_when_too_few_rules(
        self, safety_gate, stable_certainty, no_conflict_result, psoriasis_evidence_result
    ):
        gate = ClinicalSafetyGate(min_activated_rules=100)
        report = gate.evaluate(stable_certainty, no_conflict_result, psoriasis_evidence_result)
        i2 = next(r for r in report.invariant_results if r.gate_id == "I2")
        assert i2.triggered
        assert i2.cap == TriageRecommendation.AMBIGUOUS_PRESENTATION

    def test_i2_does_not_trigger_with_sufficient_rules(
        self, safety_gate, stable_certainty, no_conflict_result, psoriasis_evidence_result
    ):
        # psoriasis_evidence_result has 4 active rules; default min is 2
        report = safety_gate.evaluate(stable_certainty, no_conflict_result, psoriasis_evidence_result)
        i2 = next(r for r in report.invariant_results if r.gate_id == "I2")
        assert not i2.triggered


# ── Invariant I3 — Entropy Escalation Ceiling ────────────────────────────────

class TestI3EntropyCeiling:
    def test_i3_triggers_on_high_entropy(
        self, safety_gate, ambiguous_certainty, no_conflict_result, psoriasis_evidence_result
    ):
        # ambiguous_certainty has entropy=2.30 > ceiling=1.50
        report = safety_gate.evaluate(
            ambiguous_certainty, no_conflict_result, psoriasis_evidence_result
        )
        i3 = next(r for r in report.invariant_results if r.gate_id == "I3")
        assert i3.triggered
        assert i3.cap == TriageRecommendation.BIOPSY_RECOMMENDED

    def test_i3_does_not_trigger_on_low_entropy(
        self, safety_gate, stable_certainty, no_conflict_result, psoriasis_evidence_result
    ):
        report = safety_gate.evaluate(stable_certainty, no_conflict_result, psoriasis_evidence_result)
        i3 = next(r for r in report.invariant_results if r.gate_id == "I3")
        assert not i3.triggered


# ── Gate G1 — Single-Source Dominance ────────────────────────────────────────

class TestG1SingleSourceDominance:
    def test_g1_triggers_when_single_rule_active(
        self, safety_gate, stable_certainty, no_conflict_result, psoriasis_evidence_result
    ):
        # Artificially force single-rule scenario
        from src.reasoning.evidence_evaluator import DiseaseEvidenceVector, EvidenceEvaluationResult
        from tests.reasoning.conftest import _make_rule_result

        vectors = {}
        for d in ["psoriasis", "seborrheic_dermatitis", "lichen_planus",
                  "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"]:
            vectors[d] = DiseaseEvidenceVector(
                disease=d, raw_evidence_score=0.0,
                tier_a_score=0.0, tier_b_score=0.0, tier_d_score=0.0,
            )

        pso = vectors["psoriasis"]
        pso.activated_rules = [_make_rule_result("PSO_001", "psoriasis", "A", 0.85,
                                                  features=["koebner_phenomenon"])]
        pso.active_rule_count = 1
        pso.raw_evidence_score = 0.85
        pso.has_pathognomonic = True

        single_rule_evidence = EvidenceEvaluationResult(
            disease_vectors=vectors,
            evaluated_tiers=["A"],
            total_rules_checked=1,
            total_rules_active=1,
            leading_disease="psoriasis",
            second_disease="seborrheic_dermatitis",
        )
        report = safety_gate.evaluate(stable_certainty, no_conflict_result, single_rule_evidence)
        g1 = next(r for r in report.gate_results if r.gate_id == "G1")
        assert g1.triggered


# ── Gate G2 — Pathognomonic Absence ──────────────────────────────────────────

class TestG2PathognomicAbsence:
    def test_g2_triggers_high_certainty_without_tier_a(
        self, safety_gate, no_conflict_result, psoriasis_evidence_result
    ):
        from src.reasoning.certainty_propagator import CertaintyDistribution
        from tests.reasoning.conftest import _make_hyp

        # High certainty (>0.75) but modify evidence to have no pathognomonic
        from src.reasoning.evidence_evaluator import DiseaseEvidenceVector, EvidenceEvaluationResult
        vectors = {}
        for d in ["psoriasis", "seborrheic_dermatitis", "lichen_planus",
                  "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"]:
            vectors[d] = DiseaseEvidenceVector(
                disease=d, raw_evidence_score=0.0,
                tier_a_score=0.0, tier_b_score=0.0, tier_d_score=0.0,
            )
        pso = vectors["psoriasis"]
        pso.raw_evidence_score = 1.0
        pso.active_rule_count = 2
        pso.has_pathognomonic = False  # no Tier-A

        no_patho_evidence = EvidenceEvaluationResult(
            disease_vectors=vectors,
            evaluated_tiers=["B"],
            total_rules_checked=2,
            total_rules_active=2,
            leading_disease="psoriasis",
            second_disease="seborrheic_dermatitis",
        )

        # certainty > 0.75 without pathognomonic → G2 triggers
        high_no_patho = CertaintyDistribution(
            hypotheses=[_make_hyp("psoriasis", 0.80, 1, tier_a=0, patho=False)]
                      + [_make_hyp(d, 0.04, i, tier_a=0, patho=False)
                         for i, d in enumerate(
                             ["seborrheic_dermatitis","lichen_planus",
                              "pityriasis_rosea","chronic_dermatitis","pityriasis_rubra_pilaris"], 2)],
            leading_disease="psoriasis",
            second_disease="seborrheic_dermatitis",
            max_certainty=0.80,
            certainty_gap=0.76,
            ambiguity_index=0.50,
            contradiction_load=0.0,
            contradiction_dampened=False,
            is_stable=True, is_highly_certain=True, is_ambiguous=False,
        )
        report = safety_gate.evaluate(high_no_patho, no_conflict_result, no_patho_evidence)
        g2 = next(r for r in report.gate_results if r.gate_id == "G2")
        assert g2.triggered
        assert g2.cap == TriageRecommendation.MODERATE_CERTAINTY


# ── Gate G3 — Critical Feature Missingness ───────────────────────────────────

class TestG3CriticalMissingness:
    def test_g3_triggers_when_three_or_more_critical_missing(
        self, safety_gate, stable_certainty, no_conflict_result, psoriasis_evidence_result
    ):
        missing = [
            "koebner_phenomenon", "polygonal_papules",
            "follicular_papules",  # 3 critical features missing → triggers
        ]
        report = safety_gate.evaluate(
            stable_certainty, no_conflict_result, psoriasis_evidence_result,
            missing_features=missing,
        )
        g3 = next(r for r in report.gate_results if r.gate_id == "G3")
        assert g3.triggered
        assert g3.cap == TriageRecommendation.AMBIGUOUS_PRESENTATION

    def test_g3_does_not_trigger_with_few_missing(
        self, safety_gate, stable_certainty, no_conflict_result, psoriasis_evidence_result
    ):
        report = safety_gate.evaluate(
            stable_certainty, no_conflict_result, psoriasis_evidence_result,
            missing_features=["koebner_phenomenon"],
        )
        g3 = next(r for r in report.gate_results if r.gate_id == "G3")
        assert not g3.triggered


# ── Gate G5 — Overconfidence Prevention ──────────────────────────────────────

class TestG5Overconfidence:
    def test_g5_triggers_high_certainty_with_contradiction(
        self, safety_gate, moderate_conflict_result, psoriasis_evidence_result
    ):
        from src.reasoning.certainty_propagator import CertaintyDistribution
        from tests.reasoning.conftest import _make_hyp

        overconf = CertaintyDistribution(
            hypotheses=[_make_hyp("psoriasis", 0.95, 1)]
                      + [_make_hyp(d, 0.01, i, tier_a=0, patho=False)
                         for i, d in enumerate(
                             ["seborrheic_dermatitis","lichen_planus",
                              "pityriasis_rosea","chronic_dermatitis","pityriasis_rubra_pilaris"], 2)],
            leading_disease="psoriasis",
            second_disease="seborrheic_dermatitis",
            max_certainty=0.95,
            certainty_gap=0.94,
            ambiguity_index=0.20,
            contradiction_load=0.30,    # >= 0.10 threshold
            contradiction_dampened=True,
            is_stable=True, is_highly_certain=True, is_ambiguous=False,
        )
        report = safety_gate.evaluate(overconf, moderate_conflict_result, psoriasis_evidence_result)
        g5 = next(r for r in report.gate_results if r.gate_id == "G5")
        assert g5.triggered
        assert g5.cap == TriageRecommendation.MODERATE_CERTAINTY


# ── Effective cap ─────────────────────────────────────────────────────────────

class TestEffectiveCap:
    def test_effective_cap_is_most_severe(
        self, safety_gate, ambiguous_certainty, high_conflict_result, psoriasis_evidence_result
    ):
        report = safety_gate.evaluate(
            ambiguous_certainty, high_conflict_result, psoriasis_evidence_result
        )
        # Both I1 (BIOPSY) and I3 (BIOPSY) triggered; max severity is BIOPSY
        assert report.effective_cap == TriageRecommendation.BIOPSY_RECOMMENDED

    def test_apply_cap_escalates_base_recommendation(self):
        report = SafetyGateReport(
            effective_cap=TriageRecommendation.BIOPSY_RECOMMENDED,
            any_triggered=True,
        )
        result = report.apply_cap(TriageRecommendation.SAFE_NON_INVASIVE_TRIAGE)
        assert result == TriageRecommendation.BIOPSY_RECOMMENDED

    def test_apply_cap_does_not_downgrade(self):
        report = SafetyGateReport(
            effective_cap=TriageRecommendation.MODERATE_CERTAINTY,
            any_triggered=True,
        )
        # Base is already more severe
        result = report.apply_cap(TriageRecommendation.BIOPSY_RECOMMENDED)
        assert result == TriageRecommendation.BIOPSY_RECOMMENDED

    def test_no_cap_returns_base(self):
        report = SafetyGateReport(effective_cap=None, any_triggered=False)
        result = report.apply_cap(TriageRecommendation.SAFE_NON_INVASIVE_TRIAGE)
        assert result == TriageRecommendation.SAFE_NON_INVASIVE_TRIAGE
