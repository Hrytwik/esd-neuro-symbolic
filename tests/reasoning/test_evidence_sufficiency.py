"""
Tests for EvidenceSufficiencyAnalyzer — evidence quality and coverage.

Validates anatomical domain coverage, tier diversity scoring, rule adequacy,
consistency computation, aggregate sufficiency, and biopsy-free decision.
"""

import pytest

from src.reasoning.evidence_sufficiency import EvidenceSufficiencyAnalyzer, SufficiencyReport


# ── Sufficient evidence scenario ──────────────────────────────────────────────

class TestSufficientEvidence:
    def test_analyze_returns_sufficiency_report(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(
            evidence=psoriasis_evidence_result,
            certainty=stable_certainty,
        )
        assert isinstance(report, SufficiencyReport)

    def test_disease_identified_correctly(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert report.disease == "psoriasis"

    def test_tier_a_evidence_detected(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert report.has_tier_a_evidence

    def test_tier_b_evidence_detected(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert report.has_tier_b_evidence

    def test_tier_diversity_both_tiers(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert report.tier_diversity_score == pytest.approx(1.0)

    def test_aggregate_sufficiency_in_range(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert 0.0 <= report.aggregate_sufficiency <= 1.0

    def test_rule_count_captured(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert report.active_rule_count == 3  # psoriasis has 3 active rules


# ── Empty/insufficient evidence scenario ──────────────────────────────────────

class TestInsufficientEvidence:
    def test_empty_report_for_no_active_rules(
        self, sufficiency_analyzer, ambiguous_certainty
    ):
        from src.reasoning.evidence_evaluator import DiseaseEvidenceVector, EvidenceEvaluationResult

        # Build evidence with zero active rules for leading disease
        vectors = {
            "psoriasis": DiseaseEvidenceVector(
                disease="psoriasis", raw_evidence_score=0.0,
                tier_a_score=0.0, tier_b_score=0.0, tier_d_score=0.0,
                active_rule_count=0,
            )
        }
        for d in ["seborrheic_dermatitis", "lichen_planus", "pityriasis_rosea",
                  "chronic_dermatitis", "pityriasis_rubra_pilaris"]:
            vectors[d] = DiseaseEvidenceVector(
                disease=d, raw_evidence_score=0.0,
                tier_a_score=0.0, tier_b_score=0.0, tier_d_score=0.0,
            )

        empty_evidence = EvidenceEvaluationResult(
            disease_vectors=vectors,
            evaluated_tiers=["A", "B"],
            total_rules_checked=0,
            total_rules_active=0,
            leading_disease="psoriasis",
            second_disease="seborrheic_dermatitis",
        )
        report = sufficiency_analyzer.analyze(empty_evidence, ambiguous_certainty)
        assert not report.is_biopsy_free_sufficient
        assert report.fragility_risk == "high"
        assert len(report.insufficiency_reasons) >= 1


# ── Domain coverage ───────────────────────────────────────────────────────────

class TestDomainCoverage:
    def test_domain_coverage_fraction_in_range(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert 0.0 <= report.domain_coverage_fraction <= 1.0

    def test_anatomical_domains_covered_is_list(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert isinstance(report.anatomical_domains_covered, list)


# ── Fragility risk ────────────────────────────────────────────────────────────

class TestFragilityRisk:
    def test_fragility_risk_valid_values(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert report.fragility_risk in ("low", "moderate", "high")

    def test_low_aggregate_gives_high_fragility(
        self, sufficiency_analyzer, ambiguous_certainty
    ):
        from src.reasoning.evidence_evaluator import DiseaseEvidenceVector, EvidenceEvaluationResult

        # Minimal evidence — only 1 weak rule
        vectors = {}
        for d in ["psoriasis", "seborrheic_dermatitis", "lichen_planus",
                  "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"]:
            vectors[d] = DiseaseEvidenceVector(
                disease=d, raw_evidence_score=0.0,
                tier_a_score=0.0, tier_b_score=0.0, tier_d_score=0.0,
            )
        from tests.reasoning.conftest import _make_rule_result
        pso = vectors["psoriasis"]
        pso.activated_rules = [_make_rule_result("PSO_005", "psoriasis", "B", 0.12, status="partial")]
        pso.tier_b_score = 0.12
        pso.raw_evidence_score = 0.12
        pso.active_rule_count = 1
        pso.tier_b_count = 1

        minimal_evidence = EvidenceEvaluationResult(
            disease_vectors=vectors,
            evaluated_tiers=["B"],
            total_rules_checked=1,
            total_rules_active=1,
            leading_disease="psoriasis",
            second_disease="seborrheic_dermatitis",
        )
        report = sufficiency_analyzer.analyze(minimal_evidence, ambiguous_certainty)
        assert report.fragility_risk in ("moderate", "high")


# ── Consistency score ─────────────────────────────────────────────────────────

class TestConsistencyScore:
    def test_consistency_in_range(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        assert 0.0 <= report.consistency_score <= 1.0

    def test_single_rule_consistency_is_partial(self):
        """Single active rule → consistency = 0.5 (per implementation)."""
        from src.reasoning.evidence_sufficiency import EvidenceSufficiencyAnalyzer
        from tests.reasoning.conftest import _make_rule_result
        rule = _make_rule_result("PSO_001", "psoriasis", "A", 0.85, features=["koebner_phenomenon"])
        score = EvidenceSufficiencyAnalyzer._activation_consistency([rule])
        assert score == pytest.approx(0.5)

    def test_empty_active_rules_consistency_zero(self):
        from src.reasoning.evidence_sufficiency import EvidenceSufficiencyAnalyzer
        assert EvidenceSufficiencyAnalyzer._activation_consistency([]) == 0.0


# ── Summary property ──────────────────────────────────────────────────────────

class TestSummaryProperty:
    def test_summary_is_string(
        self, sufficiency_analyzer, psoriasis_evidence_result, stable_certainty
    ):
        report = sufficiency_analyzer.analyze(psoriasis_evidence_result, stable_certainty)
        summary = report.summary
        assert isinstance(summary, str)
        assert "psoriasis" in summary.lower()
