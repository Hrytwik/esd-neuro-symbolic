"""
Tests for DiagnosticNarrativeGenerator — Stage 6 clinical narrative.

Validates section generation, full_text concatenation, safe behaviour
with no contradictions, and narrative content correctness.
"""

import pytest

from src.reasoning.narrative_generator import ClinicalNarrative, DiagnosticNarrativeGenerator
from src.reasoning.state_tracker import DiagnosticState


# ── Fixture: complete narrative ───────────────────────────────────────────────

@pytest.fixture
def complete_narrative(
    narrative_generator, psoriasis_evidence_result, no_conflict_result,
    stable_certainty, safe_safety_report, safe_triage_decision
):
    return narrative_generator.generate(
        evidence=psoriasis_evidence_result,
        conflict=no_conflict_result,
        certainty=stable_certainty,
        safety_report=safe_safety_report,
        decision=safe_triage_decision,
        final_state=DiagnosticState.SAFE_TRIAGE,
    )


# ── Section content ───────────────────────────────────────────────────────────

class TestNarrativeSections:
    def test_generate_returns_clinical_narrative(self, complete_narrative):
        assert isinstance(complete_narrative, ClinicalNarrative)

    def test_presentation_summary_not_empty(self, complete_narrative):
        assert len(complete_narrative.presentation_summary) > 0

    def test_evidence_interpretation_not_empty(self, complete_narrative):
        assert len(complete_narrative.evidence_interpretation) > 0

    def test_contradiction_summary_not_empty(self, complete_narrative):
        assert len(complete_narrative.contradiction_summary) > 0

    def test_certainty_evolution_not_empty(self, complete_narrative):
        assert len(complete_narrative.certainty_evolution) > 0

    def test_safety_assessment_not_empty(self, complete_narrative):
        assert len(complete_narrative.safety_assessment) > 0

    def test_triage_rationale_not_empty(self, complete_narrative):
        assert len(complete_narrative.triage_rationale) > 0


# ── Presentation summary ──────────────────────────────────────────────────────

class TestPresentationSummary:
    def test_mentions_feature_count(self, complete_narrative):
        # Should mention the number of contributing features
        assert "feature" in complete_narrative.presentation_summary.lower()

    def test_mentions_rule_count(self, complete_narrative):
        assert "rule" in complete_narrative.presentation_summary.lower()


# ── Contradiction summary ─────────────────────────────────────────────────────

class TestContradictionSummary:
    def test_no_contradiction_message_when_clean(self, complete_narrative):
        # no_conflict_result → contradiction-free message
        assert "no cross-disease contradiction" in complete_narrative.contradiction_summary.lower()

    def test_contradiction_summary_with_conflicts(
        self, narrative_generator, psoriasis_evidence_result, moderate_conflict_result,
        stable_certainty, safe_safety_report, safe_triage_decision
    ):
        narrative = narrative_generator.generate(
            evidence=psoriasis_evidence_result,
            conflict=moderate_conflict_result,
            certainty=stable_certainty,
            safety_report=safe_safety_report,
            decision=safe_triage_decision,
            final_state=DiagnosticState.PARTIAL_ALIGNMENT,
        )
        # Should mention contradiction load
        assert "contradiction" in narrative.contradiction_summary.lower()
        assert "koebner" in narrative.contradiction_summary.lower().replace("_", " ") \
            or "koebner phenomenon" in narrative.contradiction_summary.lower()


# ── Certainty evolution ───────────────────────────────────────────────────────

class TestCertaintyEvolution:
    def test_mentions_leading_disease(self, complete_narrative):
        assert "psoriasis" in complete_narrative.certainty_evolution.lower()

    def test_mentions_certainty_value(self, complete_narrative):
        # Certainty=0.72 should appear in text
        assert "0.72" in complete_narrative.certainty_evolution


# ── Safety assessment ─────────────────────────────────────────────────────────

class TestSafetyAssessment:
    def test_no_gate_message_when_no_gates_triggered(self, complete_narrative):
        assert "all safety invariant" in complete_narrative.safety_assessment.lower() \
            or "no safety" in complete_narrative.safety_assessment.lower()

    def test_gate_mentioned_when_triggered(
        self, narrative_generator, psoriasis_evidence_result, no_conflict_result,
        ambiguous_certainty, safe_triage_decision
    ):
        from src.reasoning.safety_gate import SafetyGateReport, GateResult, TriageRecommendation

        gate = GateResult(
            gate_id="I3", gate_name="Entropy Escalation Ceiling",
            triggered=True, cap=TriageRecommendation.BIOPSY_RECOMMENDED,
            rationale="Entropy exceeds ceiling.",
        )
        triggered_report = SafetyGateReport(
            invariant_results=[gate], gate_results=[],
            effective_cap=TriageRecommendation.BIOPSY_RECOMMENDED,
            any_triggered=True,
        )
        narrative = narrative_generator.generate(
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=ambiguous_certainty,
            safety_report=triggered_report,
            decision=safe_triage_decision,
            final_state=DiagnosticState.BIOPSY_ESCALATION,
        )
        assert "I3" in narrative.safety_assessment


# ── Triage rationale ──────────────────────────────────────────────────────────

class TestTriageRationale:
    def test_triage_rationale_mentions_recommendation(self, complete_narrative):
        assert "safe_non_invasive_triage" in complete_narrative.triage_rationale.upper() \
            or "safe non invasive" in complete_narrative.triage_rationale.lower()

    def test_triage_rationale_mentions_leading_disease(self, complete_narrative):
        assert "psoriasis" in complete_narrative.triage_rationale.lower()


# ── full_text ─────────────────────────────────────────────────────────────────

class TestFullText:
    def test_full_text_contains_all_headings(self, complete_narrative):
        text = complete_narrative.full_text()
        assert "[Clinical Presentation]" in text
        assert "[Evidence Interpretation]" in text
        assert "[Contradiction Analysis]" in text
        assert "[Certainty Assessment]" in text
        assert "[Safety Assessment]" in text
        assert "[Triage Rationale]" in text

    def test_full_text_custom_separator(self, complete_narrative):
        text = complete_narrative.full_text(separator="\n---\n")
        assert "\n---\n" in text

    def test_full_text_is_string(self, complete_narrative):
        assert isinstance(complete_narrative.full_text(), str)
