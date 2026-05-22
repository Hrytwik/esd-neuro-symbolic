"""
Tests for ClinicalEscalationEngine — terminal triage decision integration.

Validates each triage outcome path (SAFE, MODERATE, AMBIGUOUS, BIOPSY,
HIGH_RISK), safety gate cap integration, and decision rationale content.
"""

import pytest

from src.reasoning.escalation_engine import ClinicalEscalationEngine, TriageDecision
from src.reasoning.safety_gate import TriageRecommendation
from src.reasoning.state_tracker import DiagnosticState


# ── SAFE_NON_INVASIVE_TRIAGE path ─────────────────────────────────────────────

class TestSafeTriage:
    def test_safe_triage_at_high_certainty_low_contradiction(
        self, escalation_engine, high_certainty_dist, no_conflict_result,
        psoriasis_evidence_result, safe_safety_report
    ):
        decision = escalation_engine.decide(
            certainty=high_certainty_dist,
            conflict=no_conflict_result,
            evidence=psoriasis_evidence_result,
            safety_report=safe_safety_report,
            final_state=DiagnosticState.CERTAINTY_STABILIZATION,
        )
        assert decision.recommendation == TriageRecommendation.SAFE_NON_INVASIVE_TRIAGE

    def test_is_safe_triage_property(
        self, escalation_engine, high_certainty_dist, no_conflict_result,
        psoriasis_evidence_result, safe_safety_report
    ):
        decision = escalation_engine.decide(
            high_certainty_dist, no_conflict_result,
            psoriasis_evidence_result, safe_safety_report,
            DiagnosticState.CERTAINTY_STABILIZATION,
        )
        assert decision.is_safe_triage
        assert not decision.requires_biopsy


# ── BIOPSY_RECOMMENDED path (FSM override) ────────────────────────────────────

class TestBiopsyRecommended:
    def test_biopsy_escalation_state_forces_biopsy(
        self, escalation_engine, stable_certainty, no_conflict_result,
        psoriasis_evidence_result, safe_safety_report
    ):
        decision = escalation_engine.decide(
            certainty=stable_certainty,
            conflict=no_conflict_result,
            evidence=psoriasis_evidence_result,
            safety_report=safe_safety_report,
            final_state=DiagnosticState.BIOPSY_ESCALATION,
        )
        assert decision.recommendation == TriageRecommendation.BIOPSY_RECOMMENDED
        assert decision.requires_biopsy

    def test_requires_biopsy_property_for_biopsy_decision(self, biopsy_triage_decision):
        assert biopsy_triage_decision.requires_biopsy


# ── HIGH_RISK_CONTRADICTION path ──────────────────────────────────────────────

class TestHighRiskContradiction:
    def test_very_high_contradiction_triggers_high_risk(
        self, escalation_engine, stable_certainty,
        psoriasis_evidence_result, safe_safety_report
    ):
        from src.reasoning.conflict_analyzer import ConflictAnalysisResult, ActiveContradiction

        very_high_conflict = ConflictAnalysisResult(
            active_contradictions=[],
            pair_tensions=[],
            penalty_by_disease={},
            contradiction_load=0.65,  # > high_risk_contradiction_ceiling=0.60
            confusion_zone_active=[],
            instability_contribution=0.65,
            mandatory_escalation=True,
        )
        decision = escalation_engine.decide(
            certainty=stable_certainty,
            conflict=very_high_conflict,
            evidence=psoriasis_evidence_result,
            safety_report=safe_safety_report,
            final_state=DiagnosticState.PARTIAL_ALIGNMENT,
        )
        assert decision.recommendation == TriageRecommendation.HIGH_RISK_CONTRADICTION


# ── MODERATE_CERTAINTY path ───────────────────────────────────────────────────

class TestModerateCertainty:
    def test_moderate_certainty_at_threshold(
        self, escalation_engine, stable_certainty, no_conflict_result,
        psoriasis_evidence_result, safe_safety_report
    ):
        # stable_certainty has max_certainty=0.72 >= moderate_min=0.65
        # and gap=0.62 >= moderate_gap=0.35
        decision = escalation_engine.decide(
            certainty=stable_certainty,
            conflict=no_conflict_result,
            evidence=psoriasis_evidence_result,
            safety_report=safe_safety_report,
            final_state=DiagnosticState.PARTIAL_ALIGNMENT,
        )
        assert decision.recommendation == TriageRecommendation.MODERATE_CERTAINTY


# ── Safety gate cap integration ───────────────────────────────────────────────

class TestSafetyGateCap:
    def test_safety_gate_cap_overrides_base_recommendation(
        self, escalation_engine, high_certainty_dist, no_conflict_result,
        psoriasis_evidence_result
    ):
        from src.reasoning.safety_gate import SafetyGateReport

        # Force a BIOPSY cap from safety gate
        capping_report = SafetyGateReport(
            effective_cap=TriageRecommendation.BIOPSY_RECOMMENDED,
            any_triggered=True,
        )
        decision = escalation_engine.decide(
            certainty=high_certainty_dist,
            conflict=no_conflict_result,
            evidence=psoriasis_evidence_result,
            safety_report=capping_report,
            final_state=DiagnosticState.CERTAINTY_STABILIZATION,
        )
        # Base would be SAFE, but cap overrides to BIOPSY
        assert decision.recommendation == TriageRecommendation.BIOPSY_RECOMMENDED
        assert decision.safety_gate_applied

    def test_safety_gate_ids_captured(
        self, escalation_engine, high_certainty_dist, no_conflict_result,
        psoriasis_evidence_result
    ):
        from src.reasoning.safety_gate import SafetyGateReport, GateResult

        gate_result = GateResult(
            gate_id="I1",
            gate_name="Contradiction Safety Ceiling",
            triggered=True,
            cap=TriageRecommendation.BIOPSY_RECOMMENDED,
            rationale="Test.",
        )
        report = SafetyGateReport(
            invariant_results=[gate_result],
            gate_results=[],
            effective_cap=TriageRecommendation.BIOPSY_RECOMMENDED,
            any_triggered=True,
        )
        decision = escalation_engine.decide(
            high_certainty_dist, no_conflict_result, psoriasis_evidence_result,
            report, DiagnosticState.CERTAINTY_STABILIZATION,
        )
        assert "I1" in decision.applied_gate_ids


# ── Decision properties ───────────────────────────────────────────────────────

class TestDecisionProperties:
    def test_decision_captures_leading_disease(
        self, escalation_engine, high_certainty_dist, no_conflict_result,
        psoriasis_evidence_result, safe_safety_report
    ):
        decision = escalation_engine.decide(
            high_certainty_dist, no_conflict_result, psoriasis_evidence_result,
            safe_safety_report, DiagnosticState.CERTAINTY_STABILIZATION,
        )
        assert decision.leading_disease == "psoriasis"

    def test_decision_rationale_not_empty(
        self, escalation_engine, high_certainty_dist, no_conflict_result,
        psoriasis_evidence_result, safe_safety_report
    ):
        decision = escalation_engine.decide(
            high_certainty_dist, no_conflict_result, psoriasis_evidence_result,
            safe_safety_report, DiagnosticState.CERTAINTY_STABILIZATION,
        )
        assert len(decision.decision_rationale) > 0
