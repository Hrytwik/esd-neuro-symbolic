"""
ClinicalEscalationEngine — Stage 6 terminal triage decision.

Integrates certainty distribution, contradiction analysis, safety gate
report, evidence sufficiency, and diagnostic state to produce the final
biopsy triage recommendation. This is the PRIMARY clinical output of the
reasoning engine — not a disease prediction.

Triage recommendations
----------------------
  SAFE_NON_INVASIVE_TRIAGE  — High certainty, low contradiction, stable.
                               Biopsy not required for differential diagnosis.
  MODERATE_CERTAINTY        — Reasonable certainty but with caveats.
                               Biopsy optional; clinical judgement required.
  AMBIGUOUS_PRESENTATION    — Multiple competing hypotheses; evidence
                               insufficient for safe non-invasive diagnosis.
  BIOPSY_RECOMMENDED        — Contradictions or ambiguity require histological
                               confirmation for safe management.
  HIGH_RISK_CONTRADICTION   — Severe cross-disease contradiction load or
                               critical safety invariant violated. Immediate
                               biopsy required.

Decision integration priority
------------------------------
1. Safety gate effective cap (hard override, escalation-only)
2. Contradiction load thresholds
3. Certainty and certainty_gap thresholds
4. Diagnostic state (FSM terminal state)
5. Evidence sufficiency
"""

from __future__ import annotations

from dataclasses import dataclass

from src.reasoning.certainty_propagator import CertaintyDistribution
from src.reasoning.conflict_analyzer import ConflictAnalysisResult
from src.reasoning.evidence_evaluator import EvidenceEvaluationResult
from src.reasoning.safety_gate import SafetyGateReport, TriageRecommendation
from src.reasoning.state_tracker import DiagnosticState


# ── Triage decision ───────────────────────────────────────────────────────────

@dataclass
class TriageDecision:
    """
    Terminal triage decision for a single clinical case.
    Primary output of ClinicalEscalationEngine.decide().
    """

    recommendation:      TriageRecommendation
    leading_disease:     str
    second_disease:      str
    max_certainty:       float
    certainty_gap:       float
    contradiction_load:  float
    ambiguity_index:     float
    final_state:         DiagnosticState
    safety_gate_applied: bool
    applied_gate_ids:    list[str]
    decision_rationale:  str

    @property
    def is_safe_triage(self) -> bool:
        return self.recommendation == TriageRecommendation.SAFE_NON_INVASIVE_TRIAGE

    @property
    def requires_biopsy(self) -> bool:
        return self.recommendation in (
            TriageRecommendation.BIOPSY_RECOMMENDED,
            TriageRecommendation.HIGH_RISK_CONTRADICTION,
        )


# ── Escalation engine ─────────────────────────────────────────────────────────

class ClinicalEscalationEngine:
    """
    Produces the terminal biopsy triage recommendation by integrating all
    available reasoning signals.

    Parameters
    ----------
    safe_min_certainty:
        Minimum certainty for SAFE_NON_INVASIVE_TRIAGE. Default: 0.82.
    safe_min_gap:
        Minimum certainty_gap for SAFE_NON_INVASIVE_TRIAGE. Default: 0.40.
    safe_max_contradiction:
        Maximum contradiction_load for SAFE_NON_INVASIVE_TRIAGE. Default: 0.20.
    moderate_min_certainty:
        Minimum certainty for MODERATE_CERTAINTY. Default: 0.65.
    moderate_min_gap:
        Minimum gap for MODERATE_CERTAINTY. Default: 0.35.
    high_risk_contradiction_ceiling:
        Contradiction load above which HIGH_RISK_CONTRADICTION fires. Default: 0.60.
    """

    def __init__(
        self,
        safe_min_certainty: float = 0.82,
        safe_min_gap: float = 0.40,
        safe_max_contradiction: float = 0.20,
        moderate_min_certainty: float = 0.65,
        moderate_min_gap: float = 0.35,
        ambiguous_min_certainty: float = 0.45,
        high_risk_contradiction_ceiling: float = 0.60,
    ) -> None:
        self._safe_cert       = safe_min_certainty
        self._safe_gap        = safe_min_gap
        self._safe_max_contra = safe_max_contradiction
        self._mod_cert        = moderate_min_certainty
        self._mod_gap         = moderate_min_gap
        self._amb_cert        = ambiguous_min_certainty
        self._high_risk       = high_risk_contradiction_ceiling

    # ── Public API ────────────────────────────────────────────────────────────

    def decide(
        self,
        certainty: CertaintyDistribution,
        conflict: ConflictAnalysisResult,
        evidence: EvidenceEvaluationResult,
        safety_report: SafetyGateReport,
        final_state: DiagnosticState,
    ) -> TriageDecision:
        """
        Produce the terminal triage recommendation.

        The decision integrates safety gate caps (hard override), contradiction
        load severity, and certainty stability in descending priority.
        """
        # Apply confusion zone penalty to effective certainty
        eff_certainty   = max(certainty.max_certainty - safety_report.certainty_penalty, 0.0)
        eff_gap         = max(certainty.certainty_gap  - safety_report.certainty_penalty, 0.0)
        contradiction   = conflict.contradiction_load
        entropy         = certainty.ambiguity_index

        applied_gates: list[str] = [
            r.gate_id for r in safety_report.all_results if r.triggered
        ]

        # ── Step 1: terminal FSM state overrides ─────────────────────────────
        if final_state == DiagnosticState.BIOPSY_ESCALATION:
            base = TriageRecommendation.BIOPSY_RECOMMENDED
            rationale = "Diagnostic state machine reached BIOPSY_ESCALATION state."
            return self._finalise(base, certainty, conflict, final_state,
                                  safety_report, applied_gates, rationale, eff_certainty, eff_gap)

        # ── Step 2: HIGH_RISK_CONTRADICTION (critical contradiction load) ─────
        if contradiction >= self._high_risk:
            base = TriageRecommendation.HIGH_RISK_CONTRADICTION
            rationale = (
                f"Contradiction load {contradiction:.3f} exceeds high-risk ceiling "
                f"{self._high_risk}. Histological confirmation required."
            )
            return self._finalise(base, certainty, conflict, final_state,
                                  safety_report, applied_gates, rationale, eff_certainty, eff_gap)

        # ── Step 3: FSM SAFE_TRIAGE + threshold criteria ──────────────────────
        if (
            eff_certainty >= self._safe_cert
            and eff_gap     >= self._safe_gap
            and contradiction < self._safe_max_contra
            and final_state not in (
                DiagnosticState.CONTRADICTION_DETECTED,
                DiagnosticState.AMBIGUITY_ESCALATION,
                DiagnosticState.BIOPSY_ESCALATION,
                DiagnosticState.UNSTABLE_REASONING,
            )
        ):
            base = TriageRecommendation.SAFE_NON_INVASIVE_TRIAGE
            rationale = (
                f"Certainty={eff_certainty:.3f} >= {self._safe_cert}, "
                f"gap={eff_gap:.3f} >= {self._safe_gap}, "
                f"contradiction={contradiction:.3f} < {self._safe_max_contra}. "
                f"Non-invasive triage is supported."
            )

        # ── Step 4: MODERATE_CERTAINTY ────────────────────────────────────────
        elif (
            eff_certainty >= self._mod_cert
            and eff_gap   >= self._mod_gap
            and final_state not in (
                DiagnosticState.BIOPSY_ESCALATION,
                DiagnosticState.AMBIGUITY_ESCALATION,
            )
        ):
            base = TriageRecommendation.MODERATE_CERTAINTY
            rationale = (
                f"Certainty={eff_certainty:.3f} and gap={eff_gap:.3f} meet "
                f"moderate thresholds. Clinical judgement required for biopsy decision."
            )

        # ── Step 5: AMBIGUOUS_PRESENTATION ───────────────────────────────────
        elif eff_certainty >= self._amb_cert:
            base = TriageRecommendation.AMBIGUOUS_PRESENTATION
            rationale = (
                f"Certainty={eff_certainty:.3f} in ambiguous range; "
                f"entropy={entropy:.3f} bits. Biopsy preferred for safe management."
            )

        # ── Step 6: BIOPSY_RECOMMENDED (default fallback) ────────────────────
        else:
            base = TriageRecommendation.BIOPSY_RECOMMENDED
            rationale = (
                f"Certainty={eff_certainty:.3f} below minimum thresholds. "
                f"Histological confirmation required."
            )

        return self._finalise(base, certainty, conflict, final_state,
                              safety_report, applied_gates, rationale, eff_certainty, eff_gap)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _finalise(
        self,
        base: TriageRecommendation,
        certainty: CertaintyDistribution,
        conflict: ConflictAnalysisResult,
        final_state: DiagnosticState,
        safety_report: SafetyGateReport,
        applied_gates: list[str],
        rationale: str,
        eff_certainty: float,
        eff_gap: float,
    ) -> TriageDecision:
        """Apply safety gate cap and return the final TriageDecision."""
        final = safety_report.apply_cap(base)
        if final != base:
            rationale += (
                f" [Safety gate cap applied: {safety_report.effective_cap.value}]"
            )
        return TriageDecision(
            recommendation=final,
            leading_disease=certainty.leading_disease,
            second_disease=certainty.second_disease,
            max_certainty=eff_certainty,
            certainty_gap=eff_gap,
            contradiction_load=conflict.contradiction_load,
            ambiguity_index=certainty.ambiguity_index,
            final_state=final_state,
            safety_gate_applied=safety_report.any_triggered,
            applied_gate_ids=applied_gates,
            decision_rationale=rationale,
        )
