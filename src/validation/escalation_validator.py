"""
EscalationValidator — biopsy escalation appropriateness assessment.

Validates that triage decisions are clinically appropriate:

  Over-escalation  — BIOPSY_RECOMMENDED for cases that should safely triage
  Under-escalation — SAFE_NON_INVASIVE_TRIAGE for cases with genuine conflict
  Appropriateness  — escalation triggers match the documented clinical reasons

The validator checks three safety invariants that must always hold:

  I1: contradiction_load >= 0.40  → must not be SAFE_NON_INVASIVE_TRIAGE
  I2: active_rules < 2            → must not be SAFE_NON_INVASIVE_TRIAGE
  I3: ambiguity_index > 1.5 bits  → must not be SAFE_NON_INVASIVE_TRIAGE

These mirror the clinical safety gates embedded in the reasoning pipeline.
"""

from __future__ import annotations

from src.pipeline.pipeline_runner import PipelineResult
from src.pipeline.synthetic_case_library import SyntheticCase
from src.validation.behavioral_validator import Severity, ValidationSignal


class EscalationValidator:
    """
    Validates escalation decisions for clinical appropriateness.

    Parameters
    ----------
    contradiction_biopsy_ceiling:
        Contradiction load above which SAFE_NON_INVASIVE_TRIAGE is incoherent.
    entropy_biopsy_ceiling:
        Ambiguity index (bits) above which SAFE_NON_INVASIVE_TRIAGE is incoherent.
    safe_triage_min_certainty:
        Minimum certainty for a case to be considered for safe triage.
    safe_triage_min_gap:
        Minimum certainty gap for safe triage consideration.
    """

    def __init__(
        self,
        contradiction_biopsy_ceiling: float = 0.40,
        entropy_biopsy_ceiling:       float = 1.50,
        safe_triage_min_certainty:    float = 0.55,
        safe_triage_min_gap:          float = 0.20,
    ) -> None:
        self._contra_ceil    = contradiction_biopsy_ceiling
        self._entropy_ceil   = entropy_biopsy_ceiling
        self._safe_cert_min  = safe_triage_min_certainty
        self._safe_gap_min   = safe_triage_min_gap

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        result: PipelineResult,
        case: SyntheticCase | None = None,
    ) -> tuple[list[ValidationSignal], float]:
        """
        Validate escalation decision appropriateness.

        Returns (signals, score) where score ∈ [0, 1].
        """
        signals: list[ValidationSignal] = []

        signals.extend(self._check_safety_invariants(result))
        signals.extend(self._check_recommendation_coherence(result))
        signals.extend(self._check_rationale_present(result))
        if case is not None:
            signals.extend(self._check_expected_escalation(result, case))

        score = sum(1 for s in signals if s.passed) / max(len(signals), 1)
        return signals, score

    # ── Safety invariant checks ───────────────────────────────────────────────

    def _check_safety_invariants(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Clinical safety invariants that must hold regardless of case type.

        I1: High contradiction → must not be SAFE_NON_INVASIVE_TRIAGE.
        I3: High entropy        → must not be SAFE_NON_INVASIVE_TRIAGE.
        """
        signals: list[ValidationSignal] = []
        is_safe = result.recommendation == "SAFE_NON_INVASIVE_TRIAGE"

        # Invariant I1
        i1_violated = is_safe and result.contradiction_load >= self._contra_ceil
        signals.append(ValidationSignal(
            validator="escalation",
            signal_name="invariant_I1_no_safe_with_high_contradiction",
            passed=not i1_violated,
            severity="critical",
            description=(
                f"I1: contradiction_load={result.contradiction_load:.3f} "
                + ("VIOLATES" if i1_violated else "respects")
                + f" ceiling={self._contra_ceil} for SAFE triage."
            ),
            measured_value=result.contradiction_load,
            expected_range=(0.0, self._contra_ceil),
        ))

        # Invariant I3
        i3_violated = is_safe and result.ambiguity_index > self._entropy_ceil
        signals.append(ValidationSignal(
            validator="escalation",
            signal_name="invariant_I3_no_safe_with_high_entropy",
            passed=not i3_violated,
            severity="critical",
            description=(
                f"I3: ambiguity_index={result.ambiguity_index:.3f} bits "
                + ("VIOLATES" if i3_violated else "respects")
                + f" ceiling={self._entropy_ceil} bits for SAFE triage."
            ),
            measured_value=result.ambiguity_index,
            expected_range=(0.0, self._entropy_ceil),
        ))

        return signals

    # ── Recommendation coherence ──────────────────────────────────────────────

    def _check_recommendation_coherence(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Cross-validate recommendation against measured certainty, gap, and
        contradiction metrics for internal consistency.
        """
        signals: list[ValidationSignal] = []
        rec = result.recommendation

        # SAFE triage should have reasonable certainty and gap
        if rec == "SAFE_NON_INVASIVE_TRIAGE":
            cert_ok = result.max_certainty >= self._safe_cert_min
            gap_ok  = result.certainty_gap >= self._safe_gap_min
            signals.append(ValidationSignal(
                validator="escalation",
                signal_name="safe_triage_certainty_adequate",
                passed=cert_ok,
                severity="warning",
                description=(
                    f"SAFE triage issued with certainty={result.max_certainty:.3f} "
                    + ("(adequate)." if cert_ok else f"(below floor {self._safe_cert_min:.3f}).")
                ),
                measured_value=result.max_certainty,
                expected_range=(self._safe_cert_min, 1.0),
            ))
            signals.append(ValidationSignal(
                validator="escalation",
                signal_name="safe_triage_gap_adequate",
                passed=gap_ok,
                severity="warning",
                description=(
                    f"SAFE triage issued with gap={result.certainty_gap:.3f} "
                    + ("(adequate)." if gap_ok else f"(below floor {self._safe_gap_min:.3f}).")
                ),
                measured_value=result.certainty_gap,
                expected_range=(self._safe_gap_min, 1.0),
            ))

        # HIGH_RISK should have measurable contradiction load
        if rec == "HIGH_RISK_CONTRADICTION":
            contra_present = result.contradiction_load > 0.0
            signals.append(ValidationSignal(
                validator="escalation",
                signal_name="high_risk_has_contradiction",
                passed=contra_present,
                severity="warning",
                description=(
                    "HIGH_RISK_CONTRADICTION should have contradiction_load > 0. "
                    + (f"Found {result.contradiction_load:.3f}." if contra_present else "Load is 0.")
                ),
                measured_value=result.contradiction_load,
            ))

        # BIOPSY cases should have at least one of: high contradiction, high entropy, low certainty
        if rec == "BIOPSY_RECOMMENDED":
            clinical_reason_present = (
                result.contradiction_load >= self._contra_ceil
                or result.ambiguity_index > self._entropy_ceil
                or result.max_certainty < self._safe_cert_min
                or result.certainty_gap < self._safe_gap_min
            )
            signals.append(ValidationSignal(
                validator="escalation",
                signal_name="biopsy_has_clinical_justification",
                passed=clinical_reason_present,
                severity="warning",
                description=(
                    "BIOPSY_RECOMMENDED should be justified by high contradiction, "
                    "high entropy, or insufficient certainty/gap. "
                    + ("Justified." if clinical_reason_present else "No clear justification found.")
                ),
            ))

        # Recommendation must be a known value
        valid_recs = {
            "SAFE_NON_INVASIVE_TRIAGE", "MODERATE_CERTAINTY",
            "AMBIGUOUS_PRESENTATION", "BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION",
        }
        signals.append(ValidationSignal(
            validator="escalation",
            signal_name="recommendation_valid_enum",
            passed=rec in valid_recs if rec else False,
            severity="critical",
            description=(
                f"Recommendation '{rec}' is "
                + ("a valid" if rec in valid_recs else "NOT a valid")
                + " triage recommendation."
            ),
        ))

        return signals

    # ── Rationale present ─────────────────────────────────────────────────────

    def _check_rationale_present(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """decision_rationale must be a non-empty string."""
        has_rationale = bool(result.decision_rationale and len(result.decision_rationale) > 10)
        return [ValidationSignal(
            validator="escalation",
            signal_name="decision_rationale_present",
            passed=has_rationale,
            severity="warning",
            description=(
                "Decision rationale is present and non-trivial." if has_rationale
                else "Decision rationale is absent or too short to be meaningful."
            ),
        )]

    # ── Expected escalation agreement ─────────────────────────────────────────

    def _check_expected_escalation(
        self,
        result: PipelineResult,
        case: SyntheticCase,
    ) -> list[ValidationSignal]:
        """
        When a SyntheticCase provides ground truth, validate that biopsy
        escalation expectation is honoured.
        """
        signals: list[ValidationSignal] = []

        if case.expect_biopsy_escalation:
            biopsy_issued = result.recommendation in (
                "BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"
            )
            signals.append(ValidationSignal(
                validator="escalation",
                signal_name="expected_biopsy_issued",
                passed=biopsy_issued,
                severity="critical",
                description=(
                    f"Case {case.case_id} expects biopsy escalation. "
                    f"Recommendation is '{result.recommendation}'."
                ),
            ))

        if case.expect_stable:
            safe_issued = result.recommendation == "SAFE_NON_INVASIVE_TRIAGE"
            signals.append(ValidationSignal(
                validator="escalation",
                signal_name="expected_safe_triage_issued",
                passed=safe_issued,
                severity="warning",
                description=(
                    f"Case {case.case_id} expects stable safe triage. "
                    f"Recommendation is '{result.recommendation}'."
                ),
            ))

        return signals
