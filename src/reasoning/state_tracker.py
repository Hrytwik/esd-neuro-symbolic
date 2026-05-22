"""
DiagnosticStateTracker — 9-state finite-state machine for reasoning state.

Maintains the current epistemic state of the diagnostic reasoning process,
advancing through well-defined guard conditions at each reasoning stage.
Every transition is recorded with its triggering condition, producing
a fully traceable reasoning trajectory.

State taxonomy
--------------
INITIAL_EVIDENCE      — Sparse activation; no hypothesis has emerged.
PARTIAL_ALIGNMENT     — Evidence beginning to accumulate; one disease leading.
REINFORCING_ALIGNMENT — Multiple rules reinforcing the leading hypothesis.
CONTRADICTION_DETECTED — Active cross-disease contradictions present.
AMBIGUITY_ESCALATION  — High entropy; multiple competing hypotheses.
CERTAINTY_STABILIZATION — Certainty converging toward one hypothesis.
BIOPSY_ESCALATION     — Safety invariant or contradiction ceiling triggered.
SAFE_TRIAGE           — High certainty, low contradiction; biopsy-free safe.
UNSTABLE_REASONING    — Oscillating or contradictory reasoning trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.reasoning.certainty_propagator import CertaintyDistribution
from src.reasoning.conflict_analyzer import ConflictAnalysisResult
from src.reasoning.evidence_evaluator import EvidenceEvaluationResult


# ── State enumeration ─────────────────────────────────────────────────────────

class DiagnosticState(str, Enum):
    INITIAL_EVIDENCE       = "INITIAL_EVIDENCE"
    PARTIAL_ALIGNMENT      = "PARTIAL_ALIGNMENT"
    REINFORCING_ALIGNMENT  = "REINFORCING_ALIGNMENT"
    CONTRADICTION_DETECTED = "CONTRADICTION_DETECTED"
    AMBIGUITY_ESCALATION   = "AMBIGUITY_ESCALATION"
    CERTAINTY_STABILIZATION = "CERTAINTY_STABILIZATION"
    BIOPSY_ESCALATION      = "BIOPSY_ESCALATION"
    SAFE_TRIAGE            = "SAFE_TRIAGE"
    UNSTABLE_REASONING     = "UNSTABLE_REASONING"


# ── State transition record ───────────────────────────────────────────────────

@dataclass(frozen=True)
class StateTransition:
    """A single recorded state transition with triggering context."""

    from_state:    DiagnosticState
    to_state:      DiagnosticState
    stage:         int              # reasoning pipeline stage (0–6)
    trigger:       str              # human-readable guard condition
    is_escalation: bool             # True if transitioning to BIOPSY_ESCALATION


# ── State tracker ─────────────────────────────────────────────────────────────

class DiagnosticStateTracker:
    """
    Manages the 9-state diagnostic state machine.

    The tracker evaluates guard conditions at each reasoning stage and
    advances the state deterministically. Transitions are append-only —
    the state can only escalate or stabilize; it cannot retreat to a
    prior state except through explicit unstable_reasoning detection.

    Parameters
    ----------
    min_rules_partial:
        Minimum activated rule count to leave INITIAL_EVIDENCE. Default: 2.
    min_rules_reinforcing:
        Minimum rules for leading disease to enter REINFORCING_ALIGNMENT.
    contradiction_detection_threshold:
        Contradiction load above which CONTRADICTION_DETECTED is triggered.
    instability_threshold:
        Instability index above which UNSTABLE_REASONING is entered.
    """

    def __init__(
        self,
        min_rules_partial: int = 2,
        min_rules_reinforcing: int = 3,
        contradiction_detection_threshold: float = 0.10,
        ambiguity_escalation_entropy: float = 1.00,
        certainty_stabilization_gap: float = 0.20,
        certainty_stabilization_min: float = 0.55,
        safe_triage_gap: float = 0.35,
        safe_triage_min: float = 0.65,
        biopsy_contradiction_ceiling: float = 0.40,
        biopsy_entropy_ceiling: float = 1.50,
        instability_threshold: float = 0.60,
    ) -> None:
        self._min_partial        = min_rules_partial
        self._min_reinforcing    = min_rules_reinforcing
        self._contra_threshold   = contradiction_detection_threshold
        self._ambiguity_entropy  = ambiguity_escalation_entropy
        self._stab_gap           = certainty_stabilization_gap
        self._stab_min           = certainty_stabilization_min
        self._safe_gap           = safe_triage_gap
        self._safe_min           = safe_triage_min
        self._biopsy_contra      = biopsy_contradiction_ceiling
        self._biopsy_entropy     = biopsy_entropy_ceiling
        self._instability_thresh = instability_threshold

        self._current: DiagnosticState = DiagnosticState.INITIAL_EVIDENCE
        self._transitions: list[StateTransition] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def current_state(self) -> DiagnosticState:
        return self._current

    @property
    def transition_history(self) -> list[StateTransition]:
        return list(self._transitions)

    @property
    def is_terminal(self) -> bool:
        return self._current in (
            DiagnosticState.BIOPSY_ESCALATION,
            DiagnosticState.SAFE_TRIAGE,
            DiagnosticState.UNSTABLE_REASONING,
        )

    def advance(
        self,
        stage: int,
        evidence: EvidenceEvaluationResult,
        conflict: ConflictAnalysisResult,
        certainty: CertaintyDistribution,
        instability_index: float = 0.0,
    ) -> DiagnosticState:
        """
        Evaluate all guard conditions for the current state and advance
        if a transition condition is satisfied.

        Returns the current state after evaluation (may be unchanged).
        """
        if self.is_terminal:
            return self._current

        # Safety override — always checked first
        if self._check_biopsy_escalation(conflict, certainty):
            return self._transition(
                DiagnosticState.BIOPSY_ESCALATION, stage,
                trigger=self._biopsy_trigger(conflict, certainty),
                escalation=True,
            )

        # Instability detection — checked before all others
        if instability_index >= self._instability_thresh:
            return self._transition(
                DiagnosticState.UNSTABLE_REASONING, stage,
                trigger=f"instability_index={instability_index:.3f} >= {self._instability_thresh}",
                escalation=False,
            )

        # State-dependent transitions
        s = self._current

        if s == DiagnosticState.INITIAL_EVIDENCE:
            if evidence.total_rules_active >= self._min_partial:
                return self._transition(
                    DiagnosticState.PARTIAL_ALIGNMENT, stage,
                    trigger=f"activated_rules={evidence.total_rules_active} >= {self._min_partial}",
                )

        elif s == DiagnosticState.PARTIAL_ALIGNMENT:
            leading_vec = evidence.get(evidence.leading_disease)
            leading_rules = leading_vec.active_rule_count if leading_vec else 0
            if conflict.contradiction_load > self._contra_threshold:
                return self._transition(
                    DiagnosticState.CONTRADICTION_DETECTED, stage,
                    trigger=f"contradiction_load={conflict.contradiction_load:.3f} > {self._contra_threshold}",
                )
            if leading_rules >= self._min_reinforcing:
                return self._transition(
                    DiagnosticState.REINFORCING_ALIGNMENT, stage,
                    trigger=f"leading_disease_rules={leading_rules} >= {self._min_reinforcing}",
                )

        elif s == DiagnosticState.REINFORCING_ALIGNMENT:
            if conflict.contradiction_load > self._contra_threshold:
                return self._transition(
                    DiagnosticState.CONTRADICTION_DETECTED, stage,
                    trigger=f"contradiction_load={conflict.contradiction_load:.3f} > {self._contra_threshold}",
                )
            if certainty.certainty_gap >= self._stab_gap and certainty.max_certainty >= self._stab_min:
                return self._transition(
                    DiagnosticState.CERTAINTY_STABILIZATION, stage,
                    trigger=(f"gap={certainty.certainty_gap:.3f} >= {self._stab_gap}, "
                             f"certainty={certainty.max_certainty:.3f} >= {self._stab_min}"),
                )

        elif s == DiagnosticState.CONTRADICTION_DETECTED:
            if certainty.ambiguity_index >= self._ambiguity_entropy:
                return self._transition(
                    DiagnosticState.AMBIGUITY_ESCALATION, stage,
                    trigger=f"entropy={certainty.ambiguity_index:.3f} >= {self._ambiguity_entropy}",
                )
            # If contradiction resolves and certainty stabilizes
            if (conflict.contradiction_load <= self._contra_threshold
                    and certainty.certainty_gap >= self._stab_gap):
                return self._transition(
                    DiagnosticState.CERTAINTY_STABILIZATION, stage,
                    trigger="contradiction resolved and certainty stabilized",
                )

        elif s == DiagnosticState.AMBIGUITY_ESCALATION:
            # Ambiguity can stabilize if entropy drops
            if certainty.ambiguity_index < self._ambiguity_entropy and certainty.is_stable:
                return self._transition(
                    DiagnosticState.CERTAINTY_STABILIZATION, stage,
                    trigger=f"entropy reduced to {certainty.ambiguity_index:.3f} and certainty stabilized",
                )

        elif s == DiagnosticState.CERTAINTY_STABILIZATION:
            if (certainty.certainty_gap >= self._safe_gap
                    and certainty.max_certainty >= self._safe_min
                    and conflict.contradiction_load < 0.20):
                return self._transition(
                    DiagnosticState.SAFE_TRIAGE, stage,
                    trigger=(f"gap={certainty.certainty_gap:.3f} >= {self._safe_gap}, "
                             f"certainty={certainty.max_certainty:.3f} >= {self._safe_min}, "
                             f"contradiction_load={conflict.contradiction_load:.3f} < 0.20"),
                )

        return self._current

    def reset(self) -> None:
        """Reset to initial state (for new case processing)."""
        self._current     = DiagnosticState.INITIAL_EVIDENCE
        self._transitions = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _transition(
        self,
        to: DiagnosticState,
        stage: int,
        trigger: str,
        escalation: bool = False,
    ) -> DiagnosticState:
        record = StateTransition(
            from_state=self._current,
            to_state=to,
            stage=stage,
            trigger=trigger,
            is_escalation=escalation,
        )
        self._transitions.append(record)
        self._current = to
        return to

    def _check_biopsy_escalation(
        self,
        conflict: ConflictAnalysisResult,
        certainty: CertaintyDistribution,
    ) -> bool:
        """Safety invariants that always override other transitions."""
        if conflict.contradiction_load >= self._biopsy_contra:
            return True
        if certainty.ambiguity_index >= self._biopsy_entropy:
            return True
        return False

    def _biopsy_trigger(
        self,
        conflict: ConflictAnalysisResult,
        certainty: CertaintyDistribution,
    ) -> str:
        reasons = []
        if conflict.contradiction_load >= self._biopsy_contra:
            reasons.append(
                f"contradiction_load={conflict.contradiction_load:.3f} >= {self._biopsy_contra}"
            )
        if certainty.ambiguity_index >= self._biopsy_entropy:
            reasons.append(
                f"ambiguity_index={certainty.ambiguity_index:.3f} >= {self._biopsy_entropy}"
            )
        return "; ".join(reasons)
