"""
DiagnosticTrajectoryMemory — replayable reasoning trace.

Records a complete, ordered sequence of reasoning snapshots — one per
pipeline stage — capturing the evolution of evidence, contradictions,
certainty, and state. The trajectory supports:

  · Frontend replay mode (step-by-step reasoning visualisation)
  · Publication figure generation (state transition diagrams)
  · Counterfactual analysis (perturbation starting from any stage)
  · Audit logging (traceability for clinical decision support)

Snapshot granularity
--------------------
Each snapshot is a frozen record of all quantitative reasoning signals
at the moment it was captured. Snapshots are immutable once recorded;
the trajectory is append-only during inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.reasoning.certainty_propagator import CertaintyDistribution
from src.reasoning.conflict_analyzer import ConflictAnalysisResult
from src.reasoning.escalation_engine import TriageDecision
from src.reasoning.evidence_evaluator import EvidenceEvaluationResult
from src.reasoning.safety_gate import SafetyGateReport
from src.reasoning.state_tracker import DiagnosticState


# ── Reasoning snapshot ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReasoningSnapshot:
    """
    Immutable point-in-time record of the reasoning engine state
    after completing a single pipeline stage.
    """

    stage:              int
    stage_name:         str
    state:              DiagnosticState
    leading_disease:    str
    max_certainty:      float
    certainty_gap:      float
    contradiction_load: float
    ambiguity_index:    float
    active_rule_count:  int
    tier_a_count:       int
    safety_triggered:   bool
    triage_so_far:      str | None      # TriageRecommendation.value or None
    delta_description:  str             # human-readable stage delta


# ── Delta record (stage-to-stage change) ──────────────────────────────────────

@dataclass(frozen=True)
class StageDelta:
    """Quantitative change between two consecutive reasoning snapshots."""

    from_stage:           int
    to_stage:             int
    certainty_delta:      float    # positive = increasing
    gap_delta:            float
    contradiction_delta:  float    # positive = more contradictions
    entropy_delta:        float    # positive = more ambiguous
    state_changed:        bool
    from_state:           DiagnosticState
    to_state:             DiagnosticState


# ── Full trajectory ───────────────────────────────────────────────────────────

@dataclass
class DiagnosticTrajectory:
    """
    Complete ordered reasoning trajectory for a single case.
    Produced by DiagnosticTrajectoryMemory.
    """

    case_id:     str
    run_id:      str
    snapshots:   list[ReasoningSnapshot] = field(default_factory=list)
    final_decision: TriageDecision | None = None
    created_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def stage_count(self) -> int:
        return len(self.snapshots)

    @property
    def final_state(self) -> DiagnosticState | None:
        return self.snapshots[-1].state if self.snapshots else None

    @property
    def final_certainty(self) -> float:
        return self.snapshots[-1].max_certainty if self.snapshots else 0.0

    def get_stage(self, stage: int) -> ReasoningSnapshot | None:
        for s in self.snapshots:
            if s.stage == stage:
                return s
        return None

    def deltas(self) -> list[StageDelta]:
        """Compute stage-to-stage quantitative deltas."""
        if len(self.snapshots) < 2:
            return []
        result = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i - 1]
            curr = self.snapshots[i]
            result.append(StageDelta(
                from_stage=prev.stage,
                to_stage=curr.stage,
                certainty_delta=curr.max_certainty - prev.max_certainty,
                gap_delta=curr.certainty_gap - prev.certainty_gap,
                contradiction_delta=curr.contradiction_load - prev.contradiction_load,
                entropy_delta=curr.ambiguity_index - prev.ambiguity_index,
                state_changed=(curr.state != prev.state),
                from_state=prev.state,
                to_state=curr.state,
            ))
        return result

    def certainty_series(self) -> list[float]:
        return [s.max_certainty for s in self.snapshots]

    def contradiction_series(self) -> list[float]:
        return [s.contradiction_load for s in self.snapshots]

    def state_sequence(self) -> list[str]:
        return [s.state.value for s in self.snapshots]


# ── Trajectory memory ─────────────────────────────────────────────────────────

class DiagnosticTrajectoryMemory:
    """
    Records reasoning snapshots at each pipeline stage, building a
    replayable trajectory for a single case.

    Usage
    -----
    memory = DiagnosticTrajectoryMemory(case_id="case_001", run_id="run-abc123")
    memory.record(stage=0, stage_name="Clinical Grading", state=..., ...)
    # ... after all stages ...
    trajectory = memory.finalise(decision)
    """

    def __init__(self, case_id: str, run_id: str) -> None:
        self._trajectory = DiagnosticTrajectory(case_id=case_id, run_id=run_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        stage: int,
        stage_name: str,
        state: DiagnosticState,
        evidence: EvidenceEvaluationResult | None = None,
        conflict: ConflictAnalysisResult | None = None,
        certainty: CertaintyDistribution | None = None,
        safety_report: SafetyGateReport | None = None,
        delta_description: str = "",
    ) -> ReasoningSnapshot:
        """
        Record a reasoning snapshot for the given stage.
        Returns the snapshot for immediate use.
        """
        leading    = certainty.leading_disease if certainty else "unknown"
        max_cert   = certainty.max_certainty   if certainty else 0.0
        gap        = certainty.certainty_gap   if certainty else 0.0
        contra     = conflict.contradiction_load if conflict else 0.0
        entropy    = certainty.ambiguity_index if certainty else 0.0
        rules      = evidence.total_rules_active if evidence else 0
        tier_a     = (
            sum(v.tier_a_count for v in evidence.disease_vectors.values())
            if evidence else 0
        )
        safe_trig  = safety_report.any_triggered if safety_report else False

        snapshot = ReasoningSnapshot(
            stage=stage,
            stage_name=stage_name,
            state=state,
            leading_disease=leading,
            max_certainty=max_cert,
            certainty_gap=gap,
            contradiction_load=contra,
            ambiguity_index=entropy,
            active_rule_count=rules,
            tier_a_count=tier_a,
            safety_triggered=safe_trig,
            triage_so_far=None,
            delta_description=delta_description or stage_name,
        )
        self._trajectory.snapshots.append(snapshot)
        return snapshot

    def finalise(self, decision: TriageDecision | None = None) -> DiagnosticTrajectory:
        """
        Mark the trajectory as complete and attach the terminal decision.
        Returns the completed DiagnosticTrajectory.
        """
        self._trajectory.final_decision = decision
        return self._trajectory

    @property
    def trajectory(self) -> DiagnosticTrajectory:
        return self._trajectory
