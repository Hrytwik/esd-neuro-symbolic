"""
PipelineContext — mutable execution context for a single diagnostic case.

The context object is the shared carrier passed through all pipeline stages.
Each stage reads from the context what it needs, writes its outputs back,
and advances the stage counter. This design:

  · Preserves strict stage ordering (stages fail fast if prerequisites absent)
  · Enables replay-safe execution (full state at any stage is capturable)
  · Decouples stages from each other (no direct stage-to-stage dependencies)
  · Supports partial execution and incremental updates for debugging

Context evolution
-----------------
Stage 0 — Grading:               feature_values → grading_result
Stage 1 — Activation:            grading_result + rules → evidence_result
Stage 2 — Contradiction:         feature_values → conflict_result
Stage 3 — Certainty propagation: evidence + conflict → certainty_dist
Stage 4 — Competition:           certainty + evidence + conflict → competition_result
Stage 5 — Sufficiency:           evidence + certainty → sufficiency_report
Stage 6 — Instability:           certainty + conflict → instability updated
Stage 7 — FSM + Escalation:      all signals → state, safety_report, triage_decision
Stage 8 — Narrative:             all outputs → narrative
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.reasoning.certainty_propagator import CertaintyDistribution
from src.reasoning.clinical_grading import GradingResult
from src.reasoning.conflict_analyzer import ConflictAnalysisResult
from src.reasoning.differential_competition import CompetitionResult
from src.reasoning.escalation_engine import TriageDecision
from src.reasoning.evidence_evaluator import EvidenceEvaluationResult
from src.reasoning.evidence_sufficiency import SufficiencyReport
from src.reasoning.instability_monitor import InstabilityReport
from src.reasoning.narrative_generator import ClinicalNarrative
from src.reasoning.safety_gate import SafetyGateReport
from src.reasoning.state_tracker import DiagnosticState
from src.reasoning.trajectory_memory import DiagnosticTrajectoryMemory


# ── Binary feature classification (pipeline-wide constant) ───────────────────

BINARY_FEATURES: frozenset[str] = frozenset({
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history",
})


# ── Pipeline context ──────────────────────────────────────────────────────────

@dataclass
class PipelineContext:
    """
    Mutable shared state for a single pipeline execution pass.

    All fields are initialised to None / empty; each stage populates
    the fields it produces. Stage execution functions assert required
    upstream fields are populated before proceeding.

    Parameters
    ----------
    case_id:
        Unique identifier for the clinical case being processed.
    run_id:
        Unique identifier for this execution run (for replay tracing).
    feature_values:
        Raw clinical feature dictionary {name: value}.
    config_snapshot:
        Optional copy of the PipelineConfig used for this run (for replay).
    """

    case_id:       str
    run_id:        str
    feature_values: dict[str, Any]
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # ── Stage outputs (populated progressively) ───────────────────────────────

    grading_result:     GradingResult            | None = None
    evidence_result:    EvidenceEvaluationResult | None = None
    conflict_result:    ConflictAnalysisResult   | None = None
    certainty_dist:     CertaintyDistribution    | None = None
    competition_result: CompetitionResult        | None = None
    sufficiency_report: SufficiencyReport        | None = None
    instability_report: InstabilityReport        | None = None
    current_state:      DiagnosticState                 = DiagnosticState.INITIAL_EVIDENCE
    safety_report:      SafetyGateReport         | None = None
    triage_decision:    TriageDecision           | None = None
    narrative:          ClinicalNarrative        | None = None

    # ── Missing feature tracking ──────────────────────────────────────────────
    missing_features: list[str] = field(default_factory=list)

    # ── Execution tracking ────────────────────────────────────────────────────
    stage_index:       int        = 0
    completed_stages:  list[str]  = field(default_factory=list)
    stage_errors:      list[str]  = field(default_factory=list)

    # ── Trajectory memory (initialised on first use) ──────────────────────────
    _trajectory_memory: DiagnosticTrajectoryMemory | None = field(
        default=None, repr=False
    )

    # ── Public helpers ────────────────────────────────────────────────────────

    @property
    def trajectory_memory(self) -> DiagnosticTrajectoryMemory:
        """Lazy-initialise trajectory memory on first access."""
        if self._trajectory_memory is None:
            self._trajectory_memory = DiagnosticTrajectoryMemory(
                case_id=self.case_id,
                run_id=self.run_id,
            )
        return self._trajectory_memory

    @property
    def is_complete(self) -> bool:
        """True if the pipeline has reached a terminal decision."""
        return self.triage_decision is not None

    @property
    def has_error(self) -> bool:
        return len(self.stage_errors) > 0

    def mark_stage_complete(self, stage_name: str) -> None:
        """Register a stage as complete and advance the stage counter."""
        self.completed_stages.append(stage_name)
        self.stage_index += 1

    def record_stage_error(self, stage_name: str, error: str) -> None:
        self.stage_errors.append(f"[{stage_name}] {error}")

    def require(self, *fields: str) -> None:
        """
        Assert that required upstream fields are populated.
        Raises RuntimeError if any prerequisite is missing.
        """
        missing = [f for f in fields if getattr(self, f, None) is None]
        if missing:
            raise RuntimeError(
                f"PipelineContext prerequisite fields not yet populated: {missing}. "
                f"Completed stages: {self.completed_stages}"
            )

    def current_contradiction_load(self) -> float:
        if self.conflict_result is not None:
            return self.conflict_result.contradiction_load
        return 0.0

    def current_max_certainty(self) -> float:
        if self.certainty_dist is not None:
            return self.certainty_dist.max_certainty
        return 0.0

    def current_leading_disease(self) -> str:
        if self.certainty_dist is not None:
            return self.certainty_dist.leading_disease
        if self.evidence_result is not None:
            return self.evidence_result.leading_disease
        return "unknown"

    def summary_line(self) -> str:
        """One-line status summary for logging."""
        cert  = f"{self.current_max_certainty():.3f}"
        contra = f"{self.current_contradiction_load():.3f}"
        lead  = self.current_leading_disease()
        state = self.current_state.value
        return (
            f"case={self.case_id} | leading={lead} | "
            f"certainty={cert} | contradiction={contra} | state={state}"
        )
