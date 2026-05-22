"""
PipelineConfig — centralised execution configuration for the symbolic pipeline.

Eliminates scattered constants and provides named configuration profiles
for the four primary execution modes:

  standard    — production execution with full outputs
  debug       — verbose per-stage logging and trace dumps
  validation  — deterministic batch execution across synthetic cases
  replay      — re-execution from a saved snapshot with full trace export

All subsystem thresholds can be overridden here to avoid hardcoded constants
spreading across the orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ExecutionMode = Literal["standard", "debug", "validation", "replay"]


@dataclass
class PipelineConfig:
    """
    Complete configuration for a single pipeline execution run.

    Parameters
    ----------
    execution_mode:
        Controls verbosity, tracing, and output behaviour.
    enable_trace:
        If True, a per-stage ReasoningSnapshot is recorded to the trajectory.
    enable_narrative:
        If True, the DiagnosticNarrativeGenerator runs at Stage 8.
    enable_replay_export:
        If True, a replay-safe JSON snapshot is written to outputs/.
    enable_counterfactual:
        If True, the LightweightCounterfactualReasoner is invoked after triage.
    rules_dir:
        Path to the YAML rule base directory.
    output_dir:
        Root directory for all pipeline outputs.
    log_level:
        Structlog level string ("DEBUG", "INFO", "WARNING").
    """

    # ── Execution mode ────────────────────────────────────────────────────────
    execution_mode:        ExecutionMode = "standard"
    enable_trace:          bool = True
    enable_narrative:      bool = True
    enable_replay_export:  bool = True
    enable_counterfactual: bool = False   # lightweight; off by default

    # ── Paths ─────────────────────────────────────────────────────────────────
    rules_dir:   Path = field(default_factory=lambda: Path("rules"))
    output_dir:  Path = field(default_factory=lambda: Path("outputs"))

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Grading thresholds ────────────────────────────────────────────────────
    ordinal_grade_map: dict[int, float] = field(default_factory=lambda: {
        0: 0.00, 1: 0.33, 2: 0.67, 3: 1.00,
    })
    significance_threshold:  int   = 2
    grading_dormant_threshold: float = 0.05

    # ── Evidence evaluation ───────────────────────────────────────────────────
    min_rule_activation:  float = 0.10
    evaluate_tiers:       list[str] = field(default_factory=lambda: ["A", "B", "D"])

    # ── Certainty propagation ─────────────────────────────────────────────────
    softmax_temperature:          float = 1.0
    contradiction_damping_threshold: float = 0.20
    certainty_decay_rate:         float = 0.15
    stability_gap_threshold:      float = 0.20
    stability_certainty_threshold: float = 0.55
    high_certainty_gap_threshold: float = 0.35
    high_certainty_threshold:     float = 0.65

    # ── Conflict analysis ─────────────────────────────────────────────────────
    contradiction_escalation_ceiling: float = 0.40

    # ── State machine ─────────────────────────────────────────────────────────
    min_rules_partial:                int   = 2
    min_rules_reinforcing:            int   = 3
    contradiction_detection_threshold: float = 0.10
    ambiguity_escalation_entropy:     float = 1.00
    certainty_stabilization_gap:      float = 0.20
    certainty_stabilization_min:      float = 0.55
    safe_triage_gap:                  float = 0.35
    safe_triage_min:                  float = 0.65
    biopsy_contradiction_ceiling:     float = 0.40
    biopsy_entropy_ceiling:           float = 1.50
    instability_threshold:            float = 0.60

    # ── Safety gate ───────────────────────────────────────────────────────────
    safety_contradiction_ceiling:  float = 0.40
    safety_min_activated_rules:    int   = 2
    safety_entropy_ceiling:        float = 1.50
    safety_single_rule_dominance:  float = 0.60
    safety_patho_certainty:        float = 0.75
    safety_max_critical_missing:   int   = 2
    safety_confusion_max_gap:      float = 0.30
    safety_confusion_penalty:      float = 0.15
    safety_overconf_certainty:     float = 0.92
    safety_overconf_contradiction: float = 0.10

    # ── Escalation engine ─────────────────────────────────────────────────────
    escalation_safe_min_certainty:   float = 0.72
    escalation_safe_min_gap:         float = 0.40
    escalation_safe_max_contra:      float = 0.20
    escalation_moderate_min_cert:    float = 0.65
    escalation_moderate_min_gap:     float = 0.35
    escalation_ambiguous_min_cert:   float = 0.45
    escalation_high_risk_ceiling:    float = 0.60

    # ── Instability monitor ───────────────────────────────────────────────────
    instability_oscillation_window: int = 3

    # ── Competition engine ────────────────────────────────────────────────────
    competition_tier_a_weight:     float = 0.35
    competition_contra_amplify:    float = 1.20

    # ── Sufficiency analyzer ──────────────────────────────────────────────────
    sufficiency_min_domains:          int   = 2
    sufficiency_min_rules:            int   = 3
    sufficiency_biopsy_free_threshold: float = 0.60

    # ── Counterfactual ────────────────────────────────────────────────────────
    counterfactual_drop_threshold:  float = 0.05
    counterfactual_search_depth:    int   = 4

    # ── Output directories ────────────────────────────────────────────────────
    @property
    def traces_dir(self) -> Path:
        return self.output_dir / "reasoning_traces"

    @property
    def narratives_dir(self) -> Path:
        return self.output_dir / "narratives"

    @property
    def escalation_reports_dir(self) -> Path:
        return self.output_dir / "escalation_reports"

    @property
    def replay_snapshots_dir(self) -> Path:
        return self.output_dir / "replay_snapshots"

    @property
    def validation_runs_dir(self) -> Path:
        return self.output_dir / "validation_runs"

    def ensure_output_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for d in (
            self.traces_dir, self.narratives_dir,
            self.escalation_reports_dir, self.replay_snapshots_dir,
            self.validation_runs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ── Named profiles ────────────────────────────────────────────────────────

    @classmethod
    def debug_profile(cls) -> "PipelineConfig":
        """Full debug profile: all outputs enabled, trace-level logging."""
        return cls(
            execution_mode="debug",
            enable_counterfactual=True,
            log_level="DEBUG",
        )

    @classmethod
    def validation_profile(cls) -> "PipelineConfig":
        """Validation profile: deterministic batch execution, no file exports."""
        return cls(
            execution_mode="validation",
            enable_replay_export=False,
            enable_counterfactual=False,
            log_level="WARNING",
        )

    @classmethod
    def replay_profile(cls) -> "PipelineConfig":
        """Replay profile: full trace + snapshot export."""
        return cls(
            execution_mode="replay",
            enable_trace=True,
            enable_replay_export=True,
            log_level="INFO",
        )
