"""
PipelineRunner — central orchestrator for the symbolic diagnostic pipeline.

Coordinates all reasoning subsystems in a deterministic execution sequence,
threading the PipelineContext through each stage and producing a
PipelineResult capturing the full reasoning output.

The runner owns subsystem instantiation — it reads PipelineConfig and
constructs every subsystem with the configured parameters. This ensures
single-source-of-truth configuration and full reproducibility.

Execution sequence
------------------
  Stage 0 — Clinical grading (fuzzy feature conversion)
  Stage 1 — Evidence activation (rule evaluation)
  Stage 2 — Contradiction analysis
  Stage 3 — Certainty propagation
  Stage 4 — Differential competition
  Stage 5 — Evidence sufficiency
  Stage 6 — Instability monitoring
  Stage 7 — FSM + Safety gate + Escalation
  [snapshot recorded after Stage 7]
  Stage 8 — Narrative generation (if enabled)
  Stage 9 — Output export (if replay export enabled)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.reasoning.certainty_propagator import HypothesisCertaintyPropagator
from src.reasoning.clinical_grading import ClinicalGradingModule
from src.reasoning.conflict_analyzer import DiagnosticConflictAnalyzer
from src.reasoning.differential_competition import DifferentialCompetitionEngine
from src.reasoning.escalation_engine import ClinicalEscalationEngine
from src.reasoning.evidence_evaluator import DiagnosticEvidenceEvaluator
from src.reasoning.evidence_sufficiency import EvidenceSufficiencyAnalyzer
from src.reasoning.instability_monitor import DiagnosticInstabilityMonitor
from src.reasoning.narrative_generator import DiagnosticNarrativeGenerator
from src.reasoning.safety_gate import ClinicalSafetyGate
from src.reasoning.state_tracker import DiagnosticStateTracker
from src.reasoning.trajectory_memory import DiagnosticTrajectory
from src.symbolic_engine.rule_registry import DiagnosticRuleRepository
from src.utils.logger import get_logger
from src.pipeline.pipeline_config import PipelineConfig
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.execution_stages import (
    GradingStage,
    EvidenceActivationStage,
    ContradictionAnalysisStage,
    CertaintyPropagationStage,
    CompetitionAnalysisStage,
    EvidenceSufficiencyStage,
    InstabilityStage,
    EscalationStage,
    NarrativeStage,
    TrajectorySnapshotStage,
    StageResult,
)
from src.pipeline.pipeline_outputs import (
    ReasoningTraceExporter,
    NarrativeExporter,
    EscalationReportExporter,
    ReplaySnapshotExporter,
)

log = get_logger(__name__, subsystem="PipelineRunner")


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Complete output of a single pipeline execution.
    Captures all reasoning outputs in a structured, queryable form.
    """

    case_id:         str
    run_id:          str
    success:         bool

    # Terminal outputs
    recommendation:  str | None      = None    # TriageRecommendation.value
    leading_disease: str | None      = None
    max_certainty:   float           = 0.0
    certainty_gap:   float           = 0.0
    contradiction_load: float        = 0.0
    ambiguity_index: float           = 0.0
    final_state:     str | None      = None    # DiagnosticState.value
    decision_rationale: str          = ""

    # Stage execution log
    stage_results:   list[StageResult] = field(default_factory=list)
    completed_stages: list[str]        = field(default_factory=list)
    stage_errors:    list[str]         = field(default_factory=list)

    # Trajectory
    trajectory:      DiagnosticTrajectory | None = None

    # Exported file paths
    trace_path:      Path | None = None
    narrative_path:  Path | None = None
    escalation_path: Path | None = None
    replay_path:     Path | None = None

    @property
    def requires_biopsy(self) -> bool:
        return self.recommendation in ("BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION")

    @property
    def is_safe_triage(self) -> bool:
        return self.recommendation == "SAFE_NON_INVASIVE_TRIAGE"

    @property
    def has_errors(self) -> bool:
        return len(self.stage_errors) > 0

    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"[{status}] case={self.case_id} "
            f"| {self.recommendation or 'INCOMPLETE'} "
            f"| {self.leading_disease or '?'} "
            f"| certainty={self.max_certainty:.3f} "
            f"| gap={self.certainty_gap:.3f} "
            f"| contradiction={self.contradiction_load:.3f} "
            f"| state={self.final_state or '?'}"
        )


# ── Pipeline runner ───────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Orchestrates the full symbolic diagnostic reasoning pipeline.

    Parameters
    ----------
    config:
        PipelineConfig instance controlling all thresholds and behaviour.
    rule_repository:
        Pre-initialised DiagnosticRuleRepository. If not provided, one
        is created from config.rules_dir.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        rule_repository: DiagnosticRuleRepository | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._rule_repo = rule_repository or DiagnosticRuleRepository(
            rules_dir=self.config.rules_dir
        )
        self._init_subsystems()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        case_id: str,
        feature_values: dict[str, Any],
        run_id: str | None = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline for a single clinical case.

        Parameters
        ----------
        case_id:
            Unique identifier for the case.
        feature_values:
            Raw clinical feature dict {name: raw_value}.
        run_id:
            Optional run identifier for tracing. Auto-generated if None.
        """
        run_id = run_id or self._generate_run_id()
        log.info("Pipeline execution started", case_id=case_id, run_id=run_id)

        # Reset per-run stateful subsystems so that sequential calls on the
        # same runner instance do not carry over accumulated history from
        # previous cases (FSM state, instability signal window, etc.).
        self._state_tracker.reset()
        self._instability_monitor.reset()

        ctx = PipelineContext(
            case_id=case_id,
            run_id=run_id,
            feature_values=dict(feature_values),
        )

        stage_results: list[StageResult] = []

        # ── Stage 0: Clinical grading ─────────────────────────────────────────
        r = GradingStage.execute(ctx, self._grading)
        stage_results.append(r)
        self._log_stage(r)
        if not r.success:
            return self._abort(ctx, stage_results, "Clinical grading failed.")

        # ── Stage 1: Evidence activation ──────────────────────────────────────
        rules = self._rule_repo.all_rules()
        r = EvidenceActivationStage.execute(ctx, self._evaluator, rules)
        stage_results.append(r)
        self._log_stage(r)
        if not r.success:
            return self._abort(ctx, stage_results, "Evidence activation failed.")

        # ── Stage 2: Contradiction analysis ───────────────────────────────────
        r = ContradictionAnalysisStage.execute(ctx, self._conflict_analyzer)
        stage_results.append(r)
        self._log_stage(r)
        if not r.success:
            return self._abort(ctx, stage_results, "Contradiction analysis failed.")

        # ── Stage 3: Certainty propagation ────────────────────────────────────
        r = CertaintyPropagationStage.execute(ctx, self._propagator)
        stage_results.append(r)
        self._log_stage(r)
        if not r.success:
            return self._abort(ctx, stage_results, "Certainty propagation failed.")

        # ── Stage 4: Differential competition ────────────────────────────────
        r = CompetitionAnalysisStage.execute(ctx, self._competition)
        stage_results.append(r)
        self._log_stage(r)
        # Competition is non-critical; pipeline continues on failure

        # ── Stage 5: Evidence sufficiency ────────────────────────────────────
        r = EvidenceSufficiencyStage.execute(ctx, self._sufficiency)
        stage_results.append(r)
        self._log_stage(r)
        # Sufficiency is non-critical; pipeline continues on failure

        # ── Stage 6: Instability monitoring ──────────────────────────────────
        r = InstabilityStage.execute(ctx, self._instability_monitor, stage_index=6)
        stage_results.append(r)
        self._log_stage(r)

        # ── Stage 7: FSM + Safety gate + Escalation ───────────────────────────
        r = EscalationStage.execute(
            ctx, self._state_tracker, self._safety_gate,
            self._escalation_engine, stage_index=7,
        )
        stage_results.append(r)
        self._log_stage(r)
        if not r.success:
            return self._abort(ctx, stage_results, "Escalation stage failed.")

        # ── Trajectory snapshot ───────────────────────────────────────────────
        if self.config.enable_trace:
            TrajectorySnapshotStage.execute(
                ctx, stage_index=7,
                stage_name="terminal_reasoning",
                delta_description=r.summary,
            )

        # ── Stage 8: Narrative generation (optional) ──────────────────────────
        if self.config.enable_narrative:
            r = NarrativeStage.execute(ctx, self._narrative_generator)
            stage_results.append(r)
            self._log_stage(r)

        # ── Finalise trajectory ───────────────────────────────────────────────
        trajectory = ctx.trajectory_memory.finalise(
            decision=ctx.triage_decision
        )

        # ── Build result ──────────────────────────────────────────────────────
        decision = ctx.triage_decision
        result = PipelineResult(
            case_id=case_id,
            run_id=run_id,
            success=True,
            recommendation=decision.recommendation.value if decision else None,
            leading_disease=decision.leading_disease if decision else None,
            max_certainty=decision.max_certainty if decision else 0.0,
            certainty_gap=decision.certainty_gap if decision else 0.0,
            contradiction_load=decision.contradiction_load if decision else 0.0,
            ambiguity_index=decision.ambiguity_index if decision else 0.0,
            final_state=ctx.current_state.value,
            decision_rationale=decision.decision_rationale if decision else "",
            stage_results=stage_results,
            completed_stages=list(ctx.completed_stages),
            stage_errors=list(ctx.stage_errors),
            trajectory=trajectory,
        )

        # ── Output export ─────────────────────────────────────────────────────
        if self.config.enable_replay_export and decision:
            self.config.ensure_output_dirs()
            result.trace_path = ReasoningTraceExporter.export(
                trajectory, self.config.traces_dir
            )
            result.escalation_path = EscalationReportExporter.export(
                decision, case_id, run_id, self.config.escalation_reports_dir
            )
            result.replay_path = ReplaySnapshotExporter.export(
                trajectory, feature_values, self.config.replay_snapshots_dir
            )
            if ctx.narrative and self.config.enable_narrative:
                result.narrative_path = NarrativeExporter.export(
                    ctx.narrative, case_id, run_id, self.config.narratives_dir
                )

        log.info(
            "Pipeline execution complete",
            case_id=case_id,
            run_id=run_id,
            recommendation=result.recommendation,
            leading=result.leading_disease,
            certainty=f"{result.max_certainty:.3f}",
        )
        return result

    # ── Subsystem initialisation ──────────────────────────────────────────────

    def _init_subsystems(self) -> None:
        """Instantiate all reasoning subsystems from config."""
        cfg = self.config

        self._grading = ClinicalGradingModule(
            grade_map=cfg.ordinal_grade_map,
            significance_threshold=cfg.significance_threshold,
            dormant_threshold=cfg.grading_dormant_threshold,
        )
        self._evaluator = DiagnosticEvidenceEvaluator(
            grading_module=self._grading,
            min_activation_threshold=cfg.min_rule_activation,
        )
        matrix = self._rule_repo.contradiction_matrix()
        self._conflict_analyzer = DiagnosticConflictAnalyzer.from_matrix(
            matrix,
            escalation_ceiling=cfg.contradiction_escalation_ceiling,
        )
        self._propagator = HypothesisCertaintyPropagator(
            softmax_temperature=cfg.softmax_temperature,
            contradiction_damping_threshold=cfg.contradiction_damping_threshold,
            certainty_decay_rate=cfg.certainty_decay_rate,
            stability_gap_threshold=cfg.stability_gap_threshold,
            stability_certainty_threshold=cfg.stability_certainty_threshold,
            high_certainty_gap_threshold=cfg.high_certainty_gap_threshold,
            high_certainty_threshold=cfg.high_certainty_threshold,
        )
        self._competition = DifferentialCompetitionEngine(
            tier_a_specificity_weight=cfg.competition_tier_a_weight,
            contradiction_amplification=cfg.competition_contra_amplify,
        )
        self._sufficiency = EvidenceSufficiencyAnalyzer(
            min_anatomical_domains=cfg.sufficiency_min_domains,
            min_active_rules=cfg.sufficiency_min_rules,
            biopsy_free_sufficiency_threshold=cfg.sufficiency_biopsy_free_threshold,
        )
        self._instability_monitor = DiagnosticInstabilityMonitor(
            instability_threshold=cfg.instability_threshold,
            oscillation_window=cfg.instability_oscillation_window,
        )
        self._state_tracker = DiagnosticStateTracker(
            min_rules_partial=cfg.min_rules_partial,
            min_rules_reinforcing=cfg.min_rules_reinforcing,
            contradiction_detection_threshold=cfg.contradiction_detection_threshold,
            ambiguity_escalation_entropy=cfg.ambiguity_escalation_entropy,
            certainty_stabilization_gap=cfg.certainty_stabilization_gap,
            certainty_stabilization_min=cfg.certainty_stabilization_min,
            safe_triage_gap=cfg.safe_triage_gap,
            safe_triage_min=cfg.safe_triage_min,
            biopsy_contradiction_ceiling=cfg.biopsy_contradiction_ceiling,
            biopsy_entropy_ceiling=cfg.biopsy_entropy_ceiling,
            instability_threshold=cfg.instability_threshold,
        )
        self._safety_gate = ClinicalSafetyGate(
            contradiction_ceiling=cfg.safety_contradiction_ceiling,
            min_activated_rules=cfg.safety_min_activated_rules,
            entropy_ceiling=cfg.safety_entropy_ceiling,
            single_rule_dominance=cfg.safety_single_rule_dominance,
            pathognomonic_certainty_threshold=cfg.safety_patho_certainty,
            max_critical_missing=cfg.safety_max_critical_missing,
            confusion_zone_max_gap=cfg.safety_confusion_max_gap,
            confusion_zone_penalty=cfg.safety_confusion_penalty,
            overconfidence_certainty=cfg.safety_overconf_certainty,
            overconfidence_min_contradiction=cfg.safety_overconf_contradiction,
        )
        self._escalation_engine = ClinicalEscalationEngine(
            safe_min_certainty=cfg.escalation_safe_min_certainty,
            safe_min_gap=cfg.escalation_safe_min_gap,
            safe_max_contradiction=cfg.escalation_safe_max_contra,
            moderate_min_certainty=cfg.escalation_moderate_min_cert,
            moderate_min_gap=cfg.escalation_moderate_min_gap,
            ambiguous_min_certainty=cfg.escalation_ambiguous_min_cert,
            high_risk_contradiction_ceiling=cfg.escalation_high_risk_ceiling,
        )
        self._narrative_generator = DiagnosticNarrativeGenerator()

        log.debug(
            "Subsystems initialised",
            rules_loaded=self._rule_repo.rule_count(),
            discriminators=len(self._rule_repo.discriminators()),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _abort(
        self,
        ctx: PipelineContext,
        stage_results: list[StageResult],
        reason: str,
    ) -> PipelineResult:
        log.error("Pipeline aborted", case_id=ctx.case_id, reason=reason)
        return PipelineResult(
            case_id=ctx.case_id,
            run_id=ctx.run_id,
            success=False,
            final_state=ctx.current_state.value,
            stage_results=stage_results,
            completed_stages=list(ctx.completed_stages),
            stage_errors=list(ctx.stage_errors) + [reason],
        )

    @staticmethod
    def _generate_run_id() -> str:
        return str(uuid.uuid4())[:12]

    @staticmethod
    def _log_stage(result: StageResult) -> None:
        status = "✓" if result.success else "✗"
        log.debug(
            f"Stage {status} [{result.stage_name}]",
            summary=result.summary,
            error=result.error,
        )
