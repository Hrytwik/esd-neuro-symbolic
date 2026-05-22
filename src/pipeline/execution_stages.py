"""
Execution stages — typed, independently testable pipeline stage functions.

Each stage is a callable class with a single `execute(context, ...)` method.
Stages:
  · Accept a PipelineContext and their dedicated subsystem instance(s)
  · Assert required upstream context fields are populated
  · Write their output back into the context
  · Record a trajectory snapshot (if tracing is enabled)
  · Return a StageResult carrying the stage name, success flag, and summary

Stages do NOT contain reasoning logic — they delegate entirely to the
subsystem instances injected by PipelineRunner.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from src.pipeline.pipeline_context import BINARY_FEATURES, PipelineContext


# ── Stage result ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageResult:
    """Outcome of a single pipeline stage execution."""

    stage_name: str
    success:    bool
    summary:    str
    error:      str | None = None


# ── Stage 0 — Clinical Grading ────────────────────────────────────────────────

class GradingStage:
    """
    Stage 0: Convert raw feature values to fuzzy grades.
    Input  → feature_values (already on context)
    Output → context.grading_result, context.missing_features
    """

    NAME = "clinical_grading"

    @staticmethod
    def execute(
        context: PipelineContext,
        grading_module: ClinicalGradingModule,
    ) -> StageResult:
        try:
            result = grading_module.grade_vector(
                context.feature_values,
                binary_features=BINARY_FEATURES,
            )
            context.grading_result = result
            context.missing_features = [
                f.feature_name for f in result.missing_features
            ]
            context.mark_stage_complete(GradingStage.NAME)
            return StageResult(
                stage_name=GradingStage.NAME,
                success=True,
                summary=(
                    f"{len(result.present_features)} feature(s) present; "
                    f"{len(result.missing_features)} missing; "
                    f"completeness={result.completeness_score:.2f}"
                ),
            )
        except Exception as exc:
            context.record_stage_error(GradingStage.NAME, str(exc))
            return StageResult(
                stage_name=GradingStage.NAME, success=False,
                summary="Grading failed.", error=str(exc),
            )


# ── Stage 1 — Evidence Activation ────────────────────────────────────────────

class EvidenceActivationStage:
    """
    Stage 1: Evaluate symbolic rules against graded features.
    Input  → context.grading_result + rules list
    Output → context.evidence_result
    """

    NAME = "evidence_activation"

    @staticmethod
    def execute(
        context: PipelineContext,
        evaluator: DiagnosticEvidenceEvaluator,
        rules: list[dict],
    ) -> StageResult:
        try:
            context.require("grading_result")
            result = evaluator.evaluate(
                context.grading_result,  # type: ignore[arg-type]
                rules,
            )
            context.evidence_result = result
            context.mark_stage_complete(EvidenceActivationStage.NAME)
            return StageResult(
                stage_name=EvidenceActivationStage.NAME,
                success=True,
                summary=(
                    f"{result.total_rules_active} rule(s) active / "
                    f"{result.total_rules_checked} checked; "
                    f"leading={result.leading_disease}"
                ),
            )
        except Exception as exc:
            context.record_stage_error(EvidenceActivationStage.NAME, str(exc))
            return StageResult(
                stage_name=EvidenceActivationStage.NAME, success=False,
                summary="Evidence activation failed.", error=str(exc),
            )


# ── Stage 2 — Contradiction Analysis ─────────────────────────────────────────

class ContradictionAnalysisStage:
    """
    Stage 2: Detect cross-disease contradictions and compute penalty load.
    Input  → context.feature_values
    Output → context.conflict_result
    """

    NAME = "contradiction_analysis"

    @staticmethod
    def execute(
        context: PipelineContext,
        conflict_analyzer: DiagnosticConflictAnalyzer,
    ) -> StageResult:
        try:
            result = conflict_analyzer.analyze(context.feature_values)
            context.conflict_result = result
            context.mark_stage_complete(ContradictionAnalysisStage.NAME)
            return StageResult(
                stage_name=ContradictionAnalysisStage.NAME,
                success=True,
                summary=(
                    f"{len(result.active_contradictions)} contradiction(s); "
                    f"load={result.contradiction_load:.3f}; "
                    f"mandatory_escalation={result.mandatory_escalation}"
                ),
            )
        except Exception as exc:
            context.record_stage_error(ContradictionAnalysisStage.NAME, str(exc))
            return StageResult(
                stage_name=ContradictionAnalysisStage.NAME, success=False,
                summary="Contradiction analysis failed.", error=str(exc),
            )


# ── Stage 3 — Certainty Propagation ──────────────────────────────────────────

class CertaintyPropagationStage:
    """
    Stage 3: Propagate evidence scores through softmax normalisation with
    contradiction dampening.
    Input  → context.evidence_result, context.conflict_result
    Output → context.certainty_dist
    """

    NAME = "certainty_propagation"

    @staticmethod
    def execute(
        context: PipelineContext,
        propagator: HypothesisCertaintyPropagator,
    ) -> StageResult:
        try:
            context.require("evidence_result", "conflict_result")
            dist = propagator.propagate(
                context.evidence_result,   # type: ignore[arg-type]
                context.conflict_result,   # type: ignore[arg-type]
            )
            context.certainty_dist = dist
            context.mark_stage_complete(CertaintyPropagationStage.NAME)
            return StageResult(
                stage_name=CertaintyPropagationStage.NAME,
                success=True,
                summary=(
                    f"leading={dist.leading_disease} "
                    f"certainty={dist.max_certainty:.3f} "
                    f"gap={dist.certainty_gap:.3f} "
                    f"entropy={dist.ambiguity_index:.3f} bits "
                    f"dampened={dist.contradiction_dampened}"
                ),
            )
        except Exception as exc:
            context.record_stage_error(CertaintyPropagationStage.NAME, str(exc))
            return StageResult(
                stage_name=CertaintyPropagationStage.NAME, success=False,
                summary="Certainty propagation failed.", error=str(exc),
            )


# ── Stage 4 — Competition Analysis ───────────────────────────────────────────

class CompetitionAnalysisStage:
    """
    Stage 4: Compute inter-hypothesis suppression and competition scores.
    Input  → context.certainty_dist, context.evidence_result, context.conflict_result
    Output → context.competition_result
    """

    NAME = "competition_analysis"

    @staticmethod
    def execute(
        context: PipelineContext,
        engine: DifferentialCompetitionEngine,
    ) -> StageResult:
        try:
            context.require("certainty_dist", "evidence_result", "conflict_result")
            result = engine.evaluate(
                context.certainty_dist,   # type: ignore[arg-type]
                context.evidence_result,  # type: ignore[arg-type]
                context.conflict_result,  # type: ignore[arg-type]
            )
            context.competition_result = result
            context.mark_stage_complete(CompetitionAnalysisStage.NAME)
            return StageResult(
                stage_name=CompetitionAnalysisStage.NAME,
                success=True,
                summary=(
                    f"leading_by_competition={result.leading_by_competition} "
                    f"gap={result.competition_gap:.3f} "
                    f"divergence_amplified={result.divergence_amplified}"
                ),
            )
        except Exception as exc:
            context.record_stage_error(CompetitionAnalysisStage.NAME, str(exc))
            return StageResult(
                stage_name=CompetitionAnalysisStage.NAME, success=False,
                summary="Competition analysis failed.", error=str(exc),
            )


# ── Stage 5 — Evidence Sufficiency ───────────────────────────────────────────

class EvidenceSufficiencyStage:
    """
    Stage 5: Assess evidence quality and coverage for the leading hypothesis.
    Input  → context.evidence_result, context.certainty_dist
    Output → context.sufficiency_report
    """

    NAME = "evidence_sufficiency"

    @staticmethod
    def execute(
        context: PipelineContext,
        analyzer: EvidenceSufficiencyAnalyzer,
    ) -> StageResult:
        try:
            context.require("evidence_result", "certainty_dist")
            report = analyzer.analyze(
                context.evidence_result,  # type: ignore[arg-type]
                context.certainty_dist,   # type: ignore[arg-type]
            )
            context.sufficiency_report = report
            context.mark_stage_complete(EvidenceSufficiencyStage.NAME)
            return StageResult(
                stage_name=EvidenceSufficiencyStage.NAME,
                success=True,
                summary=report.summary,
            )
        except Exception as exc:
            context.record_stage_error(EvidenceSufficiencyStage.NAME, str(exc))
            return StageResult(
                stage_name=EvidenceSufficiencyStage.NAME, success=False,
                summary="Sufficiency analysis failed.", error=str(exc),
            )


# ── Stage 6 — Instability Monitoring ─────────────────────────────────────────

class InstabilityStage:
    """
    Stage 6: Update instability monitor and assess reasoning trajectory volatility.
    Input  → context.certainty_dist, context.conflict_result, context.current_state
    Output → context.instability_report (via monitor update + assess)
    """

    NAME = "instability_monitoring"

    @staticmethod
    def execute(
        context: PipelineContext,
        monitor: DiagnosticInstabilityMonitor,
        stage_index: int,
    ) -> StageResult:
        try:
            context.require("certainty_dist", "conflict_result")
            dist    = context.certainty_dist   # type: ignore[union-attr]
            conflict = context.conflict_result  # type: ignore[union-attr]

            monitor.update(
                stage=stage_index,
                max_certainty=dist.max_certainty,
                contradiction_load=conflict.contradiction_load,
                ambiguity_index=dist.ambiguity_index,
                leading_disease=dist.leading_disease,
                state=context.current_state.value,
            )
            report = monitor.assess()
            context.instability_report = report
            context.mark_stage_complete(InstabilityStage.NAME)
            return StageResult(
                stage_name=InstabilityStage.NAME,
                success=True,
                summary=(
                    f"index={report.instability_index:.3f} "
                    f"signals={report.signal_count} "
                    f"is_unstable={report.is_unstable}"
                ),
            )
        except Exception as exc:
            context.record_stage_error(InstabilityStage.NAME, str(exc))
            return StageResult(
                stage_name=InstabilityStage.NAME, success=False,
                summary="Instability monitoring failed.", error=str(exc),
            )


# ── Stage 7 — FSM + Safety Gate + Escalation ─────────────────────────────────

class EscalationStage:
    """
    Stage 7: Advance the diagnostic FSM, evaluate safety gates, and produce
    the terminal triage recommendation.
    Input  → context.evidence_result, context.conflict_result,
             context.certainty_dist, context.instability_report
    Output → context.current_state, context.safety_report, context.triage_decision
    """

    NAME = "escalation"

    @staticmethod
    def execute(
        context: PipelineContext,
        state_tracker: DiagnosticStateTracker,
        safety_gate: ClinicalSafetyGate,
        escalation_engine: ClinicalEscalationEngine,
        stage_index: int,
    ) -> StageResult:
        try:
            context.require("evidence_result", "conflict_result", "certainty_dist")
            instability_idx = (
                context.instability_report.instability_index
                if context.instability_report else 0.0
            )

            # FSM advance
            new_state = state_tracker.advance(
                stage=stage_index,
                evidence=context.evidence_result,     # type: ignore[arg-type]
                conflict=context.conflict_result,      # type: ignore[arg-type]
                certainty=context.certainty_dist,      # type: ignore[arg-type]
                instability_index=instability_idx,
            )
            context.current_state = new_state

            # Safety gate evaluation
            safety_report = safety_gate.evaluate(
                certainty=context.certainty_dist,      # type: ignore[arg-type]
                conflict=context.conflict_result,      # type: ignore[arg-type]
                evidence=context.evidence_result,      # type: ignore[arg-type]
                missing_features=context.missing_features,
            )
            context.safety_report = safety_report

            # Triage decision
            decision = escalation_engine.decide(
                certainty=context.certainty_dist,      # type: ignore[arg-type]
                conflict=context.conflict_result,      # type: ignore[arg-type]
                evidence=context.evidence_result,      # type: ignore[arg-type]
                safety_report=safety_report,
                final_state=new_state,
            )
            context.triage_decision = decision
            context.mark_stage_complete(EscalationStage.NAME)

            return StageResult(
                stage_name=EscalationStage.NAME,
                success=True,
                summary=(
                    f"state={new_state.value} | "
                    f"recommendation={decision.recommendation.value} | "
                    f"certainty={decision.max_certainty:.3f} | "
                    f"safety_triggered={safety_report.any_triggered}"
                ),
            )
        except Exception as exc:
            context.record_stage_error(EscalationStage.NAME, str(exc))
            return StageResult(
                stage_name=EscalationStage.NAME, success=False,
                summary="Escalation stage failed.", error=str(exc),
            )


# ── Stage 8 — Narrative Generation ───────────────────────────────────────────

class NarrativeStage:
    """
    Stage 8: Generate the structured clinical reasoning narrative.
    Input  → context.evidence_result, context.conflict_result,
             context.certainty_dist, context.safety_report,
             context.triage_decision, context.current_state
    Output → context.narrative
    """

    NAME = "narrative_generation"

    @staticmethod
    def execute(
        context: PipelineContext,
        generator: DiagnosticNarrativeGenerator,
    ) -> StageResult:
        try:
            context.require(
                "evidence_result", "conflict_result", "certainty_dist",
                "safety_report", "triage_decision",
            )
            narrative = generator.generate(
                evidence=context.evidence_result,      # type: ignore[arg-type]
                conflict=context.conflict_result,      # type: ignore[arg-type]
                certainty=context.certainty_dist,      # type: ignore[arg-type]
                safety_report=context.safety_report,   # type: ignore[arg-type]
                decision=context.triage_decision,      # type: ignore[arg-type]
                final_state=context.current_state,
            )
            context.narrative = narrative
            context.mark_stage_complete(NarrativeStage.NAME)
            return StageResult(
                stage_name=NarrativeStage.NAME,
                success=True,
                summary="Clinical reasoning narrative generated.",
            )
        except Exception as exc:
            context.record_stage_error(NarrativeStage.NAME, str(exc))
            return StageResult(
                stage_name=NarrativeStage.NAME, success=False,
                summary="Narrative generation failed.", error=str(exc),
            )


# ── Stage 9 — Trajectory Snapshot ────────────────────────────────────────────

class TrajectorySnapshotStage:
    """
    Stage 9: Record the completed reasoning state to the trajectory memory.
    Input  → all populated context fields
    Output → snapshot recorded to context.trajectory_memory
    """

    NAME = "trajectory_snapshot"

    @staticmethod
    def execute(
        context: PipelineContext,
        stage_index: int,
        stage_name: str,
        delta_description: str = "",
    ) -> StageResult:
        try:
            context.trajectory_memory.record(
                stage=stage_index,
                stage_name=stage_name,
                state=context.current_state,
                evidence=context.evidence_result,
                conflict=context.conflict_result,
                certainty=context.certainty_dist,
                safety_report=context.safety_report,
                delta_description=delta_description,
            )
            return StageResult(
                stage_name=TrajectorySnapshotStage.NAME,
                success=True,
                summary=f"Snapshot recorded at stage={stage_index}.",
            )
        except Exception as exc:
            context.record_stage_error(TrajectorySnapshotStage.NAME, str(exc))
            return StageResult(
                stage_name=TrajectorySnapshotStage.NAME, success=False,
                summary="Trajectory snapshot failed.", error=str(exc),
            )
