"""
backend_refinement_report.py
===============================
Comprehensive backend refinement report for the CASDRE clinical inference
pipeline.

Aggregates signals from all 10 refinement modules into a unified maturity
progression report that defines whether the backend is ready for frontend
transition.

The report tracks:
  - Model B and C accuracy progress
  - Symbolic recovery improvements
  - Escalation selectivity improvements
  - Contradiction handling improvements
  - Disease-wise discrimination improvements
  - Trajectory realism improvements
  - Overall backend maturity progression
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class RefinementPhase(str, Enum):
    PHASE_INITIAL    = "phase_initial"     # < 0.55
    PHASE_DEVELOPING = "phase_developing"  # 0.55–0.70
    PHASE_STABLE     = "phase_stable"      # 0.70–0.85
    PHASE_STRONG     = "phase_strong"      # ≥ 0.85


class FrontendTransitionDecision(str, Enum):
    NOT_READY              = "not_ready"
    CONDITIONALLY_READY    = "conditionally_ready"
    READY                  = "ready"


# ──────────────────────────────────────────────────────────────────────────────
# Sub-report stubs (accepted from other modules or provided inline)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProgressEntry:
    model_label: str        # "Model B" / "Model C"
    current_accuracy: float
    target_low: float
    target_high: float
    symbolic_lift_pp: float
    target_met: bool
    progress_note: str


@dataclass
class SubsystemRefinementEntry:
    subsystem: str
    score: float             # [0, 1]
    key_finding: str
    action_needed: bool


@dataclass
class ProgressionTimeline:
    """Tracks the improvement trajectory across refinement iterations."""
    iterations: List[Dict[str, float]]   # each dict: {"iteration": n, "accuracy_c": x, ...}
    total_iterations: int
    best_accuracy_c: float
    best_symbolic_lift_pp: float
    iterations_to_target: Optional[int]  # None if target not yet reached


@dataclass
class BackendRefinementReport:
    """Master backend refinement report."""
    # Model progress
    model_progress: List[ModelProgressEntry]

    # Subsystem refinement entries
    subsystem_entries: List[SubsystemRefinementEntry]

    # Optional progression tracking
    progression_timeline: Optional[ProgressionTimeline]

    # Overall scores
    overall_refinement_score: float   # [0, 1]
    refinement_phase: RefinementPhase

    # Safety compliance
    contradiction_ceiling_compliant: bool   # 0.40 ceiling — must be True
    escalation_safety_passed: bool          # zero unsafe stabilisations
    all_safety_constraints_met: bool

    # Frontend transition
    frontend_transition: FrontendTransitionDecision
    frontend_blockers: List[str]
    frontend_conditionals: List[str]

    # Publication readiness
    publication_ready: bool
    publication_gaps: List[str]

    # Summary checklist
    checklist: List[Dict[str, Any]]

    # Recommendations
    priority_actions: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "BACKEND REFINEMENT REPORT — CASDRE CLINICAL INFERENCE PIPELINE",
            "=" * 70,
            f"  Refinement phase       : {self.refinement_phase.value.upper()}",
            f"  Overall score          : {self.overall_refinement_score:.3f}",
            f"  Frontend transition    : {self.frontend_transition.value.upper()}",
            f"  Publication ready      : {'YES' if self.publication_ready else 'NO'}",
            "",
            "  ── Model Progress ────────────────────────────────────────────",
            f"  {'Model':<14s}  {'Current':>8s}  {'Target':>14s}  "
            f"{'Lift':>8s}  {'Met':>5s}",
            f"  {'-'*14}  {'-'*8}  {'-'*14}  {'-'*8}  {'-'*5}",
        ]
        for mp in self.model_progress:
            target_str = f"{mp.target_low:.0%}–{mp.target_high:.0%}"
            met_str    = "✓" if mp.target_met else "✗"
            lines.append(
                f"  {mp.model_label:<14s}  {mp.current_accuracy:>8.3f}  "
                f"{target_str:>14s}  {mp.symbolic_lift_pp:>+7.2f}pp  {met_str:>5s}"
            )
            if mp.progress_note:
                lines.append(f"    → {mp.progress_note}")
        lines += [
            "",
            "  ── Subsystem Refinement Scores ───────────────────────────────",
        ]
        for se in sorted(self.subsystem_entries, key=lambda s: s.score):
            action = "  ← ACTION NEEDED" if se.action_needed else ""
            lines.append(
                f"    {se.subsystem:<42s}  {se.score:.3f}{action}"
            )
            if se.key_finding:
                lines.append(f"      └ {se.key_finding}")
        lines += [
            "",
            "  ── Safety Compliance ─────────────────────────────────────────",
            f"    Contradiction ceiling (0.40)  : "
            f"{'✓ COMPLIANT' if self.contradiction_ceiling_compliant else '✗ VIOLATION'}",
            f"    Escalation safety audit       : "
            f"{'✓ PASSED' if self.escalation_safety_passed else '✗ FAILED'}",
            f"    All safety constraints        : "
            f"{'✓ MET' if self.all_safety_constraints_met else '✗ NOT MET'}",
        ]
        if self.frontend_blockers:
            lines += ["", "  ── Frontend Blockers ─────────────────────────────────────────"]
            for b in self.frontend_blockers:
                lines.append(f"    ✗  {b}")
        if self.frontend_conditionals:
            lines += ["", "  ── Frontend Conditionals ────────────────────────────────────"]
            for c in self.frontend_conditionals:
                lines.append(f"    ⚠  {c}")
        if self.publication_gaps:
            lines += ["", "  ── Publication Gaps ──────────────────────────────────────────"]
            for g in self.publication_gaps:
                lines.append(f"    –  {g}")
        lines += ["", "  ── Priority Actions ──────────────────────────────────────────"]
        for i, act in enumerate(self.priority_actions, 1):
            lines.append(f"    {i}. {act}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refinement_phase": self.refinement_phase.value,
            "overall_refinement_score": self.overall_refinement_score,
            "frontend_transition": self.frontend_transition.value,
            "publication_ready": self.publication_ready,
            "model_progress": [
                {
                    "model": mp.model_label,
                    "accuracy": mp.current_accuracy,
                    "target_met": mp.target_met,
                    "lift_pp": mp.symbolic_lift_pp,
                }
                for mp in self.model_progress
            ],
            "safety": {
                "contradiction_ceiling_compliant": self.contradiction_ceiling_compliant,
                "escalation_safety_passed": self.escalation_safety_passed,
                "all_constraints_met": self.all_safety_constraints_met,
            },
            "subsystem_scores": {
                se.subsystem: se.score for se in self.subsystem_entries
            },
            "frontend_blockers": self.frontend_blockers,
            "publication_gaps": self.publication_gaps,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Compiler
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_B_TARGET   = (0.85, 0.87)
_MODEL_C_TARGET   = (0.88, 0.91)
_MODEL_A_TARGET   = (0.98, 1.00)

_PHASE_INITIAL    = 0.55
_PHASE_DEVELOPING = 0.70
_PHASE_STABLE     = 0.85


def _refinement_phase(score: float) -> RefinementPhase:
    if score >= _PHASE_STABLE:
        return RefinementPhase.PHASE_STRONG
    elif score >= _PHASE_DEVELOPING:
        return RefinementPhase.PHASE_STABLE
    elif score >= _PHASE_INITIAL:
        return RefinementPhase.PHASE_DEVELOPING
    return RefinementPhase.PHASE_INITIAL


class BackendRefinementReporter:
    """
    Compiles the master backend refinement report.

    All parameters are optional with sensible defaults so the report can
    be produced incrementally as refinement modules are completed.
    """

    def compile(
        self,
        *,
        model_a_accuracy: float = 0.9818,
        model_b_accuracy: float = 0.80,
        model_c_accuracy: float = 0.8182,
        escalation_rate: float  = 0.35,
        symbolic_recovery_rate: float   = 0.30,
        trajectory_realism_score: float = 0.70,
        contradiction_ceiling_compliant: bool = True,
        escalation_safety_passed: bool        = True,
        subsystem_scores: Optional[Dict[str, float]] = None,
        subsystem_findings: Optional[Dict[str, str]] = None,
        progression_timeline: Optional[ProgressionTimeline] = None,
    ) -> BackendRefinementReport:
        """Compile the full refinement report."""

        subsystem_scores   = subsystem_scores or {}
        subsystem_findings = subsystem_findings or {}

        # ── Model progress ──────────────────────────────────────────────
        lift_b   = 0.0                              # Model B = clinical baseline
        lift_c   = (model_c_accuracy - model_b_accuracy) * 100.0

        b_met = _MODEL_B_TARGET[0] <= model_b_accuracy <= _MODEL_B_TARGET[1]
        c_met = _MODEL_C_TARGET[0] <= model_c_accuracy <= _MODEL_C_TARGET[1]
        a_met = model_a_accuracy >= _MODEL_A_TARGET[0]

        b_note = (
            "Within target range." if b_met
            else f"Gap: {(_MODEL_B_TARGET[0] - model_b_accuracy)*100:.1f} pp to floor."
        )
        c_note = (
            "Within target range." if c_met
            else f"Gap: {(_MODEL_C_TARGET[0] - model_c_accuracy)*100:.1f} pp to floor."
        )

        model_progress = [
            ModelProgressEntry("Model A", model_a_accuracy, 0.98, 1.00, 0.0, a_met,
                               "Biopsy-complete baseline." if a_met else "Investigate."),
            ModelProgressEntry("Model B", model_b_accuracy, *_MODEL_B_TARGET, lift_b, b_met, b_note),
            ModelProgressEntry("Model C", model_c_accuracy, *_MODEL_C_TARGET, lift_c, c_met, c_note),
        ]

        # ── Subsystem entries ───────────────────────────────────────────
        _default_scores = {
            "model_c_optimizer":               model_c_accuracy,
            "symbolic_recovery_refinement":    symbolic_recovery_rate,
            "disease_discrimination_refinement": 0.78,
            "escalation_behavior_refinement":  min(1.0, escalation_rate / 0.50),
            "contradiction_competition_refinement": 0.80,
            "trajectory_realism_refinement":   trajectory_realism_score,
            "rare_disease_refinement":         0.72,
            "symbolic_rule_refinement_v2":     0.80,
            "certainty_behavior_refinement":   0.75,
            "publication_evaluation_suite":    0.83,
        }
        _default_scores.update(subsystem_scores)

        subsystem_entries = [
            SubsystemRefinementEntry(
                subsystem=name,
                score=score,
                key_finding=subsystem_findings.get(name, ""),
                action_needed=(score < 0.70),
            )
            for name, score in _default_scores.items()
        ]

        # ── Overall score ───────────────────────────────────────────────
        model_score   = statistics.mean([
            min(model_b_accuracy / _MODEL_B_TARGET[1], 1.0),
            min(model_c_accuracy / _MODEL_C_TARGET[1], 1.0),
        ])
        sys_score     = statistics.mean(se.score for se in subsystem_entries)
        safety_score  = (1.0 if contradiction_ceiling_compliant else 0.0) * \
                        (1.0 if escalation_safety_passed else 0.5)

        overall = model_score * 0.35 + sys_score * 0.40 + safety_score * 0.25
        phase   = _refinement_phase(overall)

        # ── Safety ──────────────────────────────────────────────────────
        all_safe = contradiction_ceiling_compliant and escalation_safety_passed

        # ── Frontend transition ─────────────────────────────────────────
        blockers: List[str] = []
        conditionals: List[str] = []

        if not contradiction_ceiling_compliant:
            blockers.append("Contradiction ceiling (0.40) violation — CRITICAL blocker.")
        if not escalation_safety_passed:
            blockers.append("Unsafe stabilisations detected — CRITICAL blocker.")
        if not c_met:
            conditionals.append(
                f"Model C ({model_c_accuracy:.1%}) below 88–91 % target — "
                "strengthen symbolic features before frontend."
            )
        if symbolic_recovery_rate < 0.40:
            conditionals.append(
                f"Symbolic recovery rate ({symbolic_recovery_rate:.1%}) below 40 % — "
                "recovery must be substantial to justify symbolic contribution."
            )
        if trajectory_realism_score < 0.65:
            conditionals.append(
                "Trajectory realism below 0.65 — smooth trajectories before "
                "frontend visualisation."
            )

        if blockers:
            transition = FrontendTransitionDecision.NOT_READY
        elif conditionals:
            transition = FrontendTransitionDecision.CONDITIONALLY_READY
        else:
            transition = FrontendTransitionDecision.READY

        # ── Publication readiness ───────────────────────────────────────
        pub_gaps: List[str] = []
        if not c_met:
            pub_gaps.append(
                "Model C accuracy below publication-grade 88 % floor."
            )
        if symbolic_recovery_rate < 0.35:
            pub_gaps.append(
                "Symbolic recovery rate too low for meaningful scientific claim."
            )
        if not b_met:
            pub_gaps.append(
                "Model B accuracy below 85 % — biopsy-free baseline insufficient."
            )
        pub_ready = len(pub_gaps) == 0

        # ── Checklist ───────────────────────────────────────────────────
        checklist = [
            {"item": "Model B ≥ 85 %",                        "done": b_met,     "critical": False},
            {"item": "Model C ≥ 88 %",                        "done": c_met,     "critical": False},
            {"item": "Symbolic lift ≥ 5 pp",                  "done": lift_c >= 5.0, "critical": False},
            {"item": "Symbolic recovery rate ≥ 40 %",        "done": symbolic_recovery_rate >= 0.40, "critical": False},
            {"item": "Contradiction ceiling compliant",        "done": contradiction_ceiling_compliant, "critical": True},
            {"item": "Escalation safety passed",              "done": escalation_safety_passed, "critical": True},
            {"item": "Escalation rate 20–70 %",              "done": 0.20 <= escalation_rate <= 0.70, "critical": False},
            {"item": "Trajectory realism ≥ 0.70",            "done": trajectory_realism_score >= 0.70, "critical": False},
            {"item": "All subsystems scoring ≥ 0.70",        "done": all(se.score >= 0.70 for se in subsystem_entries), "critical": False},
        ]

        # ── Priority actions ────────────────────────────────────────────
        priority: List[str] = []
        if not c_met:
            priority.append(
                f"Run model_c_optimizer with full symbolic feature set — "
                f"current {model_c_accuracy:.1%}, target 88–91 %."
            )
        if symbolic_recovery_rate < 0.40:
            priority.append(
                "Strengthen symbolic recovery taxonomy — "
                "especially CONTRADICTION_RESOLUTION and SIGNATURE_MATCH mechanisms."
            )
        weak_sys = sorted(
            (se for se in subsystem_entries if se.action_needed),
            key=lambda s: s.score,
        )
        for se in weak_sys[:2]:
            priority.append(
                f"Improve '{se.subsystem}' (score={se.score:.3f}) — "
                f"{se.key_finding or 'score below 0.70 threshold'}."
            )
        if not priority:
            priority.append(
                "Backend refinement on track — continue iterative evaluation "
                "until all subsystems reach ≥ 0.80."
            )

        return BackendRefinementReport(
            model_progress=model_progress,
            subsystem_entries=subsystem_entries,
            progression_timeline=progression_timeline,
            overall_refinement_score=overall,
            refinement_phase=phase,
            contradiction_ceiling_compliant=contradiction_ceiling_compliant,
            escalation_safety_passed=escalation_safety_passed,
            all_safety_constraints_met=all_safe,
            frontend_transition=transition,
            frontend_blockers=blockers,
            frontend_conditionals=conditionals,
            publication_ready=pub_ready,
            publication_gaps=pub_gaps,
            checklist=checklist,
            priority_actions=priority[:4],
        )
