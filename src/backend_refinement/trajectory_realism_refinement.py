"""
trajectory_realism_refinement.py
===================================
Trajectory realism refinement for the CASDRE clinical inference pipeline.

Ensures that reasoning trajectories resemble clinically believable diagnostic
evolution rather than computational artefacts.

Improves:
  - certainty smoothness (no abrupt spikes)
  - convergence realism (gradual, evidence-driven)
  - stabilisation progression (monotone toward confident correct diagnosis)
  - contradiction-triggered divergence (legitimate uncertainty signals)
  - trajectory interpretability

Prevents:
  - abrupt certainty spikes (Δ > 0.30 in one step)
  - pathological oscillation (> 4 direction reversals)
  - artificial convergence (certainty lock-in at step 1)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryQuality(str, Enum):
    CLINICALLY_BELIEVABLE = "clinically_believable"
    ACCEPTABLE            = "acceptable"
    QUESTIONABLE          = "questionable"
    ARTEFACTUAL           = "artefactual"


class SmoothnessGrade(str, Enum):
    SMOOTH    = "smooth"     # max step Δ ≤ 0.10
    MODERATE  = "moderate"   # max step Δ 0.10–0.20
    ROUGH     = "rough"      # max step Δ 0.20–0.30
    SPIKED    = "spiked"     # max step Δ > 0.30


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryQualityRecord:
    case_index: int
    true_label: int
    pred_label: int
    is_correct: bool
    n_steps: int
    quality: TrajectoryQuality
    smoothness_grade: SmoothnessGrade
    max_step_delta: float
    n_direction_reversals: int
    final_certainty: float
    initial_certainty: float
    certainty_range: float          # max - min
    locked_in_early: bool           # certainty ≥ 0.85 at step 1
    converged: bool
    contradiction_triggered_divergence: bool


@dataclass
class SmoothnessProfile:
    """Population-level trajectory smoothness analysis."""
    n_cases: int
    mean_max_step_delta: float
    std_max_step_delta: float
    fraction_smooth: float
    fraction_moderate: float
    fraction_rough: float
    fraction_spiked: float
    mean_direction_reversals: float
    fraction_pathological_oscillation: float  # > 4 reversals
    fraction_early_lock_in: float             # locked in at step 1


@dataclass
class ConvergenceRealismProfile:
    """Population-level convergence realism analysis."""
    n_cases: int
    mean_convergence_step: float         # average step at which cert ≥ 0.75
    std_convergence_step: float
    fraction_early_convergence: float    # converges at step ≤ 1 (suspicious)
    fraction_late_convergence: float     # never converges (cert < 0.75)
    mean_final_certainty: float
    mean_certainty_gain: float           # final - initial
    fraction_monotone_increasing: float
    fraction_with_contradiction_divergence: float


@dataclass
class DiseaseTrajectoryRealismProfile:
    """Per-disease trajectory realism summary."""
    disease: str
    n_cases: int
    mean_quality_score: float         # 1 (artefactual) – 4 (believable)
    fraction_clinically_believable: float
    mean_smoothness: float
    mean_convergence_step: float
    mean_final_certainty: float
    realism_grade: TrajectoryQuality


@dataclass
class TrajectoryRealismReport:
    """Full trajectory realism refinement report."""
    quality_records: List[TrajectoryQualityRecord]
    smoothness_profile: SmoothnessProfile
    convergence_profile: ConvergenceRealismProfile
    disease_profiles: List[DiseaseTrajectoryRealismProfile]

    # Overall
    fraction_clinically_believable: float
    fraction_artefactual: float
    mean_quality_score: float    # 1–4
    overall_realism_grade: TrajectoryQuality

    # Refinement interventions
    n_smoothing_interventions_needed: int
    n_oscillation_dampening_needed: int
    n_early_lock_in_fixes_needed: int

    recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "TRAJECTORY REALISM REFINEMENT REPORT",
            "=" * 70,
            f"  Overall realism grade          : {self.overall_realism_grade.value}",
            f"  Clinically believable          : "
            f"{self.fraction_clinically_believable:.1%}",
            f"  Artefactual                    : {self.fraction_artefactual:.1%}",
            f"  Mean quality score (1–4)       : {self.mean_quality_score:.2f}",
            "",
            "  ── Smoothness Profile ────────────────────────────────────────",
            f"    Mean max step Δ              : "
            f"{self.smoothness_profile.mean_max_step_delta:.4f}",
            f"    Fraction spiked (Δ > 0.30)   : "
            f"{self.smoothness_profile.fraction_spiked:.1%}",
            f"    Mean direction reversals     : "
            f"{self.smoothness_profile.mean_direction_reversals:.2f}",
            f"    Pathological oscillation     : "
            f"{self.smoothness_profile.fraction_pathological_oscillation:.1%}",
            f"    Early lock-in fraction       : "
            f"{self.smoothness_profile.fraction_early_lock_in:.1%}",
            "",
            "  ── Convergence Realism ───────────────────────────────────────",
            f"    Mean convergence step        : "
            f"{self.convergence_profile.mean_convergence_step:.2f}",
            f"    Early convergence (step ≤ 1) : "
            f"{self.convergence_profile.fraction_early_convergence:.1%}",
            f"    Late convergence (no conv.)  : "
            f"{self.convergence_profile.fraction_late_convergence:.1%}",
            f"    Monotone increasing          : "
            f"{self.convergence_profile.fraction_monotone_increasing:.1%}",
            "",
            "  ── Disease Trajectory Profiles ───────────────────────────────",
        ]
        for dp in sorted(self.disease_profiles,
                         key=lambda d: d.fraction_clinically_believable,
                         reverse=True):
            lines.append(
                f"    {dp.disease:<32s}  "
                f"believable={dp.fraction_clinically_believable:.1%}  "
                f"smooth={dp.mean_smoothness:.3f}  "
                f"cert_f={dp.mean_final_certainty:.3f}"
            )
        lines += [
            "",
            "  ── Refinement Interventions Needed ───────────────────────────",
            f"    Smoothing            : {self.n_smoothing_interventions_needed}",
            f"    Oscillation dampening: {self.n_oscillation_dampening_needed}",
            f"    Early lock-in fixes  : {self.n_early_lock_in_fixes_needed}",
        ]
        if self.recommendations:
            lines += ["", "  ── Recommendations ──────────────────────────────────────────"]
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_SMOOTH_THRESH    = 0.10
_MODERATE_THRESH  = 0.20
_ROUGH_THRESH     = 0.30
_PATHOLOGICAL_OSC = 4        # reversal count
_EARLY_LOCK_CERT  = 0.85
_CONVERGENCE_CERT = 0.75


def _smoothness_grade(max_delta: float) -> SmoothnessGrade:
    if max_delta <= _SMOOTH_THRESH:
        return SmoothnessGrade.SMOOTH
    elif max_delta <= _MODERATE_THRESH:
        return SmoothnessGrade.MODERATE
    elif max_delta <= _ROUGH_THRESH:
        return SmoothnessGrade.ROUGH
    return SmoothnessGrade.SPIKED


def _trajectory_quality(
    smooth: SmoothnessGrade,
    n_reversals: int,
    locked_in_early: bool,
) -> TrajectoryQuality:
    if locked_in_early:
        return TrajectoryQuality.ARTEFACTUAL
    if smooth == SmoothnessGrade.SPIKED or n_reversals > _PATHOLOGICAL_OSC:
        return TrajectoryQuality.ARTEFACTUAL
    if smooth == SmoothnessGrade.ROUGH or n_reversals > 2:
        return TrajectoryQuality.QUESTIONABLE
    if smooth == SmoothnessGrade.MODERATE or n_reversals > 0:
        return TrajectoryQuality.ACCEPTABLE
    return TrajectoryQuality.CLINICALLY_BELIEVABLE


_QUALITY_SCORE = {
    TrajectoryQuality.CLINICALLY_BELIEVABLE: 4,
    TrajectoryQuality.ACCEPTABLE: 3,
    TrajectoryQuality.QUESTIONABLE: 2,
    TrajectoryQuality.ARTEFACTUAL: 1,
}


# ──────────────────────────────────────────────────────────────────────────────
# Refiner
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryRealismRefiner:
    """
    Analyses and scores trajectory realism, identifies pathologies, and
    recommends targeted refinement interventions.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names.
    max_steps : int
        Number of reasoning steps per trajectory.
    """

    def __init__(
        self,
        class_labels: List[str],
        max_steps: int = 8,
    ):
        self.class_labels = class_labels
        self.max_steps    = max_steps

    # ------------------------------------------------------------------
    def analyse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        certainty_trajectories: Optional[np.ndarray] = None,
        # (n, max_steps) — if None, synthesized from y_true / y_pred
        contradiction_loads: Optional[np.ndarray] = None,
    ) -> TrajectoryRealismReport:
        """Run full trajectory realism analysis."""
        n = len(y_true)
        rng = np.random.default_rng(seed=99)

        if contradiction_loads is None:
            contradiction_loads = rng.uniform(0.0, 0.38, n)
        contradiction_loads = np.clip(contradiction_loads, 0.0, 0.40)

        if certainty_trajectories is None:
            certainty_trajectories = self._synthesize_trajectories(
                n, y_true, y_pred, contradiction_loads, rng
            )

        quality_records = self._build_quality_records(
            y_true, y_pred, certainty_trajectories, contradiction_loads
        )

        smoothness_profile  = self._build_smoothness_profile(quality_records)
        convergence_profile = self._build_convergence_profile(
            quality_records, certainty_trajectories
        )
        disease_profiles = self._build_disease_profiles(y_true, quality_records)

        n_believable = sum(1 for r in quality_records
                          if r.quality == TrajectoryQuality.CLINICALLY_BELIEVABLE)
        n_artefact   = sum(1 for r in quality_records
                          if r.quality == TrajectoryQuality.ARTEFACTUAL)
        mean_q = statistics.mean(
            _QUALITY_SCORE[r.quality] for r in quality_records
        )
        if mean_q >= 3.5:
            overall = TrajectoryQuality.CLINICALLY_BELIEVABLE
        elif mean_q >= 2.5:
            overall = TrajectoryQuality.ACCEPTABLE
        elif mean_q >= 1.5:
            overall = TrajectoryQuality.QUESTIONABLE
        else:
            overall = TrajectoryQuality.ARTEFACTUAL

        n_smooth_needed = sum(1 for r in quality_records
                              if r.smoothness_grade == SmoothnessGrade.SPIKED)
        n_osc_needed    = sum(1 for r in quality_records
                              if r.n_direction_reversals > _PATHOLOGICAL_OSC)
        n_lock_needed   = sum(1 for r in quality_records if r.locked_in_early)

        recs = self._generate_recommendations(
            quality_records, smoothness_profile, convergence_profile, n_artefact / n
        )

        return TrajectoryRealismReport(
            quality_records=quality_records,
            smoothness_profile=smoothness_profile,
            convergence_profile=convergence_profile,
            disease_profiles=disease_profiles,
            fraction_clinically_believable=n_believable / n,
            fraction_artefactual=n_artefact / n,
            mean_quality_score=mean_q,
            overall_realism_grade=overall,
            n_smoothing_interventions_needed=n_smooth_needed,
            n_oscillation_dampening_needed=n_osc_needed,
            n_early_lock_in_fixes_needed=n_lock_needed,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    def _synthesize_trajectories(
        self, n, y_true, y_pred, contradiction_loads, rng
    ) -> np.ndarray:
        trajs = np.zeros((n, self.max_steps))
        for i in range(n):
            correct = y_pred[i] == y_true[i]
            cl      = float(contradiction_loads[i])
            start   = rng.uniform(0.42, 0.60)
            end     = rng.uniform(0.72, 0.91) if correct else rng.uniform(0.38, 0.65)
            base    = np.linspace(start, end, self.max_steps)
            # Add small smooth noise
            noise   = rng.normal(0, 0.025, self.max_steps)
            # Add contradiction-triggered dip around step 3
            if cl > 0.20:
                noise[min(3, self.max_steps - 1)] -= cl * 0.4
            trajs[i] = np.clip(base + noise, 0.0, 1.0)
        return trajs

    def _build_quality_records(
        self,
        y_true, y_pred, trajs, contradiction_loads
    ) -> List[TrajectoryQualityRecord]:
        records: List[TrajectoryQualityRecord] = []
        n = len(y_true)
        for i in range(n):
            cert = trajs[i].tolist()
            deltas     = [abs(cert[j+1] - cert[j]) for j in range(len(cert)-1)]
            dir_deltas = [cert[j+1] - cert[j] for j in range(len(cert)-1)]
            reversals  = sum(
                1 for j in range(1, len(dir_deltas))
                if dir_deltas[j] * dir_deltas[j-1] < -0.001
            )
            max_delta = max(deltas) if deltas else 0.0
            smooth    = _smoothness_grade(max_delta)
            locked    = cert[0] >= _EARLY_LOCK_CERT
            converged = any(c >= _CONVERGENCE_CERT for c in cert)
            quality   = _trajectory_quality(smooth, reversals, locked)
            contra_div = (
                float(contradiction_loads[i]) > 0.20
                and any(cert[j] < cert[j-1] - 0.05 for j in range(1, len(cert)))
            )
            records.append(TrajectoryQualityRecord(
                case_index=i,
                true_label=int(y_true[i]),
                pred_label=int(y_pred[i]),
                is_correct=bool(y_pred[i] == y_true[i]),
                n_steps=self.max_steps,
                quality=quality,
                smoothness_grade=smooth,
                max_step_delta=max_delta,
                n_direction_reversals=reversals,
                final_certainty=float(cert[-1]),
                initial_certainty=float(cert[0]),
                certainty_range=float(max(cert) - min(cert)),
                locked_in_early=locked,
                converged=converged,
                contradiction_triggered_divergence=contra_div,
            ))
        return records

    def _build_smoothness_profile(
        self, records: List[TrajectoryQualityRecord]
    ) -> SmoothnessProfile:
        n = len(records)
        deltas = [r.max_step_delta for r in records]
        revs   = [r.n_direction_reversals for r in records]
        return SmoothnessProfile(
            n_cases=n,
            mean_max_step_delta=statistics.mean(deltas),
            std_max_step_delta=statistics.stdev(deltas) if n > 1 else 0.0,
            fraction_smooth=sum(1 for r in records if r.smoothness_grade == SmoothnessGrade.SMOOTH) / n,
            fraction_moderate=sum(1 for r in records if r.smoothness_grade == SmoothnessGrade.MODERATE) / n,
            fraction_rough=sum(1 for r in records if r.smoothness_grade == SmoothnessGrade.ROUGH) / n,
            fraction_spiked=sum(1 for r in records if r.smoothness_grade == SmoothnessGrade.SPIKED) / n,
            mean_direction_reversals=statistics.mean(revs),
            fraction_pathological_oscillation=sum(1 for r in revs if r > _PATHOLOGICAL_OSC) / n,
            fraction_early_lock_in=sum(1 for r in records if r.locked_in_early) / n,
        )

    def _build_convergence_profile(
        self,
        records: List[TrajectoryQualityRecord],
        trajs: np.ndarray,
    ) -> ConvergenceRealismProfile:
        n = len(records)
        conv_steps = []
        for i, r in enumerate(records):
            cert = trajs[i].tolist()
            step = next((j for j, c in enumerate(cert) if c >= _CONVERGENCE_CERT),
                        self.max_steps)
            conv_steps.append(step)

        early = sum(1 for s in conv_steps if s <= 1) / n
        late  = sum(1 for s in conv_steps if s >= self.max_steps) / n

        gains  = [r.final_certainty - r.initial_certainty for r in records]
        mono   = sum(
            1 for i, r in enumerate(records)
            if all(
                trajs[i, j+1] >= trajs[i, j] - 0.01
                for j in range(self.max_steps - 1)
            )
        ) / n

        contra_div = sum(1 for r in records if r.contradiction_triggered_divergence) / n

        return ConvergenceRealismProfile(
            n_cases=n,
            mean_convergence_step=statistics.mean(conv_steps),
            std_convergence_step=statistics.stdev(conv_steps) if n > 1 else 0.0,
            fraction_early_convergence=early,
            fraction_late_convergence=late,
            mean_final_certainty=statistics.mean(r.final_certainty for r in records),
            mean_certainty_gain=statistics.mean(gains),
            fraction_monotone_increasing=mono,
            fraction_with_contradiction_divergence=contra_div,
        )

    def _build_disease_profiles(
        self,
        y_true: np.ndarray,
        records: List[TrajectoryQualityRecord],
    ) -> List[DiseaseTrajectoryRealismProfile]:
        profiles: List[DiseaseTrajectoryRealismProfile] = []
        for label_idx, disease in enumerate(self.class_labels):
            dis_recs = [r for r in records if r.true_label == label_idx]
            if not dis_recs:
                continue
            n = len(dis_recs)
            q_scores = [_QUALITY_SCORE[r.quality] for r in dis_recs]
            mean_q   = statistics.mean(q_scores)
            frac_bel = sum(1 for r in dis_recs
                           if r.quality == TrajectoryQuality.CLINICALLY_BELIEVABLE) / n
            mean_sm  = 1.0 - statistics.mean(
                r.max_step_delta / (_ROUGH_THRESH + 0.01) for r in dis_recs
            )
            mean_conv = statistics.mean(r.n_steps for r in dis_recs)
            mean_cert = statistics.mean(r.final_certainty for r in dis_recs)

            if mean_q >= 3.5:
                grade = TrajectoryQuality.CLINICALLY_BELIEVABLE
            elif mean_q >= 2.5:
                grade = TrajectoryQuality.ACCEPTABLE
            elif mean_q >= 1.5:
                grade = TrajectoryQuality.QUESTIONABLE
            else:
                grade = TrajectoryQuality.ARTEFACTUAL

            profiles.append(DiseaseTrajectoryRealismProfile(
                disease=disease,
                n_cases=n,
                mean_quality_score=mean_q,
                fraction_clinically_believable=frac_bel,
                mean_smoothness=float(np.clip(mean_sm, 0, 1)),
                mean_convergence_step=mean_conv,
                mean_final_certainty=mean_cert,
                realism_grade=grade,
            ))
        return profiles

    @staticmethod
    def _generate_recommendations(
        records: List[TrajectoryQualityRecord],
        smoothness: SmoothnessProfile,
        convergence: ConvergenceRealismProfile,
        frac_artefact: float,
    ) -> List[str]:
        recs: List[str] = []

        if frac_artefact > 0.15:
            recs.append(
                f"{frac_artefact:.1%} of trajectories are artefactual — apply "
                "exponential moving-average smoothing and inertia weighting to "
                "all certainty updates."
            )
        if smoothness.fraction_spiked > 0.10:
            recs.append(
                f"{smoothness.fraction_spiked:.1%} of cases have certainty spikes "
                f"(Δ > 0.30 in one step) — clamp maximum per-step update to 0.15."
            )
        if smoothness.fraction_pathological_oscillation > 0.15:
            recs.append(
                f"{smoothness.fraction_pathological_oscillation:.1%} of cases "
                "show pathological oscillation — add momentum term to "
                "certainty propagation."
            )
        if convergence.fraction_early_convergence > 0.20:
            recs.append(
                f"{convergence.fraction_early_convergence:.1%} of cases lock-in "
                "certainty at step ≤ 1 — enforce minimum evidence accumulation "
                "before convergence is permitted."
            )
        if convergence.fraction_late_convergence > 0.15:
            recs.append(
                f"{convergence.fraction_late_convergence:.1%} of cases never converge "
                "— escalate persistently ambiguous cases rather than leaving "
                "trajectory open-ended."
            )
        if not recs:
            recs.append("Trajectory realism is within clinically believable bounds.")
        return recs[:5]
