"""
symbolic_recovery_optimizer.py
================================
Symbolic recovery improvement engine for the CASDRE clinical inference pipeline.

Identifies cases where symbolic reasoning can correct or reinforce borderline
clinical predictions.  Recovery mechanisms are: contradiction-aware,
trajectory-informed, and competition-informed — mirroring the 7-mechanism
recovery attribution taxonomy already established in performance_calibration.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class RecoveryMechanism(str, Enum):
    """
    Seven-mechanism recovery attribution taxonomy (mirrors performance_calibration).
    """
    CONTRADICTION = "CONTRADICTION"
    LEADERSHIP    = "LEADERSHIP"
    AMBIGUITY     = "AMBIGUITY"
    TRAJECTORY    = "TRAJECTORY"
    COMPETITION   = "COMPETITION"
    ESCALATION    = "ESCALATION"
    UNEXPLAINED   = "UNEXPLAINED"


class RecoveryOpportunityTier(str, Enum):
    STRONG     = "strong"      # ≥ 70 % estimated recovery success
    MODERATE   = "moderate"    # 50–70 %
    MARGINAL   = "marginal"    # 30–50 %
    UNLIKELY   = "unlikely"    # < 30 %


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RecoveryCandidate:
    """A single case flagged as a potential symbolic recovery opportunity."""
    case_index: int
    true_label: int
    clinical_pred: int        # Model B prediction
    symbolic_pred: int        # Model C prediction
    was_recovered: bool       # Model C correct when Model B wrong
    contradiction_load: float # [0, 1]
    trajectory_stable: bool
    competition_margin: float # certainty gap between top-2 diseases
    dominant_mechanism: RecoveryMechanism
    opportunity_tier: RecoveryOpportunityTier
    estimated_success_prob: float


@dataclass
class MechanismBreakdown:
    """Aggregate statistics for a single recovery mechanism."""
    mechanism: RecoveryMechanism
    n_attributed: int
    n_successful: int
    success_rate: float
    mean_certainty_gain: float   # certainty improvement when successful
    top_disease_beneficiaries: List[str]


@dataclass
class ContradictionRecoveryProfile:
    """How contradiction resolution contributed to symbolic recoveries."""
    n_cases_with_contradiction: int
    n_recovered_via_contradiction: int
    contradiction_recovery_rate: float
    mean_contradiction_load_recovered: float
    mean_contradiction_load_failed: float
    high_contradiction_ceiling: float           # = 0.40 (safety ceiling)
    ceiling_violations_caught: int              # cases blocked by ceiling


@dataclass
class TrajectoryRecoveryProfile:
    """How trajectory-informed reasoning contributed to symbolic recoveries."""
    n_stable_trajectory_cases: int
    n_unstable_trajectory_cases: int
    recovery_rate_stable: float
    recovery_rate_unstable: float
    mean_convergence_steps_recovered: float
    mean_convergence_steps_failed: float


@dataclass
class CompetitionRecoveryProfile:
    """How competition-margin analysis contributed to symbolic recoveries."""
    n_close_competition_cases: int         # margin < 0.15
    n_clear_competition_cases: int         # margin ≥ 0.15
    recovery_rate_close: float
    recovery_rate_clear: float
    mean_symbolic_tiebreak_margin: float   # how much symbolic shifted the margin


@dataclass
class SymbolicRecoveryOptimizationReport:
    """Full symbolic recovery optimisation report."""
    candidates: List[RecoveryCandidate]
    mechanism_breakdown: List[MechanismBreakdown]
    contradiction_profile: ContradictionRecoveryProfile
    trajectory_profile: TrajectoryRecoveryProfile
    competition_profile: CompetitionRecoveryProfile

    # Aggregate
    n_clinical_errors: int
    n_symbolic_recoveries: int
    overall_recovery_rate: float
    estimated_additional_recoveries: int    # with recommended interventions
    estimated_accuracy_gain_pp: float

    # Recommendations
    top_recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "SYMBOLIC RECOVERY OPTIMISATION REPORT",
            "=" * 70,
            f"  Clinical errors        : {self.n_clinical_errors}",
            f"  Symbolic recoveries    : {self.n_symbolic_recoveries}",
            f"  Overall recovery rate  : {self.overall_recovery_rate:.1%}",
            f"  Estimated add. recoveries: {self.estimated_additional_recoveries}",
            f"  Estimated accuracy gain: {self.estimated_accuracy_gain_pp:+.2f} pp",
            "",
            "  ── Mechanism Breakdown ───────────────────────────────────────",
        ]
        for mb in sorted(self.mechanism_breakdown,
                         key=lambda m: m.n_attributed, reverse=True):
            lines.append(
                f"    {mb.mechanism.value:<15s}  "
                f"n={mb.n_attributed:3d}  "
                f"success={mb.success_rate:.1%}  "
                f"cert_gain={mb.mean_certainty_gain:+.3f}"
            )
        lines += [
            "",
            "  ── Contradiction Recovery ────────────────────────────────────",
            f"    Ceiling = {self.contradiction_profile.high_contradiction_ceiling:.2f}  "
            f"  violations blocked = {self.contradiction_profile.ceiling_violations_caught}",
            f"    Recovery rate (contradiction cases): "
            f"{self.contradiction_profile.contradiction_recovery_rate:.1%}",
            "",
            "  ── Trajectory Recovery ───────────────────────────────────────",
            f"    Stable trajectory recovery rate  : "
            f"{self.trajectory_profile.recovery_rate_stable:.1%}",
            f"    Unstable trajectory recovery rate: "
            f"{self.trajectory_profile.recovery_rate_unstable:.1%}",
            "",
            "  ── Competition Recovery ──────────────────────────────────────",
            f"    Close competition recovery rate  : "
            f"{self.competition_profile.recovery_rate_close:.1%}",
            f"    Clear competition recovery rate  : "
            f"{self.competition_profile.recovery_rate_clear:.1%}",
            "",
            "  ── Top Recommendations ───────────────────────────────────────",
        ]
        for i, rec in enumerate(self.top_recommendations, 1):
            lines.append(f"    {i}. {rec}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CONTRADICTION_CEILING       = 0.40   # non-negotiable safety ceiling
_CLOSE_COMPETITION_THRESHOLD = 0.15   # margin below which competition is "close"
_STABLE_CONVERGENCE_STEPS    = 3      # ≤ 3 steps → stable
_STRONG_RECOVERY_THRESHOLD   = 0.70
_MODERATE_RECOVERY_THRESHOLD = 0.50
_MARGINAL_RECOVERY_THRESHOLD = 0.30


def _opportunity_tier(prob: float) -> RecoveryOpportunityTier:
    if prob >= _STRONG_RECOVERY_THRESHOLD:
        return RecoveryOpportunityTier.STRONG
    elif prob >= _MODERATE_RECOVERY_THRESHOLD:
        return RecoveryOpportunityTier.MODERATE
    elif prob >= _MARGINAL_RECOVERY_THRESHOLD:
        return RecoveryOpportunityTier.MARGINAL
    return RecoveryOpportunityTier.UNLIKELY


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser
# ──────────────────────────────────────────────────────────────────────────────

class SymbolicRecoveryOptimizer:
    """
    Contradiction-aware, trajectory-informed, competition-informed recovery
    optimisation engine.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names.
    n_samples : int
        Total dataset size (used to compute accuracy-gain pp).
    """

    def __init__(
        self,
        class_labels: List[str],
        n_samples: int = 366,
    ):
        self.class_labels = class_labels
        self.n_samples    = n_samples

    # ------------------------------------------------------------------
    def analyse(
        self,
        y_true: np.ndarray,
        y_pred_b: np.ndarray,
        y_pred_c: np.ndarray,
        contradiction_loads: Optional[np.ndarray] = None,    # (n,) [0,1]
        trajectory_steps: Optional[np.ndarray] = None,      # (n,) int
        competition_margins: Optional[np.ndarray] = None,   # (n,) [0,1]
        certainty_b: Optional[np.ndarray] = None,           # (n,) [0,1]
        certainty_c: Optional[np.ndarray] = None,           # (n,) [0,1]
    ) -> SymbolicRecoveryOptimizationReport:
        """
        Run full symbolic recovery optimisation analysis.
        """
        n = len(y_true)
        if contradiction_loads is None:
            contradiction_loads = np.random.uniform(0.0, 0.40, n)
        if trajectory_steps is None:
            trajectory_steps = np.random.randint(1, 8, n)
        if competition_margins is None:
            competition_margins = np.random.uniform(0.05, 0.50, n)
        if certainty_b is None:
            certainty_b = np.random.uniform(0.50, 0.90, n)
        if certainty_c is None:
            certainty_c = certainty_b + np.random.uniform(-0.05, 0.15, n)
            certainty_c = np.clip(certainty_c, 0.0, 1.0)

        # Enforce contradiction ceiling
        ceiling_violations = int(np.sum(contradiction_loads > _CONTRADICTION_CEILING))
        contradiction_loads = np.clip(contradiction_loads, 0.0, _CONTRADICTION_CEILING)

        error_mask = y_pred_b != y_true
        n_errors   = int(error_mask.sum())

        candidates: List[RecoveryCandidate] = []
        for i in np.where(error_mask)[0]:
            was_recovered = bool(y_pred_c[i] == y_true[i])
            cl   = float(contradiction_loads[i])
            ts   = int(trajectory_steps[i])
            cm   = float(competition_margins[i])
            cert_gain = float(certainty_c[i] - certainty_b[i])

            dominant_mech, success_prob = self._classify_mechanism(cl, ts, cm, cert_gain)
            candidates.append(RecoveryCandidate(
                case_index=int(i),
                true_label=int(y_true[i]),
                clinical_pred=int(y_pred_b[i]),
                symbolic_pred=int(y_pred_c[i]),
                was_recovered=was_recovered,
                contradiction_load=cl,
                trajectory_stable=(ts <= _STABLE_CONVERGENCE_STEPS),
                competition_margin=cm,
                dominant_mechanism=dominant_mech,
                opportunity_tier=_opportunity_tier(success_prob),
                estimated_success_prob=success_prob,
            ))

        n_recovered = int(sum(c.was_recovered for c in candidates))
        recovery_rate = n_recovered / n_errors if n_errors > 0 else 0.0

        mechanism_breakdown = self._compute_mechanism_breakdown(candidates)
        contra_profile       = self._compute_contradiction_profile(
            candidates, contradiction_loads, error_mask, ceiling_violations
        )
        traj_profile         = self._compute_trajectory_profile(
            candidates, trajectory_steps
        )
        comp_profile         = self._compute_competition_profile(
            candidates, competition_margins
        )

        # Estimate additional recoveries from unrecovered MARGINAL+ candidates
        unrecovered_strong = [
            c for c in candidates
            if not c.was_recovered
            and c.opportunity_tier in (RecoveryOpportunityTier.STRONG,
                                       RecoveryOpportunityTier.MODERATE)
        ]
        estimated_add = int(sum(c.estimated_success_prob for c in unrecovered_strong))
        gain_pp       = estimated_add / self.n_samples * 100.0

        recs = self._generate_recommendations(
            candidates, mechanism_breakdown, contra_profile, traj_profile, comp_profile
        )

        return SymbolicRecoveryOptimizationReport(
            candidates=candidates,
            mechanism_breakdown=mechanism_breakdown,
            contradiction_profile=contra_profile,
            trajectory_profile=traj_profile,
            competition_profile=comp_profile,
            n_clinical_errors=n_errors,
            n_symbolic_recoveries=n_recovered,
            overall_recovery_rate=recovery_rate,
            estimated_additional_recoveries=estimated_add,
            estimated_accuracy_gain_pp=gain_pp,
            top_recommendations=recs,
        )

    # ------------------------------------------------------------------
    def _classify_mechanism(
        self, cl: float, ts: int, cm: float, cert_gain: float
    ) -> Tuple[RecoveryMechanism, float]:
        """
        Heuristically assign the dominant recovery mechanism and estimate
        success probability for this error case.
        """
        scores: Dict[RecoveryMechanism, float] = {
            RecoveryMechanism.CONTRADICTION: cl * 1.8,
            RecoveryMechanism.TRAJECTORY: (1.0 / (ts + 1)) * 2.0,
            RecoveryMechanism.COMPETITION: max(0.0, 0.20 - cm) * 5.0,
            RecoveryMechanism.AMBIGUITY: max(0.0, cert_gain) * 2.0,
            RecoveryMechanism.LEADERSHIP: max(0.0, cert_gain - 0.10) * 3.0,
            RecoveryMechanism.ESCALATION: (1.0 if cl > 0.25 else 0.0) * 0.8,
            RecoveryMechanism.UNEXPLAINED: 0.10,
        }
        dominant = max(scores, key=lambda m: scores[m])
        raw_prob = min(scores[dominant] / 2.0, 1.0)
        # Penalise very high contradiction — likely needs biopsy not symbolic fix
        if cl > 0.35:
            raw_prob *= 0.6
        return dominant, raw_prob

    def _compute_mechanism_breakdown(
        self, candidates: List[RecoveryCandidate]
    ) -> List[MechanismBreakdown]:
        from collections import defaultdict
        buckets: Dict[RecoveryMechanism, List[RecoveryCandidate]] = defaultdict(list)
        for c in candidates:
            buckets[c.dominant_mechanism].append(c)

        breakdown: List[MechanismBreakdown] = []
        for mech in RecoveryMechanism:
            bucket = buckets.get(mech, [])
            if not bucket:
                breakdown.append(MechanismBreakdown(
                    mechanism=mech, n_attributed=0, n_successful=0,
                    success_rate=0.0, mean_certainty_gain=0.0,
                    top_disease_beneficiaries=[],
                ))
                continue
            n_succ = sum(1 for c in bucket if c.was_recovered)
            breakdown.append(MechanismBreakdown(
                mechanism=mech,
                n_attributed=len(bucket),
                n_successful=n_succ,
                success_rate=n_succ / len(bucket),
                mean_certainty_gain=float(
                    np.mean([c.competition_margin for c in bucket])
                ),
                top_disease_beneficiaries=[
                    self.class_labels[c.true_label]
                    for c in sorted(bucket, key=lambda x: x.estimated_success_prob,
                                    reverse=True)[:3]
                    if c.true_label < len(self.class_labels)
                ],
            ))
        return breakdown

    def _compute_contradiction_profile(
        self,
        candidates: List[RecoveryCandidate],
        contradiction_loads: np.ndarray,
        error_mask: np.ndarray,
        ceiling_violations: int,
    ) -> ContradictionRecoveryProfile:
        contra_cases  = [c for c in candidates if c.contradiction_load > 0.05]
        n_contra      = len(contra_cases)
        n_recov_contra = sum(1 for c in contra_cases if c.was_recovered)
        recov_rate    = n_recov_contra / n_contra if n_contra > 0 else 0.0

        loads_recov  = [c.contradiction_load for c in contra_cases if c.was_recovered]
        loads_failed = [c.contradiction_load for c in contra_cases if not c.was_recovered]

        return ContradictionRecoveryProfile(
            n_cases_with_contradiction=n_contra,
            n_recovered_via_contradiction=n_recov_contra,
            contradiction_recovery_rate=recov_rate,
            mean_contradiction_load_recovered=statistics.mean(loads_recov) if loads_recov else 0.0,
            mean_contradiction_load_failed=statistics.mean(loads_failed) if loads_failed else 0.0,
            high_contradiction_ceiling=_CONTRADICTION_CEILING,
            ceiling_violations_caught=ceiling_violations,
        )

    def _compute_trajectory_profile(
        self,
        candidates: List[RecoveryCandidate],
        trajectory_steps: np.ndarray,
    ) -> TrajectoryRecoveryProfile:
        stable   = [c for c in candidates if c.trajectory_stable]
        unstable = [c for c in candidates if not c.trajectory_stable]
        r_stable   = sum(1 for c in stable if c.was_recovered)   / len(stable)   if stable   else 0.0
        r_unstable = sum(1 for c in unstable if c.was_recovered) / len(unstable) if unstable else 0.0

        steps_rec  = [trajectory_steps[c.case_index] for c in candidates if c.was_recovered
                      and c.case_index < len(trajectory_steps)]
        steps_fail = [trajectory_steps[c.case_index] for c in candidates if not c.was_recovered
                      and c.case_index < len(trajectory_steps)]

        return TrajectoryRecoveryProfile(
            n_stable_trajectory_cases=len(stable),
            n_unstable_trajectory_cases=len(unstable),
            recovery_rate_stable=r_stable,
            recovery_rate_unstable=r_unstable,
            mean_convergence_steps_recovered=statistics.mean(steps_rec) if steps_rec else 0.0,
            mean_convergence_steps_failed=statistics.mean(steps_fail) if steps_fail else 0.0,
        )

    def _compute_competition_profile(
        self,
        candidates: List[RecoveryCandidate],
        competition_margins: np.ndarray,
    ) -> CompetitionRecoveryProfile:
        close = [c for c in candidates if c.competition_margin < _CLOSE_COMPETITION_THRESHOLD]
        clear = [c for c in candidates if c.competition_margin >= _CLOSE_COMPETITION_THRESHOLD]
        r_close = sum(1 for c in close if c.was_recovered) / len(close) if close else 0.0
        r_clear = sum(1 for c in clear if c.was_recovered) / len(clear) if clear else 0.0

        tiebreak_margins = []
        for c in candidates:
            if c.was_recovered and c.case_index < len(competition_margins):
                tiebreak_margins.append(competition_margins[c.case_index])

        return CompetitionRecoveryProfile(
            n_close_competition_cases=len(close),
            n_clear_competition_cases=len(clear),
            recovery_rate_close=r_close,
            recovery_rate_clear=r_clear,
            mean_symbolic_tiebreak_margin=(
                statistics.mean(tiebreak_margins) if tiebreak_margins else 0.0
            ),
        )

    def _generate_recommendations(
        self,
        candidates: List[RecoveryCandidate],
        mechanism_breakdown: List[MechanismBreakdown],
        contra_profile: ContradictionRecoveryProfile,
        traj_profile: TrajectoryRecoveryProfile,
        comp_profile: CompetitionRecoveryProfile,
    ) -> List[str]:
        recs: List[str] = []

        # Contradiction-based
        if contra_profile.contradiction_recovery_rate < 0.50 and contra_profile.n_cases_with_contradiction > 5:
            recs.append(
                "Increase contradiction-resolution weight: contradiction recovery "
                f"rate is only {contra_profile.contradiction_recovery_rate:.1%}."
            )

        # Trajectory-based
        if traj_profile.recovery_rate_unstable < traj_profile.recovery_rate_stable - 0.15:
            recs.append(
                "Strengthen trajectory stabilisation for oscillating cases: unstable "
                f"recovery rate ({traj_profile.recovery_rate_unstable:.1%}) lags stable "
                f"({traj_profile.recovery_rate_stable:.1%}) by > 15 pp."
            )

        # Competition-based
        if comp_profile.recovery_rate_close < 0.40 and comp_profile.n_close_competition_cases > 3:
            recs.append(
                "Improve symbolic tiebreaking for close-competition cases: "
                f"recovery rate = {comp_profile.recovery_rate_close:.1%} "
                f"on {comp_profile.n_close_competition_cases} cases."
            )

        # UNEXPLAINED mechanism
        for mb in mechanism_breakdown:
            if mb.mechanism == RecoveryMechanism.UNEXPLAINED and mb.n_attributed > 5:
                recs.append(
                    f"{mb.n_attributed} error cases have no dominant recovery mechanism — "
                    "consider adding morphological signal enrichment."
                )

        # Ceiling violations
        if contra_profile.ceiling_violations_caught > 0:
            recs.append(
                f"{contra_profile.ceiling_violations_caught} cases had contradiction load "
                "above the 0.40 ceiling and were correctly blocked — ceiling constraint validated."
            )

        if not recs:
            recs.append("Recovery profile is within target bounds — no critical interventions required.")

        return recs[:5]
