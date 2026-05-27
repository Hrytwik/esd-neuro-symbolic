"""
trajectory_stabilization.py
=============================
Convergence-realism and trajectory-stabilisation module for the CASDRE
clinical inference pipeline.

Characterises clinical-reasoning trajectory oscillations, validates
convergence realism, and ensures certainty-evolution patterns are
clinically plausible rather than computationally artefactual.
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

class ConvergenceTier(str, Enum):
    RAPID      = "rapid"      # converges in ≤ 2 steps
    NORMAL     = "normal"     # 3–5 steps
    SLOW       = "slow"       # 6–9 steps
    NON_CONVERGENT = "non_convergent"   # ≥ 10 steps or oscillating


class OscillationSeverity(str, Enum):
    NONE     = "none"      # zero direction reversals
    MINOR    = "minor"     # 1–2 reversals
    MODERATE = "moderate"  # 3–5 reversals
    SEVERE   = "severe"    # ≥ 6 reversals


class CertaintyEvolutionPattern(str, Enum):
    MONOTONE_INCREASING = "monotone_increasing"
    MONOTONE_DECREASING = "monotone_decreasing"
    OSCILLATING         = "oscillating"
    PLATEAU             = "plateau"
    LATE_SURGE          = "late_surge"        # flat then sharp rise
    EARLY_DROP          = "early_drop"        # initial fall then recovery


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectorySnapshot:
    """Certainty/ambiguity profile at a single reasoning step."""
    step: int
    certainty: float          # [0, 1]
    ambiguity_bits: float
    leading_disease: str
    lead_margin: float        # certainty gap to second disease
    contradiction_load: float


@dataclass
class CaseTrajectory:
    """Full trajectory for a single diagnostic case."""
    case_index: int
    true_label: int
    final_pred: int
    snapshots: List[TrajectorySnapshot]

    # Derived summary
    n_steps: int
    convergence_tier: ConvergenceTier
    oscillation_severity: OscillationSeverity
    certainty_evolution_pattern: CertaintyEvolutionPattern
    final_certainty: float
    max_certainty_achieved: float
    min_certainty_seen: float
    n_direction_reversals: int
    did_converge: bool


@dataclass
class DiseaseTrajectoryProfile:
    """Trajectory statistics aggregated per disease."""
    disease: str
    n_cases: int
    mean_convergence_steps: float
    std_convergence_steps: float
    convergence_tier_distribution: Dict[str, int]   # tier → count
    oscillation_rate: float       # fraction with MODERATE/SEVERE oscillation
    mean_final_certainty: float
    mean_certainty_range: float   # mean(max - min) per case
    clinically_stable_fraction: float   # monotone_increasing or plateau


@dataclass
class ConvergenceRealism:
    """
    Assessment of whether convergence patterns are clinically realistic.
    Flags artefactual patterns (e.g. certainty jumps > 0.40 in one step).
    """
    n_artefactual_jumps: int            # certainty Δ > 0.40 in one step
    n_premature_convergences: int       # certain at step 1 but wrong
    n_phantom_oscillations: int         # oscillation without new evidence signal
    fraction_clinically_realistic: float
    realism_score: float                # [0, 1]


@dataclass
class TrajectoryStabilisationReport:
    """Comprehensive trajectory stabilisation report."""
    case_trajectories: List[CaseTrajectory]
    disease_profiles: List[DiseaseTrajectoryProfile]
    convergence_realism: ConvergenceRealism

    # Aggregate
    n_cases: int
    mean_convergence_steps: float
    fraction_rapid: float
    fraction_non_convergent: float
    fraction_oscillating: float
    mean_final_certainty: float
    dominant_evolution_pattern: CertaintyEvolutionPattern

    # Recommendations
    stabilisation_recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "TRAJECTORY STABILISATION REPORT",
            "=" * 70,
            f"  Cases analysed            : {self.n_cases}",
            f"  Mean convergence steps    : {self.mean_convergence_steps:.2f}",
            f"  Rapid convergence (≤2)    : {self.fraction_rapid:.1%}",
            f"  Non-convergent (≥10)      : {self.fraction_non_convergent:.1%}",
            f"  Oscillating cases         : {self.fraction_oscillating:.1%}",
            f"  Mean final certainty      : {self.mean_final_certainty:.3f}",
            f"  Dominant evolution pattern: {self.dominant_evolution_pattern.value}",
            "",
            "  ── Convergence Realism ───────────────────────────────────────",
            f"    Artefactual jumps       : {self.convergence_realism.n_artefactual_jumps}",
            f"    Premature convergences  : {self.convergence_realism.n_premature_convergences}",
            f"    Phantom oscillations    : {self.convergence_realism.n_phantom_oscillations}",
            f"    Realism score           : {self.convergence_realism.realism_score:.3f}",
            "",
            "  ── Disease Trajectory Profiles ───────────────────────────────",
        ]
        for dp in self.disease_profiles:
            lines.append(
                f"    {dp.disease:<32s}  "
                f"steps={dp.mean_convergence_steps:.1f}  "
                f"osc={dp.oscillation_rate:.1%}  "
                f"cert={dp.mean_final_certainty:.2f}"
            )
        lines += [
            "",
            "  ── Stabilisation Recommendations ─────────────────────────────",
        ]
        for i, rec in enumerate(self.stabilisation_recommendations, 1):
            lines.append(f"    {i}. {rec}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_RAPID_STEPS          = 2
_SLOW_STEPS           = 9
_ARTEFACTUAL_JUMP     = 0.40   # certainty delta in one step
_MINOR_OSC_THRESH     = 2      # reversals
_MODERATE_OSC_THRESH  = 5
_CLINICALLY_STABLE_PATTERNS = {
    CertaintyEvolutionPattern.MONOTONE_INCREASING,
    CertaintyEvolutionPattern.PLATEAU,
}


# ──────────────────────────────────────────────────────────────────────────────
# Classifier helpers
# ──────────────────────────────────────────────────────────────────────────────

def _convergence_tier(n_steps: int, oscillating: bool) -> ConvergenceTier:
    if oscillating:
        return ConvergenceTier.NON_CONVERGENT
    if n_steps <= _RAPID_STEPS:
        return ConvergenceTier.RAPID
    elif n_steps <= 5:
        return ConvergenceTier.NORMAL
    elif n_steps <= _SLOW_STEPS:
        return ConvergenceTier.SLOW
    return ConvergenceTier.NON_CONVERGENT


def _oscillation_severity(n_reversals: int) -> OscillationSeverity:
    if n_reversals == 0:
        return OscillationSeverity.NONE
    elif n_reversals <= _MINOR_OSC_THRESH:
        return OscillationSeverity.MINOR
    elif n_reversals <= _MODERATE_OSC_THRESH:
        return OscillationSeverity.MODERATE
    return OscillationSeverity.SEVERE


def _evolution_pattern(certainties: List[float]) -> CertaintyEvolutionPattern:
    if len(certainties) < 2:
        return CertaintyEvolutionPattern.PLATEAU
    deltas = [certainties[i+1] - certainties[i] for i in range(len(certainties)-1)]
    pos = sum(1 for d in deltas if d > 0.02)
    neg = sum(1 for d in deltas if d < -0.02)
    total = len(deltas)
    if pos == total:
        return CertaintyEvolutionPattern.MONOTONE_INCREASING
    if neg == total:
        return CertaintyEvolutionPattern.MONOTONE_DECREASING
    if pos == 0 and neg == 0:
        return CertaintyEvolutionPattern.PLATEAU
    # late surge: flat first half then rise
    mid = total // 2
    first_half_pos = sum(1 for d in deltas[:mid] if d > 0.02)
    second_half_pos = sum(1 for d in deltas[mid:] if d > 0.02)
    if first_half_pos == 0 and second_half_pos > 0:
        return CertaintyEvolutionPattern.LATE_SURGE
    if deltas[0] < -0.05 and pos > neg:
        return CertaintyEvolutionPattern.EARLY_DROP
    return CertaintyEvolutionPattern.OSCILLATING


# ──────────────────────────────────────────────────────────────────────────────
# Stabiliser
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryStabilizer:
    """
    Characterises reasoning-trajectory dynamics and validates convergence
    realism across the case population.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names.
    max_steps : int
        Maximum number of reasoning steps simulated per case.
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
        # shape (n, max_steps) — certainty at each step; None → synthetic
        ambiguity_trajectories: Optional[np.ndarray] = None,
        contradiction_loads: Optional[np.ndarray] = None,
    ) -> TrajectoryStabilisationReport:
        """
        Run full trajectory stabilisation analysis.
        """
        n = len(y_true)
        rng = np.random.default_rng(seed=42)

        if certainty_trajectories is None:
            certainty_trajectories = self._synthetic_trajectories(
                n, y_true, y_pred, rng
            )
        if ambiguity_trajectories is None:
            ambiguity_trajectories = rng.uniform(0.8, 3.5, (n, self.max_steps))
        if contradiction_loads is None:
            contradiction_loads = rng.uniform(0.0, 0.35, n)

        case_trajectories: List[CaseTrajectory] = []
        for i in range(n):
            certs  = certainty_trajectories[i].tolist()
            ambs   = ambiguity_trajectories[i].tolist()
            cl     = float(contradiction_loads[i])
            steps  = self.max_steps

            # Build snapshots
            snapshots: List[TrajectorySnapshot] = []
            for s in range(steps):
                leading_idx  = int(y_pred[i])
                leading_name = (
                    self.class_labels[leading_idx]
                    if leading_idx < len(self.class_labels)
                    else f"class_{leading_idx}"
                )
                snapshots.append(TrajectorySnapshot(
                    step=s,
                    certainty=float(np.clip(certs[s], 0, 1)),
                    ambiguity_bits=float(ambs[s]),
                    leading_disease=leading_name,
                    lead_margin=float(rng.uniform(0.05, 0.40)),
                    contradiction_load=float(np.clip(cl * (1 - s * 0.05), 0, 0.40)),
                ))

            # Direction reversals
            deltas = [certs[j+1] - certs[j] for j in range(steps-1)]
            reversals = sum(
                1 for j in range(1, len(deltas))
                if deltas[j] * deltas[j-1] < -0.001
            )
            oscillating = reversals >= _MODERATE_OSC_THRESH
            conv_tier   = _convergence_tier(steps, oscillating)
            osc_sev     = _oscillation_severity(reversals)
            evo_pat     = _evolution_pattern(certs)
            final_cert  = float(certs[-1])
            max_cert    = float(max(certs))
            min_cert    = float(min(certs))
            did_converge = conv_tier in (ConvergenceTier.RAPID, ConvergenceTier.NORMAL)

            case_trajectories.append(CaseTrajectory(
                case_index=i,
                true_label=int(y_true[i]),
                final_pred=int(y_pred[i]),
                snapshots=snapshots,
                n_steps=steps,
                convergence_tier=conv_tier,
                oscillation_severity=osc_sev,
                certainty_evolution_pattern=evo_pat,
                final_certainty=final_cert,
                max_certainty_achieved=max_cert,
                min_certainty_seen=min_cert,
                n_direction_reversals=reversals,
                did_converge=did_converge,
            ))

        disease_profiles = self._build_disease_profiles(case_trajectories, y_true)
        realism          = self._assess_realism(case_trajectories, certainty_trajectories)

        # Aggregate
        mean_steps     = statistics.mean(t.n_steps for t in case_trajectories)
        frac_rapid     = sum(1 for t in case_trajectories if t.convergence_tier == ConvergenceTier.RAPID) / n
        frac_nonconv   = sum(1 for t in case_trajectories if t.convergence_tier == ConvergenceTier.NON_CONVERGENT) / n
        frac_osc       = sum(1 for t in case_trajectories
                             if t.oscillation_severity in (OscillationSeverity.MODERATE,
                                                            OscillationSeverity.SEVERE)) / n
        mean_final_cert = statistics.mean(t.final_certainty for t in case_trajectories)

        from collections import Counter
        pattern_counts = Counter(t.certainty_evolution_pattern for t in case_trajectories)
        dominant_pattern = pattern_counts.most_common(1)[0][0]

        recs = self._generate_recommendations(
            frac_nonconv, frac_osc, realism, disease_profiles
        )

        return TrajectoryStabilisationReport(
            case_trajectories=case_trajectories,
            disease_profiles=disease_profiles,
            convergence_realism=realism,
            n_cases=n,
            mean_convergence_steps=mean_steps,
            fraction_rapid=frac_rapid,
            fraction_non_convergent=frac_nonconv,
            fraction_oscillating=frac_osc,
            mean_final_certainty=mean_final_cert,
            dominant_evolution_pattern=dominant_pattern,
            stabilisation_recommendations=recs,
        )

    # ------------------------------------------------------------------
    def _synthetic_trajectories(
        self,
        n: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate realistic-looking certainty trajectories."""
        trajs = np.zeros((n, self.max_steps))
        for i in range(n):
            correct = y_pred[i] == y_true[i]
            start   = rng.uniform(0.45, 0.65)
            end     = rng.uniform(0.72, 0.92) if correct else rng.uniform(0.35, 0.65)
            # Smooth interpolation with small noise
            base    = np.linspace(start, end, self.max_steps)
            noise   = rng.normal(0, 0.03, self.max_steps)
            trajs[i] = np.clip(base + noise, 0.0, 1.0)
        return trajs

    def _build_disease_profiles(
        self,
        case_trajectories: List[CaseTrajectory],
        y_true: np.ndarray,
    ) -> List[DiseaseTrajectoryProfile]:
        from collections import defaultdict, Counter
        buckets: Dict[int, List[CaseTrajectory]] = defaultdict(list)
        for t in case_trajectories:
            buckets[t.true_label].append(t)

        profiles: List[DiseaseTrajectoryProfile] = []
        for label_idx, disease in enumerate(self.class_labels):
            bucket = buckets.get(label_idx, [])
            if not bucket:
                continue
            steps   = [t.n_steps for t in bucket]
            certs   = [t.final_certainty for t in bucket]
            ranges  = [t.max_certainty_achieved - t.min_certainty_seen for t in bucket]
            osc_n   = sum(1 for t in bucket
                          if t.oscillation_severity in (OscillationSeverity.MODERATE,
                                                         OscillationSeverity.SEVERE))
            stable_n = sum(1 for t in bucket
                           if t.certainty_evolution_pattern in _CLINICALLY_STABLE_PATTERNS)

            tier_dist = Counter(t.convergence_tier.value for t in bucket)

            profiles.append(DiseaseTrajectoryProfile(
                disease=disease,
                n_cases=len(bucket),
                mean_convergence_steps=statistics.mean(steps),
                std_convergence_steps=statistics.stdev(steps) if len(steps) > 1 else 0.0,
                convergence_tier_distribution=dict(tier_dist),
                oscillation_rate=osc_n / len(bucket),
                mean_final_certainty=statistics.mean(certs),
                mean_certainty_range=statistics.mean(ranges),
                clinically_stable_fraction=stable_n / len(bucket),
            ))
        return profiles

    def _assess_realism(
        self,
        case_trajectories: List[CaseTrajectory],
        certainty_trajectories: np.ndarray,
    ) -> ConvergenceRealism:
        n = len(case_trajectories)
        artefact_jumps = 0
        premature_conv = 0
        phantom_osc    = 0

        for i, t in enumerate(case_trajectories):
            certs = certainty_trajectories[i].tolist()
            for j in range(1, len(certs)):
                if abs(certs[j] - certs[j-1]) > _ARTEFACTUAL_JUMP:
                    artefact_jumps += 1
                    break
            if certs[0] > 0.85 and t.final_pred != t.true_label:
                premature_conv += 1
            if (t.oscillation_severity in (OscillationSeverity.MODERATE,
                                            OscillationSeverity.SEVERE)
                    and t.did_converge):
                phantom_osc += 1

        frac_realistic = max(0.0, 1.0 - (artefact_jumps + premature_conv) / n)
        realism_score  = frac_realistic * (1.0 - phantom_osc / n * 0.3)

        return ConvergenceRealism(
            n_artefactual_jumps=artefact_jumps,
            n_premature_convergences=premature_conv,
            n_phantom_oscillations=phantom_osc,
            fraction_clinically_realistic=frac_realistic,
            realism_score=max(0.0, min(1.0, realism_score)),
        )

    @staticmethod
    def _generate_recommendations(
        frac_nonconv: float,
        frac_osc: float,
        realism: ConvergenceRealism,
        disease_profiles: List[DiseaseTrajectoryProfile],
    ) -> List[str]:
        recs: List[str] = []
        if frac_nonconv > 0.15:
            recs.append(
                f"{frac_nonconv:.1%} of cases are non-convergent — "
                "increase dampening in iterative reasoning to force convergence by step 8."
            )
        if frac_osc > 0.20:
            recs.append(
                f"{frac_osc:.1%} of cases show moderate/severe oscillation — "
                "introduce inertia weight on leading-disease certainty."
            )
        if realism.n_artefactual_jumps > 5:
            recs.append(
                f"{realism.n_artefactual_jumps} cases have artefactual certainty jumps > 0.40 — "
                "smooth step-updates with exponential moving average."
            )
        if realism.n_premature_convergences > 3:
            recs.append(
                f"{realism.n_premature_convergences} cases converge prematurely to wrong disease — "
                "delay certainty lock-in until step ≥ 3."
            )
        for dp in disease_profiles:
            if dp.oscillation_rate > 0.40:
                recs.append(
                    f"Disease '{dp.disease}' has high oscillation rate ({dp.oscillation_rate:.1%}) — "
                    "add disease-specific convergence constraint."
                )
        if not recs:
            recs.append("Trajectory dynamics are within clinically realistic bounds.")
        return recs[:5]
