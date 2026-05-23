"""
TrajectoryEvaluator — reasoning trajectory stability and convergence analysis.

Measures the qualitative dynamics of the symbolic diagnostic reasoning process
as it evolves across pipeline stages for each patient. The trajectory is not
just the final decision — it is the PATH to that decision.

Key trajectory metrics
-----------------------
  convergence_index:
    Final certainty / peak certainty [0, 1].
    1.0 = certainty stayed at its maximum → clean, converging reasoning.
    < 0.70 = significant post-peak decay → unstable or oscillating reasoning.

  oscillation_count:
    Number of certainty direction reversals.
    0 = monotone convergence.
    ≥ 2 = clinically meaningful instability.

  stabilisation_stage:
    Stage at which certainty first met stability criteria
    (certainty ≥ 0.55 AND gap ≥ 0.20 simultaneously).
    -1 = never stabilised.

  leadership_changes:
    How many times the leading disease switched.
    0 = stable differential from first stage.
    ≥ 1 = evidence re-weighting caused hypothesis shift.

Clinical interpretation
-----------------------
Diseases with strong pathognomonic clinical markers (e.g. psoriasis:
koebner + knee/elbow involvement + scaling) should show:
  · Fast convergence (early stabilisation_stage)
  · Low oscillation counts
  · Leadership stable from stage 1

Diseases in confusion zones (e.g. psoriasis ↔ PRP, LP ↔ psoriasis) should:
  · Show higher oscillation counts
  · Converge later (higher stabilisation_stage or never)
  · Exhibit leadership changes before settlement

Usage
-----
  from src.evaluation_pipeline.trajectory_evaluator import TrajectoryEvaluator

  result = TrajectoryEvaluator.evaluate_vectors(test_vectors)
  print(result.summary())
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Per-patient trajectory profile ────────────────────────────────────────────

@dataclass(frozen=True)
class TrajectoryProfile:
    """
    Trajectory dynamics for a single patient.

    Attributes
    ----------
    patient_id:
        Source patient identifier.
    disease_label:
        Ground-truth disease label.
    trajectory_length:
        Number of pipeline stages (snapshots).
    convergence_index:
        Final / peak certainty ratio [0, 1].
    oscillation_count:
        Number of certainty direction reversals.
    peak_certainty:
        Maximum certainty observed across the trajectory.
    final_certainty:
        Certainty at the terminal stage.
    certainty_delta_total:
        Change from first to final stage (final − initial).
    was_dampened:
        True if contradiction dampening suppressed certainty at any stage.
    leadership_changed:
        True if the leading disease changed during the trajectory.
    leadership_changes_count:
        Number of leadership transitions.
    stabilisation_stage:
        Stage index where certainty first stabilised (-1 if never).
    recommendation:
        Terminal triage recommendation.
    """

    patient_id:               str
    disease_label:            str
    trajectory_length:        int
    convergence_index:        float
    oscillation_count:        int
    peak_certainty:           float
    final_certainty:          float
    certainty_delta_total:    float
    was_dampened:             bool
    leadership_changed:       bool
    leadership_changes_count: int
    stabilisation_stage:      int
    recommendation:           str

    @property
    def is_convergent(self) -> bool:
        """True if convergence_index ≥ 0.80 (strong final certainty vs peak)."""
        return self.convergence_index >= 0.80

    @property
    def is_stable(self) -> bool:
        """True if reasoning was stable (no oscillation, no leadership changes)."""
        return self.oscillation_count == 0 and not self.leadership_changed

    @property
    def is_oscillating(self) -> bool:
        """True if certainty reversed direction ≥ 2 times."""
        return self.oscillation_count >= 2

    @property
    def stabilised_early(self) -> bool:
        """True if certainty stabilised within the first 3 stages."""
        return 0 <= self.stabilisation_stage <= 3


# ── Trajectory evaluation result ─────────────────────────────────────────────

@dataclass
class TrajectoryEvaluationResult:
    """
    Dataset-level trajectory stability and convergence analysis.

    Attributes
    ----------
    total_cases:
        Total evaluated patients.
    mean_convergence_index:
        Mean convergence_index across all cases.
    std_convergence_index:
        Standard deviation of convergence_index.
    mean_oscillation_count:
        Mean number of oscillations per case.
    oscillating_case_count:
        Cases with ≥ 2 oscillations.
    stable_case_count:
        Cases with no oscillations and no leadership changes.
    convergent_case_count:
        Cases with convergence_index ≥ 0.80.
    dampened_case_count:
        Cases where certainty dampening was active.
    leadership_changed_count:
        Cases where the leading disease changed at least once.
    mean_peak_certainty:
        Mean peak certainty across all cases.
    mean_final_certainty:
        Mean final certainty across all cases.
    mean_certainty_delta:
        Mean certainty change from first to final stage.
    stabilised_count:
        Cases where certainty stabilised (stabilisation_stage ≥ 0).
    mean_stabilisation_stage:
        Mean stabilisation stage index among stabilised cases.
    per_disease_convergence_index:
        Mean convergence_index per disease.
    per_disease_oscillation_count:
        Mean oscillation count per disease.
    per_disease_stable_fraction:
        Fraction of stable trajectories per disease.
    per_disease_leadership_change_rate:
        Fraction of trajectories with leadership change per disease.
    profiles:
        All per-patient TrajectoryProfile instances.
    """

    total_cases:                        int
    mean_convergence_index:             float = 0.0
    std_convergence_index:              float = 0.0
    mean_oscillation_count:             float = 0.0
    oscillating_case_count:             int   = 0
    stable_case_count:                  int   = 0
    convergent_case_count:              int   = 0
    dampened_case_count:                int   = 0
    leadership_changed_count:           int   = 0
    mean_peak_certainty:                float = 0.0
    mean_final_certainty:               float = 0.0
    mean_certainty_delta:               float = 0.0
    stabilised_count:                   int   = 0
    mean_stabilisation_stage:           float = 0.0
    per_disease_convergence_index:      dict[str, float] = field(default_factory=dict)
    per_disease_oscillation_count:      dict[str, float] = field(default_factory=dict)
    per_disease_stable_fraction:        dict[str, float] = field(default_factory=dict)
    per_disease_leadership_change_rate: dict[str, float] = field(default_factory=dict)
    profiles:                           list[TrajectoryProfile] = field(default_factory=list)

    def summary(self) -> str:
        n = max(self.total_cases, 1)
        lines = [
            "TRAJECTORY EVALUATION",
            f"  Total cases              : {self.total_cases}",
            f"  Mean convergence index   : {self.mean_convergence_index:.4f} "
            f"(std={self.std_convergence_index:.4f})",
            f"  Convergent (idx >= 0.80) : {self.convergent_case_count} "
            f"({self.convergent_case_count/n:.1%})",
            f"  Oscillating (>= 2)       : {self.oscillating_case_count} "
            f"({self.oscillating_case_count/n:.1%})",
            f"  Stable trajectories      : {self.stable_case_count} "
            f"({self.stable_case_count/n:.1%})",
            f"  Dampened                 : {self.dampened_case_count} "
            f"({self.dampened_case_count/n:.1%})",
            f"  Leadership changes       : {self.leadership_changed_count} "
            f"({self.leadership_changed_count/n:.1%})",
            f"  Stabilised cases         : {self.stabilised_count} "
            f"({self.stabilised_count/n:.1%})",
            f"  Mean stab stage          : {self.mean_stabilisation_stage:.1f}",
            f"  Mean peak certainty      : {self.mean_peak_certainty:.4f}",
            f"  Mean final certainty     : {self.mean_final_certainty:.4f}",
            f"  Mean certainty delta     : {self.mean_certainty_delta:+.4f}",
        ]
        return "\n".join(lines)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TrajectoryEvaluator:
    """
    Stateless trajectory stability and convergence analyser.
    """

    @classmethod
    def evaluate_vectors(
        cls,
        vectors: list[SymbolicFeatureVector],
        disease_labels: list[str] | None = None,
    ) -> TrajectoryEvaluationResult:
        """
        Analyse trajectory dynamics across all symbolic reasoning vectors.

        Parameters
        ----------
        vectors:
            Symbolic reasoning outputs, one per patient.
        disease_labels:
            Ground-truth labels (optional override).
        """
        labels   = disease_labels or [v.disease_label for v in vectors]
        profiles = [
            cls._build_profile(v, lbl)
            for v, lbl in zip(vectors, labels)
        ]
        return cls._aggregate(profiles)

    @classmethod
    def _build_profile(
        cls,
        v: SymbolicFeatureVector,
        disease_label: str,
    ) -> TrajectoryProfile:
        return TrajectoryProfile(
            patient_id=v.patient_id,
            disease_label=disease_label,
            trajectory_length=v.trajectory_length,
            convergence_index=v.convergence_index,
            oscillation_count=v.oscillation_count,
            peak_certainty=v.peak_certainty,
            final_certainty=v.certainty,
            certainty_delta_total=v.certainty_delta_total,
            was_dampened=v.was_dampened,
            leadership_changed=v.leadership_changed,
            leadership_changes_count=v.leadership_changes_count,
            stabilisation_stage=v.stabilisation_stage,
            recommendation=v.recommendation,
        )

    @classmethod
    def _aggregate(
        cls,
        profiles: list[TrajectoryProfile],
    ) -> TrajectoryEvaluationResult:
        n = len(profiles)
        if n == 0:
            return TrajectoryEvaluationResult(total_cases=0)

        conv_vals  = [p.convergence_index for p in profiles]
        osc_vals   = [p.oscillation_count for p in profiles]
        peak_vals  = [p.peak_certainty for p in profiles]
        final_vals = [p.final_certainty for p in profiles]
        delta_vals = [p.certainty_delta_total for p in profiles]

        stab_profiles = [p for p in profiles if p.stabilisation_stage >= 0]
        mean_stab = (
            statistics.mean(p.stabilisation_stage for p in stab_profiles)
            if stab_profiles else 0.0
        )

        by_disease: dict[str, list[TrajectoryProfile]] = {}
        for p in profiles:
            by_disease.setdefault(p.disease_label, []).append(p)

        per_conv:  dict[str, float] = {}
        per_osc:   dict[str, float] = {}
        per_stable: dict[str, float] = {}
        per_lead:  dict[str, float] = {}

        for dis, dp in by_disease.items():
            nd = len(dp)
            per_conv[dis]   = statistics.mean(p.convergence_index for p in dp)
            per_osc[dis]    = statistics.mean(float(p.oscillation_count) for p in dp)
            per_stable[dis] = sum(1 for p in dp if p.is_stable) / nd
            per_lead[dis]   = sum(1 for p in dp if p.leadership_changed) / nd

        return TrajectoryEvaluationResult(
            total_cases=n,
            mean_convergence_index=statistics.mean(conv_vals),
            std_convergence_index=statistics.stdev(conv_vals) if len(conv_vals) > 1 else 0.0,
            mean_oscillation_count=statistics.mean(float(o) for o in osc_vals),
            oscillating_case_count=sum(1 for p in profiles if p.is_oscillating),
            stable_case_count=sum(1 for p in profiles if p.is_stable),
            convergent_case_count=sum(1 for p in profiles if p.is_convergent),
            dampened_case_count=sum(1 for p in profiles if p.was_dampened),
            leadership_changed_count=sum(1 for p in profiles if p.leadership_changed),
            mean_peak_certainty=statistics.mean(peak_vals),
            mean_final_certainty=statistics.mean(final_vals),
            mean_certainty_delta=statistics.mean(delta_vals),
            stabilised_count=len(stab_profiles),
            mean_stabilisation_stage=mean_stab,
            per_disease_convergence_index=per_conv,
            per_disease_oscillation_count=per_osc,
            per_disease_stable_fraction=per_stable,
            per_disease_leadership_change_rate=per_lead,
            profiles=profiles,
        )
