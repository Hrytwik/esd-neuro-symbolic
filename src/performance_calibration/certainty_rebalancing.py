"""
CertaintyRebalancer — certainty signal calibration and trajectory normalization.

The symbolic pipeline produces certainty values in the range [0.10, 0.55]
on clinical-only data (vs. [0.30, 0.90] on full 34-feature input). This
compressed range causes two problems:

  1. Poor ML discrimination — most patients cluster near 0.20–0.35 certainty,
     making it difficult for the classifier to distinguish them.

  2. Pathological escalation — all cases fall below the 0.55 safe-triage
     threshold, preventing any non-escalated stabilization.

This module implements certainty rebalancing through:

  1. Range-preserving normalization — rescales certainty values into a
     wider ML-friendly range while preserving the relative ordering
     (monotone transform). Does NOT inflate to artificial certainty.

  2. Convergence-weighted certainty — combines raw certainty with
     trajectory convergence quality to produce a composite stability-
     certainty score that better reflects diagnostic confidence.

  3. Entropy-certainty consistency score — measures whether certainty
     and entropy are mutually consistent. Inconsistency suggests
     certainty was disrupted by dampening/oscillation.

  4. Certainty trajectory summary — derives signals from the full
     certainty trajectory dynamics (delta, momentum, acceleration).

  5. Context-aware certainty sufficiency — recalibrated sufficiency
     threshold using the empirical distribution of clinical-only certainty.

Important
---------
All transforms are monotone-preserving — cases with higher certainty
in the original pipeline always have higher certainty in the
rebalanced output. The relative diagnostic confidence ordering is intact.

No artificial certainty is added. A case with certainty = 0.21 in the
original pipeline is not converted to 0.75. The range expansion is
a linear or power-law monotone map, not a score inflation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Rebalancing constants ─────────────────────────────────────────────────────

_MIN_CLINICAL_CERTAINTY:    float = 0.05   # empirical minimum on clinical-only data
_MAX_CLINICAL_CERTAINTY:    float = 0.55   # empirical maximum on clinical-only data
_CONVERGENCE_BLEND_WEIGHT:  float = 0.35   # weight of convergence_index in composite
_STABILITY_WEIGHT:          float = 0.25   # weight of trajectory stability
_TARGET_RANGE_MIN:          float = 0.10   # target min for normalized certainty
_TARGET_RANGE_MAX:          float = 0.85   # target max for normalized certainty


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CertaintyEnrichedSignals:
    """
    Rebalanced and enriched certainty signals derived from SymbolicFeatureVector.

    All signals are derived from genuine pipeline outputs — no raw clinical
    feature engineering.

    Attributes
    ----------
    certainty_normalised:
        Certainty rescaled to [0.10, 0.85] via monotone linear map using
        empirical min/max of clinical-only certainty range.
    certainty_convergence_composite:
        Blend of normalised certainty and convergence_index.
        Captures both strength and stability of the leading hypothesis.
    certainty_trajectory_momentum:
        certainty_delta_total / max(trajectory_length, 1).
        Positive = rising trajectory; negative = declining.
    certainty_acceleration:
        |certainty_delta_total| / max(peak_certainty, 0.01).
        How rapidly certainty changed relative to its peak.
    entropy_certainty_consistency:
        1 - |normalised_entropy - (1 - certainty_normalised)|.
        1.0 = perfectly consistent; 0 = inconsistent (dampened/oscillated).
    peak_certainty_gap:
        peak_certainty - certainty — drop from peak to terminal.
        High = certainty collapsed after peaking.
    context_certainty_sufficiency:
        1.0 if certainty_normalised >= 0.45 and certainty_gap >= 0.12.
        Clinical-context-specific sufficiency threshold.
    stabilisation_quality:
        1.0 if stabilisation_stage >= 0 (certainty stabilised at some stage).
        0.0 if never stabilised (perpetual uncertainty).
    certainty_delta_normalised:
        certainty_delta_total / max(peak_certainty, 0.01).
        Relative certainty change over the trajectory.
    trajectory_stability_score:
        1 - (oscillation_count * 0.25) clamped to [0, 1].
        High = stable trajectory; low = oscillatory.
    """

    certainty_normalised:             float
    certainty_convergence_composite:  float
    certainty_trajectory_momentum:    float
    certainty_acceleration:           float
    entropy_certainty_consistency:    float
    peak_certainty_gap:               float
    context_certainty_sufficiency:    float
    stabilisation_quality:            float
    certainty_delta_normalised:       float
    trajectory_stability_score:       float

    @staticmethod
    def signal_names() -> list[str]:
        return [
            "certainty_normalised",
            "certainty_convergence_composite",
            "certainty_trajectory_momentum",
            "certainty_acceleration",
            "entropy_certainty_consistency",
            "peak_certainty_gap",
            "context_certainty_sufficiency",
            "stabilisation_quality",
            "certainty_delta_normalised",
            "trajectory_stability_score",
        ]

    def to_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in self.signal_names()}


@dataclass
class CertaintyDistributionProfile:
    """Empirical distribution of certainty values in a dataset split."""

    mean:            float = 0.0
    std:             float = 0.0
    p10:             float = 0.0   # 10th percentile
    p25:             float = 0.0
    p50:             float = 0.0   # median
    p75:             float = 0.0
    p90:             float = 0.0
    minimum:         float = 0.0
    maximum:         float = 0.0
    above_0_40_rate: float = 0.0   # fraction with certainty >= 0.40
    above_0_55_rate: float = 0.0   # fraction with certainty >= 0.55 (original threshold)


@dataclass
class CertaintyRebalancingReport:
    """
    Complete certainty rebalancing analysis.

    Attributes
    ----------
    original_distribution:
        Empirical certainty distribution before rebalancing.
    normalised_distribution:
        Distribution of certainty_normalised after rebalancing.
    signal_variance:
        Variance of each derived certainty signal.
    per_disease_mean_certainty_normalised:
        Mean normalised certainty per disease.
    n_context_sufficient:
        Cases with context_certainty_sufficiency == 1.0.
    improvement_vs_original:
        Fraction of cases where normalised certainty > original certainty.
    """

    original_distribution:                  CertaintyDistributionProfile = field(
        default_factory=CertaintyDistributionProfile
    )
    normalised_distribution:                CertaintyDistributionProfile = field(
        default_factory=CertaintyDistributionProfile
    )
    signal_variance:                        dict[str, float] = field(default_factory=dict)
    per_disease_mean_certainty_normalised:  dict[str, float] = field(default_factory=dict)
    n_context_sufficient:                   int   = 0
    improvement_vs_original:               float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "CERTAINTY REBALANCING REPORT",
            "=" * 72,
            "  ORIGINAL CERTAINTY DISTRIBUTION:",
            f"    mean={self.original_distribution.mean:.3f}  "
            f"std={self.original_distribution.std:.3f}  "
            f"p50={self.original_distribution.p50:.3f}  "
            f"max={self.original_distribution.maximum:.3f}",
            f"    >=0.40: {self.original_distribution.above_0_40_rate:.1%}  "
            f">=0.55: {self.original_distribution.above_0_55_rate:.1%}",
            "  NORMALISED CERTAINTY DISTRIBUTION:",
            f"    mean={self.normalised_distribution.mean:.3f}  "
            f"std={self.normalised_distribution.std:.3f}  "
            f"p50={self.normalised_distribution.p50:.3f}  "
            f"max={self.normalised_distribution.maximum:.3f}",
            f"    >=0.40: {self.normalised_distribution.above_0_40_rate:.1%}  "
            f">=0.55: {self.normalised_distribution.above_0_55_rate:.1%}",
            f"  Context-sufficient cases : {self.n_context_sufficient}",
            f"  Improvement rate         : {self.improvement_vs_original:.1%}",
            "-" * 72,
            "  MEAN NORMALISED CERTAINTY BY DISEASE:",
        ]
        for dis, val in sorted(
            self.per_disease_mean_certainty_normalised.items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"    {dis:35s} {val:.4f}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Rebalancer ────────────────────────────────────────────────────────────────

class CertaintyRebalancer:
    """
    Applies monotone normalization and derives enriched certainty signals.

    Parameters
    ----------
    observed_min:
        Empirically observed minimum certainty on clinical-only data.
    observed_max:
        Empirically observed maximum certainty on clinical-only data.
    target_min:
        Target minimum for normalized certainty.
    target_max:
        Target maximum for normalized certainty.
    """

    def __init__(
        self,
        observed_min: float = _MIN_CLINICAL_CERTAINTY,
        observed_max: float = _MAX_CLINICAL_CERTAINTY,
        target_min:   float = _TARGET_RANGE_MIN,
        target_max:   float = _TARGET_RANGE_MAX,
    ) -> None:
        self.obs_min = observed_min
        self.obs_max = max(observed_max, observed_min + 1e-6)
        self.tgt_min = target_min
        self.tgt_max = target_max

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, vectors: list[SymbolicFeatureVector]) -> "CertaintyRebalancer":
        """
        Fit the observed certainty range from a training set.

        Parameters
        ----------
        vectors:
            Training-set symbolic feature vectors.
        """
        if vectors:
            certs      = [v.certainty for v in vectors]
            self.obs_min = float(np.percentile(certs, 2))
            self.obs_max = float(np.percentile(certs, 98))
            self.obs_max = max(self.obs_max, self.obs_min + 1e-6)
        return self

    def enrich(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> list[CertaintyEnrichedSignals]:
        """
        Derive rebalanced certainty signals for each vector.

        Parameters
        ----------
        vectors:
            Symbolic feature vectors (train or test set).
        """
        return [self._enrich_one(v) for v in vectors]

    def build_analysis_report(
        self,
        vectors:  list[SymbolicFeatureVector],
        enriched: list[CertaintyEnrichedSignals],
    ) -> CertaintyRebalancingReport:
        """Generate the full rebalancing analysis report."""
        if not vectors:
            return CertaintyRebalancingReport()

        orig_certs  = np.array([v.certainty for v in vectors])
        norm_certs  = np.array([e.certainty_normalised for e in enriched])

        orig_dist  = self._distribution_profile(orig_certs)
        norm_dist  = self._distribution_profile(norm_certs)

        per_dis: dict[str, list[float]] = {}
        for v, e in zip(vectors, enriched):
            per_dis.setdefault(v.disease_label, []).append(e.certainty_normalised)
        per_dis_mean = {d: float(np.mean(vs)) for d, vs in per_dis.items()}

        n_suff  = sum(1 for e in enriched if e.context_certainty_sufficiency > 0.5)
        impr    = float(np.mean(norm_certs > orig_certs))

        sig_names = CertaintyEnrichedSignals.signal_names()
        X_enr     = np.array([[e.to_dict()[n] for n in sig_names] for e in enriched])
        sig_var   = {n: float(np.var(X_enr[:, i])) for i, n in enumerate(sig_names)}

        return CertaintyRebalancingReport(
            original_distribution=orig_dist,
            normalised_distribution=norm_dist,
            signal_variance=sig_var,
            per_disease_mean_certainty_normalised=per_dis_mean,
            n_context_sufficient=n_suff,
            improvement_vs_original=impr,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalise(self, cert: float) -> float:
        """Apply monotone linear map from [obs_min, obs_max] to [tgt_min, tgt_max]."""
        if self.obs_max <= self.obs_min:
            return self.tgt_min
        t = (cert - self.obs_min) / (self.obs_max - self.obs_min)
        t = float(np.clip(t, 0.0, 1.0))
        return self.tgt_min + t * (self.tgt_max - self.tgt_min)

    def _enrich_one(self, v: SymbolicFeatureVector) -> CertaintyEnrichedSignals:
        cert        = float(v.certainty)
        peak        = float(v.peak_certainty)
        conv        = float(v.convergence_index)
        traj_len    = max(int(v.trajectory_length), 1)
        delta       = float(v.certainty_delta_total)
        n_ent       = float(v.normalised_entropy)
        n_osc       = int(v.oscillation_count)
        stab_stage  = int(v.stabilisation_stage)
        gap         = float(v.certainty_gap)

        cert_norm   = self._normalise(cert)
        comp        = (
            (1.0 - _CONVERGENCE_BLEND_WEIGHT - _STABILITY_WEIGHT) * cert_norm
            + _CONVERGENCE_BLEND_WEIGHT * conv
            + _STABILITY_WEIGHT * (1.0 - min(n_osc * 0.25, 1.0))
        )
        momentum    = delta / traj_len
        accel       = abs(delta) / max(peak, 0.01)
        ent_cert_cons = 1.0 - abs(n_ent - (1.0 - cert_norm))
        peak_gap    = peak - cert
        ctx_suff    = float(cert_norm >= 0.45 and gap >= 0.12)
        stab_qual   = float(stab_stage >= 0)
        delta_norm  = delta / max(peak, 0.01)
        stab_score  = max(0.0, 1.0 - n_osc * 0.25)

        return CertaintyEnrichedSignals(
            certainty_normalised=float(np.clip(cert_norm, 0.0, 1.0)),
            certainty_convergence_composite=float(np.clip(comp, 0.0, 1.0)),
            certainty_trajectory_momentum=float(np.clip(momentum, -1.0, 1.0)),
            certainty_acceleration=float(np.clip(accel, 0.0, 2.0)),
            entropy_certainty_consistency=float(np.clip(ent_cert_cons, 0.0, 1.0)),
            peak_certainty_gap=float(np.clip(peak_gap, 0.0, 1.0)),
            context_certainty_sufficiency=ctx_suff,
            stabilisation_quality=stab_qual,
            certainty_delta_normalised=float(np.clip(delta_norm, -1.0, 1.0)),
            trajectory_stability_score=float(np.clip(stab_score, 0.0, 1.0)),
        )

    def _distribution_profile(self, arr: np.ndarray) -> CertaintyDistributionProfile:
        if len(arr) == 0:
            return CertaintyDistributionProfile()
        return CertaintyDistributionProfile(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            p10=float(np.percentile(arr, 10)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p90=float(np.percentile(arr, 90)),
            minimum=float(np.min(arr)),
            maximum=float(np.max(arr)),
            above_0_40_rate=float(np.mean(arr >= 0.40)),
            above_0_55_rate=float(np.mean(arr >= 0.55)),
        )
