"""
ContradictionRebalancer — severity-tiered contradiction signal calibration.

The Phase 5 diagnostic audit showed that the symbolic pipeline applies a
binary escalation trigger for contradiction_load > 0.40, but the current
scoring distributes contradiction load in the range [0, ~4.0] with many
cases in the 0.5–1.5 range that trigger mandatory certainty dampening
without producing a clinical escalation.

This module:

  1. Classifies contradiction load into four severity tiers:
       NONE     : load == 0.0             (no contradiction observed)
       MINOR    : 0.0 < load < 0.15      (borderline, low clinical impact)
       MODERATE : 0.15 <= load < 0.40    (significant, dampening appropriate)
       CRITICAL : load >= 0.40           (mandatory escalation — non-negotiable)

  2. Derives tier-adjusted contradiction features:
       contradiction_tier            — ordinal encoding [0, 3]
       contradiction_adjusted_load   — tier-scaled continuous value
       contradiction_severity_index  — non-linear transform for ML discrimination
       bilateral_asymmetry_index     — abs(contr_load - mean_expected_load)
       dampening_necessity           — was_dampened AND tier >= MODERATE

  3. Analyses contradiction prevalence by disease and tier.

  4. Generates a rebalancing report showing which tier drives the most
     false certainty collapse and which diseases are most affected.

Safety constraint
-----------------
  The CRITICAL tier (load >= 0.40) always triggers escalation.
  This cannot be modified. The rebalancing only affects how MINOR and
  MODERATE tier contradictions contribute to the ML feature representation
  — it does not suppress escalation for critical cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Contradiction severity tiers ──────────────────────────────────────────────

class ContradictionTier(IntEnum):
    NONE     = 0
    MINOR    = 1
    MODERATE = 2
    CRITICAL = 3


_TIER_THRESHOLDS: dict[ContradictionTier, tuple[float, float]] = {
    ContradictionTier.NONE:     (0.00, 0.001),
    ContradictionTier.MINOR:    (0.001, 0.15),
    ContradictionTier.MODERATE: (0.15,  0.40),
    ContradictionTier.CRITICAL: (0.40,  float("inf")),
}

_TIER_SCALE_FACTORS: dict[ContradictionTier, float] = {
    ContradictionTier.NONE:     0.00,
    ContradictionTier.MINOR:    0.20,   # down-weight minor contradiction
    ContradictionTier.MODERATE: 0.70,   # moderate → meaningful signal
    ContradictionTier.CRITICAL: 1.00,   # critical → maximum signal
}


def _classify_tier(load: float) -> ContradictionTier:
    """Return the contradiction severity tier for a given load value."""
    for tier, (lo, hi) in _TIER_THRESHOLDS.items():
        if lo <= load < hi:
            return tier
    return ContradictionTier.CRITICAL


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ContradictionEnrichedSignals:
    """
    Tier-calibrated contradiction signals derived from SymbolicFeatureVector.

    Attributes
    ----------
    contradiction_tier:
        Ordinal tier encoding [0=NONE, 1=MINOR, 2=MODERATE, 3=CRITICAL].
    contradiction_adjusted_load:
        Load * tier_scale_factor — down-weights minor contradictions.
    contradiction_severity_index:
        Non-linear transform: 1 - exp(-2 * adjusted_load).
        Range [0, 1]; saturates near 1 for critical contradictions.
    bilateral_asymmetry_index:
        |load - tier_midpoint| — deviation from tier centre.
        High = load is at extreme of its tier.
    dampening_necessity:
        1.0 if was_dampened AND tier >= MODERATE, else 0.0.
        Captures whether dampening was warranted given contradiction severity.
    escalation_justified:
        1.0 if tier == CRITICAL (i.e., escalation was non-negotiable).
    minor_contradiction_flag:
        1.0 if tier == MINOR (possible over-dampening candidate).
    load_log_transform:
        log(1 + contradiction_load) — variance-stabilising transform.
    """

    contradiction_tier:         int
    contradiction_adjusted_load: float
    contradiction_severity_index: float
    bilateral_asymmetry_index:  float
    dampening_necessity:        float
    escalation_justified:       float
    minor_contradiction_flag:   float
    load_log_transform:         float

    @staticmethod
    def signal_names() -> list[str]:
        return [
            "contradiction_tier",
            "contradiction_adjusted_load",
            "contradiction_severity_index",
            "bilateral_asymmetry_index",
            "dampening_necessity",
            "escalation_justified",
            "minor_contradiction_flag",
            "load_log_transform",
        ]

    def to_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in self.signal_names()}


@dataclass
class ContradictionTierStats:
    """Statistics for one contradiction severity tier."""
    tier:          ContradictionTier
    count:         int
    fraction:      float
    per_disease:   dict[str, int] = field(default_factory=dict)
    mean_certainty: float = 0.0
    mean_accuracy:  float = 0.0   # fraction correctly classified within tier


@dataclass
class ContradictionRebalancingReport:
    """
    Complete contradiction rebalancing analysis.

    Attributes
    ----------
    tier_stats:
        Per-tier count and disease breakdown.
    minor_over_dampening_count:
        Cases where dampening was applied but tier was MINOR.
    critical_correct_escalation_count:
        Cases where critical tier correctly triggered escalation.
    mean_load_by_tier:
        Mean contradiction load per tier.
    per_disease_tier_distribution:
        For each disease, fraction of cases in each tier.
    rebalancing_signal_variance:
        Variance of each derived contradiction signal.
    """

    tier_stats:                      dict[str, ContradictionTierStats] = field(default_factory=dict)
    minor_over_dampening_count:      int = 0
    critical_correct_escalation_count: int = 0
    mean_load_by_tier:               dict[str, float] = field(default_factory=dict)
    per_disease_tier_distribution:   dict[str, dict[str, float]] = field(default_factory=dict)
    rebalancing_signal_variance:     dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "CONTRADICTION REBALANCING REPORT",
            "=" * 72,
            "  TIER DISTRIBUTION:",
        ]
        for tier_name, stats in self.tier_stats.items():
            lines.append(
                f"    {tier_name:10s}: {stats.count:4d} cases ({stats.fraction:.1%})  "
                f"mean_cert={stats.mean_certainty:.3f}"
            )
        lines += [
            f"  Minor over-dampening cases   : {self.minor_over_dampening_count}",
            f"  Critical correct escalations : {self.critical_correct_escalation_count}",
            "-" * 72,
            "  TIER DISTRIBUTION BY DISEASE:",
        ]
        for dis, tiers in self.per_disease_tier_distribution.items():
            tier_str = " | ".join(
                f"{t[:3]}={v:.0%}" for t, v in tiers.items()
            )
            lines.append(f"    {dis:30s} {tier_str}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Rebalancer ────────────────────────────────────────────────────────────────

class ContradictionRebalancer:
    """
    Derives tier-calibrated contradiction signals from symbolic vectors.

    This module does NOT modify the symbolic pipeline or suppress critical
    escalations. It generates enhanced ML features that allow the classifier
    to distinguish between minor, moderate, and critical contradiction cases.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> list[ContradictionEnrichedSignals]:
        """
        Derive tier-calibrated contradiction signals for each vector.

        Parameters
        ----------
        vectors:
            Symbolic feature vectors from the pipeline.
        """
        return [self._enrich_one(v) for v in vectors]

    def build_analysis_report(
        self,
        vectors:   list[SymbolicFeatureVector],
        enriched:  list[ContradictionEnrichedSignals],
        y_pred:    np.ndarray | None = None,
        y_true:    np.ndarray | None = None,
    ) -> ContradictionRebalancingReport:
        """Generate the full rebalancing analysis report."""
        if not vectors:
            return ContradictionRebalancingReport()

        # Tier stats
        tier_stats: dict[str, ContradictionTierStats] = {}
        for tier in ContradictionTier:
            mask = [_classify_tier(v.contradiction_load) == tier for v in vectors]
            tier_vecs = [v for v, m in zip(vectors, mask) if m]
            per_dis: dict[str, int] = {}
            for v in tier_vecs:
                per_dis[v.disease_label] = per_dis.get(v.disease_label, 0) + 1
            mean_cert = float(np.mean([v.certainty for v in tier_vecs])) if tier_vecs else 0.0
            acc = 0.0
            if y_pred is not None and y_true is not None and tier_vecs:
                idxs = [i for i, m in enumerate(mask) if m]
                if idxs:
                    acc = float(np.mean(y_pred[idxs] == y_true[idxs]))
            tier_stats[tier.name] = ContradictionTierStats(
                tier=tier,
                count=len(tier_vecs),
                fraction=len(tier_vecs) / len(vectors),
                per_disease=per_dis,
                mean_certainty=mean_cert,
                mean_accuracy=acc,
            )

        # Over-dampening and correct escalation counts
        minor_damp = sum(
            1 for v, e in zip(vectors, enriched)
            if e.minor_contradiction_flag > 0 and v.was_dampened
        )
        crit_esc = sum(1 for e in enriched if e.escalation_justified > 0)

        # Mean load by tier
        mean_load: dict[str, float] = {}
        for tier in ContradictionTier:
            loads = [v.contradiction_load for v in vectors
                     if _classify_tier(v.contradiction_load) == tier]
            mean_load[tier.name] = float(np.mean(loads)) if loads else 0.0

        # Per-disease tier distribution
        diseases = list({v.disease_label for v in vectors})
        per_dis_tiers: dict[str, dict[str, float]] = {}
        for dis in diseases:
            dis_vecs = [v for v in vectors if v.disease_label == dis]
            n = max(len(dis_vecs), 1)
            per_dis_tiers[dis] = {
                t.name: sum(
                    1 for v in dis_vecs if _classify_tier(v.contradiction_load) == t
                ) / n
                for t in ContradictionTier
            }

        # Signal variance
        sig_names = ContradictionEnrichedSignals.signal_names()
        X_enrich  = np.array([[e.to_dict()[n] for n in sig_names] for e in enriched])
        sig_var   = {n: float(np.var(X_enrich[:, i])) for i, n in enumerate(sig_names)}

        return ContradictionRebalancingReport(
            tier_stats=tier_stats,
            minor_over_dampening_count=minor_damp,
            critical_correct_escalation_count=crit_esc,
            mean_load_by_tier=mean_load,
            per_disease_tier_distribution=per_dis_tiers,
            rebalancing_signal_variance=sig_var,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _enrich_one(self, v: SymbolicFeatureVector) -> ContradictionEnrichedSignals:
        """Derive tier-calibrated signals for a single vector."""
        load    = float(v.contradiction_load)
        damped  = bool(v.was_dampened)
        tier    = _classify_tier(load)
        scale   = _TIER_SCALE_FACTORS[tier]

        adj_load     = load * scale
        sev_index    = 1.0 - math.exp(-2.0 * adj_load)
        lo, hi       = _TIER_THRESHOLDS[tier]
        tier_mid     = (lo + min(hi, 2.0)) / 2.0
        asym_index   = abs(load - tier_mid)
        damp_nec     = float(damped and tier.value >= ContradictionTier.MODERATE)
        esc_just     = float(tier == ContradictionTier.CRITICAL)
        minor_flag   = float(tier == ContradictionTier.MINOR)
        log_load     = math.log1p(load)

        return ContradictionEnrichedSignals(
            contradiction_tier=int(tier),
            contradiction_adjusted_load=float(np.clip(adj_load, 0.0, 2.0)),
            contradiction_severity_index=float(np.clip(sev_index, 0.0, 1.0)),
            bilateral_asymmetry_index=float(np.clip(asym_index, 0.0, 2.0)),
            dampening_necessity=damp_nec,
            escalation_justified=esc_just,
            minor_contradiction_flag=minor_flag,
            load_log_transform=float(np.clip(log_load, 0.0, 3.0)),
        )
