"""
DiscriminativeOptimizer — biopsy-free discriminative performance improvement.

Targets the two primary sources of Model B / C under-performance:

  1. Feature-level overlap — diseases share overlapping clinical feature
     distributions, making classification boundaries ambiguous.

  2. Certainty gap insufficiency — the symbolic pipeline does not produce
     sufficient certainty differential between the leading and competing
     hypothesis in many cases.

This module analyses the discriminative landscape and generates:
  · Per-feature Fisher discriminant ratios for every disease pair
  · Per-disease confusion risk profiles
  · Certainty gap adequacy assessment
  · Targeted optimization recommendations
  · Expected accuracy improvement estimates per intervention

The analysis is purely observational — it does not modify pipeline state.
All recommendations are framed as rule-weight or threshold adjustments
that can be applied through existing calibration infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_CASES_FOR_ANALYSIS: int   = 3
_STRONG_F_RATIO:         float = 10.0
_MODERATE_F_RATIO:       float = 3.0
_WEAK_F_RATIO:           float = 1.0
_ADEQUATE_CERTAINTY_GAP: float = 0.15
_MIN_RECOVERY_THRESHOLD: float = 0.20   # min recovery rate to flag as opportunity


# ── Dataclasses ───────────────────────────────────────────────────────────────

class DiscriminationTier(str, Enum):
    STRONG   = "strong"     # F >= 10
    MODERATE = "moderate"   # F >= 3
    WEAK     = "weak"       # F >= 1
    NEGLIGIBLE = "negligible"  # F < 1


@dataclass
class FeaturePairSeparation:
    """
    Discriminative power of one clinical feature for one disease pair.

    Attributes
    ----------
    feature_name:
        Clinical feature name (e.g. 'polygonal_papules').
    disease_a:
        First disease in pair.
    disease_b:
        Second disease in pair.
    f_ratio:
        Fisher discriminant ratio (between-group / within-group variance).
    mean_a:
        Mean feature value for disease_a cases.
    mean_b:
        Mean feature value for disease_b cases.
    delta:
        |mean_a - mean_b| — absolute separation.
    overlap_coefficient:
        Estimated proportion of overlapping feature distributions.
        0.0 = no overlap (ideal); 1.0 = complete overlap (no discrimination).
    tier:
        Discrimination quality tier.
    """

    feature_name:       str
    disease_a:          str
    disease_b:          str
    f_ratio:            float
    mean_a:             float
    mean_b:             float
    delta:              float
    overlap_coefficient: float
    tier:               DiscriminationTier

    def is_useful(self) -> bool:
        return self.tier in (DiscriminationTier.STRONG, DiscriminationTier.MODERATE)


@dataclass
class DiseaseDiscriminativeProfile:
    """
    Full discriminative profile for a single disease.

    Attributes
    ----------
    disease:
        Canonical disease name.
    n_cases:
        Test cases for this disease.
    mean_certainty_gap:
        Mean certainty gap (leading vs. second hypothesis) for cases of this disease.
    gap_adequate_rate:
        Fraction of cases where certainty gap >= _ADEQUATE_CERTAINTY_GAP.
    top_separating_features:
        Features with highest average F-ratio separating this disease from others.
    hardest_confusion_partners:
        Diseases most frequently confused with this one, ranked by confusion rate.
    confusion_rate:
        Fraction of cases misclassified (requires y_pred to be provided).
    symbolic_advantage:
        Additional certainty gain from symbolic signals vs. clinical-only.
    improvement_potential:
        Estimated accuracy gain achievable through targeted optimization (0–1).
    """

    disease:                  str
    n_cases:                  int   = 0
    mean_certainty_gap:       float = 0.0
    gap_adequate_rate:        float = 0.0
    top_separating_features:  list[str] = field(default_factory=list)
    hardest_confusion_partners: list[tuple[str, float]] = field(default_factory=list)
    confusion_rate:           float = 0.0
    symbolic_advantage:       float = 0.0
    improvement_potential:    float = 0.0

    def discrimination_tier(self) -> str:
        if self.gap_adequate_rate >= 0.60:
            return "well_separated"
        if self.gap_adequate_rate >= 0.30:
            return "partially_separated"
        return "poorly_separated"


@dataclass
class OptimizationRecommendation:
    """
    Specific discriminative optimization recommendation.

    Attributes
    ----------
    priority:
        1 = highest priority.
    target_diseases:
        Diseases this recommendation addresses.
    intervention_type:
        "feature_weight" / "certainty_threshold" / "rule_weight" / "signal_enrichment"
    description:
        Clinical rationale.
    expected_accuracy_gain_pp:
        Estimated accuracy improvement in percentage points.
    implementation_note:
        How to implement via existing calibration infrastructure.
    """

    priority:               int
    target_diseases:        list[str]
    intervention_type:      str
    description:            str
    expected_accuracy_gain_pp: float
    implementation_note:    str


@dataclass
class DiscriminativeOptimizationReport:
    """
    Full discriminative optimization analysis.

    Attributes
    ----------
    disease_profiles:
        Per-disease discriminative profiles.
    pair_separations:
        Feature-level separation for all disease × pair combinations.
    overall_mean_certainty_gap:
        Mean certainty gap across all test cases.
    gap_adequate_rate:
        Fraction of cases with certainty gap >= 0.15.
    poorly_separated_pairs:
        Disease pairs with no strong separating features.
    recommendations:
        Prioritized optimization recommendations.
    estimated_total_gain_pp:
        Sum of non-overlapping estimated gains.
    """

    disease_profiles:           list[DiseaseDiscriminativeProfile] = field(default_factory=list)
    pair_separations:           list[FeaturePairSeparation] = field(default_factory=list)
    overall_mean_certainty_gap: float = 0.0
    gap_adequate_rate:          float = 0.0
    poorly_separated_pairs:     list[tuple[str, str]] = field(default_factory=list)
    recommendations:            list[OptimizationRecommendation] = field(default_factory=list)
    estimated_total_gain_pp:    float = 0.0

    def summary(self) -> str:
        sep  = "=" * 72
        dash = "-" * 72
        lines = [
            sep,
            "DISCRIMINATIVE OPTIMIZATION REPORT",
            sep,
            f"  Overall mean certainty gap : {self.overall_mean_certainty_gap:.4f}",
            f"  Gap-adequate rate (>=0.15) : {self.gap_adequate_rate:.1%}",
            f"  Poorly separated pairs     : {len(self.poorly_separated_pairs)}",
            f"  Estimated total gain       : +{self.estimated_total_gain_pp:.1f} pp",
            dash,
            "  PER-DISEASE DISCRIMINATION STATUS:",
        ]
        for p in sorted(self.disease_profiles, key=lambda x: x.gap_adequate_rate):
            tier_str = p.discrimination_tier()
            lines.append(
                f"    [{tier_str:18s}] {p.disease:32s} "
                f"gap_adequate={p.gap_adequate_rate:.1%}  "
                f"confusion={p.confusion_rate:.1%}"
            )
        if self.poorly_separated_pairs:
            lines += [dash, "  POORLY SEPARATED DISEASE PAIRS:"]
            for a, b in self.poorly_separated_pairs[:6]:
                lines.append(f"    {a:32s} vs {b}")
        if self.recommendations:
            lines += [dash, "  OPTIMIZATION RECOMMENDATIONS (by priority):"]
            for r in sorted(self.recommendations, key=lambda x: x.priority)[:6]:
                diseases = ", ".join(r.target_diseases[:2])
                lines.append(
                    f"    [P{r.priority}][{r.intervention_type:20s}] "
                    f"{diseases:30s} +{r.expected_accuracy_gain_pp:.1f} pp"
                )
                lines.append(f"         {r.description}")
        lines.append(sep)
        return "\n".join(lines)


# ── Optimizer ─────────────────────────────────────────────────────────────────

class DiscriminativeOptimizer:
    """
    Analyses discriminative performance gaps and generates targeted
    optimization recommendations.

    Parameters
    ----------
    class_labels:
        Ordered canonical disease names.
    clinical_feature_names:
        Names of the 12 clinical features in X_clinical column order.
    """

    def __init__(
        self,
        class_labels:           list[str],
        clinical_feature_names: list[str] | None = None,
    ) -> None:
        self.class_labels      = class_labels
        self.feature_names     = clinical_feature_names or [
            "erythema", "scaling", "definite_borders", "itching",
            "koebner_phenomenon", "polygonal_papules", "follicular_papules",
            "oral_mucosal_involvement", "knee_and_elbow_involvement",
            "scalp_involvement", "family_history", "age",
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        symbolic_vectors: list[SymbolicFeatureVector],
        X_clinical:       np.ndarray,
        y_true:           np.ndarray,
        y_pred:           np.ndarray | None = None,
    ) -> DiscriminativeOptimizationReport:
        """
        Run full discriminative optimization analysis.

        Parameters
        ----------
        symbolic_vectors:
            Test-set symbolic feature vectors.
        X_clinical:
            Clinical feature matrix (n × 12).
        y_true:
            True class labels (0-based integers).
        y_pred:
            Predicted class labels (optional, for confusion analysis).
        """
        # Accept either a list of SymbolicFeatureVector objects or a raw numpy
        # symbolic matrix (n × n_symbolic).  When a numpy array is provided,
        # derive proxy certainty_gap / disease_label values from it.
        if isinstance(symbolic_vectors, np.ndarray):
            symbolic_vectors = self._matrix_to_proxy_vecs(
                symbolic_vectors, y_true
            )

        if len(symbolic_vectors) == 0:
            return DiscriminativeOptimizationReport()

        n = len(symbolic_vectors)
        gaps    = [v.certainty_gap for v in symbolic_vectors]
        mean_gap = float(np.mean(gaps))
        gap_ok   = float(np.mean([g >= _ADEQUATE_CERTAINTY_GAP for g in gaps]))

        # Per-feature pair separations
        pair_seps = self._compute_pair_separations(X_clinical, y_true)

        # Poorly separated pairs (no strong feature)
        all_pairs = list({
            (min(a, b), max(a, b))
            for fp in pair_seps
            for a, b in [(fp.disease_a, fp.disease_b)]
        })
        poor_pairs = self._find_poor_pairs(pair_seps, all_pairs)

        # Disease profiles
        profiles = self._build_profiles(
            symbolic_vectors, X_clinical, y_true, y_pred, pair_seps
        )

        # Recommendations
        recs = self._generate_recommendations(profiles, pair_seps, poor_pairs)

        total_gain = min(
            sum(r.expected_accuracy_gain_pp for r in recs[:4]) * 0.6,   # overlap factor
            12.0
        )

        return DiscriminativeOptimizationReport(
            disease_profiles           = profiles,
            pair_separations           = pair_seps,
            overall_mean_certainty_gap = mean_gap,
            gap_adequate_rate          = gap_ok,
            poorly_separated_pairs     = poor_pairs,
            recommendations            = recs,
            estimated_total_gain_pp    = total_gain,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _matrix_to_proxy_vecs(
        self,
        matrix: np.ndarray,   # (n, n_symbolic)
        y_true: np.ndarray,
    ) -> list:
        """
        Convert a raw symbolic feature matrix into lightweight proxy objects
        that expose the `certainty_gap` and `disease_label` attributes used
        by the rest of the analysis.
        """
        class _ProxyVec:
            __slots__ = ("certainty_gap", "certainty", "disease_label")
            def __init__(self, row: np.ndarray, label: str):
                self.certainty_gap  = float(np.max(row) - np.mean(row))
                self.certainty      = float(np.max(row))
                self.disease_label  = label

        proxies = []
        for i, row in enumerate(matrix):
            label_idx = int(y_true[i])
            label = (
                self.class_labels[label_idx]
                if label_idx < len(self.class_labels)
                else f"class_{label_idx}"
            )
            proxies.append(_ProxyVec(row, label))
        return proxies

    def _compute_pair_separations(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[FeaturePairSeparation]:
        """Compute per-feature Fisher discriminant for all disease pairs."""
        labels  = self.class_labels
        results: list[FeaturePairSeparation] = []

        for fi, feat in enumerate(self.feature_names):
            if fi >= X.shape[1]:
                break
            col = X[:, fi].astype(float)
            for ai, la in enumerate(labels):
                for bi, lb in enumerate(labels):
                    if bi <= ai:
                        continue
                    a_vals = col[y == ai]
                    b_vals = col[y == bi]
                    if len(a_vals) < _MIN_CASES_FOR_ANALYSIS or len(b_vals) < _MIN_CASES_FOR_ANALYSIS:
                        continue

                    ma, mb   = float(np.mean(a_vals)), float(np.mean(b_vals))
                    va, vb   = float(np.var(a_vals))  , float(np.var(b_vals))
                    pooled_v = (va * len(a_vals) + vb * len(b_vals)) / (len(a_vals) + len(b_vals))
                    f_ratio  = ((ma - mb) ** 2) / max(pooled_v, 1e-9)
                    delta    = abs(ma - mb)

                    # Overlap coefficient (Weitzman measure approximation)
                    sa, sb   = max(float(np.std(a_vals)), 1e-6), max(float(np.std(b_vals)), 1e-6)
                    overlap  = float(np.clip(1.0 - delta / (sa + sb + 1e-9), 0.0, 1.0))

                    if f_ratio >= _STRONG_F_RATIO:
                        tier = DiscriminationTier.STRONG
                    elif f_ratio >= _MODERATE_F_RATIO:
                        tier = DiscriminationTier.MODERATE
                    elif f_ratio >= _WEAK_F_RATIO:
                        tier = DiscriminationTier.WEAK
                    else:
                        tier = DiscriminationTier.NEGLIGIBLE

                    results.append(FeaturePairSeparation(
                        feature_name      = feat,
                        disease_a         = la,
                        disease_b         = lb,
                        f_ratio           = f_ratio,
                        mean_a            = ma,
                        mean_b            = mb,
                        delta             = delta,
                        overlap_coefficient = overlap,
                        tier              = tier,
                    ))
        return results

    def _find_poor_pairs(
        self,
        pair_seps: list[FeaturePairSeparation],
        all_pairs: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Identify pairs with no strong or moderate separating feature."""
        poor: list[tuple[str, str]] = []
        for a, b in all_pairs:
            pair_f = [
                fp for fp in pair_seps
                if {fp.disease_a, fp.disease_b} == {a, b}
            ]
            best = max((fp.f_ratio for fp in pair_f), default=0.0)
            if best < _MODERATE_F_RATIO:
                poor.append((a, b))
        return sorted(poor)

    def _build_profiles(
        self,
        vecs:      list[SymbolicFeatureVector],
        X:         np.ndarray,
        y_true:    np.ndarray,
        y_pred:    np.ndarray | None,
        pair_seps: list[FeaturePairSeparation],
    ) -> list[DiseaseDiscriminativeProfile]:
        profiles: list[DiseaseDiscriminativeProfile] = []
        for di, dis in enumerate(self.class_labels):
            idxs = [i for i, v in enumerate(vecs) if v.disease_label == dis]
            if not idxs:
                continue

            dis_vecs  = [vecs[i] for i in idxs]
            gaps      = [v.certainty_gap for v in dis_vecs]
            certs     = [v.certainty     for v in dis_vecs]
            mean_gap  = float(np.mean(gaps))
            gap_ok    = float(np.mean([g >= _ADEQUATE_CERTAINTY_GAP for g in gaps]))

            confusion_rate = 0.0
            if y_pred is not None:
                confusion_rate = float(np.mean(y_pred[idxs] != y_true[idxs]))

            # Top separating features for this disease vs. all others
            dis_pair_seps = [
                fp for fp in pair_seps
                if fp.disease_a == dis or fp.disease_b == dis
            ]
            feat_best: dict[str, float] = {}
            for fp in dis_pair_seps:
                feat_best[fp.feature_name] = max(
                    feat_best.get(fp.feature_name, 0.0), fp.f_ratio
                )
            top_feats = sorted(feat_best, key=lambda f: -feat_best[f])[:4]

            # Hardest confusion partners (paired F-ratio lowest)
            pair_max: dict[str, float] = {}
            for fp in dis_pair_seps:
                partner = fp.disease_b if fp.disease_a == dis else fp.disease_a
                pair_max[partner] = max(pair_max.get(partner, 0.0), fp.f_ratio)
            hardest = sorted(pair_max.items(), key=lambda x: x[1])[:3]

            # Symbolic advantage: compare mean certainty vs gap
            sym_adv = float(np.mean(certs)) * gap_ok

            # Improvement potential: inverse of gap adequacy × confusion rate
            potential = min(1.0, (1.0 - gap_ok) * 0.5 + confusion_rate * 0.5)

            profiles.append(DiseaseDiscriminativeProfile(
                disease                   = dis,
                n_cases                   = len(idxs),
                mean_certainty_gap        = mean_gap,
                gap_adequate_rate         = gap_ok,
                top_separating_features   = top_feats,
                hardest_confusion_partners = hardest,
                confusion_rate            = confusion_rate,
                symbolic_advantage        = sym_adv,
                improvement_potential     = potential,
            ))
        return profiles

    def _generate_recommendations(
        self,
        profiles:   list[DiseaseDiscriminativeProfile],
        pair_seps:  list[FeaturePairSeparation],
        poor_pairs: list[tuple[str, str]],
    ) -> list[OptimizationRecommendation]:
        recs: list[OptimizationRecommendation] = []
        priority = 1

        # P1: Address poorly separated disease pairs
        for a, b in poor_pairs[:3]:
            recs.append(OptimizationRecommendation(
                priority             = priority,
                target_diseases      = [a, b],
                intervention_type    = "rule_weight",
                description          = (
                    f"Disease pair {a}/{b} lacks a strong separating clinical feature. "
                    f"Increase rule weights for auxiliary differentiating signs to "
                    f"widen the certainty gap between these competing hypotheses."
                ),
                expected_accuracy_gain_pp = 1.5,
                implementation_note  = (
                    "Apply DiseaseSignatureRefiner recommendations to adjust "
                    "rule confidence weights in the YAML rule definitions."
                ),
            ))
            priority += 1

        # P2: Address diseases with low gap adequacy
        for p in sorted(profiles, key=lambda x: x.gap_adequate_rate)[:2]:
            if p.gap_adequate_rate < 0.30:
                recs.append(OptimizationRecommendation(
                    priority             = priority,
                    target_diseases      = [p.disease],
                    intervention_type    = "certainty_threshold",
                    description          = (
                        f"{p.disease}: only {p.gap_adequate_rate:.0%} of cases produce "
                        f"an adequate certainty gap (>= 0.15). "
                        f"Strengthen differential competition weighting for "
                        f"the strongest separating features: "
                        f"{', '.join(p.top_separating_features[:2])}."
                    ),
                    expected_accuracy_gain_pp = 2.0,
                    implementation_note  = (
                        "Apply CompetitionSharpener.gap_gamma increase (2.0 -> 2.5) "
                        "for this disease. Recalibrate certainty_floor threshold."
                    ),
                ))
                priority += 1

        # P3: Signal enrichment for high-confusion diseases
        high_confusion = [p for p in profiles if p.confusion_rate > 0.25]
        if high_confusion:
            recs.append(OptimizationRecommendation(
                priority             = priority,
                target_diseases      = [p.disease for p in high_confusion[:3]],
                intervention_type    = "signal_enrichment",
                description          = (
                    "High-confusion diseases benefit from enriched trajectory and "
                    "competition signals. Certainty velocity, convergence stability, "
                    "and leadership persistence signals should be prioritised in "
                    "the Model C feature set."
                ),
                expected_accuracy_gain_pp = 3.0,
                implementation_note  = (
                    "SymbolicSignalEnricherV2 already generates these signals. "
                    "Ensure they receive appropriate weight in AdvancedBaselineCalibrator "
                    "feature selection."
                ),
            ))
            priority += 1

        # P4: Recalibrated ambiguity threshold for remaining escalated cases
        recs.append(OptimizationRecommendation(
            priority             = priority,
            target_diseases      = list({p.disease for p in profiles}),
            intervention_type    = "certainty_threshold",
            description          = (
                "Remaining pathological escalation prevents symbolic signals from "
                "contributing to Model C predictions. Progressive threshold relaxation "
                "(ambiguity ceiling 2.50 -> 3.00) for coherent low-contradiction cases "
                "will unlock additional safe-triage capacity."
            ),
            expected_accuracy_gain_pp = 2.5,
            implementation_note  = (
                "Apply ThresholdRecalibrator with ambiguity_ceiling=3.00 and "
                "certainty_floor=0.35 on training data with safety audit."
            ),
        ))

        return recs
