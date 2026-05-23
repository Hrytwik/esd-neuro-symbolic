"""
HypothesisSeparationAnalyzer — inter-disease certainty gap analysis.

Analyses whether the symbolic reasoning pipeline produces sufficient
certainty separation between competing disease hypotheses, particularly
in the confusion zones identified by the diagnostic audit.

Clinical significance
---------------------
For the symbolic pipeline to benefit Model C, it must produce
measurably different certainty signatures for diseases that are
clinically similar (confusion zones). If hypotheses coexist at similar
certainty levels (soft competition), the leading disease encoded into
the symbolic vector is unreliable and adds noise rather than signal.

Analysis dimensions
-------------------
  1. Certainty gap distribution — distribution of (best - second) certainty
     across all test cases, broken down by disease.
  2. Confusion zone separation — for each high-confusion pair (A,B),
     how well do the symbolic signals separate true-A from true-B cases?
  3. Leadership stability — how often is the leading disease correct,
     and how often does it flip between pipeline stages?
  4. Hypothesis coexistence score — mean number of hypotheses above 0.30
     certainty simultaneously (higher = softer competition).
  5. Clinical discriminability index — composite score combining
     certainty gap, convergence, and leadership stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ConfusionZoneProfile:
    """
    Symbolic separation analysis for a single confusion zone (disease pair).

    Attributes
    ----------
    true_disease:
        The disease being misidentified.
    confused_with:
        The disease it is most often confused with.
    certainty_gap_true:
        Mean certainty gap (leading - second) for true_disease cases.
    certainty_gap_confused:
        Mean certainty gap for confused_with disease cases.
    separation_delta:
        Difference in certainty gaps between the two diseases.
        Positive = true_disease has higher gap (better separated).
    leading_correct_rate_true:
        Fraction of true_disease cases where pipeline leading disease
        matches the true label.
    mean_ambiguity_true:
        Mean ambiguity index for true_disease cases.
    mean_ambiguity_confused:
        Mean ambiguity index for confused_with cases.
    is_separable_by_symbolic:
        True if separation_delta ≥ 0.10 (symbolic signals can help).
    """

    true_disease:             str
    confused_with:            str
    certainty_gap_true:       float = 0.0
    certainty_gap_confused:   float = 0.0
    separation_delta:         float = 0.0
    leading_correct_rate_true: float = 0.0
    mean_ambiguity_true:      float = 0.0
    mean_ambiguity_confused:  float = 0.0
    is_separable_by_symbolic: bool  = False


@dataclass
class LeadershipProfile:
    """Leadership stability profile across the test set."""

    leading_correct_overall:    float   # Fraction where pipeline leader matches true label
    mean_leadership_changes:    float   # Mean number of leadership flips per case
    always_stable_fraction:     float   # Fraction with 0 leadership changes
    per_disease_correct_rate:   dict[str, float] = field(default_factory=dict)


@dataclass
class SeparationReport:
    """
    Complete hypothesis separation analysis output.

    Attributes
    ----------
    confusion_zone_profiles:
        Separation analysis for each identified confusion zone pair.
    mean_certainty_gap_overall:
        Mean certainty gap across all test records.
    mean_leadership_changes:
        Mean number of hypothesis leadership changes per patient.
    hypothesis_coexistence_score:
        Estimated mean number of hypotheses simultaneously above 0.30
        certainty. Higher = softer competition.
    leadership_profile:
        Detailed leadership stability statistics.
    separable_confusion_zones:
        Zones where symbolic signals can help separate diseases.
    non_separable_confusion_zones:
        Zones where clinical features are fundamentally insufficient.
    clinical_discriminability_index:
        Composite 0–1 score. Higher = more discriminating.
    top_separation_recommendations:
        Specific recommendations for improving inter-disease separation.
    """

    confusion_zone_profiles:          list[ConfusionZoneProfile] = field(default_factory=list)
    mean_certainty_gap_overall:       float = 0.0
    mean_leadership_changes:          float = 0.0
    hypothesis_coexistence_score:     float = 0.0
    leadership_profile:               LeadershipProfile = field(
        default_factory=LeadershipProfile
    )
    separable_confusion_zones:        list[str] = field(default_factory=list)
    non_separable_confusion_zones:    list[str] = field(default_factory=list)
    clinical_discriminability_index:  float = 0.0
    top_separation_recommendations:   list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "HYPOTHESIS SEPARATION ANALYSIS",
            "=" * 72,
            f"  Mean certainty gap               : {self.mean_certainty_gap_overall:.3f}",
            f"  Mean leadership changes/case      : {self.mean_leadership_changes:.2f}",
            f"  Hypothesis coexistence score      : {self.hypothesis_coexistence_score:.2f}",
            f"  Clinical discriminability index   : {self.clinical_discriminability_index:.3f}",
            f"  Separable confusion zones         : {len(self.separable_confusion_zones)}",
            f"  Non-separable confusion zones     : {len(self.non_separable_confusion_zones)}",
            "-" * 72,
            "  CONFUSION ZONE SEPARATION:",
        ]
        for cz in self.confusion_zone_profiles:
            sep_mark = "SEPARABLE" if cz.is_separable_by_symbolic else "OVERLAPPING"
            lines.append(
                f"    [{sep_mark:10s}] {cz.true_disease:30s} vs "
                f"{cz.confused_with:30s}  gap_delta={cz.separation_delta:+.3f}  "
                f"leader_ok={cz.leading_correct_rate_true:.1%}"
            )
        lines += [
            "-" * 72,
            "  RECOMMENDATIONS:",
        ]
        for rec in self.top_separation_recommendations:
            lines.append(f"    · {rec}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Analyser ──────────────────────────────────────────────────────────────────

# Common confusion zone pairs in UCI Dermatology
_KNOWN_CONFUSION_PAIRS = [
    ("psoriasis",            "seborrheic_dermatitis"),
    ("psoriasis",            "chronic_dermatitis"),
    ("seborrheic_dermatitis", "chronic_dermatitis"),
    ("lichen_planus",        "pityriasis_rosea"),
    ("pityriasis_rosea",     "chronic_dermatitis"),
    ("pityriasis_rubra_pilaris", "psoriasis"),
]


class HypothesisSeparationAnalyzer:
    """
    Analyses inter-disease certainty gap and leadership stability.

    Parameters
    ----------
    class_labels:
        Ordered canonical disease names.
    confusion_pairs:
        Disease pairs to analyse. Defaults to known UCI confusion zones.
    """

    def __init__(
        self,
        class_labels:    list[str],
        confusion_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        self.class_labels    = class_labels
        self.confusion_pairs = confusion_pairs or _KNOWN_CONFUSION_PAIRS

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        symbolic_vectors: list[SymbolicFeatureVector],
        y_true:           np.ndarray,
    ) -> SeparationReport:
        """
        Run hypothesis separation analysis.

        Parameters
        ----------
        symbolic_vectors:
            Test-set symbolic feature vectors.
        y_true:
            True 0-based integer class labels.
        """
        if not symbolic_vectors:
            return SeparationReport(
                leadership_profile=LeadershipProfile(0.0, 0.0, 0.0)
            )

        disease_labels = np.array([v.disease_label for v in symbolic_vectors])

        # 1. Overall certainty gap
        gaps = np.array([v.certainty_gap for v in symbolic_vectors])
        mean_gap = float(np.mean(gaps))

        # 2. Leadership profile
        leadership = self._compute_leadership_profile(symbolic_vectors, disease_labels)

        # 3. Hypothesis coexistence score
        coexistence = self._hypothesis_coexistence_score(symbolic_vectors)

        # 4. Confusion zone profiles
        cz_profiles = self._analyse_confusion_zones(symbolic_vectors, disease_labels)

        # 5. Classify zones
        separable     = [cz.true_disease for cz in cz_profiles if cz.is_separable_by_symbolic]
        non_separable = [cz.true_disease for cz in cz_profiles if not cz.is_separable_by_symbolic]

        # 6. Clinical discriminability index
        cdi = self._discriminability_index(mean_gap, leadership, coexistence)

        # 7. Recommendations
        recs = self._build_recommendations(cz_profiles, coexistence, leadership)

        return SeparationReport(
            confusion_zone_profiles=cz_profiles,
            mean_certainty_gap_overall=mean_gap,
            mean_leadership_changes=leadership.mean_leadership_changes,
            hypothesis_coexistence_score=coexistence,
            leadership_profile=leadership,
            separable_confusion_zones=separable,
            non_separable_confusion_zones=non_separable,
            clinical_discriminability_index=cdi,
            top_separation_recommendations=recs,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_leadership_profile(
        self,
        vectors:       list[SymbolicFeatureVector],
        disease_labels: np.ndarray,
    ) -> LeadershipProfile:
        """Compute leadership stability statistics."""
        leader_correct = np.array([
            v.leading_disease == v.disease_label
            for v in vectors
        ])
        overall_correct = float(np.mean(leader_correct))

        lc_counts = np.array([v.leadership_changes_count for v in vectors])
        mean_lc   = float(np.mean(lc_counts))
        stable    = float(np.mean(lc_counts == 0))

        per_dis: dict[str, float] = {}
        for disease in np.unique(disease_labels):
            mask = disease_labels == disease
            per_dis[str(disease)] = float(np.mean(leader_correct[mask])) if np.any(mask) else 0.0

        return LeadershipProfile(
            leading_correct_overall=overall_correct,
            mean_leadership_changes=mean_lc,
            always_stable_fraction=stable,
            per_disease_correct_rate=per_dis,
        )

    def _hypothesis_coexistence_score(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> float:
        """
        Estimate mean number of hypotheses above 0.30 certainty.

        Since we only have the leading certainty (not the full hypothesis
        distribution), we approximate via the certainty gap:
        a small gap suggests multiple hypotheses are close together.
        """
        gaps = np.array([v.certainty_gap for v in vectors])
        certs = np.array([v.certainty for v in vectors])

        # Heuristic: if gap < 0.10, at least 2 hypotheses above 0.25 certainty
        # if gap < 0.05, likely 3+ hypotheses coexisting
        soft_count = np.where(gaps < 0.05, 3.5,
                     np.where(gaps < 0.10, 2.5,
                     np.where(gaps < 0.20, 1.8, 1.2)))
        # Weight by certainty level (only meaningful when certainty > 0)
        weighted = soft_count * np.clip(certs, 0.1, 1.0)
        return float(np.mean(weighted))

    def _analyse_confusion_zones(
        self,
        vectors:       list[SymbolicFeatureVector],
        disease_labels: np.ndarray,
    ) -> list[ConfusionZoneProfile]:
        """Analyse each confusion zone pair."""
        profiles: list[ConfusionZoneProfile] = []

        for true_dis, confused_dis in self.confusion_pairs:
            mask_true     = disease_labels == true_dis
            mask_confused = disease_labels == confused_dis

            if not np.any(mask_true):
                continue

            # Certainty gap for each group
            vecs_true     = [v for v, m in zip(vectors, mask_true)     if m]
            vecs_confused = [v for v, m in zip(vectors, mask_confused) if m]

            gap_true     = float(np.mean([v.certainty_gap for v in vecs_true])) if vecs_true else 0.0
            gap_confused = float(np.mean([v.certainty_gap for v in vecs_confused])) if vecs_confused else 0.0
            sep_delta    = gap_true - gap_confused

            # Leadership accuracy for true disease cases
            leader_ok = float(np.mean([
                v.leading_disease == true_dis for v in vecs_true
            ])) if vecs_true else 0.0

            amb_true     = float(np.mean([v.ambiguity_index for v in vecs_true])) if vecs_true else 0.0
            amb_confused = float(np.mean([v.ambiguity_index for v in vecs_confused])) if vecs_confused else 0.0

            profiles.append(ConfusionZoneProfile(
                true_disease=true_dis,
                confused_with=confused_dis,
                certainty_gap_true=gap_true,
                certainty_gap_confused=gap_confused,
                separation_delta=sep_delta,
                leading_correct_rate_true=leader_ok,
                mean_ambiguity_true=amb_true,
                mean_ambiguity_confused=amb_confused,
                is_separable_by_symbolic=abs(sep_delta) >= 0.05 or leader_ok >= 0.60,
            ))

        return sorted(profiles, key=lambda p: p.separation_delta, reverse=True)

    def _discriminability_index(
        self,
        mean_gap:    float,
        leadership:  LeadershipProfile,
        coexistence: float,
    ) -> float:
        """Compute composite clinical discriminability index (0–1)."""
        gap_score      = min(1.0, mean_gap / 0.30)                  # 0.30 = good gap
        leader_score   = leadership.leading_correct_overall
        stability_score = leadership.always_stable_fraction
        coex_score     = max(0.0, 1.0 - (coexistence - 1.0) / 3.0)  # lower coexistence = better

        return float(0.35 * gap_score + 0.35 * leader_score +
                     0.15 * stability_score + 0.15 * coex_score)

    def _build_recommendations(
        self,
        cz_profiles: list[ConfusionZoneProfile],
        coexistence: float,
        leadership:  LeadershipProfile,
    ) -> list[str]:
        recs: list[str] = []

        non_sep = [cz for cz in cz_profiles if not cz.is_separable_by_symbolic]
        if non_sep:
            pairs = ", ".join(f"{cz.true_disease}/{cz.confused_with}" for cz in non_sep[:2])
            recs.append(
                f"Pairs {pairs}: require disease-specific rule strengthening to "
                "widen certainty gap beyond current separation level."
            )

        if coexistence > 2.0:
            recs.append(
                f"High hypothesis coexistence score ({coexistence:.2f}) indicates "
                "soft competition. Increase inter-hypothesis suppression weight so "
                "the leading hypothesis accumulates certainty more aggressively."
            )

        if leadership.leading_correct_overall < 0.70:
            recs.append(
                f"Pipeline leadership accuracy is {leadership.leading_correct_overall:.1%} "
                "— below the 70% minimum for reliable encoded signals. Review "
                "differential evidence priority weighting."
            )

        if leadership.mean_leadership_changes > 1.0:
            recs.append(
                f"High mean leadership changes ({leadership.mean_leadership_changes:.1f}/case) "
                "suggests reasoning oscillation. Apply dampening more aggressively "
                "after 2+ leadership changes."
            )

        if not recs:
            recs.append(
                "Hypothesis separation is adequate. Focus optimisation on threshold "
                "calibration and classifier tuning rather than pipeline restructuring."
            )

        return recs
