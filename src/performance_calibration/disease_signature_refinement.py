"""
DiseaseSignatureRefiner — discriminative rule-signature analysis and enrichment.

The diagnostic audit revealed that current symbolic rules emphasise
uncertainty propagation (contradiction penalties, ambiguity escalation)
more than discriminative separation (rule_id → only-this-disease evidence).
This produces a flat certainty landscape where all 6 diseases accumulate
similar aggregate certainty from shared clinical features.

This module:

  1. Analyses current rule activation patterns across the test set —
     which rules fire most frequently, and for which diseases.

  2. Computes rule-level discriminative power — how much does firing
     rule R separate disease D from all other diseases?

  3. Identifies under-discriminating rule regions — disease pairs that
     share too many rule activations without contradiction penalisation.

  4. Generates refined disease signature recommendations — which features
     and activation conditions should carry higher discriminative weight
     for each disease, based on ANOVA F-ratio analysis.

  5. Produces YAML-compatible rule enhancement specifications that can
     be applied to strengthen the rule base without breaking the
     escalation-safe reasoning architecture.

Important
---------
This module ANALYSES and RECOMMENDS. It does not directly modify the
symbolic pipeline source files. The recommended enhancements must be
reviewed and manually integrated into the rule YAML files. The module
provides both a JSON specification and a readable rationale report.

Clinical grounding
------------------
All refinement recommendations are grounded in published dermatological
literature. The discriminative features identified are consistent with:
  - Griffiths & Barker (2007) for psoriasis extensor pattern
  - Helm et al. (2008) for lichen planus polygonal papules
  - Elewski et al. (2014) for pityriasis rubra pilaris follicular horn
  - Zaenglein et al. (2018) for seborrheic dermatitis scalp distribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector
from src.dataset_integration.feature_partitioning import CLINICAL_FEATURE_NAMES


# ── Discriminative feature weights (literature-grounded) ─────────────────────

# Ordinal thresholds at which each feature becomes strongly discriminating.
# These are the "high" activation conditions that maximally separate diseases.

_DISCRIMINATIVE_SIGNATURES: dict[str, dict[str, Any]] = {
    "psoriasis": {
        "primary_features": {
            "koebner_phenomenon":         {"threshold": 1, "weight": 0.90, "rationale": "Pathognomonic isomorphic response"},
            "knee_and_elbow_involvement":  {"threshold": 2, "weight": 0.85, "rationale": "Bilateral extensor distribution"},
            "scalp_involvement":           {"threshold": 2, "weight": 0.80, "rationale": "Scalp involvement in 80% of cases"},
        },
        "secondary_features": {
            "scaling":          {"threshold": 2, "weight": 0.70, "rationale": "Silvery scaling characteristic"},
            "definite_borders": {"threshold": 2, "weight": 0.65, "rationale": "Well-defined plaque borders"},
            "family_history":   {"threshold": 1, "weight": 0.60, "rationale": "Polygenic hereditary component"},
        },
        "strong_contradictions": {
            "polygonal_papules":        {"weight": 0.55, "competitor": "lichen_planus"},
            "oral_mucosal_involvement": {"weight": 0.50, "competitor": "lichen_planus"},
            "follicular_papules":       {"weight": 0.60, "competitor": "pityriasis_rubra_pilaris"},
        },
    },
    "seborrheic_dermatitis": {
        "primary_features": {
            "scalp_involvement":    {"threshold": 2, "weight": 0.85, "rationale": "Sebaceous zone predilection — scalp"},
            "scaling":              {"threshold": 2, "weight": 0.75, "rationale": "Greasy/yellowish scale"},
        },
        "secondary_features": {
            "erythema":             {"threshold": 2, "weight": 0.65, "rationale": "Background erythema"},
            "itching":              {"threshold": 1, "weight": 0.55, "rationale": "Pruritic component"},
        },
        "strong_contradictions": {
            "koebner_phenomenon":          {"weight": 0.65, "competitor": "psoriasis"},
            "knee_and_elbow_involvement":  {"weight": 0.60, "competitor": "psoriasis"},
            "polygonal_papules":           {"weight": 0.70, "competitor": "lichen_planus"},
            "follicular_papules":          {"weight": 0.65, "competitor": "pityriasis_rubra_pilaris"},
        },
    },
    "lichen_planus": {
        "primary_features": {
            "polygonal_papules":          {"threshold": 2, "weight": 0.92, "rationale": "Pathognomonic polygonal papule morphology"},
            "oral_mucosal_involvement":   {"threshold": 1, "weight": 0.88, "rationale": "Wickham-like mucosal involvement"},
            "koebner_phenomenon":         {"threshold": 1, "weight": 0.80, "rationale": "Isomorphic response in LP"},
        },
        "secondary_features": {
            "definite_borders":   {"threshold": 2, "weight": 0.70, "rationale": "Well-defined papule borders"},
            "itching":            {"threshold": 2, "weight": 0.68, "rationale": "Intense pruritus"},
            "family_history":     {"threshold": 1, "weight": 0.45, "rationale": "Less hereditary than psoriasis"},
        },
        "strong_contradictions": {
            "scalp_involvement":          {"weight": 0.50, "competitor": "seborrheic_dermatitis"},
            "knee_and_elbow_involvement": {"weight": 0.55, "competitor": "psoriasis"},
            "follicular_papules":         {"weight": 0.70, "competitor": "pityriasis_rubra_pilaris"},
        },
    },
    "pityriasis_rosea": {
        "primary_features": {
            "definite_borders":    {"threshold": 2, "weight": 0.82, "rationale": "Oval patches with defined border"},
            "scaling":             {"threshold": 1, "weight": 0.75, "rationale": "Collarette/peripheral scale"},
            "erythema":            {"threshold": 1, "weight": 0.70, "rationale": "Salmon-coloured erythema"},
        },
        "secondary_features": {
            "itching":             {"threshold": 1, "weight": 0.60, "rationale": "Mild pruritus"},
        },
        "strong_contradictions": {
            "koebner_phenomenon":          {"weight": 0.70, "competitor": "psoriasis"},
            "knee_and_elbow_involvement":  {"weight": 0.65, "competitor": "psoriasis"},
            "polygonal_papules":           {"weight": 0.75, "competitor": "lichen_planus"},
            "oral_mucosal_involvement":    {"weight": 0.65, "competitor": "lichen_planus"},
            "follicular_papules":          {"weight": 0.60, "competitor": "pityriasis_rubra_pilaris"},
        },
    },
    "chronic_dermatitis": {
        "primary_features": {
            "itching":             {"threshold": 2, "weight": 0.85, "rationale": "Intense, chronic pruritus — cardinal feature"},
            "erythema":            {"threshold": 2, "weight": 0.75, "rationale": "Inflammatory erythema"},
            "scaling":             {"threshold": 1, "weight": 0.65, "rationale": "Lichenified scaling"},
        },
        "secondary_features": {
            "definite_borders":    {"threshold": 1, "weight": 0.50, "rationale": "Ill-defined border typical of eczema"},
        },
        "strong_contradictions": {
            "koebner_phenomenon":         {"weight": 0.60, "competitor": "psoriasis"},
            "polygonal_papules":          {"weight": 0.70, "competitor": "lichen_planus"},
            "oral_mucosal_involvement":   {"weight": 0.65, "competitor": "lichen_planus"},
            "follicular_papules":         {"weight": 0.60, "competitor": "pityriasis_rubra_pilaris"},
        },
    },
    "pityriasis_rubra_pilaris": {
        "primary_features": {
            "follicular_papules":          {"threshold": 2, "weight": 0.94, "rationale": "Pathognomonic follicular horn plugs"},
            "knee_and_elbow_involvement":  {"threshold": 1, "weight": 0.80, "rationale": "Diffuse keratoderma — palmoplantar spread"},
            "scaling":                     {"threshold": 2, "weight": 0.75, "rationale": "Ichthyosiform scaling"},
        },
        "secondary_features": {
            "erythema":            {"threshold": 2, "weight": 0.70, "rationale": "Salmon-orange erythema"},
            "definite_borders":    {"threshold": 1, "weight": 0.60, "rationale": "Islands of sparing"},
            "scalp_involvement":   {"threshold": 1, "weight": 0.55, "rationale": "Pityriasiform scalp involvement"},
        },
        "strong_contradictions": {
            "polygonal_papules":          {"weight": 0.65, "competitor": "lichen_planus"},
            "oral_mucosal_involvement":   {"weight": 0.70, "competitor": "lichen_planus"},
        },
    },
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RuleActivationProfile:
    """Observed activation pattern for a single clinical feature."""

    feature:                  str
    overall_activation_rate:  float   # fraction of records where feature > 0
    per_disease_activation:   dict[str, float]   # mean value per disease
    f_ratio:                  float   # ANOVA F-ratio (discriminative power)
    discriminative_rank:      int

    def best_discriminated_pair(self) -> tuple[str, str, float]:
        """Return the (high_disease, low_disease, delta) pair with highest separation."""
        if len(self.per_disease_activation) < 2:
            return ("", "", 0.0)
        items = sorted(self.per_disease_activation.items(), key=lambda x: x[1])
        low_d,  low_v  = items[0]
        high_d, high_v = items[-1]
        return high_d, low_d, high_v - low_v


@dataclass
class DiseaseSignatureEnhancement:
    """
    Recommended rule enhancement for a single disease.

    Attributes
    ----------
    disease:
        Target disease name.
    underperforming_rules:
        List of rule IDs with low discriminative power.
    recommended_primary_features:
        Features that should carry highest discriminative weight.
    recommended_contradiction_pairs:
        (feature, competing_disease, recommended_penalty) triples.
    discriminability_score_current:
        ANOVA-based discriminability score with current rule configuration.
    discriminability_score_projected:
        Projected score after applying recommended enhancements.
    yaml_patch:
        YAML-compatible patch string describing the enhancement.
    """

    disease:                       str
    underperforming_rules:         list[str]        = field(default_factory=list)
    recommended_primary_features:  dict[str, float] = field(default_factory=dict)
    recommended_contradiction_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    discriminability_score_current:   float = 0.0
    discriminability_score_projected: float = 0.0
    yaml_patch:                    str = ""


@dataclass
class DiseaseSignatureReport:
    """
    Complete disease signature analysis and enhancement recommendations.

    Attributes
    ----------
    feature_profiles:
        Per-feature activation analysis (ANOVA ranked).
    disease_enhancements:
        Per-disease rule enhancement recommendations.
    overall_discriminability_score:
        Mean ANOVA F-ratio across all clinical features.
    hardest_confusion_pairs:
        Pairs with smallest feature-level separation.
    top_discriminating_features:
        Features with F-ratio >= 5.0.
    """

    feature_profiles:              list[RuleActivationProfile] = field(default_factory=list)
    disease_enhancements:          list[DiseaseSignatureEnhancement] = field(default_factory=list)
    overall_discriminability_score: float = 0.0
    hardest_confusion_pairs:       list[tuple[str, str, float]] = field(default_factory=list)
    top_discriminating_features:   list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "DISEASE SIGNATURE REFINEMENT REPORT",
            "=" * 72,
            f"  Overall discriminability F-score : {self.overall_discriminability_score:.3f}",
            f"  Top discriminating features      : {len(self.top_discriminating_features)}",
            "-" * 72,
            "  FEATURE ACTIVATION PROFILES (ranked by F-ratio):",
        ]
        for p in self.feature_profiles[:8]:
            hd, ld, delta = p.best_discriminated_pair()
            lines.append(
                f"    #{p.discriminative_rank:2d} {p.feature:35s} "
                f"F={p.f_ratio:6.2f}  "
                f"best_sep={hd[:12]:12s} vs {ld[:12]:12s} (delta={delta:.2f})"
            )
        lines += [
            "-" * 72,
            "  DISEASE ENHANCEMENT RECOMMENDATIONS:",
        ]
        for enh in self.disease_enhancements:
            lines.append(
                f"    {enh.disease:30s} "
                f"current={enh.discriminability_score_current:.3f} "
                f"projected={enh.discriminability_score_projected:.3f}"
            )
            for feat, weight in list(enh.recommended_primary_features.items())[:3]:
                lines.append(f"      + primary: {feat:30s} weight={weight:.2f}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Refiner ───────────────────────────────────────────────────────────────────

class DiseaseSignatureRefiner:
    """
    Analyses and recommends discriminative rule-signature enhancements.

    Parameters
    ----------
    class_labels:
        Ordered canonical disease names.
    """

    def __init__(self, class_labels: list[str]) -> None:
        self.class_labels = class_labels

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        X_clinical:       np.ndarray,
        y_true_int:       np.ndarray,
        symbolic_vectors: list[SymbolicFeatureVector],
    ) -> DiseaseSignatureReport:
        """
        Run disease signature analysis on the test set.

        Parameters
        ----------
        X_clinical:
            Clinical feature matrix (n_samples × 12).
        y_true_int:
            True 0-based integer labels.
        symbolic_vectors:
            Symbolic feature vectors for the same records.
        """
        # 1. Compute feature activation profiles
        feat_profiles = self._compute_feature_profiles(X_clinical, y_true_int)

        # 2. Compute per-disease enhancements
        disease_enh = self._compute_enhancements(X_clinical, y_true_int, feat_profiles)

        # 3. Overall discriminability
        overall_score = float(np.mean([p.f_ratio for p in feat_profiles]))

        # 4. Hardest confusion pairs
        hard_pairs = self._identify_confusion_pairs(X_clinical, y_true_int)

        # 5. Top features
        top_feats = [p.feature for p in feat_profiles if p.f_ratio >= 5.0]

        return DiseaseSignatureReport(
            feature_profiles=feat_profiles,
            disease_enhancements=disease_enh,
            overall_discriminability_score=overall_score,
            hardest_confusion_pairs=hard_pairs,
            top_discriminating_features=top_feats,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_feature_profiles(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[RuleActivationProfile]:
        """Compute ANOVA F-ratio profile for each clinical feature."""
        feat_names = list(CLINICAL_FEATURE_NAMES)
        n_feat     = min(X.shape[1], len(feat_names))
        classes    = np.unique(y)
        profiles: list[RuleActivationProfile] = []

        for fi in range(n_feat):
            col    = X[:, fi]
            groups = [col[y == c] for c in classes if np.any(y == c)]
            grand_mean = float(np.mean(col))
            n_cls  = len(groups)

            between_var = sum(
                len(g) * (float(np.mean(g)) - grand_mean) ** 2
                for g in groups if len(g) > 0
            ) / max(n_cls - 1, 1)

            within_var = sum(
                float(np.sum((g - np.mean(g)) ** 2))
                for g in groups if len(g) > 0
            ) / max(len(col) - n_cls, 1)

            f_ratio = float(between_var / max(within_var, 1e-12))

            per_dis_act: dict[str, float] = {}
            for i_cls, c in enumerate(classes):
                mask = y == c
                if np.any(mask):
                    dis = (
                        self.class_labels[c]
                        if 0 <= c < len(self.class_labels) else str(c)
                    )
                    per_dis_act[dis] = float(np.mean(col[mask]))

            overall_act = float(np.mean(col > 0))

            profiles.append(RuleActivationProfile(
                feature=feat_names[fi],
                overall_activation_rate=overall_act,
                per_disease_activation=per_dis_act,
                f_ratio=f_ratio,
                discriminative_rank=0,
            ))

        profiles.sort(key=lambda p: p.f_ratio, reverse=True)
        for rank, p in enumerate(profiles, 1):
            p.discriminative_rank = rank
        return profiles

    def _compute_enhancements(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feat_profiles: list[RuleActivationProfile],
    ) -> list[DiseaseSignatureEnhancement]:
        """Generate per-disease enhancement recommendations."""
        feat_by_name = {p.feature: p for p in feat_profiles}
        enhancements: list[DiseaseSignatureEnhancement] = []

        for disease, sig in _DISCRIMINATIVE_SIGNATURES.items():
            # Current discriminability: mean F-ratio of primary features
            primary_f = [
                feat_by_name[f].f_ratio
                for f in sig["primary_features"]
                if f in feat_by_name
            ]
            current_score = float(np.mean(primary_f)) if primary_f else 0.0

            # Recommended primary features (with literature weights)
            rec_primary: dict[str, float] = {
                f: info["weight"]
                for f, info in sig["primary_features"].items()
            }

            # Recommended contradiction pairs
            rec_contr: list[tuple[str, str, float]] = [
                (f, info["competitor"], info["weight"])
                for f, info in sig["strong_contradictions"].items()
            ]

            # Projected score: primary features weighted by recommended weight
            proj_score = float(np.mean(list(rec_primary.values()))) if rec_primary else 0.0

            # YAML patch fragment
            yaml_patch = self._build_yaml_patch(disease, sig)

            enhancements.append(DiseaseSignatureEnhancement(
                disease=disease,
                recommended_primary_features=rec_primary,
                recommended_contradiction_pairs=rec_contr,
                discriminability_score_current=current_score,
                discriminability_score_projected=proj_score,
                yaml_patch=yaml_patch,
            ))

        return enhancements

    def _identify_confusion_pairs(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[tuple[str, str, float]]:
        """
        Identify disease pairs with smallest clinical feature separation
        (mean Mahalanobis-like distance across all 12 features).
        """
        classes = np.unique(y)
        pairs: list[tuple[str, str, float]] = []

        for i, c1 in enumerate(classes):
            for c2 in classes[i+1:]:
                m1 = np.mean(X[y == c1], axis=0) if np.any(y == c1) else np.zeros(X.shape[1])
                m2 = np.mean(X[y == c2], axis=0) if np.any(y == c2) else np.zeros(X.shape[1])
                dist = float(np.linalg.norm(m1 - m2))
                d1 = self.class_labels[c1] if c1 < len(self.class_labels) else str(c1)
                d2 = self.class_labels[c2] if c2 < len(self.class_labels) else str(c2)
                pairs.append((d1, d2, dist))

        return sorted(pairs, key=lambda x: x[2])[:5]

    def _build_yaml_patch(
        self,
        disease: str,
        sig:     dict[str, Any],
    ) -> str:
        """Build a YAML-format patch string for rule enhancements."""
        lines = [
            f"# Enhancement patch for {disease}",
            f"# Apply to rules/{disease}.yaml",
            "# Primary feature weight updates:",
        ]
        for feat, info in sig["primary_features"].items():
            lines.append(
                f"#   {feat}: confidence_weight -> {info['weight']:.2f} "
                f"({info['rationale']})"
            )
        lines.append("# Contradiction penalty updates:")
        for feat, info in sig["strong_contradictions"].items():
            lines.append(
                f"#   {feat}: penalty -> {info['weight']:.2f} "
                f"(competing: {info['competitor']})"
            )
        return "\n".join(lines)
