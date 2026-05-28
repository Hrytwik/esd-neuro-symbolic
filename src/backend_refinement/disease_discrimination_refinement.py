"""
disease_discrimination_refinement.py
=======================================
Disease discrimination refinement for the CASDRE clinical inference pipeline.

Focus areas:
  - chronic_dermatitis vs seborrheic_dermatitis (highest clinical overlap)
  - PRP rare-class stabilisation (n=20, easily misclassified)
  - inflammatory overlap resolution (psoriasis / chronic_dermatitis)
  - lichen_planus differentiation (koebner / polygonal papules signals)
  - psoriasis competition refinement (scaling + knee/elbow + scalp triad)

Produces:
  - pairwise separation scores
  - disease confusion matrices
  - symbolic separability matrices
  - discrimination strength profiles
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class SeparabilityTier(str, Enum):
    STRONG      = "strong"       # pairwise accuracy ≥ 0.92
    ADEQUATE    = "adequate"     # 0.80–0.92
    BORDERLINE  = "borderline"   # 0.68–0.80
    PROBLEMATIC = "problematic"  # < 0.68


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PairwiseSeparationScore:
    disease_a: str
    disease_b: str
    # Overall discrimination quality
    balanced_accuracy: float
    confusion_a_as_b: float     # P(predict B | true A)
    confusion_b_as_a: float     # P(predict A | true B)
    separability_tier: SeparabilityTier
    # Symbolic contribution
    symbolic_separation_gain: float   # Δ accuracy from symbolic features
    top_discriminating_features: List[str]
    # Clinical risk
    clinical_risk: str               # "low" / "medium" / "high"


@dataclass
class DiseaseConfusionProfile:
    """Confusion analysis for a single disease class."""
    disease: str
    n_cases: int
    true_positive_rate: float        # recall
    false_negative_rate: float
    primary_confusion_target: str    # most common misclassification
    primary_confusion_rate: float
    secondary_confusion_target: Optional[str]
    secondary_confusion_rate: float
    is_rare: bool                    # n_cases < 30
    rare_class_correction_needed: bool


@dataclass
class SymbolicSeparabilityMatrix:
    """
    NxN matrix of symbolic separability scores.
    Row i, col j = how well symbolic features separate disease i from disease j.
    """
    class_labels: List[str]
    matrix: np.ndarray          # shape (n_classes, n_classes)
    mean_off_diagonal: float
    min_separation: float
    min_separation_pair: Tuple[str, str]
    max_separation: float
    max_separation_pair: Tuple[str, str]


@dataclass
class DiscriminationStrengthProfile:
    """Per-disease discrimination strength summary."""
    disease: str
    n_cases: int
    mean_pairwise_separation: float   # mean off-diagonal separability
    hardest_competitor: str
    hardest_competition_score: float  # lower = harder
    symbolic_advantage: float         # symbolic - clinical separability
    rule_coverage: float              # fraction of cases with active rules
    priority_for_refinement: int      # 1 = highest priority


@dataclass
class DiseaseDiscriminationReport:
    """Full discrimination refinement report."""
    pairwise_scores: List[PairwiseSeparationScore]
    confusion_profiles: List[DiseaseConfusionProfile]
    separability_matrix: SymbolicSeparabilityMatrix
    discrimination_profiles: List[DiscriminationStrengthProfile]

    # High-priority pairs for refinement
    problematic_pairs: List[Tuple[str, str]]
    borderline_pairs: List[Tuple[str, str]]

    # Aggregate
    mean_pairwise_accuracy: float
    mean_symbolic_separation_gain: float

    recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "DISEASE DISCRIMINATION REFINEMENT REPORT",
            "=" * 70,
            f"  Mean pairwise accuracy         : {self.mean_pairwise_accuracy:.3f}",
            f"  Mean symbolic separation gain  : "
            f"{self.mean_symbolic_separation_gain:+.3f}",
            "",
            "  ── Pairwise Separation Scores ────────────────────────────────",
        ]
        for ps in sorted(self.pairwise_scores,
                         key=lambda p: p.balanced_accuracy):
            lines.append(
                f"    {ps.disease_a:<22s} vs {ps.disease_b:<22s}  "
                f"{ps.balanced_accuracy:.3f}  [{ps.separability_tier.value}]"
            )
        lines += [
            "",
            "  ── Disease Confusion Profiles ────────────────────────────────",
        ]
        for cp in self.confusion_profiles:
            rare_tag = " [RARE]" if cp.is_rare else ""
            lines.append(
                f"    {cp.disease:<32s}  "
                f"TPR={cp.true_positive_rate:.3f}  "
                f"confused-as={cp.primary_confusion_target}({cp.primary_confusion_rate:.1%})"
                f"{rare_tag}"
            )
        lines += [
            "",
            "  ── Symbolic Separability Matrix (mean off-diag) ──────────────",
            f"    Mean off-diagonal separation   : "
            f"{self.separability_matrix.mean_off_diagonal:.3f}",
            f"    Hardest pair                   : "
            f"{self.separability_matrix.min_separation_pair} = "
            f"{self.separability_matrix.min_separation:.3f}",
            f"    Easiest pair                   : "
            f"{self.separability_matrix.max_separation_pair} = "
            f"{self.separability_matrix.max_separation:.3f}",
            "",
            "  ── Prioritised Refinement Targets ────────────────────────────",
        ]
        if self.problematic_pairs:
            lines.append("    PROBLEMATIC pairs (immediate attention):")
            for a, b in self.problematic_pairs:
                lines.append(f"      ✗  {a} ↔ {b}")
        if self.borderline_pairs:
            lines.append("    BORDERLINE pairs (monitor closely):")
            for a, b in self.borderline_pairs:
                lines.append(f"      ⚠  {a} ↔ {b}")
        if self.recommendations:
            lines += ["", "  ── Recommendations ──────────────────────────────────────────"]
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_RARE_CLASS_THRESHOLD  = 30
_STRONG_SEP_THRESH     = 0.92
_ADEQUATE_SEP_THRESH   = 0.80
_BORDERLINE_SEP_THRESH = 0.68


def _separability_tier(acc: float) -> SeparabilityTier:
    if acc >= _STRONG_SEP_THRESH:
        return SeparabilityTier.STRONG
    elif acc >= _ADEQUATE_SEP_THRESH:
        return SeparabilityTier.ADEQUATE
    elif acc >= _BORDERLINE_SEP_THRESH:
        return SeparabilityTier.BORDERLINE
    return SeparabilityTier.PROBLEMATIC


def _clinical_risk(disease_a: str, disease_b: str) -> str:
    # High-risk confusions: rare-class or inflammatory overlap
    high_risk_pairs = {
        frozenset({"pityriasis_rubra_pilaris", "psoriasis"}),
        frozenset({"chronic_dermatitis", "seborrheic_dermatitis"}),
        frozenset({"lichen_planus", "psoriasis"}),
    }
    medium_risk_pairs = {
        frozenset({"pityriasis_rosea", "psoriasis"}),
        frozenset({"chronic_dermatitis", "pityriasis_rosea"}),
    }
    pair = frozenset({disease_a, disease_b})
    if pair in high_risk_pairs:
        return "high"
    if pair in medium_risk_pairs:
        return "medium"
    return "low"


# ──────────────────────────────────────────────────────────────────────────────
# Refiner
# ──────────────────────────────────────────────────────────────────────────────

class DiseaseDiscriminationRefiner:
    """
    Computes pairwise disease separation, confusion analysis, and symbolic
    separability matrices.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names.
    feature_names : list[str], optional
        Names of features in X.
    """

    def __init__(
        self,
        class_labels: List[str],
        feature_names: Optional[List[str]] = None,
    ):
        self.class_labels  = class_labels
        self.feature_names = feature_names or [f"feat_{i}" for i in range(40)]

    # ------------------------------------------------------------------
    def analyse(
        self,
        X: np.ndarray,          # (n, n_features) — clinical + symbolic
        y_true: np.ndarray,
        y_pred_b: np.ndarray,   # clinical-only predictions
        y_pred_c: np.ndarray,   # symbolic-augmented predictions
    ) -> DiseaseDiscriminationReport:
        """Run full discrimination refinement analysis."""

        pairwise = self._compute_pairwise_scores(y_true, y_pred_b, y_pred_c, X)
        confusion_profiles = self._compute_confusion_profiles(y_true, y_pred_b)
        sep_matrix = self._compute_separability_matrix(X, y_true)
        disc_profiles = self._compute_discrimination_profiles(
            pairwise, confusion_profiles
        )

        problematic = [
            (ps.disease_a, ps.disease_b)
            for ps in pairwise
            if ps.separability_tier == SeparabilityTier.PROBLEMATIC
        ]
        borderline = [
            (ps.disease_a, ps.disease_b)
            for ps in pairwise
            if ps.separability_tier == SeparabilityTier.BORDERLINE
        ]

        mean_pwa = statistics.mean(ps.balanced_accuracy for ps in pairwise) if pairwise else 0.0
        mean_sym_gain = statistics.mean(
            ps.symbolic_separation_gain for ps in pairwise
        ) if pairwise else 0.0

        recs = self._generate_recommendations(
            pairwise, confusion_profiles, disc_profiles, problematic
        )

        return DiseaseDiscriminationReport(
            pairwise_scores=pairwise,
            confusion_profiles=confusion_profiles,
            separability_matrix=sep_matrix,
            discrimination_profiles=disc_profiles,
            problematic_pairs=problematic,
            borderline_pairs=borderline,
            mean_pairwise_accuracy=mean_pwa,
            mean_symbolic_separation_gain=mean_sym_gain,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    def _compute_pairwise_scores(
        self,
        y_true: np.ndarray,
        y_pred_b: np.ndarray,
        y_pred_c: np.ndarray,
        X: np.ndarray,
    ) -> List[PairwiseSeparationScore]:
        scores: List[PairwiseSeparationScore] = []
        n_classes = len(self.class_labels)
        for ai in range(n_classes):
            for bi in range(ai + 1, n_classes):
                pair_mask = (y_true == ai) | (y_true == bi)
                if pair_mask.sum() < 4:
                    continue
                yt = y_true[pair_mask]
                yb = y_pred_b[pair_mask]
                yc = y_pred_c[pair_mask]

                # Balanced accuracy on this pair (treat as binary)
                yb_bin = (yb == bi).astype(int)
                yc_bin = (yc == bi).astype(int)
                yt_bin = (yt == bi).astype(int)

                n_a = int(np.sum(yt == ai))
                n_b = int(np.sum(yt == bi))

                # Pairwise recall for each class
                tpr_a = float(np.mean(yb[yt == ai] == ai)) if n_a > 0 else 0.0
                tpr_b = float(np.mean(yb[yt == bi] == bi)) if n_b > 0 else 0.0
                bal_b = (tpr_a + tpr_b) / 2.0

                tpr_a_c = float(np.mean(yc[yt == ai] == ai)) if n_a > 0 else 0.0
                tpr_b_c = float(np.mean(yc[yt == bi] == bi)) if n_b > 0 else 0.0
                bal_c = (tpr_a_c + tpr_b_c) / 2.0

                conf_a_as_b = 1.0 - tpr_a
                conf_b_as_a = 1.0 - tpr_b
                sym_gain    = bal_c - bal_b

                # Top features by Fisher ratio
                top_feats = self._top_discriminating_features(
                    X[pair_mask], yt, ai, bi, n=4
                )

                scores.append(PairwiseSeparationScore(
                    disease_a=self.class_labels[ai],
                    disease_b=self.class_labels[bi],
                    balanced_accuracy=bal_b,
                    confusion_a_as_b=conf_a_as_b,
                    confusion_b_as_a=conf_b_as_a,
                    separability_tier=_separability_tier(bal_b),
                    symbolic_separation_gain=sym_gain,
                    top_discriminating_features=top_feats,
                    clinical_risk=_clinical_risk(
                        self.class_labels[ai], self.class_labels[bi]
                    ),
                ))
        return scores

    def _top_discriminating_features(
        self,
        X_pair: np.ndarray,
        y_pair: np.ndarray,
        ai: int, bi: int,
        n: int = 4,
    ) -> List[str]:
        if X_pair.shape[1] == 0 or X_pair.shape[0] < 4:
            return []
        f_ratios = []
        for fi in range(X_pair.shape[1]):
            col = X_pair[:, fi].astype(float)
            a_vals = col[y_pair == ai]
            b_vals = col[y_pair == bi]
            if len(a_vals) < 2 or len(b_vals) < 2:
                f_ratios.append(0.0)
                continue
            ma, mb = np.mean(a_vals), np.mean(b_vals)
            va, vb = np.var(a_vals), np.var(b_vals)
            pool   = (va * len(a_vals) + vb * len(b_vals)) / (len(a_vals) + len(b_vals))
            fr     = (ma - mb) ** 2 / max(pool, 1e-9)
            f_ratios.append(float(fr))
        top_idx = sorted(range(len(f_ratios)), key=lambda i: -f_ratios[i])[:n]
        return [
            self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
            for i in top_idx
        ]

    def _compute_confusion_profiles(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> List[DiseaseConfusionProfile]:
        n_classes = len(self.class_labels)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        profiles: List[DiseaseConfusionProfile] = []
        for i, disease in enumerate(self.class_labels):
            n_cases = int(np.sum(y_true == i))
            if n_cases == 0:
                continue
            tp   = cm[i, i]
            row  = cm[i].copy()
            row[i] = 0

            tpr  = tp / n_cases
            fnr  = 1.0 - tpr

            primary_j    = int(np.argmax(row))
            primary_rate = float(row[primary_j]) / n_cases if n_cases > 0 else 0.0
            primary_name = self.class_labels[primary_j] if primary_j != i else "none"

            row[primary_j] = 0
            secondary_j    = int(np.argmax(row))
            secondary_rate = float(row[secondary_j]) / n_cases if n_cases > 0 else 0.0
            secondary_name = (
                self.class_labels[secondary_j]
                if secondary_rate > 0.0 and secondary_j != i
                else None
            )

            is_rare = n_cases < _RARE_CLASS_THRESHOLD
            profiles.append(DiseaseConfusionProfile(
                disease=disease,
                n_cases=n_cases,
                true_positive_rate=tpr,
                false_negative_rate=fnr,
                primary_confusion_target=primary_name,
                primary_confusion_rate=primary_rate,
                secondary_confusion_target=secondary_name,
                secondary_confusion_rate=secondary_rate,
                is_rare=is_rare,
                rare_class_correction_needed=(is_rare and tpr < 0.75),
            ))
        return profiles

    def _compute_separability_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> SymbolicSeparabilityMatrix:
        n_classes = len(self.class_labels)
        mat = np.zeros((n_classes, n_classes))
        for ai in range(n_classes):
            for bi in range(n_classes):
                if ai == bi:
                    mat[ai, bi] = 1.0
                    continue
                pair_mask = (y == ai) | (y == bi)
                if pair_mask.sum() < 4:
                    mat[ai, bi] = 0.5
                    continue
                X_p = X[pair_mask]
                y_p = y[pair_mask]
                # Fisher ratio proxy: mean column-wise separability
                f_ratios = []
                for fi in range(X_p.shape[1]):
                    col  = X_p[:, fi].astype(float)
                    a_v  = col[y_p == ai]
                    b_v  = col[y_p == bi]
                    if len(a_v) < 2 or len(b_v) < 2:
                        continue
                    ma, mb = np.mean(a_v), np.mean(b_v)
                    va, vb = np.var(a_v), np.var(b_v)
                    pool   = (va * len(a_v) + vb * len(b_v)) / (len(a_v) + len(b_v))
                    fr     = (ma - mb) ** 2 / max(pool, 1e-9)
                    f_ratios.append(min(fr / 20.0, 1.0))  # normalise to [0,1]
                mat[ai, bi] = float(np.mean(f_ratios)) if f_ratios else 0.0

        # Off-diagonal stats
        off_diag = mat[np.eye(n_classes, dtype=bool) == False]
        mean_off  = float(np.mean(off_diag))
        min_val   = float(np.min(off_diag))
        max_val   = float(np.max(off_diag))
        min_idx   = np.unravel_index(
            np.argmin(mat + np.eye(n_classes) * 999), mat.shape
        )
        max_idx   = np.unravel_index(
            np.argmax(mat - np.eye(n_classes) * 999), mat.shape
        )
        min_pair = (self.class_labels[min_idx[0]], self.class_labels[min_idx[1]])
        max_pair = (self.class_labels[max_idx[0]], self.class_labels[max_idx[1]])

        return SymbolicSeparabilityMatrix(
            class_labels=self.class_labels,
            matrix=mat,
            mean_off_diagonal=mean_off,
            min_separation=min_val,
            min_separation_pair=min_pair,
            max_separation=max_val,
            max_separation_pair=max_pair,
        )

    def _compute_discrimination_profiles(
        self,
        pairwise: List[PairwiseSeparationScore],
        confusion: List[DiseaseConfusionProfile],
    ) -> List[DiscriminationStrengthProfile]:
        profiles: List[DiscriminationStrengthProfile] = []
        for disease in self.class_labels:
            relevant = [
                ps for ps in pairwise
                if ps.disease_a == disease or ps.disease_b == disease
            ]
            if not relevant:
                continue
            mean_sep = statistics.mean(ps.balanced_accuracy for ps in relevant)
            hardest  = min(relevant, key=lambda ps: ps.balanced_accuracy)
            comp_score = hardest.balanced_accuracy
            comp_name  = (hardest.disease_b
                          if hardest.disease_a == disease
                          else hardest.disease_a)
            sym_adv   = statistics.mean(ps.symbolic_separation_gain for ps in relevant)

            n_cases = next((cp.n_cases for cp in confusion if cp.disease == disease), 0)
            rule_cov = min(1.0, mean_sep + 0.1)  # proxy

            # Priority: problematic > borderline > rare
            tier_score = sum(
                1 if ps.separability_tier == SeparabilityTier.PROBLEMATIC else
                0.5 if ps.separability_tier == SeparabilityTier.BORDERLINE else 0
                for ps in relevant
            )
            is_rare_priority = 2 if n_cases < _RARE_CLASS_THRESHOLD else 0
            priority = max(1, int(5 - tier_score - is_rare_priority))

            profiles.append(DiscriminationStrengthProfile(
                disease=disease,
                n_cases=n_cases,
                mean_pairwise_separation=mean_sep,
                hardest_competitor=comp_name,
                hardest_competition_score=comp_score,
                symbolic_advantage=sym_adv,
                rule_coverage=rule_cov,
                priority_for_refinement=priority,
            ))
        profiles.sort(key=lambda p: p.priority_for_refinement)
        return profiles

    @staticmethod
    def _generate_recommendations(
        pairwise: List[PairwiseSeparationScore],
        confusion: List[DiseaseConfusionProfile],
        disc_profiles: List[DiscriminationStrengthProfile],
        problematic: List[Tuple[str, str]],
    ) -> List[str]:
        recs: List[str] = []

        if problematic:
            pair_str = "; ".join(f"{a} ↔ {b}" for a, b in problematic[:2])
            recs.append(
                f"Problematic pair(s) [{pair_str}] need targeted symbolic "
                "rule additions — add disease-specific differential rules "
                "in symbolic_rule_refinement_v2."
            )

        # Rare-class correction
        rare_needed = [cp for cp in confusion if cp.rare_class_correction_needed]
        if rare_needed:
            names = ", ".join(cp.disease for cp in rare_needed)
            recs.append(
                f"Rare-class correction needed for: {names} — apply "
                "imbalance-aware weighting in rare_disease_refinement."
            )

        # Highest-risk confusion
        high_risk = [ps for ps in pairwise if ps.clinical_risk == "high"]
        if high_risk:
            worst = min(high_risk, key=lambda ps: ps.balanced_accuracy)
            recs.append(
                f"High-risk confusion '{worst.disease_a}/{worst.disease_b}' "
                f"at only {worst.balanced_accuracy:.1%} pairwise accuracy — "
                "treat any misclassification here as a biopsy-worthy event."
            )

        # Symbolic gain champion
        best_sym = max(pairwise, key=lambda ps: ps.symbolic_separation_gain,
                       default=None)
        if best_sym and best_sym.symbolic_separation_gain > 0.05:
            recs.append(
                f"Pair '{best_sym.disease_a}/{best_sym.disease_b}' benefits most "
                f"from symbolic reasoning (+{best_sym.symbolic_separation_gain:.3f}) — "
                "use as benchmark for symbolic rule effectiveness."
            )

        if not recs:
            recs.append("Discrimination profiles are within acceptable bounds.")
        return recs[:5]
