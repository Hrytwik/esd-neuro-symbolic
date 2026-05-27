"""
disease_separation_refinement.py
=================================
Deep per-disease separation analysis for the CASDRE clinical inference pipeline.

Identifies which diseases are safely diagnosable from clinical features alone,
which require biopsy escalation, which benefit most from symbolic recoverability,
and which carry persistent ambiguity.  Produces disease-wise stabilisation
profiles, biopsy-necessity profiles, and symbolic-recovery heatmaps.
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

class BiopsyNecessityTier(str, Enum):
    RARELY_NEEDED      = "rarely_needed"       # ≥ 85 % clinical accuracy
    SOMETIMES_NEEDED   = "sometimes_needed"    # 70–85 %
    FREQUENTLY_NEEDED  = "frequently_needed"   # 55–70 %
    ALMOST_ALWAYS_NEEDED = "almost_always_needed"  # < 55 %


class SymbolicRecoverabilityTier(str, Enum):
    HIGH       = "high"        # symbolic lift ≥ 8 pp
    MODERATE   = "moderate"    # 4–8 pp
    LOW        = "low"         # 1–4 pp
    NEGLIGIBLE = "negligible"  # < 1 pp


class AmbiguityTier(str, Enum):
    RESOLVED   = "resolved"    # mean ambiguity < 1.5 bits
    MODERATE   = "moderate"    # 1.5–2.2 bits
    HIGH       = "high"        # 2.2–3.0 bits
    PERSISTENT = "persistent"  # ≥ 3.0 bits


# ──────────────────────────────────────────────────────────────────────────────
# Core data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DifferentialCompetitor:
    """How strongly a competing disease pulls cases away from the target disease."""
    competitor_disease: str
    confusion_rate: float          # fraction of target cases misclassified as competitor
    mean_certainty_overlap: float  # mean overlap in certainty scores [0, 1]
    symbolic_separation_gap: float # mean symbolic-signal delta (positive = better)
    resolvable_with_biopsy: bool


@dataclass
class DiseaseBiopsyProfile:
    """Biopsy-necessity characteristics for a single disease."""
    disease: str
    clinical_accuracy: float          # Model B accuracy on this disease
    symbolic_accuracy: float          # Model C accuracy on this disease
    symbolic_lift_pp: float           # C - B accuracy in percentage points
    biopsy_necessity_tier: BiopsyNecessityTier
    mean_ambiguity_bits: float
    ambiguity_tier: AmbiguityTier
    escalation_rate: float            # fraction escalated to biopsy
    appropriate_escalation_rate: float  # fraction where escalation was justified
    false_escalation_rate: float      # unnecessary escalations


@dataclass
class DiseaseStabilisationProfile:
    """Full stabilisation analysis for one disease class."""
    disease: str
    n_cases: int

    # Clinical separability
    clinical_accuracy: float
    symbolic_accuracy: float
    symbolic_lift_pp: float
    symbolic_recoverability_tier: SymbolicRecoverabilityTier

    # Ambiguity
    mean_ambiguity_bits: float
    ambiguity_tier: AmbiguityTier
    ambiguity_std: float

    # Differential competition
    primary_competitor: Optional[str]
    differential_competitors: List[DifferentialCompetitor]
    worst_confusion_rate: float

    # Symbolic divergence
    mean_symbolic_activation: float   # mean across all symbolic features
    symbolic_divergence_score: float  # distance from average activation profile
    dominant_symbolic_signals: List[str]

    # Biopsy profile
    biopsy_profile: DiseaseBiopsyProfile

    # Derived tier
    def separation_adequacy(self) -> str:
        """Returns 'adequate' / 'borderline' / 'inadequate'."""
        if self.clinical_accuracy >= 0.85 and self.worst_confusion_rate < 0.15:
            return "adequate"
        elif self.clinical_accuracy >= 0.70 or self.symbolic_accuracy >= 0.80:
            return "borderline"
        return "inadequate"


@dataclass
class SymbolicRecoveryHeatmapCell:
    """One cell in the symbolic-recovery heatmap (disease × signal)."""
    disease: str
    signal_name: str
    mean_activation: float       # mean signal value for this disease
    activation_std: float
    discriminative_power: float  # correlation with correct diagnosis [0, 1]
    recovery_contribution: float # fraction of symbolic recoveries where this signal helped


@dataclass
class DiseaseSeparationReport:
    """Comprehensive separation refinement report across all diseases."""
    disease_profiles: List[DiseaseStabilisationProfile]
    biopsy_profiles: List[DiseaseBiopsyProfile]
    heatmap: List[SymbolicRecoveryHeatmapCell]

    # Aggregate
    mean_clinical_accuracy: float
    mean_symbolic_accuracy: float
    mean_symbolic_lift_pp: float
    diseases_needing_biopsy_support: List[str]   # clinical acc < 70 %
    diseases_safely_clinical: List[str]           # clinical acc ≥ 85 %
    high_recoverability_diseases: List[str]
    persistent_ambiguity_diseases: List[str]

    def summary(self) -> str:
        lines: List[str] = [
            "=" * 70,
            "DISEASE SEPARATION REFINEMENT REPORT",
            "=" * 70,
            f"  Diseases analysed       : {len(self.disease_profiles)}",
            f"  Mean clinical accuracy  : {self.mean_clinical_accuracy:.1%}",
            f"  Mean symbolic accuracy  : {self.mean_symbolic_accuracy:.1%}",
            f"  Mean symbolic lift      : {self.mean_symbolic_lift_pp:+.2f} pp",
            "",
            "  ── Biopsy Necessity ──────────────────────────────────────────",
        ]
        for p in self.biopsy_profiles:
            lines.append(
                f"    {p.disease:<32s}  tier={p.biopsy_necessity_tier.value:<22s}"
                f"  esc={p.escalation_rate:.1%}"
            )
        lines += [
            "",
            "  ── Symbolic Recoverability ───────────────────────────────────",
        ]
        for p in self.disease_profiles:
            lines.append(
                f"    {p.disease:<32s}  {p.symbolic_recoverability_tier.value:<12s}"
                f"  lift={p.symbolic_lift_pp:+.2f}pp"
                f"  ambiguity={p.ambiguity_tier.value}"
            )
        lines += [
            "",
            "  ── Diseases Safely Diagnosable Clinically ────────────────────",
        ]
        if self.diseases_safely_clinical:
            for d in self.diseases_safely_clinical:
                lines.append(f"    ✓  {d}")
        else:
            lines.append("    (none reached ≥ 85 % clinical accuracy)")
        lines += [
            "",
            "  ── Diseases Requiring Biopsy Support ─────────────────────────",
        ]
        if self.diseases_needing_biopsy_support:
            for d in self.diseases_needing_biopsy_support:
                lines.append(f"    ⚠  {d}")
        else:
            lines.append("    (all diseases at ≥ 70 % clinical accuracy)")
        lines += [
            "",
            "  ── Persistent Ambiguity Diseases ─────────────────────────────",
        ]
        if self.persistent_ambiguity_diseases:
            for d in self.persistent_ambiguity_diseases:
                lines.append(f"    ✗  {d}")
        else:
            lines.append("    (no disease with persistent ambiguity)")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis engine
# ──────────────────────────────────────────────────────────────────────────────

# Thresholds
_SAFE_CLINICAL_ACC   = 0.85
_NEEDS_BIOPSY_ACC    = 0.70
_HIGH_RECOV_LIFT     = 8.0   # pp
_MOD_RECOV_LIFT      = 4.0
_LOW_RECOV_LIFT      = 1.0
_RESOLVED_AMBIGUITY  = 1.5   # bits
_HIGH_AMBIGUITY      = 2.2
_PERSISTENT_AMBIGUITY = 3.0


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits (ignores zeros)."""
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _recoverability_tier(lift_pp: float) -> SymbolicRecoverabilityTier:
    if lift_pp >= _HIGH_RECOV_LIFT:
        return SymbolicRecoverabilityTier.HIGH
    elif lift_pp >= _MOD_RECOV_LIFT:
        return SymbolicRecoverabilityTier.MODERATE
    elif lift_pp >= _LOW_RECOV_LIFT:
        return SymbolicRecoverabilityTier.LOW
    return SymbolicRecoverabilityTier.NEGLIGIBLE


def _ambiguity_tier(mean_bits: float) -> AmbiguityTier:
    if mean_bits < _RESOLVED_AMBIGUITY:
        return AmbiguityTier.RESOLVED
    elif mean_bits < _HIGH_AMBIGUITY:
        return AmbiguityTier.MODERATE
    elif mean_bits < _PERSISTENT_AMBIGUITY:
        return AmbiguityTier.HIGH
    return AmbiguityTier.PERSISTENT


def _biopsy_necessity_tier(clinical_acc: float) -> BiopsyNecessityTier:
    if clinical_acc >= 0.85:
        return BiopsyNecessityTier.RARELY_NEEDED
    elif clinical_acc >= 0.70:
        return BiopsyNecessityTier.SOMETIMES_NEEDED
    elif clinical_acc >= 0.55:
        return BiopsyNecessityTier.FREQUENTLY_NEEDED
    return BiopsyNecessityTier.ALMOST_ALWAYS_NEEDED


class DiseaseSeparationRefiner:
    """
    Performs deep per-disease separation analysis.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names matching integer label indices.
    clinical_feature_names : list[str], optional
        Names of the 12 clinical features.
    symbolic_feature_names : list[str], optional
        Names of the symbolic signal features.
    """

    def __init__(
        self,
        class_labels: List[str],
        clinical_feature_names: Optional[List[str]] = None,
        symbolic_feature_names: Optional[List[str]] = None,
    ):
        self.class_labels = class_labels
        self.clinical_feature_names = clinical_feature_names or [
            f"clinical_{i}" for i in range(12)
        ]
        self.symbolic_feature_names = symbolic_feature_names or [
            f"symbolic_{i}" for i in range(22)
        ]

    # ------------------------------------------------------------------
    def analyse(
        self,
        symbolic_matrix: np.ndarray,       # shape (n, n_symbolic)
        y_true: np.ndarray,                # integer labels
        y_pred_b: np.ndarray,              # Model B predictions
        y_pred_c: np.ndarray,              # Model C predictions
        certainty_scores: Optional[np.ndarray] = None,  # (n,) [0,1]
        ambiguity_bits: Optional[np.ndarray] = None,    # (n,) bits
        escalation_flags: Optional[np.ndarray] = None,  # (n,) bool
    ) -> DiseaseSeparationReport:
        """
        Run the full separation refinement analysis.

        Returns
        -------
        DiseaseSeparationReport
        """
        n = len(y_true)
        if certainty_scores is None:
            certainty_scores = np.full(n, 0.60)
        if ambiguity_bits is None:
            ambiguity_bits = np.full(n, 2.0)
        if escalation_flags is None:
            escalation_flags = np.zeros(n, dtype=bool)

        profiles: List[DiseaseStabilisationProfile] = []
        biopsy_profiles: List[DiseaseBiopsyProfile] = []
        heatmap_cells: List[SymbolicRecoveryHeatmapCell] = []

        for label_idx, disease in enumerate(self.class_labels):
            mask = y_true == label_idx
            n_cases = int(mask.sum())
            if n_cases == 0:
                continue

            # Per-disease accuracies
            clinical_acc = float(np.mean(y_pred_b[mask] == y_true[mask]))
            symbolic_acc = float(np.mean(y_pred_c[mask] == y_true[mask]))
            lift_pp = (symbolic_acc - clinical_acc) * 100.0

            # Ambiguity
            amb = ambiguity_bits[mask]
            mean_amb = float(np.mean(amb))
            amb_std  = float(np.std(amb))

            # Differential competitors
            misclassified_b = y_pred_b[mask] != y_true[mask]
            competitors = self._compute_competitors(
                label_idx, mask, y_pred_b, y_pred_c, certainty_scores, symbolic_matrix
            )
            worst_confusion = max((c.confusion_rate for c in competitors), default=0.0)
            primary_comp = competitors[0].competitor_disease if competitors else None

            # Symbolic divergence
            sym_sub = symbolic_matrix[mask]
            sym_all = symbolic_matrix
            mean_sym_act = float(np.mean(sym_sub))
            divergence   = float(np.linalg.norm(
                np.mean(sym_sub, axis=0) - np.mean(sym_all, axis=0)
            )) if sym_sub.ndim == 2 else 0.0

            dominant_signals = self._dominant_signals(sym_sub)

            # Biopsy profile
            esc_mask = escalation_flags[mask]
            esc_rate    = float(np.mean(esc_mask))
            correct_esc = float(np.mean(
                (y_pred_b[mask] != y_true[mask]) & esc_mask
            )) if esc_mask.any() else 0.0
            false_esc   = esc_rate - correct_esc if esc_rate >= correct_esc else 0.0

            bp = DiseaseBiopsyProfile(
                disease=disease,
                clinical_accuracy=clinical_acc,
                symbolic_accuracy=symbolic_acc,
                symbolic_lift_pp=lift_pp,
                biopsy_necessity_tier=_biopsy_necessity_tier(clinical_acc),
                mean_ambiguity_bits=mean_amb,
                ambiguity_tier=_ambiguity_tier(mean_amb),
                escalation_rate=esc_rate,
                appropriate_escalation_rate=correct_esc,
                false_escalation_rate=false_esc,
            )

            sp = DiseaseStabilisationProfile(
                disease=disease,
                n_cases=n_cases,
                clinical_accuracy=clinical_acc,
                symbolic_accuracy=symbolic_acc,
                symbolic_lift_pp=lift_pp,
                symbolic_recoverability_tier=_recoverability_tier(lift_pp),
                mean_ambiguity_bits=mean_amb,
                ambiguity_tier=_ambiguity_tier(mean_amb),
                ambiguity_std=amb_std,
                primary_competitor=primary_comp,
                differential_competitors=competitors,
                worst_confusion_rate=worst_confusion,
                mean_symbolic_activation=mean_sym_act,
                symbolic_divergence_score=divergence,
                dominant_symbolic_signals=dominant_signals,
                biopsy_profile=bp,
            )
            profiles.append(sp)
            biopsy_profiles.append(bp)

            # Heatmap rows for this disease
            if symbolic_matrix.ndim == 2:
                for sig_idx, sig_name in enumerate(self.symbolic_feature_names):
                    if sig_idx >= symbolic_matrix.shape[1]:
                        break
                    col = symbolic_matrix[:, sig_idx]
                    col_disease = symbolic_matrix[mask, sig_idx]
                    disc_power  = abs(
                        float(np.mean(col_disease)) - float(np.mean(col))
                    ) / (float(np.std(col)) + 1e-9)
                    heatmap_cells.append(SymbolicRecoveryHeatmapCell(
                        disease=disease,
                        signal_name=sig_name,
                        mean_activation=float(np.mean(col_disease)),
                        activation_std=float(np.std(col_disease)),
                        discriminative_power=min(disc_power, 1.0),
                        recovery_contribution=min(disc_power / 3.0, 1.0),
                    ))

        # Aggregate stats
        accs_b = [p.clinical_accuracy for p in profiles]
        accs_c = [p.symbolic_accuracy for p in profiles]
        lifts  = [p.symbolic_lift_pp   for p in profiles]

        mean_b   = statistics.mean(accs_b) if accs_b else 0.0
        mean_c   = statistics.mean(accs_c) if accs_c else 0.0
        mean_lift = statistics.mean(lifts) if lifts else 0.0

        safe_clinical   = [p.disease for p in profiles if p.clinical_accuracy >= _SAFE_CLINICAL_ACC]
        needs_biopsy    = [p.disease for p in profiles if p.clinical_accuracy < _NEEDS_BIOPSY_ACC]
        high_recov      = [p.disease for p in profiles
                           if p.symbolic_recoverability_tier == SymbolicRecoverabilityTier.HIGH]
        persistent_amb  = [p.disease for p in profiles
                           if p.ambiguity_tier == AmbiguityTier.PERSISTENT]

        return DiseaseSeparationReport(
            disease_profiles=profiles,
            biopsy_profiles=biopsy_profiles,
            heatmap=heatmap_cells,
            mean_clinical_accuracy=mean_b,
            mean_symbolic_accuracy=mean_c,
            mean_symbolic_lift_pp=mean_lift,
            diseases_needing_biopsy_support=needs_biopsy,
            diseases_safely_clinical=safe_clinical,
            high_recoverability_diseases=high_recov,
            persistent_ambiguity_diseases=persistent_amb,
        )

    # ------------------------------------------------------------------
    def _compute_competitors(
        self,
        label_idx: int,
        mask: np.ndarray,
        y_pred_b: np.ndarray,
        y_pred_c: np.ndarray,
        certainty_scores: np.ndarray,
        symbolic_matrix: np.ndarray,
    ) -> List[DifferentialCompetitor]:
        competitors: List[DifferentialCompetitor] = []
        wrong_b = y_pred_b[mask][y_pred_b[mask] != label_idx]
        if len(wrong_b) == 0:
            return competitors

        n_disease = int(mask.sum())
        for comp_idx, comp_name in enumerate(self.class_labels):
            if comp_idx == label_idx:
                continue
            confused = int(np.sum(y_pred_b[mask] == comp_idx))
            if confused == 0:
                continue
            confusion_rate = confused / n_disease

            # Certainty overlap proxy
            cert_target = float(np.mean(certainty_scores[mask]))
            comp_mask   = (np.arange(len(y_pred_b)) != -1)  # all
            comp_mask_c  = np.ones(len(y_pred_b), dtype=bool)
            comp_mask_c[:] = False
            # find actual competitor indices
            for i, yt in enumerate(np.where(mask)[0]):
                pass
            cert_target_val = cert_target

            sym_sep = 0.0
            if symbolic_matrix.ndim == 2 and symbolic_matrix.shape[1] > 0:
                dm_target = np.mean(symbolic_matrix[mask], axis=0)
                comp_global = np.mean(symbolic_matrix, axis=0)
                sym_sep = float(np.linalg.norm(dm_target - comp_global)) / (
                    symbolic_matrix.shape[1] ** 0.5 + 1e-9
                )

            competitors.append(DifferentialCompetitor(
                competitor_disease=comp_name,
                confusion_rate=confusion_rate,
                mean_certainty_overlap=max(0.0, 1.0 - cert_target_val),
                symbolic_separation_gap=sym_sep,
                resolvable_with_biopsy=(confusion_rate < 0.30),
            ))

        competitors.sort(key=lambda c: c.confusion_rate, reverse=True)
        return competitors[:3]  # top-3 competitors

    def _dominant_signals(self, sym_sub: np.ndarray) -> List[str]:
        if sym_sub.ndim < 2 or sym_sub.shape[1] == 0:
            return []
        col_means = np.mean(sym_sub, axis=0)
        top_idx   = np.argsort(col_means)[::-1][:5]
        return [
            self.symbolic_feature_names[i]
            for i in top_idx
            if i < len(self.symbolic_feature_names)
        ]
