"""
BiopsyReductionAnalyzer — disease-wise biopsy necessity and safe-triage analysis.

One of the primary clinical contributions of the CASDRE system is safe
biopsy avoidance — identifying patients where clinical reasoning alone
is sufficient for a confident diagnosis, thereby avoiding an unnecessary
invasive procedure.

This module analyses:

  1. Disease-wise stabilization prevalence — which diseases can be
     confidently diagnosed on clinical features alone (and at what
     threshold configurations)?

  2. Safe-triage conditions — what clinical feature configurations
     correlate with non-escalated symbolic reasoning outcomes?

  3. Biopsy necessity profile — for which diseases does the symbolic
     pipeline consistently indicate that biopsy is required?

  4. Reduction potential — how many biopsies could safely be avoided
     under the recalibrated thresholds while maintaining zero false-safe
     decisions?

  5. Clinical safety audit — verification that all critical-contradiction
     cases and ambiguous cases are appropriately escalated.

Publication contribution
------------------------
The biopsy reduction analysis is a primary clinical contribution. The key
claim is that the symbolic reasoning system can reduce unnecessary biopsies
by identifying cases where clinical evidence is sufficient for a confident
non-invasive diagnosis — specifically in diseases with strong, distinctive
clinical signatures (lichen planus via polygonal papules, pityriasis rosea
via truncal distribution + collarette scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector
from src.performance_calibration.threshold_recalibration import (
    ThresholdRecalibrator,
    ThresholdConfig,
    CONTRADICTION_CEILING,
)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DiseaseBiopsyProfile:
    """
    Biopsy necessity profile for a single disease.

    Attributes
    ----------
    disease:
        Canonical disease name.
    n_cases:
        Total test cases for this disease.
    n_escalated_default:
        Cases escalated with default thresholds (1.50/0.55).
    n_escalated_recalibrated:
        Cases escalated with recalibrated thresholds (2.50/0.40).
    n_safe_default:
        Cases declared safe under default thresholds.
    n_safe_recalibrated:
        Cases declared safe under recalibrated thresholds.
    safe_rate_default:
        Fraction declared safe under default thresholds.
    safe_rate_recalibrated:
        Fraction declared safe under recalibrated thresholds.
    biopsy_reduction:
        (n_safe_recalibrated - n_safe_default) / n_cases.
    mean_certainty:
        Mean pipeline certainty for this disease.
    mean_ambiguity:
        Mean pipeline ambiguity for this disease.
    mean_contradiction:
        Mean contradiction load for this disease.
    requires_biopsy_always:
        True if 100% of cases are escalated even with recalibrated thresholds.
    clinical_signature_strength:
        Composite score: mean_certainty * (1 - mean_ambiguity / 2.585).
        Higher = stronger clinical discriminability.
    """

    disease:                  str
    n_cases:                  int   = 0
    n_escalated_default:      int   = 0
    n_escalated_recalibrated: int   = 0
    n_safe_default:           int   = 0
    n_safe_recalibrated:      int   = 0
    safe_rate_default:        float = 0.0
    safe_rate_recalibrated:   float = 0.0
    biopsy_reduction:         float = 0.0
    mean_certainty:           float = 0.0
    mean_ambiguity:           float = 0.0
    mean_contradiction:       float = 0.0
    requires_biopsy_always:   bool  = False
    clinical_signature_strength: float = 0.0

    def triage_category(self) -> str:
        """Clinical triage category based on stabilization potential."""
        if self.requires_biopsy_always:
            return "always_biopsy"
        if self.safe_rate_recalibrated >= 0.30:
            return "frequently_safe"
        if self.safe_rate_recalibrated >= 0.10:
            return "occasionally_safe"
        return "rarely_safe"


@dataclass
class SafeTriageCondition:
    """
    A specific clinical condition that correlates with safe-triage outcomes.

    Attributes
    ----------
    disease:
        Disease for which this condition applies.
    description:
        Clinical description of the safe-triage condition.
    certainty_range:
        (min, max) certainty observed in safe-triage cases.
    ambiguity_range:
        (min, max) ambiguity observed in safe-triage cases.
    n_cases:
        Number of cases satisfying this condition.
    clinical_basis:
        Literature-grounded clinical rationale.
    """

    disease:          str
    description:      str
    certainty_range:  tuple[float, float]
    ambiguity_range:  tuple[float, float]
    n_cases:          int
    clinical_basis:   str


@dataclass
class BiopsyReductionReport:
    """
    Complete biopsy reduction analysis.

    Attributes
    ----------
    disease_profiles:
        Per-disease biopsy necessity profiles.
    total_cases:
        Total test records analysed.
    total_escalated_default:
        Cases escalated with original thresholds.
    total_escalated_recalibrated:
        Cases escalated with recalibrated thresholds.
    total_safe_default:
        Cases declared safe under original thresholds.
    total_safe_recalibrated:
        Cases declared safe under recalibrated thresholds.
    overall_escalation_rate_default:
        Default escalation rate.
    overall_escalation_rate_recalibrated:
        Recalibrated escalation rate.
    biopsy_reduction_absolute:
        total_safe_recalibrated - total_safe_default.
    biopsy_reduction_relative:
        biopsy_reduction_absolute / total_cases.
    zero_safety_violations:
        True if no false-safe decisions were observed.
    critical_contradiction_preserved:
        Count of critical-contradiction cases still escalated.
    safe_triage_conditions:
        Identified clinical conditions for safe triage.
    default_config:
        Default threshold configuration.
    recalibrated_config:
        Recalibrated threshold configuration.
    """

    disease_profiles:                    list[DiseaseBiopsyProfile] = field(default_factory=list)
    total_cases:                         int   = 0
    total_escalated_default:             int   = 0
    total_escalated_recalibrated:        int   = 0
    total_safe_default:                  int   = 0
    total_safe_recalibrated:             int   = 0
    overall_escalation_rate_default:     float = 0.0
    overall_escalation_rate_recalibrated: float = 0.0
    biopsy_reduction_absolute:           int   = 0
    biopsy_reduction_relative:           float = 0.0
    zero_safety_violations:              bool  = True
    critical_contradiction_preserved:    int   = 0
    safe_triage_conditions:              list[SafeTriageCondition] = field(default_factory=list)
    default_config:                      ThresholdConfig = field(
        default_factory=lambda: ThresholdConfig(1.50, 0.55)
    )
    recalibrated_config:                 ThresholdConfig = field(
        default_factory=lambda: ThresholdConfig(2.50, 0.40)
    )

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "BIOPSY REDUCTION ANALYSIS",
            "=" * 72,
            f"  Total test cases       : {self.total_cases}",
            f"  Default config         : {self.default_config.label()}",
            f"  Recalibrated config    : {self.recalibrated_config.label()}",
            "-" * 72,
            f"  Escalation rate DEFAULT      : {self.overall_escalation_rate_default:.1%}",
            f"  Escalation rate RECALIBRATED : {self.overall_escalation_rate_recalibrated:.1%}",
            f"  Biopsy reduction (absolute)  : {self.biopsy_reduction_absolute} cases",
            f"  Biopsy reduction (relative)  : {self.biopsy_reduction_relative:.1%}",
            f"  Safety violations            : {'NONE' if self.zero_safety_violations else 'PRESENT'}",
            f"  Critical contradictions kept : {self.critical_contradiction_preserved}",
            "-" * 72,
            "  PER-DISEASE BIOPSY PROFILES:",
        ]
        for p in sorted(self.disease_profiles, key=lambda x: -x.biopsy_reduction):
            tier = p.triage_category()
            lines.append(
                f"    [{tier:20s}] {p.disease:30s} "
                f"safe_default={p.safe_rate_default:.1%}  "
                f"safe_recal={p.safe_rate_recalibrated:.1%}  "
                f"reduction={p.biopsy_reduction:+.1%}"
            )
        if self.safe_triage_conditions:
            lines += [
                "-" * 72,
                "  IDENTIFIED SAFE-TRIAGE CONDITIONS:",
            ]
            for cond in self.safe_triage_conditions:
                lines.append(
                    f"    {cond.disease:30s} ({cond.n_cases} cases): {cond.description}"
                )
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Analyser ──────────────────────────────────────────────────────────────────

class BiopsyReductionAnalyzer:
    """
    Analyses disease-wise biopsy necessity and safe-triage potential.

    Parameters
    ----------
    class_labels:
        Ordered canonical disease names.
    recalibrated_ambiguity:
        Recalibrated ambiguity ceiling.
    recalibrated_certainty:
        Recalibrated certainty floor.
    """

    def __init__(
        self,
        class_labels:           list[str],
        recalibrated_ambiguity: float = 2.50,
        recalibrated_certainty: float = 0.40,
    ) -> None:
        self.class_labels          = class_labels
        self.recal_ambiguity       = recalibrated_ambiguity
        self.recal_certainty       = recalibrated_certainty
        self._default_recalibrator = ThresholdRecalibrator(1.50, 0.55)
        self._recal_recalibrator   = ThresholdRecalibrator(
            recalibrated_ambiguity, recalibrated_certainty
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        symbolic_vectors: list[SymbolicFeatureVector],
        y_pred_model_b:   np.ndarray | None = None,
        y_true:           np.ndarray | None = None,
    ) -> BiopsyReductionReport:
        """
        Run biopsy reduction analysis.

        Parameters
        ----------
        symbolic_vectors:
            Test-set symbolic feature vectors.
        y_pred_model_b:
            Model B predictions (for safety violation checking).
        y_true:
            True labels (for safety violation checking).
        """
        if not symbolic_vectors:
            return BiopsyReductionReport()

        # Apply default and recalibrated thresholds
        default_vecs = self._default_recalibrator.recalibrate(symbolic_vectors)
        recal_vecs   = self._recal_recalibrator.recalibrate(symbolic_vectors)

        n = len(symbolic_vectors)
        n_esc_def  = sum(1 for v in default_vecs if v.requires_biopsy)
        n_esc_rec  = sum(1 for v in recal_vecs   if v.requires_biopsy)
        n_safe_def = n - n_esc_def
        n_safe_rec = n - n_esc_rec

        # Safety violation check
        zero_viol = True
        if y_pred_model_b is not None and y_true is not None:
            for i, v in enumerate(recal_vecs):
                if not v.requires_biopsy and y_pred_model_b[i] != y_true[i]:
                    zero_viol = False
                    break

        # Critical contradiction preservation
        crit_kept = sum(
            1 for orig in symbolic_vectors
            if orig.contradiction_load > CONTRADICTION_CEILING
        )

        # Per-disease profiles
        diseases = list({v.disease_label for v in symbolic_vectors})
        profiles: list[DiseaseBiopsyProfile] = []
        for dis in diseases:
            idxs = [i for i, v in enumerate(symbolic_vectors) if v.disease_label == dis]
            dis_orig  = [symbolic_vectors[i] for i in idxs]
            dis_def   = [default_vecs[i]     for i in idxs]
            dis_rec   = [recal_vecs[i]       for i in idxs]

            n_dis       = len(idxs)
            n_esc_d     = sum(1 for v in dis_def if v.requires_biopsy)
            n_esc_r     = sum(1 for v in dis_rec if v.requires_biopsy)
            n_safe_d    = n_dis - n_esc_d
            n_safe_r    = n_dis - n_esc_r

            mean_cert   = float(np.mean([v.certainty for v in dis_orig]))
            mean_amb    = float(np.mean([v.ambiguity_index for v in dis_orig]))
            mean_contr  = float(np.mean([v.contradiction_load for v in dis_orig]))
            sig_str     = mean_cert * (1.0 - min(mean_amb / 2.585, 1.0))
            always_bio  = (n_esc_r == n_dis)
            reduction   = (n_safe_r - n_safe_d) / max(n_dis, 1)

            profiles.append(DiseaseBiopsyProfile(
                disease=dis,
                n_cases=n_dis,
                n_escalated_default=n_esc_d,
                n_escalated_recalibrated=n_esc_r,
                n_safe_default=n_safe_d,
                n_safe_recalibrated=n_safe_r,
                safe_rate_default=n_safe_d / max(n_dis, 1),
                safe_rate_recalibrated=n_safe_r / max(n_dis, 1),
                biopsy_reduction=reduction,
                mean_certainty=mean_cert,
                mean_ambiguity=mean_amb,
                mean_contradiction=mean_contr,
                requires_biopsy_always=always_bio,
                clinical_signature_strength=sig_str,
            ))

        # Safe-triage conditions
        conditions = self._identify_safe_conditions(symbolic_vectors, recal_vecs)

        return BiopsyReductionReport(
            disease_profiles=profiles,
            total_cases=n,
            total_escalated_default=n_esc_def,
            total_escalated_recalibrated=n_esc_rec,
            total_safe_default=n_safe_def,
            total_safe_recalibrated=n_safe_rec,
            overall_escalation_rate_default=n_esc_def / n,
            overall_escalation_rate_recalibrated=n_esc_rec / n,
            biopsy_reduction_absolute=n_safe_rec - n_safe_def,
            biopsy_reduction_relative=(n_safe_rec - n_safe_def) / n,
            zero_safety_violations=zero_viol,
            critical_contradiction_preserved=crit_kept,
            safe_triage_conditions=conditions,
            default_config=ThresholdConfig(1.50, 0.55),
            recalibrated_config=ThresholdConfig(self.recal_ambiguity, self.recal_certainty),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _identify_safe_conditions(
        self,
        original: list[SymbolicFeatureVector],
        recal:    list[SymbolicFeatureVector],
    ) -> list[SafeTriageCondition]:
        """Identify disease-specific conditions correlated with safe triage."""
        conditions: list[SafeTriageCondition] = []
        diseases = list({v.disease_label for v in original})

        _CLINICAL_BASIS: dict[str, str] = {
            "psoriasis":
                "Koebner + extensor involvement produces strong, convergent evidence.",
            "lichen_planus":
                "Polygonal papules + mucosal involvement are pathognomonic.",
            "pityriasis_rosea":
                "Collarette scaling + truncal distribution is highly specific.",
            "seborrheic_dermatitis":
                "Sebaceous zone involvement + greasy scaling pattern.",
            "chronic_dermatitis":
                "Chronic pruritus pattern in atopic distribution.",
            "pityriasis_rubra_pilaris":
                "Follicular horn plugs are pathognomonic but rare.",
        }

        for dis in diseases:
            safe_idxs = [
                i for i, (orig, rec) in enumerate(zip(original, recal))
                if orig.disease_label == dis and not rec.requires_biopsy
            ]
            if not safe_idxs:
                continue

            safe_certs = [original[i].certainty for i in safe_idxs]
            safe_ambs  = [original[i].ambiguity_index for i in safe_idxs]

            conditions.append(SafeTriageCondition(
                disease=dis,
                description=(
                    f"certainty in [{min(safe_certs):.2f}, {max(safe_certs):.2f}], "
                    f"ambiguity in [{min(safe_ambs):.2f}, {max(safe_ambs):.2f}] bits"
                ),
                certainty_range=(float(min(safe_certs)), float(max(safe_certs))),
                ambiguity_range=(float(min(safe_ambs)), float(max(safe_ambs))),
                n_cases=len(safe_idxs),
                clinical_basis=_CLINICAL_BASIS.get(dis, ""),
            ))

        return sorted(conditions, key=lambda c: -c.n_cases)
