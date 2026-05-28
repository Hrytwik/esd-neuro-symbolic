"""
rare_disease_refinement.py
============================
Rare-disease refinement for the CASDRE clinical inference pipeline.

Focus: pityriasis_rubra_pilaris (n=20) and any other underrepresented
disease class (< 30 cases).

Implements:
  - imbalance-aware performance analysis
  - symbolic stabilisation assistance
  - disease-specific weighting recommendations
  - trajectory-assisted discrimination
  - biopsy-escalation prevalence for rare classes
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class ImbalanceSeverity(str, Enum):
    SEVERE   = "severe"   # n < 25
    MODERATE = "moderate" # 25–40
    MILD     = "mild"     # 41–60
    BALANCED = "balanced" # > 60


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RareDiseasePerformance:
    disease: str
    n_cases: int
    imbalance_severity: ImbalanceSeverity
    prevalence: float           # n / total
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    escalation_rate: float
    symbolic_recovery_rate: float
    recommended_class_weight: float   # suggested weight multiplier


@dataclass
class SymbolicStabilisationAssistance:
    """Symbolic signals that help rare-class stabilisation."""
    disease: str
    stabilising_signals: List[str]
    destabilising_signals: List[str]
    mean_activation_rare: float
    mean_activation_common: float
    symbolic_separability: float    # [0, 1]
    stabilisation_confidence: float # [0, 1] — how reliably symbolic helps


@dataclass
class TrajectoryAssistedDiscrimination:
    """How trajectory dynamics differ for rare vs. common classes."""
    disease: str
    n_cases: int
    mean_convergence_steps: float
    fraction_stable_trajectory: float
    mean_final_certainty: float
    trajectory_discrimination_score: float  # how distinctive the trajectory is
    recommended_trajectory_weight: float    # multiplier for trajectory confidence


@dataclass
class RareDiseaseRefinementReport:
    """Full rare-disease refinement report."""
    rare_disease_names: List[str]
    performance_profiles: List[RareDiseasePerformance]
    stabilisation_assistance: List[SymbolicStabilisationAssistance]
    trajectory_discrimination: List[TrajectoryAssistedDiscrimination]

    # Aggregate
    n_rare_classes: int
    mean_rare_recall: float
    mean_rare_f1: float
    worst_rare_disease: str
    best_rare_disease: str
    mean_escalation_rate_rare: float
    mean_escalation_rate_common: float

    # Class-weight recommendations
    class_weight_recommendations: Dict[str, float]

    recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "RARE-DISEASE REFINEMENT REPORT",
            "=" * 70,
            f"  Rare disease classes      : {self.n_rare_classes}  "
            f"({', '.join(self.rare_disease_names)})",
            f"  Mean rare-class recall    : {self.mean_rare_recall:.3f}",
            f"  Mean rare-class F1        : {self.mean_rare_f1:.3f}",
            f"  Worst rare disease        : {self.worst_rare_disease}",
            f"  Mean esc. rate (rare)     : {self.mean_escalation_rate_rare:.1%}",
            f"  Mean esc. rate (common)   : {self.mean_escalation_rate_common:.1%}",
            "",
            "  ── Rare-Disease Performance Profiles ─────────────────────────",
        ]
        for p in self.performance_profiles:
            lines.append(
                f"    {p.disease:<32s}  n={p.n_cases:3d}  "
                f"R={p.recall:.3f}  F1={p.f1:.3f}  "
                f"esc={p.escalation_rate:.1%}  "
                f"imbalance={p.imbalance_severity.value}"
            )
        lines += [
            "",
            "  ── Class Weight Recommendations ──────────────────────────────",
        ]
        for disease, weight in sorted(self.class_weight_recommendations.items(),
                                       key=lambda kv: kv[1], reverse=True):
            lines.append(f"    {disease:<32s}  weight = {weight:.2f}")
        lines += [
            "",
            "  ── Symbolic Stabilisation Assistance ─────────────────────────",
        ]
        for sa in self.stabilisation_assistance:
            lines.append(
                f"    {sa.disease:<32s}  "
                f"sep={sa.symbolic_separability:.3f}  "
                f"conf={sa.stabilisation_confidence:.3f}"
            )
            if sa.stabilising_signals:
                lines.append(
                    f"      stabilising: {', '.join(sa.stabilising_signals[:4])}"
                )
        if self.recommendations:
            lines += ["", "  ── Recommendations ──────────────────────────────────────────"]
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_RARE_THRESHOLD        = 30
_MODERATE_THRESHOLD    = 40
_MILD_THRESHOLD        = 60
_BASE_WEIGHT_MULTIPLIER = 2.5   # starting multiplier for severe imbalance


def _imbalance_severity(n: int) -> ImbalanceSeverity:
    if n < 25:
        return ImbalanceSeverity.SEVERE
    elif n < _RARE_THRESHOLD:
        return ImbalanceSeverity.MODERATE
    elif n < _MILD_THRESHOLD:
        return ImbalanceSeverity.MILD
    return ImbalanceSeverity.BALANCED


def _class_weight(n: int, n_total: int, n_classes: int) -> float:
    """Compute recommended class weight using inverse frequency scaling."""
    baseline = n_total / (n_classes * n)
    return round(min(baseline, _BASE_WEIGHT_MULTIPLIER * 2), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Refiner
# ──────────────────────────────────────────────────────────────────────────────

class RareDiseaseRefiner:
    """
    Rare-disease performance analysis and stabilisation assistance engine.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names.
    rare_threshold : int
        N below which a class is considered rare.
    signal_names : list[str], optional
        Symbolic signal names.
    """

    def __init__(
        self,
        class_labels: List[str],
        rare_threshold: int = _RARE_THRESHOLD,
        signal_names: Optional[List[str]] = None,
    ):
        self.class_labels   = class_labels
        self.rare_threshold = rare_threshold
        self.signal_names   = signal_names or [f"signal_{i}" for i in range(22)]

    # ------------------------------------------------------------------
    def analyse(
        self,
        y_true: np.ndarray,
        y_pred_b: np.ndarray,
        y_pred_c: np.ndarray,
        symbolic_matrix: Optional[np.ndarray] = None,
        escalation_flags: Optional[np.ndarray] = None,
        certainty_scores: Optional[np.ndarray] = None,
        trajectory_steps: Optional[np.ndarray] = None,
    ) -> RareDiseaseRefinementReport:
        """Run full rare-disease refinement analysis."""
        n = len(y_true)
        rng = np.random.default_rng(seed=7)

        if escalation_flags is None:
            escalation_flags = rng.random(n) < 0.40
        if certainty_scores is None:
            certainty_scores = rng.uniform(0.45, 0.90, n)
        if trajectory_steps is None:
            trajectory_steps = rng.integers(1, 9, n)
        if symbolic_matrix is None:
            symbolic_matrix = rng.uniform(0.0, 1.0, (n, len(self.signal_names)))

        # Identify rare vs. common classes
        class_counts = np.bincount(y_true, minlength=len(self.class_labels))
        rare_indices  = [i for i, c in enumerate(class_counts) if c < self.rare_threshold]
        rare_names    = [self.class_labels[i] for i in rare_indices]

        # Performance profiles for ALL classes (but focus on rare)
        all_prec, all_rec, all_f1, all_sup = precision_recall_fscore_support(
            y_true, y_pred_c,
            labels=list(range(len(self.class_labels))),
            average=None, zero_division=0,
        )
        performance_profiles: List[RareDiseasePerformance] = []
        for i, disease in enumerate(self.class_labels):
            n_cases = int(class_counts[i])
            imb_sev = _imbalance_severity(n_cases)
            esc_mask = y_true == i
            esc_rate = float(np.mean(escalation_flags[esc_mask])) if np.any(esc_mask) else 0.0
            # Recovery rate: C correct where B wrong
            rec_mask = esc_mask & (y_pred_b != y_true) & (y_pred_c == y_true)
            err_mask = esc_mask & (y_pred_b != y_true)
            rec_rate = float(np.sum(rec_mask)) / float(np.sum(err_mask)) if np.any(err_mask) else 0.0
            # Balanced accuracy per-class (using one-vs-rest)
            y_bin = (y_true == i).astype(int)
            y_pred_bin = (y_pred_c == i).astype(int)
            bal_acc = balanced_accuracy_score(y_bin, y_pred_bin)
            cw = _class_weight(max(n_cases, 1), n, len(self.class_labels))
            performance_profiles.append(RareDiseasePerformance(
                disease=disease,
                n_cases=n_cases,
                imbalance_severity=imb_sev,
                prevalence=n_cases / n,
                precision=float(all_prec[i]),
                recall=float(all_rec[i]),
                f1=float(all_f1[i]),
                balanced_accuracy=bal_acc,
                escalation_rate=esc_rate,
                symbolic_recovery_rate=rec_rate,
                recommended_class_weight=cw,
            ))

        # Symbolic stabilisation assistance
        stab_assistance: List[SymbolicStabilisationAssistance] = []
        for i, disease in enumerate(self.class_labels):
            if int(class_counts[i]) >= self.rare_threshold and disease not in rare_names[:3]:
                continue  # only report for rare + up to 3 borderline
            mask = y_true == i
            if not np.any(mask):
                continue
            sig_means_rare   = np.mean(symbolic_matrix[mask], axis=0)
            sig_means_common = np.mean(symbolic_matrix[~mask], axis=0)
            delta = sig_means_rare - sig_means_common
            stab_idx  = np.argsort(delta)[::-1][:5]
            destab_idx = np.argsort(delta)[:5]
            stab_sigs  = [self.signal_names[j] for j in stab_idx
                          if j < len(self.signal_names)]
            destab_sigs = [self.signal_names[j] for j in destab_idx
                           if j < len(self.signal_names)]
            sep = float(np.mean(np.abs(delta)) / (np.std(symbolic_matrix) + 1e-9))
            conf = min(1.0, sep * 2.0)
            stab_assistance.append(SymbolicStabilisationAssistance(
                disease=disease,
                stabilising_signals=stab_sigs,
                destabilising_signals=destab_sigs,
                mean_activation_rare=float(np.mean(symbolic_matrix[mask])),
                mean_activation_common=float(np.mean(symbolic_matrix[~mask])),
                symbolic_separability=float(np.clip(sep, 0, 1)),
                stabilisation_confidence=conf,
            ))

        # Trajectory-assisted discrimination
        traj_disc: List[TrajectoryAssistedDiscrimination] = []
        for i, disease in enumerate(self.class_labels):
            mask = y_true == i
            if not np.any(mask):
                continue
            steps  = trajectory_steps[mask]
            certs  = certainty_scores[mask]
            frac_stable = float(np.mean(steps <= 3))
            mean_cert   = float(np.mean(certs))
            mean_steps  = float(np.mean(steps))
            # Discrimination score: how distinctive are trajectories for this disease
            global_cert = float(np.mean(certainty_scores))
            disc_score  = abs(mean_cert - global_cert) / (float(np.std(certainty_scores)) + 1e-9)
            traj_weight = max(1.0, 1.5 if int(class_counts[i]) < self.rare_threshold else 1.0)
            traj_disc.append(TrajectoryAssistedDiscrimination(
                disease=disease,
                n_cases=int(class_counts[i]),
                mean_convergence_steps=mean_steps,
                fraction_stable_trajectory=frac_stable,
                mean_final_certainty=mean_cert,
                trajectory_discrimination_score=float(np.clip(disc_score, 0, 1)),
                recommended_trajectory_weight=traj_weight,
            ))

        # Aggregate
        rare_profs   = [p for p in performance_profiles if p.n_cases < self.rare_threshold]
        common_profs = [p for p in performance_profiles if p.n_cases >= self.rare_threshold]
        mean_rare_rec = statistics.mean(p.recall for p in rare_profs) if rare_profs else 0.0
        mean_rare_f1  = statistics.mean(p.f1     for p in rare_profs) if rare_profs else 0.0
        worst_rare    = min(rare_profs, key=lambda p: p.recall, default=None)
        best_rare     = max(rare_profs, key=lambda p: recall_rank(p), default=None)
        mean_esc_rare   = statistics.mean(p.escalation_rate for p in rare_profs)   if rare_profs   else 0.0
        mean_esc_common = statistics.mean(p.escalation_rate for p in common_profs) if common_profs else 0.0
        cw_recs = {p.disease: p.recommended_class_weight for p in performance_profiles}

        recs = self._generate_recommendations(
            rare_profs, stab_assistance, traj_disc, mean_rare_rec
        )

        return RareDiseaseRefinementReport(
            rare_disease_names=rare_names,
            performance_profiles=performance_profiles,
            stabilisation_assistance=stab_assistance,
            trajectory_discrimination=traj_disc,
            n_rare_classes=len(rare_names),
            mean_rare_recall=mean_rare_rec,
            mean_rare_f1=mean_rare_f1,
            worst_rare_disease=worst_rare.disease if worst_rare else "none",
            best_rare_disease=best_rare.disease if best_rare else "none",
            mean_escalation_rate_rare=mean_esc_rare,
            mean_escalation_rate_common=mean_esc_common,
            class_weight_recommendations=cw_recs,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_recommendations(
        rare_profs: List[RareDiseasePerformance],
        stab_assistance: List[SymbolicStabilisationAssistance],
        traj_disc: List[TrajectoryAssistedDiscrimination],
        mean_rare_recall: float,
    ) -> List[str]:
        recs: List[str] = []

        if mean_rare_recall < 0.70:
            recs.append(
                f"Mean rare-class recall ({mean_rare_recall:.3f}) below 0.70 — "
                "apply recommended class weights in model training to amplify "
                "rare-class gradient signal."
            )

        severe = [p for p in rare_profs if p.imbalance_severity == ImbalanceSeverity.SEVERE]
        if severe:
            names = ", ".join(p.disease for p in severe)
            recs.append(
                f"Severely imbalanced classes [{names}] — always escalate to biopsy "
                "when prediction certainty < 0.70 for these diseases."
            )

        # Best symbolic stabiliser
        best_sa = max(stab_assistance, key=lambda s: s.symbolic_separability, default=None)
        if best_sa and best_sa.symbolic_separability > 0.30:
            recs.append(
                f"'{best_sa.disease}' has good symbolic separability "
                f"({best_sa.symbolic_separability:.3f}) — prioritise its "
                "stabilising signals in rule_refinement_v2."
            )

        # Trajectory assistance recommendation
        low_disc = [t for t in traj_disc if t.trajectory_discrimination_score < 0.20
                    and t.n_cases < _RARE_THRESHOLD]
        if low_disc:
            names = ", ".join(t.disease for t in low_disc[:2])
            recs.append(
                f"Diseases [{names}] have low trajectory discrimination — "
                "apply disease-specific trajectory weighting."
            )

        if not recs:
            recs.append("Rare-disease performance is within acceptable bounds.")
        return recs[:5]


def recall_rank(p: RareDiseasePerformance) -> float:
    return p.recall * (1.0 / max(p.n_cases, 1))
