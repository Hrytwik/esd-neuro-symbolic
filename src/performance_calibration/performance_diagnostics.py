"""
PerformanceDiagnostics — systematic per-disease failure audit.

Diagnoses the root causes of Model B (80%) and Model C (81.82%)
underperformance relative to the 86–91% target range, focusing on:

  1. Per-disease failure profiles — which diseases are hardest to
     classify on 12 clinical features alone, and why.
  2. Certainty-collapse analysis — where the symbolic pipeline fails
     to produce discriminating certainty signals (mean ≈ 0.27).
  3. Escalation over-activation — why 99.4% of cases are escalated
     (ambiguity ceiling 1.50 bits triggered almost universally).
  4. Confusion zone activation — which disease pairs account for the
     largest fraction of misclassifications.
  5. Clinical feature discriminability — which of the 12 clinical
     features provide the strongest between-disease separation.

Outputs are structured dataclasses that feed into:
  · FailureModeAnalyzer (root cause synthesis)
  · EscalationSensitivityAnalyzer (threshold recommendations)
  · BaselineCalibrator (optimisation targets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.evaluation_pipeline.baseline_model_a import ModelAResult
from src.evaluation_pipeline.baseline_model_b import ModelBResult
from src.evaluation_pipeline.symbolic_model_c import ModelCResult
from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector
from src.dataset_integration.feature_partitioning import CLINICAL_FEATURE_NAMES


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DiseaseFailureProfile:
    """
    Per-disease classification performance and difficulty indicators.

    Attributes
    ----------
    disease:
        Canonical disease name.
    model_a_recall, model_b_recall, model_c_recall:
        Per-disease recall for each model.
    recall_drop_b_vs_a:
        Recall lost from A to B — information cost of removing biopsy.
    recall_gain_c_vs_b:
        Recall recovered by symbolic reasoning (C vs B).
    model_b_f1, model_c_f1:
        F1 scores for each disease.
    top_confusion_targets:
        List of (disease, count) pairs showing which other diseases
        this disease is most confused with in Model B predictions.
    mean_certainty:
        Mean symbolic certainty for this disease's cases.
    mean_ambiguity:
        Mean ambiguity index for this disease's cases.
    mean_contradiction:
        Mean contradiction load for this disease's cases.
    always_escalated:
        True if 100% of this disease's cases are escalated.
    clinical_separation_score:
        0–1 score of how clinically separable this disease is from
        others using the 12 clinical features (from feature means
        deviation). Higher = more separable.
    """

    disease:                str
    model_a_recall:         float = 0.0
    model_b_recall:         float = 0.0
    model_c_recall:         float = 0.0
    recall_drop_b_vs_a:     float = 0.0
    recall_gain_c_vs_b:     float = 0.0
    model_b_f1:             float = 0.0
    model_c_f1:             float = 0.0
    top_confusion_targets:  list[tuple[str, int]] = field(default_factory=list)
    mean_certainty:         float = 0.0
    mean_ambiguity:         float = 0.0
    mean_contradiction:     float = 0.0
    always_escalated:       bool  = False
    clinical_separation_score: float = 0.0

    def difficulty_tier(self) -> str:
        """Classify diagnostic difficulty: easy / moderate / hard / critical."""
        if self.model_b_recall >= 0.90:
            return "easy"
        if self.model_b_recall >= 0.75:
            return "moderate"
        if self.model_b_recall >= 0.50:
            return "hard"
        return "critical"


@dataclass
class CertaintyCollapseProfile:
    """
    Characterises the certainty-collapse phenomenon observed in the
    symbolic pipeline when operating on clinical-only data.

    With a mean certainty of ≈ 0.27 and near-universal ambiguity
    escalation (99.4%), the symbolic signals are too weak in their
    current form to provide meaningful discrimination for most cases.
    This profile quantifies the collapse and identifies the minority
    of cases where certainty is adequate.

    Attributes
    ----------
    mean_certainty_overall:
        Mean pipeline certainty across all test records.
    std_certainty_overall:
        Standard deviation of certainty.
    certainty_above_threshold_rate:
        Fraction of records with certainty ≥ 0.55 (safe-triage floor).
    mean_ambiguity_overall:
        Mean ambiguity index (bits).
    ambiguity_below_threshold_rate:
        Fraction with ambiguity < 1.50 bits (non-escalated).
    per_disease_mean_certainty:
        Mean certainty broken down by true disease label.
    certainty_floor_barrier:
        Minimum certainty needed across the dataset — diagnoses
        which diseases never reach the safe-triage threshold.
    collapse_severity:
        "none" | "partial" | "severe" | "total"
        Total: ≥ 95% escalation; Severe: 80–95%; Partial: 50–80%.
    """

    mean_certainty_overall:          float = 0.0
    std_certainty_overall:           float = 0.0
    certainty_above_threshold_rate:  float = 0.0
    mean_ambiguity_overall:          float = 0.0
    ambiguity_below_threshold_rate:  float = 0.0
    per_disease_mean_certainty:      dict[str, float] = field(default_factory=dict)
    certainty_floor_barrier:         float = 0.0
    collapse_severity:               str   = "none"

    def is_severe(self) -> bool:
        return self.collapse_severity in ("severe", "total")


@dataclass
class ConfusionZonePair:
    """A pair of diseases that are frequently confused in Model B predictions."""

    true_disease:      str
    predicted_disease: str
    confusion_count:   int
    confusion_rate:    float   # fraction of true_disease cases predicted as predicted_disease


@dataclass
class FeatureDiscriminabilityScore:
    """How discriminating a single clinical feature is across the 6 diseases."""

    feature:            str
    between_class_var:  float   # variance of per-class feature means
    within_class_var:   float   # mean within-class feature variance
    f_ratio:            float   # between / within — higher = more discriminating
    rank:               int     # 1 = most discriminating


@dataclass
class DiagnosticReport:
    """
    Complete diagnostic output from PerformanceDiagnostics.

    Attributes
    ----------
    disease_profiles:
        Per-disease failure analysis (6 entries).
    certainty_collapse:
        Certainty-collapse characterisation.
    confusion_zones:
        Most active disease-pair confusion zones in Model B.
    feature_discriminability:
        Ranked clinical feature discriminability scores.
    overall_escalation_rate:
        Fraction of test records escalated (all models share this).
    escalation_trigger_breakdown:
        Fraction triggered by ambiguity / certainty / contradiction.
    hardest_diseases:
        Disease names with model_b_recall < 0.70.
    easiest_diseases:
        Disease names with model_b_recall ≥ 0.90.
    primary_bottleneck:
        Single-sentence root-cause summary.
    n_test:
        Number of test records analysed.
    """

    disease_profiles:           list[DiseaseFailureProfile] = field(default_factory=list)
    certainty_collapse:         CertaintyCollapseProfile = field(
        default_factory=CertaintyCollapseProfile
    )
    confusion_zones:            list[ConfusionZonePair] = field(default_factory=list)
    feature_discriminability:   list[FeatureDiscriminabilityScore] = field(default_factory=list)
    overall_escalation_rate:    float = 0.0
    escalation_trigger_breakdown: dict[str, float] = field(default_factory=dict)
    hardest_diseases:           list[str] = field(default_factory=list)
    easiest_diseases:           list[str] = field(default_factory=list)
    primary_bottleneck:         str = ""
    n_test:                     int = 0

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "PERFORMANCE DIAGNOSTIC REPORT",
            "=" * 72,
            f"  Test records analysed : {self.n_test}",
            f"  Escalation rate       : {self.overall_escalation_rate:.1%}",
            f"  Certainty collapse    : {self.certainty_collapse.collapse_severity}",
            f"  Mean certainty        : {self.certainty_collapse.mean_certainty_overall:.3f}",
            f"  Mean ambiguity        : {self.certainty_collapse.mean_ambiguity_overall:.3f} bits",
            "-" * 72,
            "  PER-DISEASE RECALL PROFILE (Model A → B → C):",
        ]
        for p in sorted(self.disease_profiles, key=lambda x: x.model_b_recall):
            tier = p.difficulty_tier()
            lines.append(
                f"    [{tier:8s}] {p.disease:30s} "
                f"A={p.model_a_recall:.3f} "
                f"B={p.model_b_recall:.3f} "
                f"C={p.model_c_recall:.3f} "
                f"(drop={p.recall_drop_b_vs_a:+.3f}, "
                f"lift={p.recall_gain_c_vs_b:+.3f})"
            )
        lines += [
            "-" * 72,
            "  TOP CONFUSION ZONES (Model B):",
        ]
        for cz in self.confusion_zones[:5]:
            lines.append(
                f"    {cz.true_disease:30s} -> {cz.predicted_disease:30s} "
                f"({cz.confusion_count} cases, {cz.confusion_rate:.1%})"
            )
        lines += [
            "-" * 72,
            "  TOP CLINICAL FEATURE DISCRIMINABILITY:",
        ]
        for fs in self.feature_discriminability[:5]:
            lines.append(
                f"    #{fs.rank:2d} {fs.feature:30s} F-ratio={fs.f_ratio:.3f}"
            )
        lines += [
            "-" * 72,
            f"  PRIMARY BOTTLENECK: {self.primary_bottleneck}",
            "=" * 72,
        ]
        return "\n".join(lines)


# ── Diagnostics engine ────────────────────────────────────────────────────────

class PerformanceDiagnostics:
    """
    Systematic diagnostic audit of Model A/B/C performance gaps.

    Inputs are the results from EvaluationRunner plus the symbolic
    feature vectors for the test set (which contain per-patient
    reasoning trajectory data needed for certainty-collapse analysis).

    Parameters
    ----------
    class_labels:
        Ordered canonical disease names (index = integer class code).
    """

    def __init__(self, class_labels: list[str]) -> None:
        self.class_labels = class_labels

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        result_a: ModelAResult,
        result_b: ModelBResult,
        result_c: ModelCResult,
        symbolic_vectors: list[SymbolicFeatureVector],
        X_test_clinical:  np.ndarray | None = None,
        y_test_int:       np.ndarray | None = None,
    ) -> DiagnosticReport:
        """
        Run the full diagnostic audit.

        Parameters
        ----------
        result_a, result_b, result_c:
            Evaluation results from the tripartite runner.
        symbolic_vectors:
            Test-set symbolic feature vectors (one per test record).
        X_test_clinical:
            Clinical feature matrix for the test set (n_test × 12).
            Used for clinical feature discriminability analysis.
        y_test_int:
            True 0-based integer labels for the test set.
        """
        n_test = result_b.n_test

        # 1. Per-disease failure profiles
        disease_profiles = self._build_disease_profiles(
            result_a, result_b, result_c, symbolic_vectors
        )

        # 2. Certainty-collapse analysis
        certainty_collapse = self._build_certainty_collapse(symbolic_vectors)

        # 3. Confusion-zone analysis from Model B confusion matrix
        confusion_zones = self._extract_confusion_zones(result_b)

        # 4. Clinical feature discriminability
        feat_disc: list[FeatureDiscriminabilityScore] = []
        if X_test_clinical is not None and y_test_int is not None:
            feat_disc = self._compute_feature_discriminability(
                X_test_clinical, y_test_int
            )

        # 5. Escalation breakdown
        esc_rate, esc_breakdown = self._escalation_breakdown(symbolic_vectors)

        # 6. Classify diseases by difficulty
        hardest  = [p.disease for p in disease_profiles if p.model_b_recall < 0.70]
        easiest  = [p.disease for p in disease_profiles if p.model_b_recall >= 0.90]

        # 7. Synthesise primary bottleneck
        bottleneck = self._synthesise_bottleneck(
            certainty_collapse, disease_profiles, esc_rate
        )

        return DiagnosticReport(
            disease_profiles=disease_profiles,
            certainty_collapse=certainty_collapse,
            confusion_zones=confusion_zones,
            feature_discriminability=feat_disc,
            overall_escalation_rate=esc_rate,
            escalation_trigger_breakdown=esc_breakdown,
            hardest_diseases=hardest,
            easiest_diseases=easiest,
            primary_bottleneck=bottleneck,
            n_test=n_test,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_disease_profiles(
        self,
        result_a: ModelAResult,
        result_b: ModelBResult,
        result_c: ModelCResult,
        vectors: list[SymbolicFeatureVector],
    ) -> list[DiseaseFailureProfile]:
        """Build per-disease failure profile from recall dicts."""
        profiles: list[DiseaseFailureProfile] = []

        # Group symbolic vectors by disease label
        vecs_by_disease: dict[str, list[SymbolicFeatureVector]] = {}
        for v in vectors:
            vecs_by_disease.setdefault(v.disease_label, []).append(v)

        for disease in self.class_labels:
            rec_a = result_a.per_class_recall.get(disease, 0.0)
            rec_b = result_b.per_class_recall.get(disease, 0.0)
            rec_c = result_c.per_class_recall.get(disease, 0.0)
            f1_b  = result_b.per_class_f1.get(disease, 0.0)
            f1_c  = result_c.per_class_f1.get(disease, 0.0)

            # Confusion targets for this disease from Model B CM
            confusion_targets = self._top_confusion_targets(disease, result_b)

            # Certainty profile for this disease's vectors
            dvecs = vecs_by_disease.get(disease, [])
            mean_cert  = float(np.mean([v.certainty for v in dvecs])) if dvecs else 0.0
            mean_amb   = float(np.mean([v.ambiguity_index for v in dvecs])) if dvecs else 0.0
            mean_contr = float(np.mean([v.contradiction_load for v in dvecs])) if dvecs else 0.0
            all_esc    = all(v.requires_biopsy for v in dvecs) if dvecs else False

            profiles.append(DiseaseFailureProfile(
                disease=disease,
                model_a_recall=rec_a,
                model_b_recall=rec_b,
                model_c_recall=rec_c,
                recall_drop_b_vs_a=rec_a - rec_b,
                recall_gain_c_vs_b=rec_c - rec_b,
                model_b_f1=f1_b,
                model_c_f1=f1_c,
                top_confusion_targets=confusion_targets,
                mean_certainty=mean_cert,
                mean_ambiguity=mean_amb,
                mean_contradiction=mean_contr,
                always_escalated=all_esc,
            ))

        return profiles

    def _top_confusion_targets(
        self,
        true_disease: str,
        result_b: ModelBResult,
    ) -> list[tuple[str, int]]:
        """
        Extract top confusion targets for a disease from the Model B
        confusion matrix.
        """
        labels = result_b.class_labels
        cm     = result_b.confusion_matrix
        if not cm or true_disease not in labels:
            return []

        row_idx = labels.index(true_disease)
        if row_idx >= len(cm):
            return []

        row = cm[row_idx]
        targets: list[tuple[str, int]] = []
        for col_idx, count in enumerate(row):
            if col_idx == row_idx or count == 0:
                continue
            pred_disease = labels[col_idx] if col_idx < len(labels) else str(col_idx)
            targets.append((pred_disease, count))

        return sorted(targets, key=lambda x: x[1], reverse=True)[:3]

    def _build_certainty_collapse(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> CertaintyCollapseProfile:
        """Characterise the certainty-collapse phenomenon."""
        if not vectors:
            return CertaintyCollapseProfile()

        certs = np.array([v.certainty for v in vectors])
        ambs  = np.array([v.ambiguity_index for v in vectors])

        mean_cert  = float(np.mean(certs))
        std_cert   = float(np.std(certs))
        cert_rate  = float(np.mean(certs >= 0.55))    # safe-triage floor
        mean_amb   = float(np.mean(ambs))
        amb_rate   = float(np.mean(ambs < 1.50))      # non-escalated fraction

        # Per-disease mean certainty
        per_dis: dict[str, float] = {}
        for v in vectors:
            per_dis.setdefault(v.disease_label, []).append(v.certainty)  # type: ignore[arg-type]
        per_dis_mean = {d: float(np.mean(c)) for d, c in per_dis.items()}  # type: ignore[arg-type]

        # Severity classification
        escalated_fraction = 1.0 - amb_rate
        if escalated_fraction >= 0.95:
            severity = "total"
        elif escalated_fraction >= 0.80:
            severity = "severe"
        elif escalated_fraction >= 0.50:
            severity = "partial"
        else:
            severity = "none"

        return CertaintyCollapseProfile(
            mean_certainty_overall=mean_cert,
            std_certainty_overall=std_cert,
            certainty_above_threshold_rate=cert_rate,
            mean_ambiguity_overall=mean_amb,
            ambiguity_below_threshold_rate=amb_rate,
            per_disease_mean_certainty=per_dis_mean,
            certainty_floor_barrier=float(np.min(certs)),
            collapse_severity=severity,
        )

    def _extract_confusion_zones(
        self,
        result_b: ModelBResult,
    ) -> list[ConfusionZonePair]:
        """Extract the most active confusion zones from Model B's CM."""
        labels = result_b.class_labels
        cm     = result_b.confusion_matrix
        if not cm:
            return []

        pairs: list[ConfusionZonePair] = []
        for i, row in enumerate(cm):
            true_disease = labels[i] if i < len(labels) else str(i)
            row_total = sum(row)
            for j, count in enumerate(row):
                if i == j or count == 0:
                    continue
                pred_disease = labels[j] if j < len(labels) else str(j)
                rate = count / row_total if row_total > 0 else 0.0
                pairs.append(ConfusionZonePair(
                    true_disease=true_disease,
                    predicted_disease=pred_disease,
                    confusion_count=count,
                    confusion_rate=rate,
                ))

        return sorted(pairs, key=lambda x: x.confusion_count, reverse=True)

    def _compute_feature_discriminability(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[FeatureDiscriminabilityScore]:
        """
        Compute one-way ANOVA F-ratio for each clinical feature across
        the 6 disease classes. Higher F-ratio = more discriminating.
        """
        feat_names = list(CLINICAL_FEATURE_NAMES)
        n_feat     = min(X.shape[1], len(feat_names))
        classes    = np.unique(y)
        scores: list[FeatureDiscriminabilityScore] = []

        for fi in range(n_feat):
            col    = X[:, fi]
            groups = [col[y == c] for c in classes if len(col[y == c]) > 0]

            grand_mean = float(np.mean(col))
            n_groups   = len(groups)

            # Between-class variance (weighted)
            between_var = sum(
                len(g) * (float(np.mean(g)) - grand_mean) ** 2
                for g in groups
            ) / max(n_groups - 1, 1)

            # Within-class variance (pooled)
            within_var = sum(
                np.sum((g - np.mean(g)) ** 2)
                for g in groups
            ) / max(len(col) - n_groups, 1)

            f_ratio = between_var / max(within_var, 1e-9)

            scores.append(FeatureDiscriminabilityScore(
                feature=feat_names[fi],
                between_class_var=float(between_var),
                within_class_var=float(within_var),
                f_ratio=float(f_ratio),
                rank=0,
            ))

        # Rank by F-ratio descending
        scores.sort(key=lambda x: x.f_ratio, reverse=True)
        for rank, s in enumerate(scores, start=1):
            s.rank = rank

        return scores

    def _escalation_breakdown(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> tuple[float, dict[str, float]]:
        """Compute escalation rate and trigger breakdown."""
        if not vectors:
            return 0.0, {}

        n_total  = len(vectors)
        n_esc    = sum(1 for v in vectors if v.requires_biopsy)
        esc_rate = n_esc / n_total

        # Approximation of trigger: ambiguity > 1.50 bits is dominant
        n_amb    = sum(1 for v in vectors if v.ambiguity_index > 1.50)
        n_cert   = sum(1 for v in vectors if v.certainty < 0.55)
        n_contr  = sum(1 for v in vectors if v.contradiction_load > 0.40)

        breakdown = {
            "ambiguity_triggered":     n_amb / n_total,
            "certainty_triggered":     n_cert / n_total,
            "contradiction_triggered": n_contr / n_total,
            "escalated":               esc_rate,
        }
        return esc_rate, breakdown

    def _synthesise_bottleneck(
        self,
        collapse: CertaintyCollapseProfile,
        profiles: list[DiseaseFailureProfile],
        esc_rate: float,
    ) -> str:
        """Produce a single-sentence root-cause summary."""
        if esc_rate >= 0.99:
            return (
                "Near-universal ambiguity escalation (≥99%) driven by insufficient "
                "clinical-feature certainty (mean ≈ {:.2f}) prevents symbolic signals "
                "from providing meaningful discrimination.".format(
                    collapse.mean_certainty_overall
                )
            )
        if collapse.is_severe():
            return (
                "Severe certainty collapse (escalation rate {:.1%}) limits symbolic "
                "feature contribution; ambiguity threshold calibration required.".format(
                    esc_rate
                )
            )
        hard = [p.disease for p in profiles if p.difficulty_tier() == "critical"]
        if hard:
            return (
                "Critical classification difficulty in {} disease(s) ({}) "
                "drives overall Model B underperformance; rule discrimination "
                "refinement targeted.".format(len(hard), ", ".join(hard))
            )
        return (
            "Moderate underperformance across multiple disease classes; "
            "ensemble calibration and threshold adjustment expected to close gap."
        )
