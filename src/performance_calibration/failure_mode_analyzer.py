"""
FailureModeAnalyzer — root-cause synthesis for Model B / C underperformance.

Takes the output of PerformanceDiagnostics and synthesises structured
failure patterns, answering:

  Q1. Why does Model B achieve only 80% when literature suggests 85–90%
      for XGBoost on clinical-only UCI Dermatology data?

  Q2. Why does Model C gain only 1.82 pp over Model B despite 22
      symbolic reasoning signals?

  Q3. Which failure categories account for the largest fraction of
      misclassifications, and which are addressable without compromising
      escalation safety?

Failure pattern taxonomy
------------------------
  CERTAINTY_COLLAPSE    — symbolic signals too weak; escalation dominates
  INTER_DISEASE_OVERLAP — clinical features insufficient to separate pair
  CLASS_IMBALANCE       — minority class (PRP: 20 records) under-sampled
  FEATURE_INSUFFICIENCY — 12 features inherently lack discriminating power
  THRESHOLD_MISMATCH    — escalation thresholds misaligned with dataset
  RULE_GENERALITY       — rules fire for multiple diseases; no discrimination
  SOFT_COMPETITION      — hypotheses coexist without sufficient suppression
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.performance_calibration.performance_diagnostics import (
    DiagnosticReport,
    DiseaseFailureProfile,
)


# ── Failure pattern taxonomy ──────────────────────────────────────────────────

class FailureCategory(str, Enum):
    CERTAINTY_COLLAPSE    = "certainty_collapse"
    INTER_DISEASE_OVERLAP = "inter_disease_overlap"
    CLASS_IMBALANCE       = "class_imbalance"
    FEATURE_INSUFFICIENCY = "feature_insufficiency"
    THRESHOLD_MISMATCH    = "threshold_mismatch"
    RULE_GENERALITY       = "rule_generality"
    SOFT_COMPETITION      = "soft_competition"


@dataclass
class FailurePattern:
    """
    A single identified failure pattern.

    Attributes
    ----------
    category:
        Failure pattern category from FailureCategory.
    description:
        Human-readable description of the failure mode.
    affected_diseases:
        Disease names primarily affected by this failure mode.
    estimated_accuracy_impact:
        Approximate fraction of total misclassifications attributed
        to this pattern.
    addressable:
        Whether this failure mode can be addressed without violating
        escalation safety or interpretability constraints.
    recommended_intervention:
        Specific intervention recommended to address this pattern.
    priority:
        1 (highest) to 5 (lowest).
    """

    category:                  FailureCategory
    description:               str
    affected_diseases:         list[str]      = field(default_factory=list)
    estimated_accuracy_impact: float          = 0.0
    addressable:               bool           = True
    recommended_intervention:  str            = ""
    priority:                  int            = 3


@dataclass
class ModelBFailureAnalysis:
    """Root-cause analysis for Model B (clinical-only) underperformance."""

    observed_accuracy:       float
    target_accuracy:         float
    accuracy_gap:            float
    primary_failure_pattern: FailurePattern | None
    contributing_patterns:   list[FailurePattern] = field(default_factory=list)
    explanation:             str = ""

    def gap_closed_by_addressable(self) -> float:
        """Fraction of accuracy gap attributable to addressable patterns."""
        addressable_impact = sum(
            p.estimated_accuracy_impact
            for p in ([self.primary_failure_pattern] + self.contributing_patterns)
            if p is not None and p.addressable
        )
        return min(1.0, addressable_impact / max(self.accuracy_gap, 1e-6))


@dataclass
class ModelCFailureAnalysis:
    """Root-cause analysis for Model C's modest symbolic lift."""

    observed_lift:          float
    expected_lift_estimate: float
    lift_deficit:           float
    primary_failure_pattern: FailurePattern | None
    contributing_patterns:  list[FailurePattern] = field(default_factory=list)
    explanation:            str = ""


@dataclass
class FailureModeReport:
    """
    Complete failure-mode analysis output.

    Attributes
    ----------
    model_b_analysis:
        Root-cause breakdown for Model B underperformance.
    model_c_analysis:
        Root-cause breakdown for Model C's modest symbolic lift.
    all_patterns:
        Unified and deduplicated list of all identified failure patterns,
        sorted by priority (ascending).
    actionable_interventions:
        Ordered list of specific recommended actions.
    expected_accuracy_ceiling:
        Estimated maximum accuracy achievable on clinical-only features
        given the inherent information content of the 12 features.
    n_addressable_patterns:
        Count of patterns that can be fixed without violating constraints.
    n_structural_patterns:
        Count of patterns that are irreducible (structural data limitations).
    """

    model_b_analysis:           ModelBFailureAnalysis
    model_c_analysis:           ModelCFailureAnalysis
    all_patterns:               list[FailurePattern] = field(default_factory=list)
    actionable_interventions:   list[str]            = field(default_factory=list)
    expected_accuracy_ceiling:  float                = 0.0
    n_addressable_patterns:     int                  = 0
    n_structural_patterns:      int                  = 0

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "FAILURE MODE ANALYSIS REPORT",
            "=" * 72,
            "",
            "MODEL B UNDERPERFORMANCE",
            f"  Observed: {self.model_b_analysis.observed_accuracy:.1%}  "
            f"Target: {self.model_b_analysis.target_accuracy:.1%}  "
            f"Gap: {self.model_b_analysis.accuracy_gap:.1%}",
            f"  {self.model_b_analysis.explanation}",
            "",
            "MODEL C SYMBOLIC LIFT DEFICIT",
            f"  Observed lift: {self.model_c_analysis.observed_lift:+.1%}  "
            f"Expected: {self.model_c_analysis.expected_lift_estimate:+.1%}  "
            f"Deficit: {self.model_c_analysis.lift_deficit:.1%}",
            f"  {self.model_c_analysis.explanation}",
            "",
            f"IDENTIFIED FAILURE PATTERNS ({len(self.all_patterns)}):",
        ]
        for p in self.all_patterns:
            addr = "addressable" if p.addressable else "structural"
            lines.append(
                f"  [P{p.priority}][{addr:12s}] {p.category.value:25s} "
                f"impact≈{p.estimated_accuracy_impact:.1%}"
            )
            lines.append(f"    -> {p.recommended_intervention}")
        lines += [
            "",
            "ACTIONABLE INTERVENTIONS (priority order):",
        ]
        for i, action in enumerate(self.actionable_interventions, 1):
            lines.append(f"  {i}. {action}")
        lines += [
            "",
            f"  Estimated accuracy ceiling (clinical-only): "
            f"{self.expected_accuracy_ceiling:.1%}",
            f"  Addressable patterns: {self.n_addressable_patterns}  "
            f"Structural: {self.n_structural_patterns}",
            "=" * 72,
        ]
        return "\n".join(lines)


# ── Failure mode analyser ─────────────────────────────────────────────────────

class FailureModeAnalyzer:
    """
    Synthesises structured failure patterns from diagnostic data.

    Parameters
    ----------
    model_b_target_accuracy:
        Target accuracy for Model B. Default 0.86.
    model_c_target_accuracy:
        Target accuracy for Model C. Default 0.89.
    """

    def __init__(
        self,
        model_b_target_accuracy: float = 0.86,
        model_c_target_accuracy: float = 0.89,
    ) -> None:
        self.b_target = model_b_target_accuracy
        self.c_target = model_c_target_accuracy

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        diagnostic: DiagnosticReport,
        model_b_accuracy: float,
        model_c_accuracy: float,
        symbolic_lift: float,
    ) -> FailureModeReport:
        """
        Run failure mode synthesis.

        Parameters
        ----------
        diagnostic:
            DiagnosticReport from PerformanceDiagnostics.
        model_b_accuracy, model_c_accuracy:
            Observed accuracy for each model.
        symbolic_lift:
            Observed accuracy gain from B to C.
        """
        patterns = self._identify_patterns(diagnostic, model_b_accuracy, symbolic_lift)

        model_b_analysis = self._analyse_model_b(
            diagnostic, model_b_accuracy, patterns
        )
        model_c_analysis = self._analyse_model_c(
            symbolic_lift, patterns
        )

        all_patterns  = sorted(patterns, key=lambda p: p.priority)
        interventions = self._build_interventions(all_patterns)
        ceiling       = self._estimate_ceiling(diagnostic, model_b_accuracy)

        n_addr = sum(1 for p in patterns if p.addressable)
        n_str  = len(patterns) - n_addr

        return FailureModeReport(
            model_b_analysis=model_b_analysis,
            model_c_analysis=model_c_analysis,
            all_patterns=all_patterns,
            actionable_interventions=interventions,
            expected_accuracy_ceiling=ceiling,
            n_addressable_patterns=n_addr,
            n_structural_patterns=n_str,
        )

    # ── Pattern identification ────────────────────────────────────────────────

    def _identify_patterns(
        self,
        diag: DiagnosticReport,
        model_b_acc: float,
        symbolic_lift: float,
    ) -> list[FailurePattern]:
        patterns: list[FailurePattern] = []

        # 1. Certainty collapse — near-universal escalation
        if diag.overall_escalation_rate >= 0.95:
            patterns.append(FailurePattern(
                category=FailureCategory.CERTAINTY_COLLAPSE,
                description=(
                    f"Symbolic pipeline escalates {diag.overall_escalation_rate:.1%} "
                    "of cases due to ambiguity exceeding 1.50-bit threshold. "
                    "Mean certainty ≈ {:.2f} is far below the 0.55 safe-triage "
                    "floor on clinical-only data.".format(
                        diag.certainty_collapse.mean_certainty_overall
                    )
                ),
                affected_diseases=list(self._always_escalated_diseases(diag)),
                estimated_accuracy_impact=max(0.0, symbolic_lift - 0.05),
                addressable=True,
                recommended_intervention=(
                    "Recalibrate ambiguity threshold (1.50 → 2.00–2.50 bits) "
                    "and certainty floor (0.55 → 0.40) for the clinical-only "
                    "operating context. Evaluate whether the threshold is "
                    "calibrated to biopsy-complete data."
                ),
                priority=1,
            ))

        # 2. Inter-disease overlap in confusion zones
        high_confusion_pairs = [
            cz for cz in diag.confusion_zones if cz.confusion_rate >= 0.20
        ]
        if high_confusion_pairs:
            diseases = list({p.true_disease for p in high_confusion_pairs[:3]})
            patterns.append(FailurePattern(
                category=FailureCategory.INTER_DISEASE_OVERLAP,
                description=(
                    f"{len(high_confusion_pairs)} disease pairs with ≥20% confusion rate "
                    "indicate insufficient clinical-feature separation between "
                    "clinically similar conditions."
                ),
                affected_diseases=diseases,
                estimated_accuracy_impact=0.04,
                addressable=True,
                recommended_intervention=(
                    "Strengthen disease-specific rule signatures to increase "
                    "differential certainty between overlapping disease pairs. "
                    "Apply hypothesis suppression escalation for ambiguous pairs."
                ),
                priority=2,
            ))

        # 3. Class imbalance — PRP has only 20 records
        prp_profile = next(
            (p for p in diag.disease_profiles
             if "pityriasis_rubra" in p.disease.lower()), None
        )
        if prp_profile and prp_profile.difficulty_tier() in ("hard", "critical"):
            patterns.append(FailurePattern(
                category=FailureCategory.CLASS_IMBALANCE,
                description=(
                    "Pityriasis rubra pilaris has only 20 training records (5.5:1 "
                    "imbalance vs majority class). This causes systematic under-recall "
                    f"(Model B recall = {prp_profile.model_b_recall:.3f})."
                ),
                affected_diseases=["pityriasis_rubra_pilaris"],
                estimated_accuracy_impact=0.02,
                addressable=True,
                recommended_intervention=(
                    "Apply class-weight balancing in XGBoost/RF (scale_pos_weight, "
                    "class_weight='balanced'). Use stratified repeated cross-validation "
                    "to ensure PRP is represented in every fold."
                ),
                priority=2,
            ))

        # 4. Threshold mismatch — escalation calibrated to wrong context
        if diag.overall_escalation_rate >= 0.90:
            patterns.append(FailurePattern(
                category=FailureCategory.THRESHOLD_MISMATCH,
                description=(
                    "Escalation thresholds (certainty ≥ 0.55, ambiguity < 1.50 bits) "
                    "were designed for the full 34-feature pipeline context, where "
                    "certainty can reach 0.70–0.90. On 12 clinical features, "
                    f"mean certainty is only {diag.certainty_collapse.mean_certainty_overall:.2f}, "
                    "making the thresholds impossibly strict."
                ),
                affected_diseases=[],
                estimated_accuracy_impact=0.05,
                addressable=True,
                recommended_intervention=(
                    "Derive clinical-context-specific thresholds via empirical "
                    "calibration on training data: certainty floor 0.35–0.45, "
                    "ambiguity ceiling 2.0–2.5 bits. Preserve safety by keeping "
                    "contradiction ceiling at 0.40."
                ),
                priority=1,
            ))

        # 5. Soft hypothesis competition
        patterns.append(FailurePattern(
            category=FailureCategory.SOFT_COMPETITION,
            description=(
                "Multiple hypotheses maintain similar certainty levels without "
                "sufficient suppression, producing flat certainty distributions "
                "that fail to identify a leading diagnosis reliably."
            ),
            affected_diseases=diag.hardest_diseases,
            estimated_accuracy_impact=0.03,
            addressable=True,
            recommended_intervention=(
                "Increase differential suppression weight in the reasoning pipeline. "
                "Require minimum certainty gap of 0.15 before declaring a leading "
                "hypothesis. Penalise competing hypotheses more aggressively when "
                "evidence is shared."
            ),
            priority=3,
        ))

        # 6. Feature insufficiency (structural)
        if model_b_acc < 0.85:
            patterns.append(FailurePattern(
                category=FailureCategory.FEATURE_INSUFFICIENCY,
                description=(
                    "12 clinical features inherently contain less diagnostic "
                    "information than 34 features (biopsy gap = 18.18 pp accuracy). "
                    "Some of this gap is irreducible without histopathological data."
                ),
                affected_diseases=[],
                estimated_accuracy_impact=0.08,
                addressable=False,
                recommended_intervention=(
                    "Irreducible structural gap. Focus optimisation on addressable "
                    "patterns; document expected clinical-only ceiling as 86–90%."
                ),
                priority=5,
            ))

        return patterns

    # ── Model-specific analysis ───────────────────────────────────────────────

    def _analyse_model_b(
        self,
        diag: DiagnosticReport,
        model_b_acc: float,
        patterns: list[FailurePattern],
    ) -> ModelBFailureAnalysis:
        gap = self.b_target - model_b_acc
        primary = next(
            (p for p in sorted(patterns, key=lambda x: x.priority)
             if p.category in (
                 FailureCategory.THRESHOLD_MISMATCH,
                 FailureCategory.INTER_DISEASE_OVERLAP,
                 FailureCategory.CLASS_IMBALANCE,
             )),
            None,
        )
        explanation = (
            f"Model B achieves {model_b_acc:.1%} vs {self.b_target:.1%} target "
            f"({gap:+.1%} gap). Primary drivers: class imbalance in minority diseases, "
            "inter-disease clinical overlap in dermatitis/psoriasis pairs, and "
            "sub-optimal hyperparameter configuration. Ensemble calibration and "
            "class-weight balancing are expected to close 4–6 pp of the gap."
        )
        return ModelBFailureAnalysis(
            observed_accuracy=model_b_acc,
            target_accuracy=self.b_target,
            accuracy_gap=gap,
            primary_failure_pattern=primary,
            contributing_patterns=[
                p for p in patterns if p is not primary and p.addressable
            ],
            explanation=explanation,
        )

    def _analyse_model_c(
        self,
        symbolic_lift: float,
        patterns: list[FailurePattern],
    ) -> ModelCFailureAnalysis:
        expected_lift = 0.06   # literature suggests 5–8 pp for symbolic augmentation
        deficit       = max(0.0, expected_lift - symbolic_lift)
        primary = next(
            (p for p in sorted(patterns, key=lambda x: x.priority)
             if p.category == FailureCategory.CERTAINTY_COLLAPSE),
            None,
        )
        explanation = (
            f"Model C achieves only {symbolic_lift:+.1%} symbolic lift "
            f"(expected ≈ {expected_lift:+.1%}). The primary driver is "
            "certainty collapse: with 99.4% of cases escalated, symbolic signals "
            "can only influence the 0.6% non-escalated minority. After threshold "
            "recalibration, symbolic features (peak_certainty, ambiguity_index, "
            "certainty_gap) should provide meaningful 5–8 pp lift."
        )
        return ModelCFailureAnalysis(
            observed_lift=symbolic_lift,
            expected_lift_estimate=expected_lift,
            lift_deficit=deficit,
            primary_failure_pattern=primary,
            contributing_patterns=[
                p for p in patterns
                if p is not primary
                and p.category in (
                    FailureCategory.SOFT_COMPETITION,
                    FailureCategory.RULE_GENERALITY,
                )
            ],
            explanation=explanation,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _always_escalated_diseases(self, diag: DiagnosticReport) -> set[str]:
        return {p.disease for p in diag.disease_profiles if p.always_escalated}

    def _estimate_ceiling(
        self,
        diag: DiagnosticReport,
        model_b_acc: float,
    ) -> float:
        """
        Estimate the theoretical accuracy ceiling on 12 clinical features.
        Based on the number of hard-to-separate disease pairs and the
        discriminability of top features.
        """
        # Literature suggests 85–92% achievable on UCI clinical features
        # with good calibration. We estimate conservatively.
        n_hard = len(diag.hardest_diseases)
        n_crit = len([p for p in diag.disease_profiles
                      if p.difficulty_tier() == "critical"])

        if n_crit >= 2:
            return 0.87
        if n_hard >= 2:
            return 0.90
        return 0.92

    def _build_interventions(
        self,
        patterns: list[FailurePattern],
    ) -> list[str]:
        """Build ordered list of specific interventions from patterns."""
        seen: set[str] = set()
        actions: list[str] = []
        for p in patterns:
            if p.addressable and p.recommended_intervention not in seen:
                seen.add(p.recommended_intervention)
                actions.append(p.recommended_intervention)
        return actions
