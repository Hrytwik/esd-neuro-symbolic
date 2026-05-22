"""
LightweightCounterfactualReasoner — feature-removal sensitivity analysis.

Answers clinically meaningful perturbation questions:
  "If koebner phenomenon were absent, would psoriasis remain the leading hypothesis?"
  "Which single feature, if removed, most destabilizes the current diagnosis?"
  "How fragile is this hypothesis to evidence loss?"

The reasoner operates by re-evaluating the evidence and certainty without
a nominated feature and comparing the resulting distribution against the
full-evidence baseline. It does not retrain or recalibrate any model —
it symbolically re-applies the activated rule set to the perturbed feature
profile, making the analysis fully transparent and deterministic.

Analysis types
--------------
1. Feature removal sensitivity   — perturb one feature at a time, observe
                                   change in leading-disease certainty
2. Hypothesis fragility          — minimum features to remove to dislodge
                                   the leading hypothesis (fragility depth)
3. Trajectory perturbation       — apply perturbation at a nominated stage
                                   and project the downstream state change
4. Critical feature identification — rank features by their individual
                                   contribution to the leading hypothesis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ── Perturbation result records ───────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureRemovalEffect:
    """Effect of removing a single feature from the clinical profile."""

    feature_name:          str
    baseline_certainty:    float    # certainty before removal
    perturbed_certainty:   float    # certainty after removal
    certainty_delta:       float    # baseline - perturbed (positive = this feature helps)
    leading_disease_stable: bool    # leading disease unchanged after removal
    alternative_leader:    str | None  # disease that takes over if unstable
    sensitivity_rank:      int       # 1 = most impactful removal


@dataclass
class HypothesisFragilityReport:
    """
    Fragility depth analysis for the leading disease hypothesis.
    Answers: how many features can be removed before the diagnosis changes?
    """

    disease:                str
    baseline_certainty:     float
    fragility_depth:        int          # min removals to dislodge leading disease
    critical_removal_set:   list[str]    # features whose joint removal dislodges
    is_robust:              bool         # fragility_depth > 1
    robustness_label:       str          # "robust" | "fragile" | "critically_fragile"


@dataclass(frozen=True)
class TrajectoryPerturbationResult:
    """Effect of a feature perturbation on trajectory state projection."""

    perturbed_feature:     str
    stage_of_perturbation: int
    baseline_state:        str
    projected_state:       str          # expected state after perturbation
    state_changed:         bool
    certainty_shift:       float        # signed delta
    clinical_implication:  str


@dataclass
class CounterfactualReport:
    """
    Complete counterfactual analysis for a single diagnostic case.
    Produced by LightweightCounterfactualReasoner.analyze().
    """

    leading_disease:       str
    baseline_certainty:    float
    feature_effects:       list[FeatureRemovalEffect]
    fragility:             HypothesisFragilityReport
    trajectory_effects:    list[TrajectoryPerturbationResult]
    most_critical_feature: str | None     # feature with highest sensitivity rank
    least_critical_feature: str | None    # feature that can be safely removed
    overall_stability:     str            # "stable" | "moderate" | "fragile"

    @property
    def stable_features(self) -> list[str]:
        """Features whose removal does NOT dislodge the leading disease."""
        return [e.feature_name for e in self.feature_effects if e.leading_disease_stable]

    @property
    def destabilizing_features(self) -> list[str]:
        """Features whose removal DOES dislodge the leading disease."""
        return [e.feature_name for e in self.feature_effects if not e.leading_disease_stable]


# ── Counterfactual reasoner ────────────────────────────────────────────────────

class LightweightCounterfactualReasoner:
    """
    Performs deterministic symbolic perturbation analysis on the diagnostic
    evidence profile.

    The reasoner accepts a re-evaluation callback rather than re-implementing
    the full evidence and certainty pipeline. This keeps it lightweight and
    decoupled from specific subsystem internals.

    Parameters
    ----------
    certainty_drop_threshold:
        Minimum certainty drop for a feature removal to be considered
        "impactful". Default: 0.05 (5 percentage points).
    fragility_search_depth:
        Maximum number of features tried in the fragility search. Default: 4.
    """

    def __init__(
        self,
        certainty_drop_threshold: float = 0.05,
        fragility_search_depth: int = 4,
    ) -> None:
        self._drop_threshold  = certainty_drop_threshold
        self._max_depth       = fragility_search_depth

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        feature_values:    dict[str, float],
        leading_disease:   str,
        baseline_certainty: float,
        second_disease:    str,
        reeval_fn:         Callable[[dict[str, float]], tuple[str, float]],
        current_state:     str = "CERTAINTY_STABILIZATION",
        stage:             int = 5,
    ) -> CounterfactualReport:
        """
        Run the full counterfactual analysis.

        Parameters
        ----------
        feature_values:
            Dict of feature_name → fuzzy_grade (0.0–1.0) for the current case.
        leading_disease:
            The current top-ranked disease hypothesis.
        baseline_certainty:
            Certainty of the leading disease under full evidence.
        second_disease:
            The second-ranked hypothesis (used to detect leadership changes).
        reeval_fn:
            Callable that accepts a perturbed feature dict and returns
            (new_leading_disease: str, new_max_certainty: float).
            The caller provides this to keep the reasoner lightweight and
            decoupled from subsystem internals.
        current_state:
            FSM state at time of perturbation (for trajectory projection).
        stage:
            Pipeline stage at which perturbation is applied.
        """
        # ── 1. Feature-removal sensitivity ───────────────────────────────────
        feature_effects = self._feature_removal_sensitivity(
            feature_values, leading_disease, baseline_certainty, reeval_fn
        )

        # ── 2. Hypothesis fragility ───────────────────────────────────────────
        fragility = self._fragility_analysis(
            feature_values, leading_disease, baseline_certainty,
            second_disease, reeval_fn
        )

        # ── 3. Trajectory perturbation ────────────────────────────────────────
        trajectory_effects = self._trajectory_perturbation(
            feature_effects, leading_disease, baseline_certainty,
            current_state, stage
        )

        # ── 4. Summary metadata ───────────────────────────────────────────────
        most_critical = (
            feature_effects[0].feature_name
            if feature_effects else None
        )
        # Least critical: smallest positive delta (i.e. least helpful, stable removal)
        stable_effects = [e for e in feature_effects if e.leading_disease_stable]
        least_critical = (
            min(stable_effects, key=lambda e: e.certainty_delta).feature_name
            if stable_effects else None
        )

        # Overall stability based on fragility depth
        overall_stability = (
            "stable"   if fragility.fragility_depth >= 3 else
            "moderate" if fragility.fragility_depth == 2 else
            "fragile"
        )

        return CounterfactualReport(
            leading_disease=leading_disease,
            baseline_certainty=baseline_certainty,
            feature_effects=feature_effects,
            fragility=fragility,
            trajectory_effects=trajectory_effects,
            most_critical_feature=most_critical,
            least_critical_feature=least_critical,
            overall_stability=overall_stability,
        )

    def feature_question(
        self,
        feature_name:      str,
        feature_values:    dict[str, float],
        leading_disease:   str,
        baseline_certainty: float,
        reeval_fn:         Callable[[dict[str, float]], tuple[str, float]],
    ) -> str:
        """
        Answer a single natural-language counterfactual question.

        Example output:
          "If koebner phenomenon were absent, psoriasis certainty would drop
           from 0.847 to 0.621 — the diagnosis remains stable."
        """
        perturbed = {k: v for k, v in feature_values.items() if k != feature_name}
        new_leader, new_cert = reeval_fn(perturbed)
        delta = baseline_certainty - new_cert
        stable = (new_leader == leading_disease)

        disease_label = leading_disease.replace("_", " ")
        feature_label = feature_name.replace("_", " ")

        if stable:
            return (
                f"If {feature_label} were absent, {disease_label} certainty "
                f"would drop from {baseline_certainty:.3f} to {new_cert:.3f} "
                f"(Δ={delta:+.3f}) — the diagnosis remains stable."
            )
        else:
            alt_label = new_leader.replace("_", " ")
            return (
                f"If {feature_label} were absent, {disease_label} certainty "
                f"would drop from {baseline_certainty:.3f} to {new_cert:.3f} "
                f"(Δ={delta:+.3f}) — leadership shifts to {alt_label}. "
                f"This feature is diagnostically critical."
            )

    # ── Internal analysis methods ─────────────────────────────────────────────

    def _feature_removal_sensitivity(
        self,
        feature_values:    dict[str, float],
        leading_disease:   str,
        baseline_certainty: float,
        reeval_fn:         Callable[[dict[str, float]], tuple[str, float]],
    ) -> list[FeatureRemovalEffect]:
        """
        Remove each feature in turn; record certainty delta and leadership stability.
        Returns effects sorted by certainty delta (most impactful first).
        """
        effects: list[FeatureRemovalEffect] = []

        for feature_name in feature_values:
            perturbed = {k: v for k, v in feature_values.items() if k != feature_name}
            new_leader, new_cert = reeval_fn(perturbed)
            delta = baseline_certainty - new_cert
            stable = (new_leader == leading_disease)
            alt = new_leader if not stable else None

            effects.append(FeatureRemovalEffect(
                feature_name=feature_name,
                baseline_certainty=baseline_certainty,
                perturbed_certainty=new_cert,
                certainty_delta=delta,
                leading_disease_stable=stable,
                alternative_leader=alt,
                sensitivity_rank=0,   # assigned below
            ))

        # Sort by descending certainty delta; assign ranks
        effects.sort(key=lambda e: e.certainty_delta, reverse=True)
        ranked = []
        for rank_idx, e in enumerate(effects, start=1):
            ranked.append(FeatureRemovalEffect(
                feature_name=e.feature_name,
                baseline_certainty=e.baseline_certainty,
                perturbed_certainty=e.perturbed_certainty,
                certainty_delta=e.certainty_delta,
                leading_disease_stable=e.leading_disease_stable,
                alternative_leader=e.alternative_leader,
                sensitivity_rank=rank_idx,
            ))
        return ranked

    def _fragility_analysis(
        self,
        feature_values:    dict[str, float],
        leading_disease:   str,
        baseline_certainty: float,
        second_disease:    str,
        reeval_fn:         Callable[[dict[str, float]], tuple[str, float]],
    ) -> HypothesisFragilityReport:
        """
        Greedy search for the minimum removal set that dislodges the leading
        disease. Removes the most-impactful feature first (greedy, not exhaustive).
        """
        remaining = dict(feature_values)
        removal_order: list[str] = []
        depth = 0

        for _ in range(self._max_depth):
            # Find single most impactful remaining feature
            best_feature: str | None = None
            best_delta = -float("inf")
            best_leader = leading_disease

            for feature_name in remaining:
                perturbed = {k: v for k, v in remaining.items() if k != feature_name}
                new_leader, new_cert = reeval_fn(perturbed)
                delta = (remaining.get(feature_name, 0.0))   # proxy: prefer higher-value features
                # Use certainty drop as primary signal
                cert_drop = baseline_certainty - new_cert
                if cert_drop > best_delta:
                    best_delta   = cert_drop
                    best_feature = feature_name
                    best_leader  = new_leader

                # Immediately stop if removal dislodges
                if new_leader != leading_disease:
                    removal_order.append(feature_name)
                    depth = len(removal_order)
                    return HypothesisFragilityReport(
                        disease=leading_disease,
                        baseline_certainty=baseline_certainty,
                        fragility_depth=depth,
                        critical_removal_set=list(removal_order),
                        is_robust=(depth > 1),
                        robustness_label=self._robustness_label(depth),
                    )

            if best_feature is None:
                break

            removal_order.append(best_feature)
            del remaining[best_feature]

        # If we exhausted all search depth without dislodging, it's robust
        depth = self._max_depth + 1  # more robust than max depth
        return HypothesisFragilityReport(
            disease=leading_disease,
            baseline_certainty=baseline_certainty,
            fragility_depth=depth,
            critical_removal_set=[],
            is_robust=True,
            robustness_label="robust",
        )

    def _trajectory_perturbation(
        self,
        feature_effects:   list[FeatureRemovalEffect],
        leading_disease:   str,
        baseline_certainty: float,
        current_state:     str,
        stage:             int,
    ) -> list[TrajectoryPerturbationResult]:
        """
        Project the expected FSM state change if each high-impact feature
        were removed at the current pipeline stage.

        Uses heuristic state projection (not full FSM re-evaluation) to remain
        lightweight — the intent is to flag which features, if absent, would
        push the trajectory toward biopsy escalation.
        """
        results: list[TrajectoryPerturbationResult] = []

        for effect in feature_effects[:5]:   # limit to top-5 for tractability
            projected_state, implication = self._project_state(
                current_state, effect.perturbed_certainty,
                effect.leading_disease_stable, effect.certainty_delta
            )
            results.append(TrajectoryPerturbationResult(
                perturbed_feature=effect.feature_name,
                stage_of_perturbation=stage,
                baseline_state=current_state,
                projected_state=projected_state,
                state_changed=(projected_state != current_state),
                certainty_shift=-effect.certainty_delta,   # negative = drop
                clinical_implication=implication,
            ))

        return results

    # ── Heuristic state projector ─────────────────────────────────────────────

    @staticmethod
    def _project_state(
        current_state: str,
        perturbed_certainty: float,
        stable: bool,
        delta: float,
    ) -> tuple[str, str]:
        """
        Heuristically project the FSM state that would result from a
        certainty drop. Returns (projected_state, clinical_implication).
        """
        if not stable:
            # Leadership change → ambiguity escalation
            return (
                "AMBIGUITY_ESCALATION",
                "Removal of this feature shifts diagnostic leadership, "
                "projecting trajectory into ambiguity escalation. "
                "Biopsy confirmation becomes likely.",
            )

        if perturbed_certainty < 0.55:
            return (
                "BIOPSY_ESCALATION",
                "Certainty would fall below safe threshold, projecting "
                "trajectory into biopsy escalation.",
            )

        if perturbed_certainty < 0.70:
            return (
                "PARTIAL_ALIGNMENT",
                "Certainty drop moves trajectory toward partial alignment — "
                "diagnosis weakened but not reversed.",
            )

        if delta > 0.10:
            return (
                "REINFORCING_ALIGNMENT",
                "Certainty remains high despite removal; current state "
                "is likely maintained with slight regression.",
            )

        # Minor or negligible effect
        return (
            current_state,
            "Negligible effect on trajectory — feature contributes "
            "marginally to the current diagnostic state.",
        )

    @staticmethod
    def _robustness_label(depth: int) -> str:
        if depth >= 3:
            return "robust"
        if depth == 2:
            return "fragile"
        return "critically_fragile"
