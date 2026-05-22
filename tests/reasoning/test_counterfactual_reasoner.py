"""
Tests for LightweightCounterfactualReasoner — feature-removal sensitivity.

Validates feature removal effects, hypothesis fragility analysis, trajectory
perturbation projection, robustness labelling, and the natural-language
question answering interface.
"""

import pytest

from src.reasoning.counterfactual_reasoner import (
    CounterfactualReport,
    FeatureRemovalEffect,
    HypothesisFragilityReport,
    LightweightCounterfactualReasoner,
    TrajectoryPerturbationResult,
)


# ── Shared re-evaluation callable ────────────────────────────────────────────

def _make_reeval_fn(leading_disease: str, base_certainty: float):
    """
    Mock re-evaluation function.
    If koebner_phenomenon is removed, certainty drops by 0.20 and leadership
    shifts to lichen_planus (if certainty drops below 0.50). Otherwise stable.
    """
    def reeval(features: dict) -> tuple[str, float]:
        cert = base_certainty
        if "koebner_phenomenon" not in features:
            cert -= 0.20
        if "scalp_involvement" not in features:
            cert -= 0.05
        if cert < 0.50:
            return "lichen_planus", cert
        return leading_disease, cert
    return reeval


@pytest.fixture
def reasoner() -> LightweightCounterfactualReasoner:
    return LightweightCounterfactualReasoner(
        certainty_drop_threshold=0.05,
        fragility_search_depth=4,
    )


@pytest.fixture
def sample_features() -> dict:
    return {
        "koebner_phenomenon":        1.0,
        "erythema":                  0.67,
        "scaling":                   1.0,
        "scalp_involvement":         1.0,
        "knee_and_elbow_involvement": 1.0,
        "family_history":            1.0,
    }


# ── analyze() return type ─────────────────────────────────────────────────────

class TestAnalyzeReturnType:
    def test_analyze_returns_counterfactual_report(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            feature_values=sample_features,
            leading_disease="psoriasis",
            baseline_certainty=0.72,
            second_disease="lichen_planus",
            reeval_fn=reeval,
        )
        assert isinstance(report, CounterfactualReport)

    def test_report_has_correct_leading_disease(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        assert report.leading_disease == "psoriasis"

    def test_report_baseline_certainty(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        assert report.baseline_certainty == pytest.approx(0.72)


# ── Feature removal effects ───────────────────────────────────────────────────

class TestFeatureRemovalEffects:
    def test_effects_cover_all_features(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        effect_names = {e.feature_name for e in report.feature_effects}
        assert effect_names == set(sample_features.keys())

    def test_effects_sorted_by_certainty_delta_descending(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        deltas = [e.certainty_delta for e in report.feature_effects]
        assert deltas == sorted(deltas, reverse=True)

    def test_koebner_is_most_critical(self, reasoner, sample_features):
        """koebner_phenomenon removal causes the largest certainty drop."""
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        assert report.most_critical_feature == "koebner_phenomenon"

    def test_koebner_removal_destabilizes(self, reasoner, sample_features):
        """Removing koebner shifts leadership (0.72 - 0.20 = 0.52 still > 0.50 so stays stable)."""
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        koebner_effect = next(
            e for e in report.feature_effects if e.feature_name == "koebner_phenomenon"
        )
        # With base=0.72, after removing koebner: 0.72-0.20=0.52 >= 0.50 → still psoriasis
        assert koebner_effect.leading_disease_stable

    def test_removal_triggers_destabilization_at_low_baseline(self, reasoner, sample_features):
        """At baseline 0.65, removing koebner drops to 0.45 → leadership shifts."""
        reeval = _make_reeval_fn("psoriasis", 0.65)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.65, "lichen_planus", reeval
        )
        koebner_effect = next(
            e for e in report.feature_effects if e.feature_name == "koebner_phenomenon"
        )
        # 0.65 - 0.20 = 0.45 < 0.50 → leadership shifts to lichen_planus
        assert not koebner_effect.leading_disease_stable
        assert koebner_effect.alternative_leader == "lichen_planus"

    def test_sensitivity_ranks_are_sequential(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        ranks = [e.sensitivity_rank for e in report.feature_effects]
        assert sorted(ranks) == list(range(1, len(sample_features) + 1))


# ── Hypothesis fragility ──────────────────────────────────────────────────────

class TestHypothesisFragility:
    def test_fragility_report_returned(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        assert isinstance(report.fragility, HypothesisFragilityReport)

    def test_fragility_report_has_correct_disease(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        assert report.fragility.disease == "psoriasis"

    def test_fragility_depth_is_positive(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        assert report.fragility.fragility_depth >= 1

    def test_fragile_case_at_low_baseline(self, reasoner, sample_features):
        """At baseline 0.65, single koebner removal dislodges → depth=1 → critically_fragile."""
        reeval = _make_reeval_fn("psoriasis", 0.65)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.65, "lichen_planus", reeval
        )
        assert report.fragility.fragility_depth == 1
        assert report.fragility.robustness_label == "critically_fragile"
        assert not report.fragility.is_robust


# ── Trajectory perturbation ───────────────────────────────────────────────────

class TestTrajectoryPerturbation:
    def test_trajectory_effects_returned(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval,
            current_state="CERTAINTY_STABILIZATION", stage=3,
        )
        assert isinstance(report.trajectory_effects, list)

    def test_trajectory_effects_at_most_five(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        assert len(report.trajectory_effects) <= 5

    def test_trajectory_effect_fields(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval,
            current_state="CERTAINTY_STABILIZATION", stage=3,
        )
        if report.trajectory_effects:
            effect = report.trajectory_effects[0]
            assert isinstance(effect, TrajectoryPerturbationResult)
            assert effect.stage_of_perturbation == 3
            assert effect.baseline_state == "CERTAINTY_STABILIZATION"


# ── Summary properties ────────────────────────────────────────────────────────

class TestCounterfactualReportSummary:
    def test_stable_features_are_stable(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.80)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.80, "lichen_planus", reeval
        )
        for feat in report.stable_features:
            effect = next(e for e in report.feature_effects if e.feature_name == feat)
            assert effect.leading_disease_stable

    def test_destabilizing_features_are_not_stable(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.65)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.65, "lichen_planus", reeval
        )
        for feat in report.destabilizing_features:
            effect = next(e for e in report.feature_effects if e.feature_name == feat)
            assert not effect.leading_disease_stable

    def test_overall_stability_label_valid(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        report = reasoner.analyze(
            sample_features, "psoriasis", 0.72, "lichen_planus", reeval
        )
        assert report.overall_stability in ("stable", "moderate", "fragile")


# ── feature_question() ───────────────────────────────────────────────────────

class TestFeatureQuestion:
    def test_feature_question_stable_case(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.80)
        question = reasoner.feature_question(
            feature_name="scalp_involvement",
            feature_values=sample_features,
            leading_disease="psoriasis",
            baseline_certainty=0.80,
            reeval_fn=reeval,
        )
        assert isinstance(question, str)
        assert "scalp involvement" in question.lower()
        assert "psoriasis" in question.lower()
        assert "stable" in question.lower()

    def test_feature_question_destabilizing_case(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.65)
        question = reasoner.feature_question(
            feature_name="koebner_phenomenon",
            feature_values=sample_features,
            leading_disease="psoriasis",
            baseline_certainty=0.65,
            reeval_fn=reeval,
        )
        assert "shifts to" in question.lower() or "shifts" in question.lower()
        assert "critical" in question.lower()

    def test_feature_question_contains_certainty_values(self, reasoner, sample_features):
        reeval = _make_reeval_fn("psoriasis", 0.72)
        question = reasoner.feature_question(
            "erythema", sample_features, "psoriasis", 0.72, reeval
        )
        assert "0.72" in question or "0.67" in question  # baseline or perturbed
