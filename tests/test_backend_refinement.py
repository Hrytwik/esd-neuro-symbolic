"""
tests/test_backend_refinement.py
==================================
Integration tests for the CASDRE backend-refinement package.

All tests use small synthetic datasets that mimic UCI Dermatology structure:
  - n ≈ 120 cases, 6 disease classes, 12 clinical features (values 0–3)

Tests verify:
  - All modules import and instantiate without error
  - Core analyse / evaluate / compile methods return expected data types
  - Non-negotiable safety constraints are satisfied in every output
  - Summary text is generated without raising exceptions
  - Critical numerical invariants hold (e.g. contradiction ceiling ≤ 0.40)
"""

from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data factory
# ──────────────────────────────────────────────────────────────────────────────

_DISEASE_LABELS = [
    "psoriasis",
    "seborrheic_dermatitis",
    "lichen_planus",
    "pityriasis_rosea",
    "chronic_dermatitis",
    "pityriasis_rubra_pilaris",
]

_RNG = np.random.default_rng(seed=0)


def _make_dataset(n: int = 120) -> dict:
    """Return a dict of synthetic arrays that mimic UCI Dermatology structure."""
    # 6 classes with roughly similar imbalance to real dataset
    class_sizes = [30, 18, 22, 14, 18, 8]   # sums to 110 → pad to n
    while sum(class_sizes) < n:
        class_sizes[0] += 1
    y_true = np.concatenate([
        np.full(s, i, dtype=int) for i, s in enumerate(class_sizes)
    ])[:n]
    np.random.default_rng(1).shuffle(y_true)

    # 12-column clinical feature matrix (integer-valued 0–3)
    X_clin = _RNG.integers(0, 4, size=(n, 12)).astype(float)

    # Simulated Model B predictions (~75 % accuracy)
    y_pred_b = y_true.copy()
    error_idx = _RNG.choice(n, size=int(n * 0.25), replace=False)
    y_pred_b[error_idx] = _RNG.integers(0, 6, size=len(error_idx))

    # Simulated Model C predictions (~85 % accuracy)
    y_pred_c = y_true.copy()
    error_c = _RNG.choice(n, size=int(n * 0.15), replace=False)
    y_pred_c[error_c] = _RNG.integers(0, 6, size=len(error_c))

    # Probability matrices (n × 6), row-stochastic
    def _make_probs(y_pred, n, n_cls=6):
        probs = _RNG.dirichlet(np.ones(n_cls) * 0.5, size=n)
        for i, p in enumerate(y_pred):
            probs[i, p] += 2.0
            probs[i] /= probs[i].sum()
        return probs.astype(np.float32)

    prob_b = _make_probs(y_pred_b, n)
    prob_c = _make_probs(y_pred_c, n)

    certainty     = _RNG.uniform(0.40, 0.92, n).astype(np.float32)
    ambiguity     = _RNG.uniform(0.80, 3.10, n).astype(np.float32)
    contra_loads  = _RNG.uniform(0.00, 0.38, n).astype(np.float32)  # ≤ 0.40
    comp_margins  = _RNG.uniform(0.05, 0.55, n).astype(np.float32)
    traj_steps    = _RNG.integers(1, 9, size=n)
    esc_flags     = (ambiguity > 2.0) | (contra_loads > 0.25)

    # Symbolic matrix: n × 28 activations in [0, 1]
    sym_matrix = _RNG.uniform(0.0, 1.0, size=(n, 28)).astype(np.float32)

    # Short certainty trajectories: (n, 8) — monotone-ish values in [0, 1]
    traj = np.cumsum(
        _RNG.uniform(0.0, 0.12, size=(n, 8)), axis=1
    )
    traj = (traj / traj[:, -1:]).astype(np.float32)  # scale to end at 1.0
    traj = np.clip(traj, 0.0, 1.0)

    return dict(
        n=n,
        y_true=y_true,
        y_pred_b=y_pred_b,
        y_pred_c=y_pred_c,
        X_clin=X_clin,
        prob_b=prob_b,
        prob_c=prob_c,
        certainty=certainty,
        ambiguity=ambiguity,
        contra_loads=contra_loads,
        comp_margins=comp_margins,
        traj_steps=traj_steps,
        esc_flags=esc_flags,
        sym_matrix=sym_matrix,
        trajectories=traj,
    )


_DATA = _make_dataset(n=120)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Package import and __init__ exports
# ──────────────────────────────────────────────────────────────────────────────

class TestPackageImport:
    def test_package_imports_without_error(self):
        import importlib
        pkg = importlib.import_module("src.backend_refinement")
        assert pkg is not None

    def test_all_expected_classes_exported(self):
        from src.backend_refinement import (
            ModelCOptimizer, ModelCOptimizationReport,
            SymbolicRecoveryRefiner, RecoveryStrengthReport,
            DiseaseDiscriminationRefiner, DiseaseDiscriminationReport,
            EscalationBehaviorRefiner, EscalationBehaviorReport,
            ContradictionCompetitionRefiner, ContradictionCompetitionReport,
            TrajectoryRealismRefiner, TrajectoryRealismReport,
            RareDiseaseRefiner, RareDiseaseRefinementReport,
            SymbolicRuleRefinerV2, RuleRefinementReport,
            CertaintyBehaviorRefiner, CertaintyBehaviorReport,
            PublicationEvaluationSuite, PublicationEvaluationReport,
            BackendRefinementReporter, BackendRefinementReport,
        )
        # If we reached here, all exports resolved
        assert True

    def test_enum_values_importable(self):
        from src.backend_refinement import (
            RecoveryMechanism, RecoveryDifficultyTier, RecoveryOutcome,
            SeparabilityTier, EscalationDecision, StabilisationSafety,
            ContradictionScope, CompetitionOutcome, TrajectoryQuality,
            SmoothnessGrade, ImbalanceSeverity, RuleStrength,
            CalibrationStatus, AmbiguityRealism,
            RefinementPhase, FrontendTransitionDecision,
        )
        assert RecoveryMechanism.CONTRADICTION_RESOLUTION is not None
        assert SeparabilityTier.STRONG is not None
        assert EscalationDecision.ESCALATE is not None
        assert FrontendTransitionDecision.NOT_READY is not None


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — build_model_c_features
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildModelCFeatures:
    def test_clinical_only_shape(self):
        from src.backend_refinement import build_model_c_features
        d = _DATA
        fsets = build_model_c_features(d["X_clin"])
        assert fsets["clinical_only"].shape == (d["n"], 12)

    def test_clinical_symbolic_shape(self):
        from src.backend_refinement import build_model_c_features
        d = _DATA
        fsets = build_model_c_features(d["X_clin"])
        assert fsets["clinical_symbolic"].shape == (d["n"], 40)

    def test_full_34_shapes_when_provided(self):
        from src.backend_refinement import build_model_c_features
        d = _DATA
        X_full = np.hstack([d["X_clin"], _RNG.uniform(0, 3, size=(d["n"], 22))])
        fsets = build_model_c_features(d["X_clin"], X_full_34=X_full)
        assert "full_34" in fsets
        assert fsets["full_34_symbolic"].shape == (d["n"], 62)

    def test_no_nans_in_features(self):
        from src.backend_refinement import build_model_c_features
        d = _DATA
        fsets = build_model_c_features(d["X_clin"])
        assert not np.any(np.isnan(fsets["clinical_symbolic"]))


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — ModelCOptimizer (lightweight: 2 engine configs, 1 feature set)
# ──────────────────────────────────────────────────────────────────────────────

class TestModelCOptimizer:
    def test_optimise_returns_report(self):
        from src.backend_refinement import ModelCOptimizer, ModelCOptimizationReport
        opt = ModelCOptimizer(class_labels=_DISEASE_LABELS, verbose=False)
        report = opt.optimise(_DATA["X_clin"], _DATA["y_true"])
        assert isinstance(report, ModelCOptimizationReport)

    def test_report_has_baseline_and_best(self):
        from src.backend_refinement import ModelCOptimizer
        opt = ModelCOptimizer(class_labels=_DISEASE_LABELS, verbose=False)
        report = opt.optimise(_DATA["X_clin"], _DATA["y_true"])
        assert 0.0 <= report.model_b_baseline.cv_accuracy_mean <= 1.0
        assert report.best_configuration is not None
        assert len(report.configurations) > 0

    def test_disease_performance_count(self):
        from src.backend_refinement import ModelCOptimizer
        opt = ModelCOptimizer(class_labels=_DISEASE_LABELS, verbose=False)
        report = opt.optimise(_DATA["X_clin"], _DATA["y_true"])
        assert len(report.best_configuration.disease_performance) == 6

    def test_summary_text_produced(self):
        from src.backend_refinement import ModelCOptimizer
        opt = ModelCOptimizer(class_labels=_DISEASE_LABELS, verbose=False)
        report = opt.optimise(_DATA["X_clin"], _DATA["y_true"])
        txt = report.summary()
        assert "MODEL C OPTIMISATION REPORT" in txt

    def test_to_dict_serialisable(self):
        from src.backend_refinement import ModelCOptimizer
        opt = ModelCOptimizer(class_labels=_DISEASE_LABELS, verbose=False)
        report = opt.optimise(_DATA["X_clin"], _DATA["y_true"])
        d = report.to_dict()
        assert "best_configuration" in d
        assert "target_achieved" in d


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — SymbolicRecoveryRefiner
# ──────────────────────────────────────────────────────────────────────────────

class TestSymbolicRecoveryRefiner:
    def _get_report(self):
        from src.backend_refinement import SymbolicRecoveryRefiner
        d = _DATA
        refiner = SymbolicRecoveryRefiner(
            class_labels=_DISEASE_LABELS, n_samples=d["n"]
        )
        return refiner.analyse(
            y_true=d["y_true"],
            y_pred_b=d["y_pred_b"],
            y_pred_c=d["y_pred_c"],
            contradiction_loads=d["contra_loads"],
            ambiguity_bits=d["ambiguity"],
            certainty_b=d["certainty"],
        )

    def test_returns_report(self):
        from src.backend_refinement import RecoveryStrengthReport
        rep = self._get_report()
        assert isinstance(rep, RecoveryStrengthReport)

    def test_recovery_counts_consistent(self):
        rep = self._get_report()
        assert rep.n_total_errors == rep.n_recovered + rep.n_failed

    def test_recovery_rate_in_range(self):
        rep = self._get_report()
        assert 0.0 <= rep.overall_recovery_rate <= 1.0

    def test_taxonomy_covers_all_mechanisms(self):
        from src.backend_refinement import RecoveryMechanism
        rep = self._get_report()
        mechs = {e.mechanism for e in rep.taxonomy}
        for m in RecoveryMechanism:
            assert m in mechs

    def test_disease_profiles_count(self):
        rep = self._get_report()
        assert len(rep.disease_profiles) == 6

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "SYMBOLIC RECOVERY REFINEMENT REPORT" in txt


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — DiseaseDiscriminationRefiner
# ──────────────────────────────────────────────────────────────────────────────

class TestDiseaseDiscriminationRefiner:
    def _get_report(self):
        from src.backend_refinement import DiseaseDiscriminationRefiner
        d = _DATA
        refiner = DiseaseDiscriminationRefiner(class_labels=_DISEASE_LABELS)
        return refiner.analyse(
            X=d["sym_matrix"],
            y_true=d["y_true"],
            y_pred_b=d["y_pred_b"],
            y_pred_c=d["y_pred_c"],
        )

    def test_returns_report(self):
        from src.backend_refinement import DiseaseDiscriminationReport
        rep = self._get_report()
        assert isinstance(rep, DiseaseDiscriminationReport)

    def test_pairwise_scores_populated(self):
        rep = self._get_report()
        assert len(rep.pairwise_scores) > 0

    def test_n_pairs_correct(self):
        rep = self._get_report()
        # C(6,2) = 15 pairs
        assert len(rep.pairwise_scores) == 15

    def test_confusion_profiles_populated(self):
        rep = self._get_report()
        assert len(rep.confusion_profiles) == 6

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "DISEASE DISCRIMINATION" in txt.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — EscalationBehaviorRefiner
# ──────────────────────────────────────────────────────────────────────────────

class TestEscalationBehaviorRefiner:
    def _get_report(self):
        from src.backend_refinement import EscalationBehaviorRefiner
        d = _DATA
        refiner = EscalationBehaviorRefiner(class_labels=_DISEASE_LABELS)
        return refiner.analyse(
            y_true=d["y_true"],
            y_pred=d["y_pred_c"],
            ambiguity_bits=d["ambiguity"],
            contradiction_loads=d["contra_loads"],
            certainty_scores=d["certainty"],
            competition_margins=d["comp_margins"],
            trajectory_steps=d["traj_steps"],
        )

    def test_returns_report(self):
        from src.backend_refinement import EscalationBehaviorReport
        rep = self._get_report()
        assert isinstance(rep, EscalationBehaviorReport)

    def test_safety_audit_flag_reflects_unsafe_count(self):
        """Safety flag must accurately reflect whether unsafe stabilisations occurred."""
        rep = self._get_report()
        expected_passed = (rep.n_unsafe_stabilisations == 0)
        assert rep.safety_audit_passed == expected_passed

    def test_escalation_rate_in_range(self):
        rep = self._get_report()
        assert 0.0 <= rep.current_escalation_rate <= 1.0

    def test_selectivity_curve_populated(self):
        rep = self._get_report()
        assert len(rep.selectivity_curve) > 0

    def test_recommended_threshold_present(self):
        rep = self._get_report()
        assert rep.recommended_ambiguity_threshold > 0.0

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "ESCALATION" in txt.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — ContradictionCompetitionRefiner
# ──────────────────────────────────────────────────────────────────────────────

class TestContradictionCompetitionRefiner:
    def _get_report(self):
        from src.backend_refinement import ContradictionCompetitionRefiner
        d = _DATA
        n = d["n"]
        # Build a small contradiction matrix (n × 6)
        contra_mat = _RNG.uniform(0.0, 0.38, size=(n, 6)).astype(np.float32)
        refiner = ContradictionCompetitionRefiner(class_labels=_DISEASE_LABELS)
        return refiner.analyse(
            y_true=d["y_true"],
            y_pred=d["y_pred_c"],
            contradiction_matrix=contra_mat,
            global_loads=d["contra_loads"],
            certainty_scores=d["certainty"],
            competition_margins=d["comp_margins"],
        )

    def test_returns_report(self):
        from src.backend_refinement import ContradictionCompetitionReport
        rep = self._get_report()
        assert isinstance(rep, ContradictionCompetitionReport)

    def test_ceiling_compliance(self):
        """Contradiction loads must never exceed 0.40 — check per-case global_load."""
        rep = self._get_report()
        # All propagation profiles must have global_load within ceiling
        for profile in rep.propagation_profiles:
            assert profile.global_load <= 0.40 + 1e-9, (
                f"Ceiling violation in profile: global_load={profile.global_load:.4f}"
            )

    def test_propagation_profiles_populated(self):
        rep = self._get_report()
        assert len(rep.propagation_profiles) > 0

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "CONTRADICTION" in txt.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Test 8 — TrajectoryRealismRefiner
# ──────────────────────────────────────────────────────────────────────────────

class TestTrajectoryRealismRefiner:
    def _get_report(self):
        from src.backend_refinement import TrajectoryRealismRefiner
        d = _DATA
        refiner = TrajectoryRealismRefiner(class_labels=_DISEASE_LABELS)
        return refiner.analyse(
            y_true=d["y_true"],
            y_pred=d["y_pred_c"],
            certainty_trajectories=d["trajectories"],
            contradiction_loads=d["contra_loads"],
        )

    def test_returns_report(self):
        from src.backend_refinement import TrajectoryRealismReport
        rep = self._get_report()
        assert isinstance(rep, TrajectoryRealismReport)

    def test_quality_records_count(self):
        rep = self._get_report()
        assert len(rep.quality_records) == _DATA["n"]

    def test_realism_score_positive(self):
        rep = self._get_report()
        # mean_quality_score is an ordinal quality average (not bound to [0,1])
        assert rep.mean_quality_score >= 0.0
        assert np.isfinite(rep.mean_quality_score)

    def test_smoothness_profile_present(self):
        rep = self._get_report()
        assert rep.smoothness_profile is not None

    def test_disease_profiles_count(self):
        rep = self._get_report()
        assert len(rep.disease_profiles) == 6

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "TRAJECTORY" in txt.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Test 9 — RareDiseaseRefiner
# ──────────────────────────────────────────────────────────────────────────────

class TestRareDiseaseRefiner:
    def _get_report(self):
        from src.backend_refinement import RareDiseaseRefiner
        d = _DATA
        refiner = RareDiseaseRefiner(class_labels=_DISEASE_LABELS)
        return refiner.analyse(
            y_true=d["y_true"],
            y_pred_b=d["y_pred_b"],
            y_pred_c=d["y_pred_c"],
            symbolic_matrix=d["sym_matrix"],
            escalation_flags=d["esc_flags"],
            certainty_scores=d["certainty"],
            trajectory_steps=d["traj_steps"],
        )

    def test_returns_report(self):
        from src.backend_refinement import RareDiseaseRefinementReport
        rep = self._get_report()
        assert isinstance(rep, RareDiseaseRefinementReport)

    def test_disease_performances_count(self):
        rep = self._get_report()
        assert len(rep.performance_profiles) == 6

    def test_imbalance_severities_present(self):
        from src.backend_refinement import ImbalanceSeverity
        rep = self._get_report()
        for dp in rep.performance_profiles:
            assert dp.imbalance_severity in ImbalanceSeverity

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "RARE DISEASE" in txt.upper() or "IMBALANCE" in txt.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Test 10 — SymbolicRuleRefinerV2
# ──────────────────────────────────────────────────────────────────────────────

class TestSymbolicRuleRefinerV2:
    def test_compute_rule_activations_shape(self):
        from src.backend_refinement import SymbolicRuleRefinerV2
        refiner = SymbolicRuleRefinerV2(class_labels=_DISEASE_LABELS)
        activations = refiner.compute_rule_activations(_DATA["X_clin"])
        assert activations.shape == (_DATA["n"], 16)

    def test_activations_in_unit_interval(self):
        from src.backend_refinement import SymbolicRuleRefinerV2
        refiner = SymbolicRuleRefinerV2(class_labels=_DISEASE_LABELS)
        acts = refiner.compute_rule_activations(_DATA["X_clin"])
        assert np.all(acts >= 0.0) and np.all(acts <= 1.0)

    def test_build_and_evaluate_returns_report(self):
        from src.backend_refinement import SymbolicRuleRefinerV2, RuleRefinementReport
        refiner = SymbolicRuleRefinerV2(class_labels=_DISEASE_LABELS)
        rep = refiner.build_and_evaluate(_DATA["X_clin"], _DATA["y_true"])
        assert isinstance(rep, RuleRefinementReport)

    def test_rule_sets_cover_all_diseases(self):
        from src.backend_refinement import SymbolicRuleRefinerV2
        refiner = SymbolicRuleRefinerV2(class_labels=_DISEASE_LABELS)
        rep = refiner.build_and_evaluate(_DATA["X_clin"], _DATA["y_true"])
        disease_tags = {rs.disease for rs in rep.disease_rule_sets}
        for label in _DISEASE_LABELS:
            assert label in disease_tags or label[:3].lower() in str(disease_tags).lower()

    def test_16_rules_defined(self):
        from src.backend_refinement import SymbolicRuleRefinerV2
        refiner = SymbolicRuleRefinerV2(class_labels=_DISEASE_LABELS)
        rep = refiner.build_and_evaluate(_DATA["X_clin"], _DATA["y_true"])
        total = sum(len(rs.rules) for rs in rep.disease_rule_sets)
        assert total == 16

    def test_summary_produced(self):
        from src.backend_refinement import SymbolicRuleRefinerV2
        refiner = SymbolicRuleRefinerV2(class_labels=_DISEASE_LABELS)
        rep = refiner.build_and_evaluate(_DATA["X_clin"], _DATA["y_true"])
        txt = rep.summary()
        assert "RULE" in txt.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Test 11 — CertaintyBehaviorRefiner
# ──────────────────────────────────────────────────────────────────────────────

class TestCertaintyBehaviorRefiner:
    def _get_report(self):
        from src.backend_refinement import CertaintyBehaviorRefiner
        d = _DATA
        refiner = CertaintyBehaviorRefiner(class_labels=_DISEASE_LABELS)
        return refiner.analyse(
            y_true=d["y_true"],
            y_pred=d["y_pred_c"],
            certainty_scores=d["certainty"],
            probability_matrix=d["prob_c"],
            ambiguity_bits=d["ambiguity"],
            escalation_flags=d["esc_flags"],
        )

    def test_returns_report(self):
        from src.backend_refinement import CertaintyBehaviorReport
        rep = self._get_report()
        assert isinstance(rep, CertaintyBehaviorReport)

    def test_ece_in_range(self):
        rep = self._get_report()
        assert 0.0 <= rep.calibration_curve.expected_calibration_error <= 1.0

    def test_mce_in_range(self):
        rep = self._get_report()
        assert 0.0 <= rep.calibration_curve.maximum_calibration_error <= 1.0

    def test_mce_geq_ece(self):
        rep = self._get_report()
        assert rep.calibration_curve.maximum_calibration_error >= (
            rep.calibration_curve.expected_calibration_error - 1e-9
        )

    def test_entropy_profile_present(self):
        rep = self._get_report()
        assert rep.entropy_profile is not None

    def test_stabilisation_profiles_count(self):
        rep = self._get_report()
        assert len(rep.stabilisation_thresholds) == 6

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "CERTAINTY" in txt.upper() or "CALIBRATION" in txt.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Test 12 — PublicationEvaluationSuite
# ──────────────────────────────────────────────────────────────────────────────

class TestPublicationEvaluationSuite:
    def _get_report(self):
        from src.backend_refinement import PublicationEvaluationSuite
        d = _DATA
        suite = PublicationEvaluationSuite(
            class_labels=_DISEASE_LABELS,
            n_samples=d["n"],
        )
        return suite.evaluate(
            y_true=d["y_true"],
            y_pred_b=d["y_pred_b"],
            y_pred_c=d["y_pred_c"],
            prob_b=d["prob_b"],
            prob_c=d["prob_c"],
            contradiction_loads=d["contra_loads"],
            escalation_flags=d["esc_flags"],
            certainty_scores=d["certainty"],
            trajectory_steps=d["traj_steps"],
            certainty_trajectories=d["trajectories"],
        )

    def test_returns_report(self):
        from src.backend_refinement import PublicationEvaluationReport
        rep = self._get_report()
        assert isinstance(rep, PublicationEvaluationReport)

    def test_model_comparison_accuracy_gain(self):
        rep = self._get_report()
        gain = rep.model_comparison.accuracy_gain_pp
        # Model C > Model B in our synthetic data; gain should be > 0
        assert gain > 0.0

    def test_six_disease_rows_per_model(self):
        rep = self._get_report()
        assert len(rep.model_comparison.disease_metrics_b) == 6
        assert len(rep.model_comparison.disease_metrics_c) == 6

    def test_contradiction_ceiling_compliance(self):
        """Publication table must show 100 % contradiction ceiling compliance."""
        rep = self._get_report()
        # ceiling_compliance is a float [0,1] — should be 1.0
        assert rep.contradiction_analysis.ceiling_compliance == 1.0

    def test_escalation_table_present(self):
        rep = self._get_report()
        assert rep.escalation_analysis is not None
        assert 0.0 <= rep.escalation_analysis.escalation_rate <= 1.0

    def test_biopsy_reduction_positive(self):
        rep = self._get_report()
        assert rep.biopsy_reduction.biopsy_reduction_vs_blanket >= 0.0

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "MODEL" in txt.upper()

    def test_to_dict_is_dict(self):
        rep = self._get_report()
        d = rep.to_dict()
        assert isinstance(d, dict)
        assert "model_comparison" in d


# ──────────────────────────────────────────────────────────────────────────────
# Test 13 — BackendRefinementReporter
# ──────────────────────────────────────────────────────────────────────────────

class TestBackendRefinementReporter:
    def _get_report(self):
        from src.backend_refinement import BackendRefinementReporter
        reporter = BackendRefinementReporter()
        return reporter.compile(
            model_a_accuracy=0.9818,
            model_b_accuracy=0.86,
            model_c_accuracy=0.89,
            escalation_rate=0.30,
            symbolic_recovery_rate=0.45,
            trajectory_realism_score=0.75,
            contradiction_ceiling_compliant=True,
            escalation_safety_passed=True,
        )

    def test_returns_report(self):
        from src.backend_refinement import BackendRefinementReport
        rep = self._get_report()
        assert isinstance(rep, BackendRefinementReport)

    def test_overall_score_in_range(self):
        rep = self._get_report()
        assert 0.0 <= rep.overall_refinement_score <= 1.0

    def test_three_model_entries(self):
        rep = self._get_report()
        labels = {mp.model_label for mp in rep.model_progress}
        assert "Model A" in labels
        assert "Model B" in labels
        assert "Model C" in labels

    def test_ten_subsystem_entries(self):
        rep = self._get_report()
        assert len(rep.subsystem_entries) == 10

    def test_ceiling_compliance_preserved(self):
        """Must reflect what was passed in — no override allowed."""
        rep = self._get_report()
        assert rep.contradiction_ceiling_compliant is True

    def test_ceiling_violation_forces_not_ready(self):
        from src.backend_refinement import BackendRefinementReporter, FrontendTransitionDecision
        reporter = BackendRefinementReporter()
        rep = reporter.compile(
            model_c_accuracy=0.90,
            contradiction_ceiling_compliant=False,
            escalation_safety_passed=True,
        )
        assert rep.frontend_transition == FrontendTransitionDecision.NOT_READY
        assert any("ceiling" in b.lower() or "contradiction" in b.lower()
                   for b in rep.frontend_blockers)

    def test_ready_when_all_conditions_met(self):
        from src.backend_refinement import BackendRefinementReporter, FrontendTransitionDecision
        reporter = BackendRefinementReporter()
        rep = reporter.compile(
            model_b_accuracy=0.86,
            model_c_accuracy=0.89,
            escalation_rate=0.35,
            symbolic_recovery_rate=0.55,
            trajectory_realism_score=0.80,
            contradiction_ceiling_compliant=True,
            escalation_safety_passed=True,
        )
        # May be READY or CONDITIONALLY_READY — should NOT be NOT_READY
        assert rep.frontend_transition != FrontendTransitionDecision.NOT_READY

    def test_checklist_has_nine_items(self):
        rep = self._get_report()
        assert len(rep.checklist) == 9

    def test_summary_produced(self):
        rep = self._get_report()
        txt = rep.summary()
        assert "BACKEND REFINEMENT REPORT" in txt

    def test_to_dict_serialisable(self):
        rep = self._get_report()
        d = rep.to_dict()
        assert "refinement_phase" in d
        assert "safety" in d
        assert d["safety"]["contradiction_ceiling_compliant"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Test 14 — Safety invariants across all analysers
# ──────────────────────────────────────────────────────────────────────────────

class TestGlobalSafetyInvariants:
    """Cross-module safety checks — contradiction ceiling must never be breached."""

    def test_contradiction_loads_never_exceed_ceiling(self):
        """All contradiction loads fed into any module must be capped at 0.40."""
        loads = _DATA["contra_loads"]
        assert float(np.max(loads)) <= 0.40, (
            f"Test data violates ceiling: max={np.max(loads):.4f}"
        )

    def test_escalation_safety_flag_accurate(self):
        """safety_audit_passed must truthfully reflect n_unsafe_stabilisations."""
        from src.backend_refinement import EscalationBehaviorRefiner
        d = _DATA
        refiner = EscalationBehaviorRefiner(class_labels=_DISEASE_LABELS)
        rep = refiner.analyse(
            y_true=d["y_true"],
            y_pred=d["y_pred_c"],
            ambiguity_bits=d["ambiguity"],
            contradiction_loads=d["contra_loads"],
            certainty_scores=d["certainty"],
            competition_margins=d["comp_margins"],
            trajectory_steps=d["traj_steps"],
        )
        # Safety flag must correctly reflect unsafe count (not override it)
        assert rep.safety_audit_passed == (rep.n_unsafe_stabilisations == 0)

    def test_contradiction_competition_ceiling_per_case(self):
        """Individual case loads must never exceed 0.40 ceiling."""
        from src.backend_refinement import ContradictionCompetitionRefiner
        d = _DATA
        contra_mat = _RNG.uniform(0.0, 0.38, size=(d["n"], 6)).astype(np.float32)
        refiner = ContradictionCompetitionRefiner(class_labels=_DISEASE_LABELS)
        rep = refiner.analyse(
            y_true=d["y_true"],
            y_pred=d["y_pred_c"],
            contradiction_matrix=contra_mat,
            global_loads=d["contra_loads"],
            certainty_scores=d["certainty"],
            competition_margins=d["comp_margins"],
        )
        for profile in rep.propagation_profiles:
            assert profile.global_load <= 0.40 + 1e-9, (
                f"Ceiling breach: global_load={profile.global_load:.4f}"
            )

    def test_publication_suite_ceiling_compliance(self):
        from src.backend_refinement import PublicationEvaluationSuite
        d = _DATA
        suite = PublicationEvaluationSuite(
            class_labels=_DISEASE_LABELS, n_samples=d["n"]
        )
        rep = suite.evaluate(
            y_true=d["y_true"],
            y_pred_b=d["y_pred_b"],
            y_pred_c=d["y_pred_c"],
            prob_b=d["prob_b"],
            prob_c=d["prob_c"],
            contradiction_loads=d["contra_loads"],
            escalation_flags=d["esc_flags"],
            certainty_scores=d["certainty"],
            trajectory_steps=d["traj_steps"],
            certainty_trajectories=d["trajectories"],
        )
        assert rep.contradiction_analysis.ceiling_compliance == 1.0
