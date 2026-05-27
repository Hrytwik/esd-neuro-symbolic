"""
Integration test for src/performance_calibration modules.

Uses a synthetic dataset (compatible with the real UCI Dermatology schema)
to verify all Phase 5 diagnostics modules execute end-to-end without errors
and produce well-formed output structures.

Run:
    cd D:/esd-neuro-symbolic
    python tests/test_performance_calibration.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Synthetic dataset generator ───────────────────────────────────────────────

def _build_synthetic_csv(seed: int = 42) -> str:
    """Return path to a temporary synthetic dermatology CSV."""
    rng = np.random.default_rng(seed)

    CLINICAL_ORDINAL = [
        "erythema", "scaling", "definite_borders", "itching",
        "koebner_phenomenon", "polygonal_papules", "follicular_papules",
        "oral_mucosal_involvement", "knee_and_elbow_involvement", "scalp_involvement",
    ]
    HISTO = [
        "melanin_incontinence", "eosinophils_in_infiltrate", "PNL_infiltrate",
        "fibrosis_of_papillary_dermis", "exocytosis", "acanthosis",
        "hyperkeratosis", "parakeratosis", "clubbing_of_rete_ridges",
        "elongation_of_rete_ridges", "thinning_of_suprapapillary_epidermis",
        "focal_hypergranulosis", "disappearance_of_granular_layer",
        "vacuolisation_and_damage_of_basal_layer", "spongiosis",
        "saw_tooth_appearance_of_retes", "inflammatory_mononuclear_infiltrate",
        "band_like_infiltrate", "spongiform_pustule", "munro_microabcess",
        "follicular_horn_plug", "perifollicular_parakeratosis",
    ]

    PROFILES = {
        "psoriasis": {
            "erythema": 2.8, "scaling": 2.9, "definite_borders": 2.5,
            "itching": 1.5, "koebner_phenomenon": 1.8, "polygonal_papules": 0.2,
            "follicular_papules": 0.1, "oral_mucosal_involvement": 0.1,
            "knee_and_elbow_involvement": 2.5, "scalp_involvement": 2.4,
            "family_history_p": 0.30, "age_mu": 45.0, "n": 112,
            "histo_high": {"acanthosis", "parakeratosis", "hyperkeratosis",
                           "spongiform_pustule", "munro_microabcess"},
        },
        "seborrheic_dermatitis": {
            "erythema": 2.0, "scaling": 2.5, "definite_borders": 1.5,
            "itching": 1.8, "koebner_phenomenon": 0.3, "polygonal_papules": 0.1,
            "follicular_papules": 0.2, "oral_mucosal_involvement": 0.1,
            "knee_and_elbow_involvement": 0.4, "scalp_involvement": 2.6,
            "family_history_p": 0.20, "age_mu": 38.0, "n": 61,
            "histo_high": {"spongiosis", "inflammatory_mononuclear_infiltrate"},
        },
        "lichen_planus": {
            "erythema": 1.8, "scaling": 1.5, "definite_borders": 2.2,
            "itching": 2.5, "koebner_phenomenon": 2.2, "polygonal_papules": 2.6,
            "follicular_papules": 0.5, "oral_mucosal_involvement": 2.2,
            "knee_and_elbow_involvement": 1.8, "scalp_involvement": 0.8,
            "family_history_p": 0.15, "age_mu": 42.0, "n": 72,
            "histo_high": {"band_like_infiltrate", "saw_tooth_appearance_of_retes",
                           "vacuolisation_and_damage_of_basal_layer"},
        },
        "pityriasis_rosea": {
            "erythema": 2.0, "scaling": 2.0, "definite_borders": 1.8,
            "itching": 1.5, "koebner_phenomenon": 0.3, "polygonal_papules": 0.2,
            "follicular_papules": 0.3, "oral_mucosal_involvement": 0.2,
            "knee_and_elbow_involvement": 0.5, "scalp_involvement": 0.5,
            "family_history_p": 0.10, "age_mu": 28.0, "n": 49,
            "histo_high": {"exocytosis"},
        },
        "chronic_dermatitis": {
            "erythema": 2.2, "scaling": 1.8, "definite_borders": 1.4,
            "itching": 2.6, "koebner_phenomenon": 0.5, "polygonal_papules": 0.2,
            "follicular_papules": 0.3, "oral_mucosal_involvement": 0.2,
            "knee_and_elbow_involvement": 0.8, "scalp_involvement": 1.2,
            "family_history_p": 0.25, "age_mu": 40.0, "n": 52,
            "histo_high": {"spongiosis", "PNL_infiltrate"},
        },
        "pityriasis_rubra_pilaris": {
            "erythema": 2.5, "scaling": 2.3, "definite_borders": 2.0,
            "itching": 1.8, "koebner_phenomenon": 0.4, "polygonal_papules": 0.3,
            "follicular_papules": 2.5, "oral_mucosal_involvement": 0.3,
            "knee_and_elbow_involvement": 2.0, "scalp_involvement": 1.5,
            "family_history_p": 0.35, "age_mu": 50.0, "n": 20,
            "histo_high": {"follicular_horn_plug", "perifollicular_parakeratosis"},
        },
    }

    disease_class_map = {
        "psoriasis": 1, "seborrheic_dermatitis": 2, "lichen_planus": 3,
        "pityriasis_rosea": 4, "chronic_dermatitis": 5, "pityriasis_rubra_pilaris": 6,
    }

    rows: list[dict] = []
    for disease, profile in PROFILES.items():
        n = profile["n"]
        histo_high: set = profile["histo_high"]  # type: ignore[assignment]
        for i in range(n):
            row: dict = {}
            for feat in CLINICAL_ORDINAL:
                mu  = float(profile[feat])  # type: ignore[call-overload]
                val = int(np.clip(round(rng.normal(mu, 0.8)), 0, 3))
                row[feat] = val
            row["family_history"] = int(rng.random() < profile["family_history_p"])
            # Inject a few missing ages
            if disease == "psoriasis" and i < 8 and rng.random() < 0.3:
                row["age"] = ""
            else:
                row["age"] = int(np.clip(rng.normal(profile["age_mu"], 12.0), 5, 80))
            for feat in HISTO:
                hmu = 2.2 if feat in histo_high else 0.5
                val  = int(np.clip(round(rng.normal(hmu, 0.7)), 0, 3))
                row[feat] = val
            row["disease_label"] = disease
            row["class"] = disease_class_map[disease]
            rows.append(row)

    rng.shuffle(rows)  # type: ignore[arg-type]

    fieldnames = (
        CLINICAL_ORDINAL
        + ["family_history", "age"]
        + HISTO
        + ["disease_label", "class"]
    )
    tmp = tempfile.mktemp(suffix=".csv")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tmp


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests() -> None:
    print("=" * 72)
    print("PERFORMANCE CALIBRATION — INTEGRATION TEST")
    print("=" * 72)

    # Step 0: Generate synthetic data
    print("\n[0] Generating synthetic UCI Dermatology dataset...")
    csv_path = _build_synthetic_csv()
    print(f"    Written to: {csv_path}")

    try:
        from src.dataset_integration.dataset_loader import (
            DermatologyDatasetLoader, CANONICAL_DISEASES,
        )
        from src.dataset_integration.dataset_splitter import DatasetSplitter
        from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureAdapter
        from src.dataset_integration.feature_partitioning import (
            CLINICAL_FEATURE_NAMES, ALL_FEATURE_NAMES,
        )
        from src.evaluation_pipeline.baseline_model_a import BaselineModelA, ModelAConfig
        from src.evaluation_pipeline.baseline_model_b import BaselineModelB, ModelBConfig
        from src.evaluation_pipeline.symbolic_model_c import SymbolicModelC, ModelCConfig

        # Step 1: Load + split
        print("\n[1] Loading and splitting dataset...")
        ds     = DermatologyDatasetLoader.load(csv_path)
        split  = DatasetSplitter(seed=42).split(ds)
        print(f"    train={split.n_train}, val={split.n_validation}, test={split.n_test}")
        assert split.n_train > 0 and split.n_test > 0

        # Step 2: Symbolic pipeline
        print("\n[2] Running symbolic reasoning pipeline...")
        adapter    = SymbolicFeatureAdapter(suppress_errors=True)
        train_vecs = adapter.adapt_batch(list(split.train_records))
        test_vecs  = adapter.adapt_batch(list(split.test_records))
        n_ok = sum(1 for v in test_vecs if v.pipeline_success)
        print(f"    success={n_ok} / {len(test_vecs)}")

        # Step 3: Build matrices
        class_labels = list(CANONICAL_DISEASES)
        X_train_all  = np.array(split.train_feature_matrix(ALL_FEATURE_NAMES))
        X_test_all   = np.array(split.test_feature_matrix(ALL_FEATURE_NAMES))
        X_train_clin = np.array(split.train_feature_matrix(CLINICAL_FEATURE_NAMES))
        X_test_clin  = np.array(split.test_feature_matrix(CLINICAL_FEATURE_NAMES))
        y_train_str  = np.array(split.train_labels)
        y_test_str   = np.array(split.test_labels)
        lbl2int      = {lbl: i for i, lbl in enumerate(class_labels)}
        y_train      = np.array([lbl2int.get(l, 0) for l in y_train_str])
        y_test       = np.array([lbl2int.get(l, 0) for l in y_test_str])

        # Step 4: Train models
        print("\n[3] Training A/B/C models...")
        result_a = BaselineModelA(ModelAConfig()).fit_and_evaluate(
            X_train_all, y_train, X_test_all, y_test,
            feature_names=list(ALL_FEATURE_NAMES), class_labels=class_labels,
        )
        model_b_fitted = BaselineModelB(ModelBConfig())
        result_b = model_b_fitted.fit_and_evaluate(
            X_train_clin, y_train, X_test_clin, y_test,
            feature_names=list(CLINICAL_FEATURE_NAMES), class_labels=class_labels,
        )
        model_c_fitted = SymbolicModelC(ModelCConfig())
        result_c = model_c_fitted.fit_and_evaluate(
            X_train_clin, train_vecs, y_train,
            X_test_clin,  test_vecs,  y_test,
            clinical_feature_names=list(CLINICAL_FEATURE_NAMES),
            class_labels=class_labels,
        )
        print(
            f"    A={result_a.accuracy:.4f}  "
            f"B={result_b.accuracy:.4f}  "
            f"C={result_c.accuracy:.4f}"
        )

        # Step 5: PerformanceDiagnostics
        print("\n[4] Running PerformanceDiagnostics...")
        from src.performance_calibration.performance_diagnostics import PerformanceDiagnostics
        diag = PerformanceDiagnostics(class_labels).analyse(
            result_a, result_b, result_c, test_vecs, X_test_clin, y_test
        )
        assert len(diag.disease_profiles) == 6, "Expected 6 disease profiles"
        assert 0.0 <= diag.overall_escalation_rate <= 1.0
        print(diag.summary())

        # Step 6: FailureModeAnalyzer
        print("\n[5] Running FailureModeAnalyzer...")
        from src.performance_calibration.failure_mode_analyzer import FailureModeAnalyzer
        sym_lift  = result_c.accuracy - result_b.accuracy
        fm_report = FailureModeAnalyzer().analyse(
            diag, result_b.accuracy, result_c.accuracy, sym_lift
        )
        assert len(fm_report.all_patterns) > 0
        print(fm_report.summary())

        # Step 7: EscalationSensitivityAnalyzer
        print("\n[6] Running EscalationSensitivityAnalyzer...")
        from src.performance_calibration.escalation_sensitivity_analysis import (
            EscalationSensitivityAnalyzer,
        )
        # Re-use the already-fitted Model B to get test predictions
        y_pred_b = np.array(model_b_fitted.predict(X_test_clin))
        sens = EscalationSensitivityAnalyzer(
            ambiguity_grid=[1.50, 2.00, 2.50],
            certainty_grid=[0.40, 0.55],
        ).analyse(test_vecs, y_pred_b, y_test)
        assert len(sens.sweep_results) > 0
        print(sens.summary())

        # Step 8: RuleDiscriminationAnalyzer
        print("\n[7] Running RuleDiscriminationAnalyzer...")
        from src.performance_calibration.rule_discrimination_analysis import (
            RuleDiscriminationAnalyzer,
        )
        rule_rpt = RuleDiscriminationAnalyzer(class_labels).analyse(
            test_vecs, y_pred_b, y_test
        )
        assert len(rule_rpt.signal_profiles) > 0
        print(rule_rpt.summary())

        # Step 9: HypothesisSeparationAnalyzer
        print("\n[8] Running HypothesisSeparationAnalyzer...")
        from src.performance_calibration.hypothesis_separation_analysis import (
            HypothesisSeparationAnalyzer,
        )
        sep_rpt = HypothesisSeparationAnalyzer(class_labels).analyse(
            test_vecs, y_test
        )
        assert 0.0 <= sep_rpt.clinical_discriminability_index <= 1.0
        print(sep_rpt.summary())

        # Step 10: BaselineCalibrator (fast_mode)
        print("\n[9] Running BaselineCalibrator (fast_mode=True)...")
        from src.performance_calibration.baseline_calibration import BaselineCalibrator
        calibrator = BaselineCalibrator(
            n_splits=3,
            n_repeats=2,
            algorithms=["xgboost", "random_forest"],
            verbose=True,
            fast_mode=True,
        )
        cal_b = calibrator.calibrate_model_b(X_train_clin, y_train)
        assert cal_b.best_trial is not None
        print(cal_b.summary())

        print("\n" + "=" * 72)
        print("ALL PHASE 5 STEP 1 DIAGNOSTIC TESTS PASSED")
        print("=" * 72)

        # ── Phase 5 Step 2 — Recalibration Modules ───────────────────────────

        print("\n" + "=" * 72)
        print("PHASE 5 STEP 2 — RECALIBRATION MODULES")
        print("=" * 72)

        # Step 11: ThresholdRecalibrator
        print("\n[10] Running ThresholdRecalibrator...")
        from src.performance_calibration.threshold_recalibration import (
            ThresholdRecalibrator,
        )
        recalibrator = ThresholdRecalibrator(
            ambiguity_ceiling=2.50,
            certainty_floor=0.40,
        )
        recal_vecs = recalibrator.recalibrate(test_vecs)
        assert len(recal_vecs) == len(test_vecs)
        # Escalation rate should be lower with recalibrated thresholds
        orig_esc = sum(1 for v in test_vecs  if v.requires_biopsy) / len(test_vecs)
        new_esc  = sum(1 for v in recal_vecs if v.requires_biopsy) / len(recal_vecs)
        print(f"    Escalation: {orig_esc:.1%} -> {new_esc:.1%}")
        thr_report = recalibrator.fit_and_report(test_vecs, y_pred_b, y_test)
        assert thr_report.best_config is not None
        print(f"    Best config: {thr_report.best_config.label()}")
        print(f"    Escalation reduction: {thr_report.escalation_reduction:.1%}")

        # Step 12: CertaintyRebalancer
        print("\n[11] Running CertaintyRebalancer...")
        from src.performance_calibration.certainty_rebalancing import CertaintyRebalancer
        cert_rebal = CertaintyRebalancer()
        cert_rebal.fit(train_vecs)
        cert_enriched = cert_rebal.enrich(test_vecs)
        assert len(cert_enriched) == len(test_vecs)
        cert_report = cert_rebal.build_analysis_report(test_vecs, cert_enriched)
        assert 0.0 <= cert_report.original_distribution.mean <= 1.0
        assert cert_report.normalised_distribution.mean >= cert_report.original_distribution.mean
        print(cert_report.summary())

        # Step 13: ContradictionRebalancer
        print("\n[12] Running ContradictionRebalancer...")
        from src.performance_calibration.contradiction_rebalancing import (
            ContradictionRebalancer,
        )
        contr_rebal   = ContradictionRebalancer()
        contr_enriched = contr_rebal.enrich(test_vecs)
        assert len(contr_enriched) == len(test_vecs)
        contr_report  = contr_rebal.build_analysis_report(
            test_vecs, contr_enriched, y_pred_b, y_test
        )
        assert "CRITICAL" in contr_report.tier_stats or "NONE" in contr_report.tier_stats
        print(contr_report.summary())

        # Step 14: CompetitionSharpener
        print("\n[13] Running CompetitionSharpener...")
        from src.performance_calibration.competition_sharpening import CompetitionSharpener
        comp_sharp   = CompetitionSharpener()
        comp_enriched = comp_sharp.enrich(test_vecs)
        assert len(comp_enriched) == len(test_vecs)
        X_comp, comp_names = comp_sharp.build_enriched_matrix(X_test_clin, test_vecs, comp_enriched)
        assert X_comp.shape[0] == len(test_vecs)
        assert X_comp.shape[1] > X_test_clin.shape[1]
        print(f"    Competition-enriched matrix: {X_comp.shape} ({len(comp_names)} features)")

        # Step 15: SymbolicSignalEnricherV2
        print("\n[14] Running SymbolicSignalEnricherV2...")
        from src.performance_calibration.symbolic_signal_enrichment_v2 import (
            SymbolicSignalEnricherV2,
        )
        enricher_v2   = SymbolicSignalEnricherV2()
        enriched_v2   = enricher_v2.enrich(test_vecs)
        assert len(enriched_v2) == len(test_vecs)
        X_v2, v2_names = enricher_v2.build_feature_matrix(X_test_clin, enriched_v2)
        assert X_v2.shape[1] == X_test_clin.shape[1] + 40
        enrichment_rpt = enricher_v2.build_report(test_vecs, enriched_v2)
        assert enrichment_rpt.n_signals_total == 40
        print(enrichment_rpt.summary())

        # Step 16: DiseaseSignatureRefiner
        print("\n[15] Running DiseaseSignatureRefiner...")
        from src.performance_calibration.disease_signature_refinement import (
            DiseaseSignatureRefiner,
        )
        sig_refiner = DiseaseSignatureRefiner(class_labels)
        sig_report  = sig_refiner.analyse(X_test_clin, y_test, test_vecs)
        assert len(sig_report.feature_profiles) > 0
        print(sig_report.summary())

        # Step 17: AdvancedBaselineCalibrator (fast mode, small grid)
        print("\n[16] Running AdvancedBaselineCalibrator (fast mode)...")
        from src.performance_calibration.advanced_baseline_calibration import (
            AdvancedBaselineCalibrator,
        )
        adv_cal = AdvancedBaselineCalibrator(
            n_splits=3,
            n_repeats=2,
            fast_mode=True,
            apply_scaling=True,
        )
        adv_b = adv_cal.calibrate_model_b(
            X_train_clin, y_train, X_test_clin, y_test
        )
        assert adv_b.best_trial is not None
        print(f"    Best Model B CV acc: {adv_b.best_trial.cv_mean_accuracy:.4f}")
        if adv_b.test_accuracy > 0:
            print(f"    Test accuracy: {adv_b.test_accuracy:.4f}")

        adv_c = adv_cal.calibrate_model_c_v2(
            X_train_clin, train_vecs, y_train,
            X_test_clin,  test_vecs,  y_test,
            ambiguity_ceiling=2.50,
            certainty_floor=0.40,
        )
        assert adv_c.best_trial is not None
        print(f"    Best Model C CV acc: {adv_c.best_trial.cv_mean_accuracy:.4f}")
        if adv_c.test_accuracy > 0:
            print(f"    Test accuracy: {adv_c.test_accuracy:.4f}")

        # Step 18: SymbolicRecoveryAnalyzer
        print("\n[17] Running SymbolicRecoveryAnalyzer...")
        from src.performance_calibration.symbolic_recovery_analysis import (
            SymbolicRecoveryAnalyzer,
        )
        # Use fitted model C to generate test predictions for recovery attribution
        y_pred_c_arr = model_c_fitted.predict(X_test_clin, test_vecs)
        recovery_rpt = SymbolicRecoveryAnalyzer(class_labels).analyse(
            test_vecs, y_pred_b, y_pred_c_arr, y_test
        )
        assert isinstance(recovery_rpt.n_recoveries, int)
        assert isinstance(recovery_rpt.symbolic_contribution_index, float)
        print(recovery_rpt.summary())

        # Step 19: BiopsyReductionAnalyzer
        print("\n[18] Running BiopsyReductionAnalyzer...")
        from src.performance_calibration.biopsy_reduction_analysis import (
            BiopsyReductionAnalyzer,
        )
        biopsy_rpt = BiopsyReductionAnalyzer(class_labels).analyse(
            test_vecs, y_pred_b, y_test
        )
        assert biopsy_rpt.total_cases == len(test_vecs)
        assert 0.0 <= biopsy_rpt.overall_escalation_rate_default <= 1.0
        print(biopsy_rpt.summary())

        # Step 20: FinalCalibrationReporter
        print("\n[19] Running FinalCalibrationReporter...")
        from src.performance_calibration.final_calibration_report import (
            FinalCalibrationReporter,
        )
        reporter  = FinalCalibrationReporter(
            class_labels=class_labels,
            model_b_accuracy_before=result_b.accuracy,
            model_c_accuracy_before=result_c.accuracy,
            model_a_reference=result_a.accuracy,
            target_model_b=0.86,
            target_model_c=0.88,
        )
        final_rpt = reporter.compile(
            threshold_report=thr_report,
            certainty_report=cert_report,
            recovery_report=recovery_rpt,
            biopsy_report=biopsy_rpt,
            contradiction_report=contr_report,
            enrichment_report=enrichment_rpt,
            model_b_result=adv_b,
            model_c_result=adv_c,
            symbolic_vectors=test_vecs,
        )
        assert final_rpt.clinical_safety_verified in (True, False)
        assert isinstance(final_rpt.performance_comparison.model_b_improvement_pp, float)
        report_text = final_rpt.summary()
        assert "SECTION 1" in report_text
        assert "SECTION 9" in report_text
        print(report_text)

        # JSON export
        report_json = final_rpt.to_json()
        import json
        parsed = json.loads(report_json)
        assert "performance_comparison" in parsed
        assert "escalation_summary" in parsed
        assert "contradiction_audit" in parsed
        print(f"    JSON export: {len(report_json)} bytes, {len(parsed)} top-level keys")

        print("\n" + "=" * 72)
        print("ALL PHASE 5 STEP 2 RECALIBRATION TESTS PASSED")
        print("=" * 72)

    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


if __name__ == "__main__":
    run_tests()
