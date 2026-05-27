"""
Targeted integration test for Phase 5 Step 2 recalibration modules.

Tests all 10 new modules end-to-end on synthetic data using a minimal CV
grid for speed. Phase 5 Step 1 modules are assumed to be working.

Run:
    cd D:/esd-neuro-symbolic
    python -X utf8 tests/test_phase5_step2.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_synthetic_csv(seed: int = 42) -> str:
    """Return path to a temporary synthetic UCI-compatible dermatology CSV."""
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
            "histo_high": {"acanthosis", "parakeratosis"},
        },
        "seborrheic_dermatitis": {
            "erythema": 2.0, "scaling": 2.5, "definite_borders": 1.5,
            "itching": 1.8, "koebner_phenomenon": 0.3, "polygonal_papules": 0.1,
            "follicular_papules": 0.2, "oral_mucosal_involvement": 0.1,
            "knee_and_elbow_involvement": 0.4, "scalp_involvement": 2.6,
            "family_history_p": 0.20, "age_mu": 38.0, "n": 61,
            "histo_high": {"spongiosis"},
        },
        "lichen_planus": {
            "erythema": 1.8, "scaling": 1.5, "definite_borders": 2.2,
            "itching": 2.5, "koebner_phenomenon": 2.2, "polygonal_papules": 2.6,
            "follicular_papules": 0.5, "oral_mucosal_involvement": 2.2,
            "knee_and_elbow_involvement": 1.8, "scalp_involvement": 0.8,
            "family_history_p": 0.15, "age_mu": 42.0, "n": 72,
            "histo_high": {"band_like_infiltrate"},
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
            "histo_high": {"spongiosis"},
        },
        "pityriasis_rubra_pilaris": {
            "erythema": 2.5, "scaling": 2.3, "definite_borders": 2.0,
            "itching": 1.8, "koebner_phenomenon": 0.4, "polygonal_papules": 0.3,
            "follicular_papules": 2.5, "oral_mucosal_involvement": 0.3,
            "knee_and_elbow_involvement": 2.0, "scalp_involvement": 1.5,
            "family_history_p": 0.35, "age_mu": 50.0, "n": 20,
            "histo_high": {"follicular_horn_plug"},
        },
    }
    disease_class_map = {
        "psoriasis": 1, "seborrheic_dermatitis": 2, "lichen_planus": 3,
        "pityriasis_rosea": 4, "chronic_dermatitis": 5, "pityriasis_rubra_pilaris": 6,
    }
    rows = []
    for disease, profile in PROFILES.items():
        histo_high = profile["histo_high"]
        for _ in range(profile["n"]):
            row = {}
            for feat in CLINICAL_ORDINAL:
                row[feat] = int(np.clip(round(rng.normal(float(profile[feat]), 0.8)), 0, 3))
            row["family_history"] = int(rng.random() < profile["family_history_p"])
            row["age"] = int(np.clip(rng.normal(profile["age_mu"], 12.0), 5, 80))
            for feat in HISTO:
                hmu = 2.2 if feat in histo_high else 0.5
                row[feat] = int(np.clip(round(rng.normal(hmu, 0.7)), 0, 3))
            row["disease_label"] = disease
            row["class"] = disease_class_map[disease]
            rows.append(row)
    rng.shuffle(rows)
    fieldnames = CLINICAL_ORDINAL + ["family_history", "age"] + HISTO + ["disease_label", "class"]
    tmp = tempfile.mktemp(suffix=".csv")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tmp


def run_tests() -> None:
    print("=" * 72)
    print("PHASE 5 STEP 2 — RECALIBRATION MODULES — INTEGRATION TEST")
    print("=" * 72)

    csv_path = _build_synthetic_csv()
    print(f"\n[setup] Synthetic dataset: {csv_path}")

    try:
        from src.dataset_integration.dataset_loader import (
            DermatologyDatasetLoader, CANONICAL_DISEASES,
        )
        from src.dataset_integration.dataset_splitter import DatasetSplitter
        from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureAdapter
        from src.dataset_integration.feature_partitioning import CLINICAL_FEATURE_NAMES
        from src.evaluation_pipeline.baseline_model_b import BaselineModelB, ModelBConfig
        from src.evaluation_pipeline.symbolic_model_c import SymbolicModelC, ModelCConfig

        ds          = DermatologyDatasetLoader.load(csv_path)
        split       = DatasetSplitter(seed=42).split(ds)
        adapter     = SymbolicFeatureAdapter(suppress_errors=True)
        train_vecs  = adapter.adapt_batch(list(split.train_records))
        test_vecs   = adapter.adapt_batch(list(split.test_records))
        class_labels = list(CANONICAL_DISEASES)
        X_train_clin = np.array(split.train_feature_matrix(CLINICAL_FEATURE_NAMES))
        X_test_clin  = np.array(split.test_feature_matrix(CLINICAL_FEATURE_NAMES))
        lbl2int      = {lbl: i for i, lbl in enumerate(class_labels)}
        y_train      = np.array([lbl2int.get(l, 0) for l in split.train_labels])
        y_test       = np.array([lbl2int.get(l, 0) for l in split.test_labels])
        model_b      = BaselineModelB(ModelBConfig()).fit(X_train_clin, y_train)
        y_pred_b     = np.array(model_b.predict(X_test_clin))
        model_c      = SymbolicModelC(ModelCConfig()).fit(X_train_clin, train_vecs, y_train)
        y_pred_c     = np.array(model_c.predict(X_test_clin, test_vecs))
        acc_b        = float(np.mean(y_pred_b == y_test))
        acc_c        = float(np.mean(y_pred_c == y_test))
        print(f"[setup] train={len(train_vecs)} test={len(test_vecs)} B={acc_b:.4f} C={acc_c:.4f}")

        # ── [10] ThresholdRecalibrator ────────────────────────────────────────
        print("\n[10] ThresholdRecalibrator...")
        from src.performance_calibration.threshold_recalibration import ThresholdRecalibrator
        recal      = ThresholdRecalibrator(2.50, 0.40)
        recal_vecs = recal.recalibrate(test_vecs)
        assert len(recal_vecs) == len(test_vecs)
        thr_report = recal.fit_and_report(test_vecs, y_pred_b, y_test)
        assert thr_report.best_config is not None
        orig_esc = sum(1 for v in test_vecs  if v.requires_biopsy) / len(test_vecs)
        new_esc  = sum(1 for v in recal_vecs if v.requires_biopsy) / len(recal_vecs)
        print(f"    PASS  esc: {orig_esc:.1%} -> {new_esc:.1%}  best={thr_report.best_config.label()}  reduction={thr_report.escalation_reduction:.1%}")

        # ── [11] CertaintyRebalancer ──────────────────────────────────────────
        print("\n[11] CertaintyRebalancer...")
        from src.performance_calibration.certainty_rebalancing import CertaintyRebalancer
        cr      = CertaintyRebalancer()
        cr.fit(train_vecs)
        ce      = cr.enrich(test_vecs)
        cert_rpt = cr.build_analysis_report(test_vecs, ce)
        assert 0.0 <= cert_rpt.original_distribution.mean <= 1.0
        print(
            f"    PASS  orig_mean={cert_rpt.original_distribution.mean:.3f}  "
            f"norm_mean={cert_rpt.normalised_distribution.mean:.3f}  "
            f"improvement={cert_rpt.improvement_vs_original:.1%}"
        )

        # ── [12] ContradictionRebalancer ──────────────────────────────────────
        print("\n[12] ContradictionRebalancer...")
        from src.performance_calibration.contradiction_rebalancing import ContradictionRebalancer
        contr     = ContradictionRebalancer()
        contr_e   = contr.enrich(test_vecs)
        contr_rpt = contr.build_analysis_report(test_vecs, contr_e, y_pred_b, y_test)
        assert "CRITICAL" in contr_rpt.tier_stats or "NONE" in contr_rpt.tier_stats
        tiers_str = " ".join(
            f"{k}={v.count}" for k, v in contr_rpt.tier_stats.items()
        )
        print(f"    PASS  tiers: {tiers_str}")

        # ── [13] CompetitionSharpener ─────────────────────────────────────────
        print("\n[13] CompetitionSharpener...")
        from src.performance_calibration.competition_sharpening import CompetitionSharpener
        cs    = CompetitionSharpener()
        cse   = cs.enrich(test_vecs)
        X_cs, cs_names = cs.build_enriched_matrix(X_test_clin, test_vecs, cse)
        assert X_cs.shape[0] == len(test_vecs)
        assert X_cs.shape[1] > X_test_clin.shape[1]
        print(f"    PASS  matrix={X_cs.shape}  n_features={len(cs_names)}")

        # ── [14] SymbolicSignalEnricherV2 ─────────────────────────────────────
        print("\n[14] SymbolicSignalEnricherV2...")
        from src.performance_calibration.symbolic_signal_enrichment_v2 import (
            SymbolicSignalEnricherV2,
        )
        ev2    = SymbolicSignalEnricherV2()
        ev2e   = ev2.enrich(test_vecs)
        X_v2, v2n = ev2.build_feature_matrix(X_test_clin, ev2e)
        enr_rpt  = ev2.build_report(test_vecs, ev2e)
        assert X_v2.shape[1] == X_test_clin.shape[1] + 40, (
            f"Expected {X_test_clin.shape[1] + 40} cols, got {X_v2.shape[1]}"
        )
        assert enr_rpt.n_signals_total == 40
        print(
            f"    PASS  matrix={X_v2.shape}  "
            f"signals={enr_rpt.n_signals_total}  "
            f"redundant_pairs={len(enr_rpt.redundancy_pairs)}"
        )

        # ── [15] DiseaseSignatureRefiner ──────────────────────────────────────
        print("\n[15] DiseaseSignatureRefiner...")
        from src.performance_calibration.disease_signature_refinement import (
            DiseaseSignatureRefiner,
        )
        dsr     = DiseaseSignatureRefiner(class_labels)
        dsr_rpt = dsr.analyse(X_test_clin, y_test, test_vecs)
        assert len(dsr_rpt.feature_profiles) > 0
        print(
            f"    PASS  n_profiles={len(dsr_rpt.feature_profiles)}  "
            f"top_disc={len(dsr_rpt.top_discriminating_features)}  "
            f"overall_F={dsr_rpt.overall_discriminability_score:.3f}"
        )

        # ── [16] AdvancedBaselineCalibrator ───────────────────────────────────
        print("\n[16] AdvancedBaselineCalibrator...")
        from src.performance_calibration.advanced_baseline_calibration import (
            AdvancedBaselineCalibrator,
        )
        adv   = AdvancedBaselineCalibrator(
            n_splits=3, n_repeats=1, fast_mode=True, apply_scaling=True,
        )
        adv_b = adv.calibrate_model_b(X_train_clin, y_train, X_test_clin, y_test)
        assert adv_b.best_trial is not None
        print(
            f"    PASS  Model B — cv={adv_b.best_trial.cv_mean_accuracy:.4f}  "
            f"test={adv_b.test_accuracy:.4f}  features={adv_b.feature_count}"
        )
        adv_c = adv.calibrate_model_c_v2(
            X_train_clin, train_vecs, y_train,
            X_test_clin,  test_vecs,  y_test,
            ambiguity_ceiling=2.50,
            certainty_floor=0.40,
        )
        assert adv_c.best_trial is not None
        print(
            f"    PASS  Model C — cv={adv_c.best_trial.cv_mean_accuracy:.4f}  "
            f"test={adv_c.test_accuracy:.4f}  features={adv_c.feature_count}"
        )

        # ── [17] SymbolicRecoveryAnalyzer ─────────────────────────────────────
        print("\n[17] SymbolicRecoveryAnalyzer...")
        from src.performance_calibration.symbolic_recovery_analysis import (
            SymbolicRecoveryAnalyzer,
        )
        rec_rpt = SymbolicRecoveryAnalyzer(class_labels).analyse(
            test_vecs, y_pred_b, y_pred_c, y_test
        )
        assert isinstance(rec_rpt.n_recoveries, int)
        assert isinstance(rec_rpt.symbolic_contribution_index, float)
        print(
            f"    PASS  recoveries={rec_rpt.n_recoveries}  "
            f"regressions={rec_rpt.n_regressions}  "
            f"sci={rec_rpt.symbolic_contribution_index:+.4f}  "
            f"rec_rate={rec_rpt.recovery_rate:.1%}"
        )

        # ── [18] BiopsyReductionAnalyzer ──────────────────────────────────────
        print("\n[18] BiopsyReductionAnalyzer...")
        from src.performance_calibration.biopsy_reduction_analysis import (
            BiopsyReductionAnalyzer,
        )
        bio_rpt = BiopsyReductionAnalyzer(class_labels).analyse(
            test_vecs, y_pred_b, y_test
        )
        assert bio_rpt.total_cases == len(test_vecs)
        assert 0.0 <= bio_rpt.overall_escalation_rate_default <= 1.0
        assert bio_rpt.zero_safety_violations, "Safety violation detected!"
        print(
            f"    PASS  reduction={bio_rpt.biopsy_reduction_absolute} cases "
            f"({bio_rpt.biopsy_reduction_relative:.1%})  "
            f"safety={'OK' if bio_rpt.zero_safety_violations else 'VIOLATION'}"
        )

        # ── [19] FinalCalibrationReporter ─────────────────────────────────────
        print("\n[19] FinalCalibrationReporter...")
        from src.performance_calibration.final_calibration_report import (
            FinalCalibrationReporter,
        )
        reporter  = FinalCalibrationReporter(
            class_labels=class_labels,
            model_b_accuracy_before=acc_b,
            model_c_accuracy_before=acc_c,
            model_a_reference=1.0,    # synthetic data placeholder
            target_model_b=0.86,
            target_model_c=0.88,
        )
        final_rpt = reporter.compile(
            threshold_report=thr_report,
            certainty_report=cert_rpt,
            recovery_report=rec_rpt,
            biopsy_report=bio_rpt,
            contradiction_report=contr_rpt,
            enrichment_report=enr_rpt,
            model_b_result=adv_b,
            model_c_result=adv_c,
            symbolic_vectors=test_vecs,
        )
        txt    = final_rpt.summary()
        parsed = json.loads(final_rpt.to_json())

        # Structural assertions
        for section in ["SECTION 1", "SECTION 2", "SECTION 3", "SECTION 4",
                         "SECTION 5", "SECTION 6", "SECTION 7", "SECTION 8", "SECTION 9"]:
            assert section in txt, f"Missing {section} in summary"
        for key in ["performance_comparison", "escalation_summary",
                    "certainty_improvement", "symbolic_lift",
                    "contradiction_audit", "disease_improvements",
                    "trajectory_stabilization"]:
            assert key in parsed, f"Missing key {key} in JSON"

        print(
            f"    PASS  report={len(txt)} chars  "
            f"json_keys={len(parsed)}  "
            f"safety={'VERIFIED' if final_rpt.clinical_safety_verified else 'FAILED'}"
        )
        print("\n--- Report excerpt (first 800 chars) ---")
        print(txt[:800])
        print("...")

        print("\n" + "=" * 72)
        print("ALL PHASE 5 STEP 2 RECALIBRATION TESTS PASSED")
        print("=" * 72)

    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


if __name__ == "__main__":
    run_tests()
