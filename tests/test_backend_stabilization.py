"""
tests/test_backend_stabilization.py
=====================================
Integration tests for src/backend_stabilization/ — all 10 modules.

Run with:
    python -X utf8 tests/test_backend_stabilization.py
"""

import sys
import os
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ──────────────────────────────────────────────────────────────────────────────
# Shared synthetic data
# ──────────────────────────────────────────────────────────────────────────────

DISEASES = [
    "psoriasis",
    "seborrheic_dermatitis",
    "lichen_planus",
    "pityriasis_rosea",
    "chronic_dermatitis",
    "pityriasis_rubra_pilaris",
]
N        = 110
N_SIG    = 22
N_CLIN   = 12

rng = np.random.default_rng(seed=0)
y_true  = rng.integers(0, 6, N)
y_pred_b = np.where(rng.random(N) < 0.80, y_true, rng.integers(0, 6, N))
y_pred_c = np.where(rng.random(N) < 0.82, y_true, rng.integers(0, 6, N))
X_clin   = rng.uniform(0, 3, (N, N_CLIN))
sym_mat  = rng.uniform(0.0, 1.0, (N, N_SIG))
contra_loads   = rng.uniform(0.0, 0.38, N)
ambig_bits     = rng.uniform(0.8, 3.5, N)
certainty_b    = rng.uniform(0.45, 0.90, N)
certainty_c    = np.clip(certainty_b + rng.uniform(-0.05, 0.15, N), 0.0, 1.0)
traj_steps     = rng.integers(1, 9, N)
comp_margins   = rng.uniform(0.05, 0.50, N)
esc_flags      = rng.random(N) < 0.60   # ~60 % escalation (still too high)

PASS = 0
FAIL = 0
ERRORS = []


def run_test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS]  {name}")
        PASS += 1
    except Exception as exc:
        print(f"  [FAIL]  {name}")
        traceback.print_exc()
        ERRORS.append((name, exc))
        FAIL += 1


# ──────────────────────────────────────────────────────────────────────────────
# [1] discriminative_optimization
# ──────────────────────────────────────────────────────────────────────────────

def test_discriminative_optimization():
    from src.backend_stabilization.discriminative_optimization import DiscriminativeOptimizer

    opt    = DiscriminativeOptimizer(class_labels=DISEASES)
    report = opt.analyse(sym_mat, X_clin, y_true, y_pred_b)

    assert len(report.disease_profiles) > 0
    assert len(report.pair_separations) >= 0
    assert 0.0 <= report.overall_mean_certainty_gap <= 1.0
    txt = report.summary()
    assert "DISCRIMINATIVE" in txt
    assert len(report.recommendations) >= 0


# ──────────────────────────────────────────────────────────────────────────────
# [2] disease_separation_refinement
# ──────────────────────────────────────────────────────────────────────────────

def test_disease_separation_refinement():
    from src.backend_stabilization.disease_separation_refinement import DiseaseSeparationRefiner

    refiner = DiseaseSeparationRefiner(class_labels=DISEASES)
    report  = refiner.analyse(
        sym_mat, y_true, y_pred_b, y_pred_c,
        certainty_scores=certainty_b,
        ambiguity_bits=ambig_bits,
        escalation_flags=esc_flags,
    )

    assert len(report.disease_profiles) == len(DISEASES)
    assert len(report.biopsy_profiles)  == len(DISEASES)
    assert 0.0 <= report.mean_clinical_accuracy <= 1.0
    assert 0.0 <= report.mean_symbolic_accuracy <= 1.0
    assert isinstance(report.diseases_safely_clinical, list)
    assert isinstance(report.diseases_needing_biopsy_support, list)
    txt = report.summary()
    assert "DISEASE SEPARATION" in txt
    # Each disease should appear in the summary
    for d in DISEASES:
        assert d in txt


# ──────────────────────────────────────────────────────────────────────────────
# [3] symbolic_recovery_optimizer
# ──────────────────────────────────────────────────────────────────────────────

def test_symbolic_recovery_optimizer():
    from src.backend_stabilization.symbolic_recovery_optimizer import SymbolicRecoveryOptimizer

    opt    = SymbolicRecoveryOptimizer(class_labels=DISEASES, n_samples=N)
    report = opt.analyse(
        y_true, y_pred_b, y_pred_c,
        contradiction_loads=contra_loads,
        trajectory_steps=traj_steps,
        competition_margins=comp_margins,
        certainty_b=certainty_b,
        certainty_c=certainty_c,
    )

    assert report.n_clinical_errors >= 0
    assert 0.0 <= report.overall_recovery_rate <= 1.0
    assert report.estimated_accuracy_gain_pp >= 0.0
    assert len(report.mechanism_breakdown) == 7   # all 7 mechanisms
    assert report.contradiction_profile.high_contradiction_ceiling == 0.40
    txt = report.summary()
    assert "SYMBOLIC RECOVERY" in txt


# ──────────────────────────────────────────────────────────────────────────────
# [4] escalation_selectivity_optimizer
# ──────────────────────────────────────────────────────────────────────────────

def test_escalation_selectivity_optimizer():
    from src.backend_stabilization.escalation_selectivity_optimizer import (
        EscalationSelectivityOptimizer, SelectivityTier
    )

    opt    = EscalationSelectivityOptimizer(class_labels=DISEASES, n_samples=N)
    report = opt.analyse(
        y_true, y_pred_b, esc_flags,
        ambiguity_bits=ambig_bits,
        contradiction_loads=contra_loads,
        certainty_scores=certainty_b,
        competition_margins=comp_margins,
    )

    assert 0.0 <= report.current_escalation_rate <= 1.0
    assert report.selectivity_tier in SelectivityTier
    assert len(report.selectivity_curve) > 0
    assert len(report.threshold_candidates) > 0
    assert report.best_threshold is not None
    assert len(report.disease_profiles) == len(DISEASES)
    assert 0.0 <= report.stabilisation_prevalence.fraction_safely_suppressible <= 1.0
    txt = report.summary()
    assert "ESCALATION SELECTIVITY" in txt


# ──────────────────────────────────────────────────────────────────────────────
# [5] trajectory_stabilization
# ──────────────────────────────────────────────────────────────────────────────

def test_trajectory_stabilization():
    from src.backend_stabilization.trajectory_stabilization import TrajectoryStabilizer

    stab   = TrajectoryStabilizer(class_labels=DISEASES, max_steps=6)
    report = stab.analyse(
        y_true, y_pred_b,
        contradiction_loads=contra_loads,
    )

    assert report.n_cases == N
    assert 0.0 <= report.fraction_rapid <= 1.0
    assert 0.0 <= report.fraction_non_convergent <= 1.0
    assert 0.0 <= report.mean_final_certainty <= 1.0
    assert 0.0 <= report.convergence_realism.realism_score <= 1.0
    assert len(report.disease_profiles) == len(DISEASES)
    txt = report.summary()
    assert "TRAJECTORY STABILISATION" in txt


# ──────────────────────────────────────────────────────────────────────────────
# [6] contradiction_localization
# ──────────────────────────────────────────────────────────────────────────────

def test_contradiction_localization():
    from src.backend_stabilization.contradiction_localization import ContradictionLocalizer

    localizer = ContradictionLocalizer(class_labels=DISEASES)
    report    = localizer.analyse(y_true, global_loads=contra_loads)

    assert report.n_cases == N
    assert len(report.signal_profiles) > 0
    assert len(report.disease_affinities) == len(DISEASES)
    # Ceiling must always be enforced
    assert report.ceiling_audit.ceiling_enforcement_rate == 1.0
    assert 0.0 <= report.mean_population_load <= 0.40
    txt = report.summary()
    assert "CONTRADICTION LOCALISATION" in txt
    assert "0.40" in txt   # ceiling must be mentioned


# ──────────────────────────────────────────────────────────────────────────────
# [7] reasoning_contract_finalizer
# ──────────────────────────────────────────────────────────────────────────────

def test_reasoning_contract_finalizer():
    from src.backend_stabilization.reasoning_contract_finalizer import (
        ReasoningContractFinalizer, REASONING_CONTRACT_VERSION, REASONING_CONTRACT_FROZEN
    )

    assert REASONING_CONTRACT_FROZEN is True

    finalizer = ReasoningContractFinalizer()

    # Valid output
    good = finalizer.canonical_empty_output("case_001")
    good["leading_diagnosis"] = "psoriasis"
    good["certainty"]         = 0.82
    result = finalizer.validate_output(good)
    assert result.is_valid, f"Expected valid, got errors: {result.errors}"

    # Contradiction ceiling violation
    bad = finalizer.canonical_empty_output("case_002")
    bad["contradiction"]["overall_load"] = 0.55   # violates ceiling
    result_bad = finalizer.validate_output(bad)
    assert not result_bad.is_valid
    assert any("ceiling" in e.lower() or "0.40" in e for e in result_bad.errors)

    # Mutual exclusion
    bad2 = finalizer.canonical_empty_output("case_003")
    bad2["requires_biopsy"]  = True
    bad2["is_safe_triage"]   = True
    result_bad2 = finalizer.validate_output(bad2)
    assert not result_bad2.is_valid

    # Batch audit
    outputs = [finalizer.canonical_empty_output(f"c_{i}") for i in range(10)]
    report  = finalizer.audit_batch(outputs)
    assert report.n_outputs_checked == 10
    assert report.n_valid == 10
    txt = report.summary()
    assert "REASONING CONTRACT" in txt


# ──────────────────────────────────────────────────────────────────────────────
# [8] replay_schema_finalizer
# ──────────────────────────────────────────────────────────────────────────────

def test_replay_schema_finalizer():
    from src.backend_stabilization.replay_schema_finalizer import (
        ReplaySchemaFinalizer, REPLAY_SCHEMA_VERSION, REPLAY_SCHEMA_FROZEN,
        ReplayEventType
    )

    assert REPLAY_SCHEMA_FROZEN is True

    finalizer = ReplaySchemaFinalizer()

    # Build a valid case
    events = [
        finalizer.canonical_event(
            case_id="c_1", step=s,
            event_type=ReplayEventType.TRAJECTORY_STEP.value,
            fsm_state="CLINICAL_ASSESSMENT",
            certainty=0.60 + s * 0.05,
            ambiguity_bits=2.0 - s * 0.1,
            leading_diagnosis="psoriasis",
            contradiction_load=0.10,
        )
        for s in range(4)
    ]
    case = finalizer.canonical_case(
        case_id="c_1",
        final_diagnosis="psoriasis",
        events=events,
        converged=True,
        requires_biopsy=False,
        is_safe_triage=True,
        final_certainty=0.80,
    )
    result = finalizer.validate_case(case)
    assert result.is_valid, f"Expected valid: {result.errors}"

    # Ceiling violation in event
    bad_event = finalizer.canonical_event(
        "c_2", 0, "clinical_eval", "INITIAL", 0.5, 2.0, "lichen_planus",
        contradiction_load=0.45,
    )
    # Manually break the ceiling (canonical_event clips it)
    bad_event["contradiction_load"] = 0.45
    errs = finalizer.validate_event(bad_event)
    assert any("ceiling" in e.lower() or "0.40" in e for e in errs)

    # Batch audit
    cases  = [case]
    report = finalizer.audit_batch(cases)
    assert report.n_valid == 1
    txt = report.summary()
    assert "REPLAY SCHEMA" in txt


# ──────────────────────────────────────────────────────────────────────────────
# [9] graph_contract_finalizer
# ──────────────────────────────────────────────────────────────────────────────

def test_graph_contract_finalizer():
    from src.backend_stabilization.graph_contract_finalizer import (
        GraphContractFinalizer, GRAPH_CONTRACT_VERSION, GRAPH_CONTRACT_FROZEN,
        GraphNodeType, GraphEdgeType
    )

    assert GRAPH_CONTRACT_FROZEN is True

    finalizer = GraphContractFinalizer()

    nodes = [
        finalizer.canonical_node("n_pso",  GraphNodeType.DISEASE.value,  "psoriasis"),
        finalizer.canonical_node("n_feat", GraphNodeType.FEATURE.value,  "scaling"),
        finalizer.canonical_node("n_sig",  GraphNodeType.SIGNAL.value,   "morph_index"),
    ]
    edges = [
        finalizer.canonical_edge("e_1", "n_feat", "n_pso", GraphEdgeType.SUPPORTS.value, 0.85),
        finalizer.canonical_edge("e_2", "n_sig",  "n_pso", GraphEdgeType.SUPPORTS.value, 0.70),
    ]
    snap = finalizer.canonical_snapshot("snap_1", nodes, edges, case_id="c_1", step=0)
    result = finalizer.validate_snapshot(snap)
    assert result.is_valid, f"Expected valid: {result.errors}"
    assert result.n_nodes == 3
    assert result.n_edges == 2

    # Invalid edge reference
    bad_edge = finalizer.canonical_edge("e_bad", "n_nonexistent", "n_pso",
                                        GraphEdgeType.SUPPORTS.value, 0.5)
    snap_bad = finalizer.canonical_snapshot("snap_2", nodes, [bad_edge])
    result_bad = finalizer.validate_snapshot(snap_bad)
    assert not result_bad.is_valid

    # Batch audit
    report = finalizer.audit_batch([snap])
    assert report.n_valid == 1
    txt = report.summary()
    assert "GRAPH CONTRACT" in txt


# ──────────────────────────────────────────────────────────────────────────────
# [10] backend_maturity_report
# ──────────────────────────────────────────────────────────────────────────────

def test_backend_maturity_report():
    from src.backend_stabilization.backend_maturity_report import (
        BackendMaturityReporter, MaturityLevel, FrontendReadiness
    )

    reporter = BackendMaturityReporter()
    report   = reporter.compile(
        model_a_accuracy=0.9818,
        model_b_accuracy=0.86,
        model_c_accuracy=0.89,
        escalation_rate=0.35,
        contradiction_ceiling_enforced=True,
        escalation_logic_intact=True,
        contradiction_handling_intact=True,
        biopsy_pathway_intact=True,
        interpretability_preserved=True,
        reasoning_validation_rate=1.0,
        replay_validation_rate=1.0,
        graph_validation_rate=1.0,
    )

    assert 0.0 <= report.overall_maturity_score <= 1.0
    assert report.maturity_level in MaturityLevel
    assert report.frontend_readiness in FrontendReadiness
    assert report.safety_audit.contradiction_ceiling_enforced is True
    assert report.safety_audit.all_constraints_satisfied is True
    assert report.model_performance.model_b_target_met is True
    assert report.model_performance.model_c_target_met is True
    assert report.model_performance.escalation_target_met is True
    assert len(report.checklist_items) >= 10
    assert len(report.frontend_blockers) == 0   # all constraints satisfied

    txt  = report.summary()
    d    = report.to_dict()
    for section in ["BACKEND MATURITY", "Model Performance", "Safety",
                    "Schema", "Subsystem", "Frontend"]:
        assert section in txt, f"Missing section: {section}"
    for key in ["overall_maturity_score", "maturity_level", "frontend_readiness",
                "safety_audit", "model_performance", "schema_readiness",
                "subsystem_audits", "checklist"]:
        assert key in d, f"Missing key: {key}"

    # Safety ceiling must ALWAYS block if violated
    report_bad = reporter.compile(contradiction_ceiling_enforced=False)
    assert report_bad.frontend_readiness == FrontendReadiness.NOT_READY
    assert len(report_bad.frontend_blockers) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# [11] package __init__ imports
# ──────────────────────────────────────────────────────────────────────────────

def test_package_init():
    import src.backend_stabilization as bs

    for attr in [
        "DiscriminativeOptimizer",
        "DiseaseSeparationRefiner",
        "SymbolicRecoveryOptimizer",
        "EscalationSelectivityOptimizer",
        "TrajectoryStabilizer",
        "ContradictionLocalizer",
        "ReasoningContractFinalizer",
        "ReplaySchemaFinalizer",
        "GraphContractFinalizer",
        "BackendMaturityReporter",
    ]:
        assert hasattr(bs, attr), f"__init__ missing export: {attr}"


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("BACKEND STABILISATION INTEGRATION TESTS")
    print("=" * 70)

    run_test("[1]  discriminative_optimization",       test_discriminative_optimization)
    run_test("[2]  disease_separation_refinement",     test_disease_separation_refinement)
    run_test("[3]  symbolic_recovery_optimizer",       test_symbolic_recovery_optimizer)
    run_test("[4]  escalation_selectivity_optimizer",  test_escalation_selectivity_optimizer)
    run_test("[5]  trajectory_stabilization",          test_trajectory_stabilization)
    run_test("[6]  contradiction_localization",        test_contradiction_localization)
    run_test("[7]  reasoning_contract_finalizer",      test_reasoning_contract_finalizer)
    run_test("[8]  replay_schema_finalizer",           test_replay_schema_finalizer)
    run_test("[9]  graph_contract_finalizer",          test_graph_contract_finalizer)
    run_test("[10] backend_maturity_report",           test_backend_maturity_report)
    run_test("[11] package __init__ imports",          test_package_init)

    print("=" * 70)
    print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    if FAIL == 0:
        print("  ALL BACKEND STABILISATION TESTS PASSED")
    else:
        print("  FAILURES:")
        for name, exc in ERRORS:
            print(f"    ✗  {name}: {exc}")
    print("=" * 70)
    sys.exit(0 if FAIL == 0 else 1)
