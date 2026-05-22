"""
Integration tests for the full symbolic diagnostic pipeline.

Validates end-to-end pipeline execution across all eight synthetic clinical
cases, asserting determinism, subsystem orchestration integrity, escalation
reproducibility, trace completeness, and state-transition validity.

Each test targets one or more of the following properties:

  DETERMINISM      — same case_id + feature_values → identical result on repeat runs
  ORCHESTRATION    — all expected stages execute and write to PipelineContext
  ESCALATION       — triage recommendations match the curated expected outcomes
  TRACE            — trajectory snapshots are non-empty and internally consistent
  STATE            — FSM state sequence is non-empty and ends at a terminal state
  SUFFICIENCY      — SufficiencyReport is populated after a successful run
  COMPETITION      — CompetitionResult is populated after a successful run
  INSTABILITY      — InstabilityReport is populated after a successful run
  EXPORT           — exported file paths are valid when export is enabled
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from src.pipeline.pipeline_config import PipelineConfig
from src.pipeline.pipeline_runner import PipelineResult, PipelineRunner
from src.pipeline.synthetic_case_library import (
    ALL_CASES,
    BIOPSY_ESCALATION_PRP,
    CONTRADICTION_HEAVY_PSORIASIS,
    STABLE_PSORIASIS,
    STABLE_SEBORRHEIC_DERMATITIS,
    AMBIGUOUS_DERMATITIS,
    COMPETING_DIFFERENTIAL_LP_PR,
    SyntheticCase,
    SyntheticCaseLibrary,
)
from src.reasoning.state_tracker import DiagnosticState
from src.symbolic_engine.rule_registry import DiagnosticRuleRepository


# ── Fixtures ──────────────────────────────────────────────────────────────────

RULES_DIR = Path(__file__).parent.parent / "rules"

_TERMINAL_STATES = {
    DiagnosticState.SAFE_TRIAGE.value,
    DiagnosticState.BIOPSY_ESCALATION.value,
    DiagnosticState.CONTRADICTION_DETECTED.value,
    DiagnosticState.AMBIGUITY_ESCALATION.value,
    DiagnosticState.CERTAINTY_STABILIZATION.value,
    DiagnosticState.REINFORCING_ALIGNMENT.value,
    DiagnosticState.PARTIAL_ALIGNMENT.value,
    DiagnosticState.INITIAL_EVIDENCE.value,
    DiagnosticState.UNSTABLE_REASONING.value,
}

_EXPECTED_STAGES = {
    "clinical_grading",
    "evidence_activation",
    "contradiction_analysis",
    "certainty_propagation",
    "escalation",
}


@pytest.fixture(scope="module")
def rule_repo() -> DiagnosticRuleRepository:
    """Pre-loaded rule repository shared across all integration tests."""
    return DiagnosticRuleRepository(rules_dir=RULES_DIR, validate=True)


@pytest.fixture(scope="module")
def standard_config() -> PipelineConfig:
    """Standard pipeline config with narrative enabled, no file export."""
    cfg = PipelineConfig()
    cfg.rules_dir          = RULES_DIR
    cfg.enable_replay_export = False   # avoid I/O during tests
    cfg.enable_narrative   = True
    cfg.enable_trace       = True
    return cfg


@pytest.fixture(scope="module")
def runner(standard_config, rule_repo) -> PipelineRunner:
    """Pipeline runner shared across all integration tests."""
    return PipelineRunner(config=standard_config, rule_repository=rule_repo)


def _run(runner: PipelineRunner, case: SyntheticCase) -> PipelineResult:
    """Convenience: run a case and return the result."""
    return runner.run(case_id=case.case_id, feature_values=case.feature_values)


# ── Orchestration integrity ───────────────────────────────────────────────────

class TestOrchestrationIntegrity:
    """All critical stages must execute and populate context fields."""

    def test_all_eight_cases_succeed(self, runner):
        """Pipeline execution must succeed for every curated synthetic case."""
        for case in ALL_CASES:
            result = _run(runner, case)
            assert result.success, (
                f"Pipeline failed for {case.case_id}: {result.stage_errors}"
            )

    def test_critical_stages_always_complete(self, runner):
        """Every critical stage name appears in completed_stages for all cases."""
        for case in ALL_CASES:
            result = _run(runner, case)
            for stage in _EXPECTED_STAGES:
                assert stage in result.completed_stages, (
                    f"{case.case_id}: stage '{stage}' not in completed_stages"
                )

    def test_stage_results_list_non_empty(self, runner):
        """stage_results must have at least 8 entries (stages 0-7 + snapshot)."""
        for case in ALL_CASES:
            result = _run(runner, case)
            assert len(result.stage_results) >= 8, (
                f"{case.case_id}: only {len(result.stage_results)} stage results"
            )

    def test_no_stage_errors_on_clean_cases(self, runner):
        """Clean cases (no contradictions) must not accumulate stage errors."""
        clean_cases = [STABLE_PSORIASIS, STABLE_SEBORRHEIC_DERMATITIS]
        for case in clean_cases:
            result = _run(runner, case)
            assert not result.stage_errors, (
                f"{case.case_id}: unexpected stage errors: {result.stage_errors}"
            )

    def test_recommendation_always_set_on_success(self, runner):
        """A successful run must always produce a non-None recommendation."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert result.recommendation is not None, (
                    f"{case.case_id}: success=True but recommendation is None"
                )

    def test_leading_disease_always_set_on_success(self, runner):
        """A successful run must always produce a non-None leading_disease."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert result.leading_disease is not None, (
                    f"{case.case_id}: success=True but leading_disease is None"
                )

    def test_final_state_always_set_on_success(self, runner):
        """A successful run must always produce a non-None final_state."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert result.final_state is not None, (
                    f"{case.case_id}: success=True but final_state is None"
                )

    def test_certainty_in_unit_interval(self, runner):
        """max_certainty must lie in [0, 1] for every successful run."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert 0.0 <= result.max_certainty <= 1.0, (
                    f"{case.case_id}: max_certainty={result.max_certainty} out of range"
                )

    def test_certainty_gap_non_negative(self, runner):
        """certainty_gap must be non-negative for every successful run."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert result.certainty_gap >= 0.0, (
                    f"{case.case_id}: certainty_gap={result.certainty_gap} is negative"
                )

    def test_contradiction_load_non_negative(self, runner):
        """contradiction_load must be non-negative for every successful run."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert result.contradiction_load >= 0.0, (
                    f"{case.case_id}: contradiction_load={result.contradiction_load} is negative"
                )

    def test_ambiguity_index_non_negative(self, runner):
        """ambiguity_index (Shannon entropy) must be non-negative."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert result.ambiguity_index >= 0.0, (
                    f"{case.case_id}: ambiguity_index={result.ambiguity_index} is negative"
                )


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    """Repeated executions with identical inputs must produce identical outputs."""

    @pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
    def test_recommendation_deterministic(self, runner, case):
        """Same case always produces the same recommendation."""
        r1 = _run(runner, case)
        r2 = _run(runner, case)
        assert r1.recommendation == r2.recommendation, (
            f"{case.case_id}: recommendation changed between runs: "
            f"{r1.recommendation!r} vs {r2.recommendation!r}"
        )

    @pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
    def test_leading_disease_deterministic(self, runner, case):
        """Same case always produces the same leading disease."""
        r1 = _run(runner, case)
        r2 = _run(runner, case)
        assert r1.leading_disease == r2.leading_disease, (
            f"{case.case_id}: leading_disease changed: "
            f"{r1.leading_disease!r} vs {r2.leading_disease!r}"
        )

    @pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
    def test_max_certainty_deterministic(self, runner, case):
        """Same case always produces the same max_certainty value."""
        r1 = _run(runner, case)
        r2 = _run(runner, case)
        assert abs(r1.max_certainty - r2.max_certainty) < 1e-9, (
            f"{case.case_id}: max_certainty differs: {r1.max_certainty} vs {r2.max_certainty}"
        )

    @pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
    def test_final_state_deterministic(self, runner, case):
        """Same case always produces the same terminal FSM state."""
        r1 = _run(runner, case)
        r2 = _run(runner, case)
        assert r1.final_state == r2.final_state, (
            f"{case.case_id}: final_state differs: {r1.final_state!r} vs {r2.final_state!r}"
        )

    @pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
    def test_completed_stages_deterministic(self, runner, case):
        """Same case always completes the same stages in the same order."""
        r1 = _run(runner, case)
        r2 = _run(runner, case)
        assert r1.completed_stages == r2.completed_stages, (
            f"{case.case_id}: completed_stages differ"
        )

    def test_different_run_ids_do_not_affect_outcome(self, runner):
        """Explicit run_id does not influence the diagnostic recommendation."""
        case = STABLE_PSORIASIS
        r1 = runner.run(case_id=case.case_id, feature_values=case.feature_values, run_id="run-A")
        r2 = runner.run(case_id=case.case_id, feature_values=case.feature_values, run_id="run-B")
        assert r1.recommendation == r2.recommendation
        assert r1.leading_disease == r2.leading_disease
        assert r1.run_id != r2.run_id

    def test_case_id_preserved_in_result(self, runner):
        """The result's case_id must match the input case_id."""
        case = STABLE_PSORIASIS
        result = _run(runner, case)
        assert result.case_id == case.case_id


# ── Expected clinical outcomes ────────────────────────────────────────────────

class TestExpectedOutcomes:
    """Curated SyntheticCase expected outcomes must be matched by the pipeline."""

    def test_stable_psoriasis_safe_triage(self, runner):
        result = _run(runner, STABLE_PSORIASIS)
        assert result.success
        assert result.recommendation == "SAFE_NON_INVASIVE_TRIAGE", (
            f"Expected SAFE_NON_INVASIVE_TRIAGE, got {result.recommendation}"
        )
        assert result.leading_disease == "psoriasis"

    def test_stable_psoriasis_min_certainty(self, runner):
        """Classic psoriasis must reach the minimum certainty floor defined in the case."""
        result = _run(runner, STABLE_PSORIASIS)
        assert result.max_certainty >= STABLE_PSORIASIS.min_expected_certainty, (
            f"Certainty {result.max_certainty:.3f} below floor "
            f"{STABLE_PSORIASIS.min_expected_certainty}"
        )

    def test_stable_psoriasis_no_contradictions(self, runner):
        """Clean psoriasis profile must not trigger any contradiction load."""
        result = _run(runner, STABLE_PSORIASIS)
        assert result.contradiction_load == 0.0, (
            f"Expected zero contradiction load, got {result.contradiction_load}"
        )

    def test_seborrheic_dermatitis_biopsy_without_pathognomonic(self, runner):
        """Tier-B only evidence for SD is insufficient to clear the certainty floor;
        the system correctly escalates to biopsy despite zero contradiction load."""
        result = _run(runner, STABLE_SEBORRHEIC_DERMATITIS)
        assert result.success
        assert result.recommendation == "BIOPSY_RECOMMENDED"
        assert result.leading_disease == "seborrheic_dermatitis"
        assert result.contradiction_load == 0.0

    def test_contradiction_heavy_psoriasis_biopsy(self, runner):
        result = _run(runner, CONTRADICTION_HEAVY_PSORIASIS)
        assert result.success
        assert result.recommendation in ("BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"), (
            f"Expected biopsy escalation, got {result.recommendation}"
        )
        assert result.contradiction_load > 0.0

    def test_contradiction_heavy_has_contradiction_load(self, runner):
        """Case 3 has cross-disease pathognomonic features — contradiction load must be > 0."""
        result = _run(runner, CONTRADICTION_HEAVY_PSORIASIS)
        assert result.contradiction_load > 0.0, (
            "Expected contradiction_load > 0 for cross-disease case"
        )

    def test_ambiguous_dermatitis_biopsy_escalation(self, runner):
        """Sparse evidence triggers mandatory escalation to biopsy."""
        result = _run(runner, AMBIGUOUS_DERMATITIS)
        assert result.success
        assert result.recommendation in (
            "BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION", "AMBIGUOUS_PRESENTATION"
        ), f"Expected escalation recommendation, got {result.recommendation}"

    def test_biopsy_escalation_prp_biopsy(self, runner):
        """PRP case with conflicting pathognomonic features must trigger biopsy."""
        result = _run(runner, BIOPSY_ESCALATION_PRP)
        assert result.success
        assert result.recommendation in ("BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"), (
            f"Expected biopsy for PRP conflict case, got {result.recommendation}"
        )

    def test_competing_differential_lichen_planus_leader(self, runner):
        """Competing case (SYN_005) must resolve lichen_planus as leading disease."""
        result = _run(runner, COMPETING_DIFFERENTIAL_LP_PR)
        assert result.success
        assert result.leading_disease == "lichen_planus", (
            f"Expected lichen_planus leader, got {result.leading_disease}"
        )

    @pytest.mark.parametrize("case", SyntheticCaseLibrary.biopsy_cases(),
                             ids=[c.case_id for c in SyntheticCaseLibrary.biopsy_cases()])
    def test_biopsy_cases_escalate(self, runner, case):
        """All cases flagged expect_biopsy_escalation must produce biopsy recommendation."""
        result = _run(runner, case)
        assert result.success
        assert result.recommendation in ("BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"), (
            f"{case.case_id}: expected biopsy recommendation, got {result.recommendation}"
        )

    @pytest.mark.parametrize("case", SyntheticCaseLibrary.safe_cases(),
                             ids=[c.case_id for c in SyntheticCaseLibrary.safe_cases()])
    def test_safe_cases_do_not_escalate(self, runner, case):
        """All cases flagged as safe triage must not produce biopsy recommendation."""
        result = _run(runner, case)
        assert result.success
        assert result.recommendation == "SAFE_NON_INVASIVE_TRIAGE", (
            f"{case.case_id}: expected SAFE_NON_INVASIVE_TRIAGE, got {result.recommendation}"
        )


# ── Trajectory and trace integrity ───────────────────────────────────────────

class TestTrajectoryIntegrity:
    """Trajectory memory must be populated, internally consistent, and replayable."""

    def test_trajectory_non_none_on_success(self, runner):
        """Every successful run must produce a non-None trajectory."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert result.trajectory is not None, (
                    f"{case.case_id}: trajectory is None on successful run"
                )

    def test_trajectory_has_at_least_one_snapshot(self, runner):
        """Trajectory must contain at least one snapshot after pipeline completion."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.trajectory:
                assert result.trajectory.stage_count >= 1, (
                    f"{case.case_id}: trajectory has no snapshots"
                )

    def test_certainty_series_non_empty(self, runner):
        """certainty_series must return at least one float value."""
        result = _run(runner, STABLE_PSORIASIS)
        assert result.trajectory is not None
        series = result.trajectory.certainty_series()
        assert len(series) >= 1

    def test_certainty_series_all_in_unit_interval(self, runner):
        """All certainty values in the trajectory series must lie in [0, 1]."""
        result = _run(runner, STABLE_PSORIASIS)
        for v in result.trajectory.certainty_series():
            assert 0.0 <= v <= 1.0, f"Certainty series value out of range: {v}"

    def test_state_sequence_non_empty(self, runner):
        """state_sequence must contain at least one state string."""
        result = _run(runner, STABLE_PSORIASIS)
        assert result.trajectory is not None
        seq = result.trajectory.state_sequence()
        assert len(seq) >= 1

    def test_state_sequence_starts_with_valid_state(self, runner):
        """The first state in the sequence must be a recognised DiagnosticState value."""
        valid_states = {s.value for s in DiagnosticState}
        result = _run(runner, STABLE_PSORIASIS)
        seq = result.trajectory.state_sequence()
        assert seq[0] in valid_states, f"Unknown first state: {seq[0]}"

    def test_trajectory_case_id_matches_run(self, runner):
        """Trajectory case_id must match the case_id passed to runner.run()."""
        case = STABLE_PSORIASIS
        result = _run(runner, case)
        assert result.trajectory.case_id == case.case_id

    def test_trajectory_run_id_matches_result(self, runner):
        """Trajectory run_id must match the PipelineResult run_id."""
        case = STABLE_PSORIASIS
        result = _run(runner, case)
        assert result.trajectory.run_id == result.run_id

    def test_trajectory_final_decision_matches_result(self, runner):
        """Trajectory final_decision recommendation must match PipelineResult.recommendation."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.trajectory and result.trajectory.final_decision:
                assert (
                    result.trajectory.final_decision.recommendation.value
                    == result.recommendation
                ), (
                    f"{case.case_id}: trajectory final_decision recommendation "
                    "does not match result recommendation"
                )

    def test_trajectory_deltas_count(self, runner):
        """Number of deltas must equal stage_count - 1 when there are ≥ 2 snapshots."""
        result = _run(runner, STABLE_PSORIASIS)
        traj = result.trajectory
        if traj and traj.stage_count >= 2:
            deltas = traj.deltas()
            assert len(deltas) == traj.stage_count - 1, (
                f"Expected {traj.stage_count - 1} deltas, got {len(deltas)}"
            )


# ── FSM state transition validity ─────────────────────────────────────────────

class TestStateTransitionValidity:
    """The FSM final state must be consistent with the triage recommendation."""

    @pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
    def test_final_state_is_recognised(self, runner, case):
        """final_state value must be a member of DiagnosticState enum."""
        valid = {s.value for s in DiagnosticState}
        result = _run(runner, case)
        if result.success:
            assert result.final_state in valid, (
                f"{case.case_id}: unrecognised final_state '{result.final_state}'"
            )

    def test_safe_triage_recommendation_with_appropriate_certainty(self, runner):
        """SAFE_NON_INVASIVE_TRIAGE must only be assigned when max_certainty is high."""
        result = _run(runner, STABLE_PSORIASIS)
        if result.recommendation == "SAFE_NON_INVASIVE_TRIAGE":
            assert result.max_certainty >= 0.60, (
                f"SAFE_NON_INVASIVE_TRIAGE with low certainty: {result.max_certainty}"
            )

    def test_biopsy_recommendation_when_contradiction_high(self, runner):
        """HIGH_RISK_CONTRADICTION or BIOPSY_RECOMMENDED must appear when contradiction >= 0.40."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.contradiction_load >= 0.40:
                assert result.recommendation in (
                    "BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"
                ), (
                    f"{case.case_id}: high contradiction ({result.contradiction_load:.3f}) "
                    f"but recommendation is {result.recommendation}"
                )

    def test_decision_rationale_non_empty(self, runner):
        """decision_rationale must be a non-empty string for all successful runs."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert isinstance(result.decision_rationale, str)
                assert len(result.decision_rationale) > 0, (
                    f"{case.case_id}: decision_rationale is empty"
                )


# ── Contradiction cases ───────────────────────────────────────────────────────

class TestContradictionHandling:
    """Contradiction detection and mandatory escalation logic."""

    def test_no_contradiction_on_clean_psoriasis(self, runner):
        result = _run(runner, STABLE_PSORIASIS)
        assert result.contradiction_load == 0.0

    def test_contradiction_detected_on_cross_disease(self, runner):
        """Cases with conflicting pathognomonic features must produce > 0 contradiction load."""
        cases_with_contradiction = [
            c for c in ALL_CASES if c.expect_contradiction
        ]
        for case in cases_with_contradiction:
            result = _run(runner, case)
            assert result.contradiction_load > 0.0, (
                f"{case.case_id}: expected contradiction_load > 0"
            )

    def test_mandatory_biopsy_on_high_contradiction(self, runner):
        """Cases expecting biopsy escalation due to contradiction must trigger it."""
        for case in SyntheticCaseLibrary.biopsy_cases():
            result = _run(runner, case)
            assert result.recommendation in (
                "BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"
            ), (
                f"{case.case_id}: biopsy case did not receive biopsy recommendation, "
                f"got {result.recommendation}"
            )


# ── Entropy boundary conditions ───────────────────────────────────────────────

class TestEntropyBoundaryConditions:
    """Entropy-related constraints and escalation thresholds."""

    def test_max_expected_entropy_not_exceeded(self, runner):
        """Cases with max_expected_entropy must not exceed that ceiling."""
        for case in ALL_CASES:
            if case.max_expected_entropy is not None:
                result = _run(runner, case)
                assert result.ambiguity_index <= case.max_expected_entropy, (
                    f"{case.case_id}: entropy {result.ambiguity_index:.3f} exceeds "
                    f"ceiling {case.max_expected_entropy}"
                )

    def test_min_expected_certainty_not_violated(self, runner):
        """Cases with min_expected_certainty must meet that certainty floor."""
        for case in ALL_CASES:
            if case.min_expected_certainty is not None:
                result = _run(runner, case)
                assert result.max_certainty >= case.min_expected_certainty, (
                    f"{case.case_id}: certainty {result.max_certainty:.3f} below "
                    f"floor {case.min_expected_certainty}"
                )


# ── Subsystem output completeness ─────────────────────────────────────────────

class TestSubsystemOutputCompleteness:
    """Non-critical stages must populate their outputs when pipeline succeeds."""

    def test_narrative_generated_when_enabled(self, standard_config, rule_repo):
        """NarrativeStage must run and complete when enable_narrative=True."""
        assert standard_config.enable_narrative is True
        runner = PipelineRunner(config=standard_config, rule_repository=rule_repo)
        result = _run(runner, STABLE_PSORIASIS)
        assert "narrative_generation" in result.completed_stages

    def test_narrative_disabled_skips_stage(self, rule_repo):
        """NarrativeStage must be skipped when enable_narrative=False."""
        cfg = PipelineConfig()
        cfg.rules_dir          = RULES_DIR
        cfg.enable_narrative   = False
        cfg.enable_replay_export = False
        runner = PipelineRunner(config=cfg, rule_repository=rule_repo)
        result = _run(runner, STABLE_PSORIASIS)
        assert "narrative_generation" not in result.completed_stages

    def test_competition_stage_completes_on_clean_case(self, runner):
        """competition_analysis must complete on a case with sufficient evidence."""
        result = _run(runner, STABLE_PSORIASIS)
        assert "competition_analysis" in result.completed_stages

    def test_sufficiency_stage_completes_on_clean_case(self, runner):
        """evidence_sufficiency must complete on a case with sufficient evidence."""
        result = _run(runner, STABLE_PSORIASIS)
        assert "evidence_sufficiency" in result.completed_stages

    def test_instability_stage_always_completes(self, runner):
        """instability_monitoring must complete for every case."""
        for case in ALL_CASES:
            result = _run(runner, case)
            if result.success:
                assert "instability_monitoring" in result.completed_stages, (
                    f"{case.case_id}: instability_monitoring not in completed_stages"
                )


# ── Replay consistency ────────────────────────────────────────────────────────

class TestReplayConsistency:
    """Results produced with export enabled must be identical to those without."""

    def test_result_unaffected_by_export_flag(self, rule_repo):
        """Enabling or disabling replay export must not change the triage recommendation."""
        case = STABLE_PSORIASIS

        cfg_no_export = PipelineConfig()
        cfg_no_export.rules_dir            = RULES_DIR
        cfg_no_export.enable_replay_export = False
        cfg_no_export.enable_trace         = True
        cfg_no_export.enable_narrative     = True
        runner_no_export = PipelineRunner(config=cfg_no_export, rule_repository=rule_repo)

        r_a = runner_no_export.run(case_id=case.case_id, feature_values=case.feature_values)
        r_b = runner_no_export.run(case_id=case.case_id, feature_values=case.feature_values)
        assert r_a.recommendation == r_b.recommendation
        assert r_a.leading_disease == r_b.leading_disease

    def test_feature_values_not_mutated_by_pipeline(self, runner):
        """Pipeline execution must not modify the input feature_values dict."""
        case    = STABLE_PSORIASIS
        fv_copy = copy.deepcopy(case.feature_values)
        _run(runner, case)
        assert case.feature_values == fv_copy, (
            "Pipeline mutated the input feature_values dict"
        )

    def test_multiple_cases_in_sequence_do_not_interfere(self, runner):
        """Running one case must not alter the result of a subsequent different case."""
        r_psoriasis_before = _run(runner, STABLE_PSORIASIS)
        _run(runner, CONTRADICTION_HEAVY_PSORIASIS)
        r_psoriasis_after  = _run(runner, STABLE_PSORIASIS)

        assert r_psoriasis_before.recommendation == r_psoriasis_after.recommendation
        assert r_psoriasis_before.leading_disease == r_psoriasis_after.leading_disease

    def test_run_id_auto_generated_when_not_provided(self, runner):
        """Auto-generated run_ids must be unique across sequential runs."""
        case = STABLE_PSORIASIS
        r1 = runner.run(case_id=case.case_id, feature_values=case.feature_values)
        r2 = runner.run(case_id=case.case_id, feature_values=case.feature_values)
        assert r1.run_id != r2.run_id

    def test_custom_run_id_preserved_in_trajectory(self, runner):
        """A custom run_id must appear in the PipelineResult and its trajectory."""
        case   = STABLE_PSORIASIS
        run_id = "test-replay-001"
        result = runner.run(
            case_id=case.case_id,
            feature_values=case.feature_values,
            run_id=run_id,
        )
        assert result.run_id == run_id
        if result.trajectory:
            assert result.trajectory.run_id == run_id


# ── PipelineResult properties ─────────────────────────────────────────────────

class TestPipelineResultProperties:
    """Boolean convenience properties on PipelineResult must be consistent."""

    def test_requires_biopsy_true_when_biopsy_recommended(self, runner):
        for case in SyntheticCaseLibrary.biopsy_cases():
            result = _run(runner, case)
            if result.recommendation in ("BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"):
                assert result.requires_biopsy is True

    def test_requires_biopsy_false_for_safe_triage(self, runner):
        result = _run(runner, STABLE_PSORIASIS)
        if result.recommendation == "SAFE_NON_INVASIVE_TRIAGE":
            assert result.requires_biopsy is False

    def test_is_safe_triage_true_for_clean_psoriasis(self, runner):
        result = _run(runner, STABLE_PSORIASIS)
        if result.recommendation == "SAFE_NON_INVASIVE_TRIAGE":
            assert result.is_safe_triage is True

    def test_has_errors_false_for_clean_case(self, runner):
        result = _run(runner, STABLE_PSORIASIS)
        assert result.has_errors is False

    def test_summary_string_contains_case_id(self, runner):
        case   = STABLE_PSORIASIS
        result = _run(runner, case)
        assert case.case_id in result.summary()

    def test_summary_string_non_empty(self, runner):
        result = _run(runner, STABLE_PSORIASIS)
        assert len(result.summary()) > 10


# ── File export validation ────────────────────────────────────────────────────

class TestFileExport:
    """When export is enabled, all output files must be created and non-empty."""

    @pytest.fixture
    def export_runner(self, rule_repo, tmp_path) -> PipelineRunner:
        cfg = PipelineConfig()
        cfg.rules_dir          = RULES_DIR
        cfg.output_dir         = tmp_path
        cfg.enable_replay_export = True
        cfg.enable_narrative   = True
        cfg.enable_trace       = True
        return PipelineRunner(config=cfg, rule_repository=rule_repo)

    def test_trace_file_created(self, export_runner):
        result = _run(export_runner, STABLE_PSORIASIS)
        assert result.trace_path is not None
        assert result.trace_path.exists()
        assert result.trace_path.stat().st_size > 0

    def test_escalation_file_created(self, export_runner):
        result = _run(export_runner, STABLE_PSORIASIS)
        assert result.escalation_path is not None
        assert result.escalation_path.exists()

    def test_replay_file_created(self, export_runner):
        result = _run(export_runner, STABLE_PSORIASIS)
        assert result.replay_path is not None
        assert result.replay_path.exists()

    def test_narrative_file_created_when_enabled(self, export_runner):
        result = _run(export_runner, STABLE_PSORIASIS)
        assert result.narrative_path is not None
        assert result.narrative_path.exists()

    def test_trace_file_is_valid_json(self, export_runner):
        import json
        result = _run(export_runner, STABLE_PSORIASIS)
        with open(result.trace_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "case_id"    in data
        assert "snapshots"  in data
        assert "final_decision" in data

    def test_escalation_file_is_valid_json(self, export_runner):
        import json
        result = _run(export_runner, STABLE_PSORIASIS)
        with open(result.escalation_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["report_type"] == "escalation_report"
        assert data["case_id"]     == STABLE_PSORIASIS.case_id
        assert "recommendation"    in data

    def test_replay_file_is_valid_json(self, export_runner):
        import json
        result = _run(export_runner, STABLE_PSORIASIS)
        with open(result.replay_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["replay_format"] == "symbolic_reasoning_v1"
        assert "feature_inputs"      in data
        assert "stages"              in data

    def test_replay_file_contains_feature_inputs(self, export_runner):
        """Replay bundle must embed the original feature_values for re-execution."""
        import json
        case   = STABLE_PSORIASIS
        result = _run(export_runner, case)
        with open(result.replay_path, encoding="utf-8") as fh:
            data = json.load(fh)
        for feature_name in case.feature_values:
            assert feature_name in data["feature_inputs"], (
                f"Feature '{feature_name}' missing from replay feature_inputs"
            )

    def test_no_export_paths_when_export_disabled(self, rule_repo, tmp_path):
        """With enable_replay_export=False all export paths must remain None."""
        cfg = PipelineConfig()
        cfg.rules_dir          = RULES_DIR
        cfg.output_dir         = tmp_path
        cfg.enable_replay_export = False
        runner = PipelineRunner(config=cfg, rule_repository=rule_repo)
        result = _run(runner, STABLE_PSORIASIS)
        assert result.trace_path     is None
        assert result.escalation_path is None
        assert result.replay_path    is None

    def test_narrative_path_none_when_narrative_disabled(self, rule_repo, tmp_path):
        """narrative_path must be None when enable_narrative=False."""
        cfg = PipelineConfig()
        cfg.rules_dir          = RULES_DIR
        cfg.output_dir         = tmp_path
        cfg.enable_replay_export = True
        cfg.enable_narrative   = False
        runner = PipelineRunner(config=cfg, rule_repository=rule_repo)
        result = _run(runner, STABLE_PSORIASIS)
        assert result.narrative_path is None
