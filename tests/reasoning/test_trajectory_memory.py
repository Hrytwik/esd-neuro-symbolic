"""
Tests for DiagnosticTrajectoryMemory — replayable reasoning trace.

Validates snapshot recording, trajectory finalisation, delta computation,
series extraction, and immutable snapshot semantics.
"""

import pytest

from src.reasoning.state_tracker import DiagnosticState
from src.reasoning.trajectory_memory import (
    DiagnosticTrajectory,
    DiagnosticTrajectoryMemory,
    ReasoningSnapshot,
    StageDelta,
)


# ── Recording snapshots ───────────────────────────────────────────────────────

class TestSnapshotRecording:
    def test_record_returns_reasoning_snapshot(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, safe_safety_report
    ):
        memory = DiagnosticTrajectoryMemory("case_001", "run-test")
        snap = memory.record(
            stage=0,
            stage_name="Clinical Grading",
            state=DiagnosticState.PARTIAL_ALIGNMENT,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
            safety_report=safe_safety_report,
        )
        assert isinstance(snap, ReasoningSnapshot)
        assert snap.stage == 0
        assert snap.stage_name == "Clinical Grading"

    def test_snapshot_captures_correct_leading_disease(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, safe_safety_report
    ):
        memory = DiagnosticTrajectoryMemory("case_002", "run-test")
        snap = memory.record(
            stage=0,
            stage_name="Stage0",
            state=DiagnosticState.PARTIAL_ALIGNMENT,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
            safety_report=safe_safety_report,
        )
        assert snap.leading_disease == "psoriasis"
        assert snap.max_certainty == pytest.approx(0.72)

    def test_multiple_snapshots_build_trajectory(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, safe_safety_report
    ):
        memory = DiagnosticTrajectoryMemory("case_003", "run-test")
        for i in range(3):
            memory.record(
                stage=i, stage_name=f"Stage{i}",
                state=DiagnosticState.PARTIAL_ALIGNMENT,
                evidence=psoriasis_evidence_result,
                conflict=no_conflict_result,
                certainty=stable_certainty,
                safety_report=safe_safety_report,
            )
        assert memory.trajectory.stage_count == 3

    def test_snapshot_is_frozen(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, safe_safety_report
    ):
        memory = DiagnosticTrajectoryMemory("case_frozen", "run-test")
        snap = memory.record(
            stage=0, stage_name="Test",
            state=DiagnosticState.INITIAL_EVIDENCE,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
            safety_report=safe_safety_report,
        )
        with pytest.raises(Exception):
            snap.stage = 99  # type: ignore[misc]


# ── Finalisation ──────────────────────────────────────────────────────────────

class TestFinalisation:
    def test_finalise_returns_trajectory(self, safe_triage_decision):
        memory = DiagnosticTrajectoryMemory("case_fin", "run-fin")
        traj = memory.finalise(decision=safe_triage_decision)
        assert isinstance(traj, DiagnosticTrajectory)
        assert traj.final_decision is not None

    def test_finalise_without_decision(self):
        memory = DiagnosticTrajectoryMemory("case_nod", "run-nod")
        traj = memory.finalise()
        assert traj.final_decision is None

    def test_trajectory_has_case_and_run_ids(self):
        memory = DiagnosticTrajectoryMemory("test_case", "test_run")
        traj = memory.finalise()
        assert traj.case_id == "test_case"
        assert traj.run_id == "test_run"


# ── Delta computation ─────────────────────────────────────────────────────────

class TestDeltaComputation:
    def _build_two_stage_memory(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, ambiguous_certainty, safe_safety_report
    ) -> DiagnosticTrajectoryMemory:
        memory = DiagnosticTrajectoryMemory("case_delta", "run-delta")
        memory.record(
            stage=0, stage_name="Stage0",
            state=DiagnosticState.PARTIAL_ALIGNMENT,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
            safety_report=safe_safety_report,
        )
        memory.record(
            stage=1, stage_name="Stage1",
            state=DiagnosticState.AMBIGUITY_ESCALATION,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=ambiguous_certainty,
            safety_report=safe_safety_report,
        )
        return memory

    def test_deltas_returns_stage_delta_list(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, ambiguous_certainty, safe_safety_report
    ):
        memory = self._build_two_stage_memory(
            psoriasis_evidence_result, no_conflict_result,
            stable_certainty, ambiguous_certainty, safe_safety_report,
        )
        deltas = memory.trajectory.deltas()
        assert len(deltas) == 1
        assert isinstance(deltas[0], StageDelta)

    def test_delta_detects_state_change(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, ambiguous_certainty, safe_safety_report
    ):
        memory = self._build_two_stage_memory(
            psoriasis_evidence_result, no_conflict_result,
            stable_certainty, ambiguous_certainty, safe_safety_report,
        )
        deltas = memory.trajectory.deltas()
        assert deltas[0].state_changed

    def test_delta_certainty_is_negative_when_certainty_drops(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, ambiguous_certainty, safe_safety_report
    ):
        memory = self._build_two_stage_memory(
            psoriasis_evidence_result, no_conflict_result,
            stable_certainty, ambiguous_certainty, safe_safety_report,
        )
        deltas = memory.trajectory.deltas()
        # certainty drops from 0.72 to 0.30
        assert deltas[0].certainty_delta < 0

    def test_no_deltas_for_single_snapshot(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, safe_safety_report
    ):
        memory = DiagnosticTrajectoryMemory("case_one", "run-one")
        memory.record(
            stage=0, stage_name="Stage0",
            state=DiagnosticState.INITIAL_EVIDENCE,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
            safety_report=safe_safety_report,
        )
        assert memory.trajectory.deltas() == []


# ── Series extraction ─────────────────────────────────────────────────────────

class TestSeriesExtraction:
    def test_certainty_series_returns_list_of_floats(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, safe_safety_report
    ):
        memory = DiagnosticTrajectoryMemory("case_s", "run-s")
        for i in range(3):
            memory.record(
                stage=i, stage_name=f"S{i}",
                state=DiagnosticState.PARTIAL_ALIGNMENT,
                evidence=psoriasis_evidence_result,
                conflict=no_conflict_result,
                certainty=stable_certainty,
                safety_report=safe_safety_report,
            )
        series = memory.trajectory.certainty_series()
        assert len(series) == 3
        assert all(isinstance(v, float) for v in series)

    def test_state_sequence_contains_state_values(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, safe_safety_report
    ):
        memory = DiagnosticTrajectoryMemory("case_ss", "run-ss")
        memory.record(
            stage=0, stage_name="S0",
            state=DiagnosticState.PARTIAL_ALIGNMENT,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
            safety_report=safe_safety_report,
        )
        seq = memory.trajectory.state_sequence()
        assert seq == ["PARTIAL_ALIGNMENT"]


# ── Trajectory metadata ───────────────────────────────────────────────────────

class TestTrajectoryMetadata:
    def test_empty_trajectory_has_no_final_state(self):
        memory = DiagnosticTrajectoryMemory("empty", "run")
        assert memory.trajectory.final_state is None
        assert memory.trajectory.final_certainty == 0.0

    def test_get_stage_returns_correct_snapshot(
        self, psoriasis_evidence_result, no_conflict_result,
        stable_certainty, safe_safety_report
    ):
        memory = DiagnosticTrajectoryMemory("case_get", "run-get")
        memory.record(
            stage=2, stage_name="Stage2",
            state=DiagnosticState.CERTAINTY_STABILIZATION,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
            safety_report=safe_safety_report,
        )
        snap = memory.trajectory.get_stage(2)
        assert snap is not None
        assert snap.stage_name == "Stage2"

    def test_get_stage_none_for_missing(self):
        memory = DiagnosticTrajectoryMemory("case_miss", "run-miss")
        assert memory.trajectory.get_stage(99) is None
