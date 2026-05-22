"""
Tests for DiagnosticStateTracker — 9-state FSM transitions.

Validates state advancement guard conditions, escalation overrides,
terminal state detection, instability-driven transitions, and the
full INITIAL_EVIDENCE → SAFE_TRIAGE pathway.
"""

import pytest

from src.reasoning.state_tracker import DiagnosticState, DiagnosticStateTracker


# ── Initial state ─────────────────────────────────────────────────────────────

class TestInitialState:
    def test_tracker_starts_in_initial_evidence(self, state_tracker):
        assert state_tracker.current_state == DiagnosticState.INITIAL_EVIDENCE

    def test_no_transitions_at_start(self, state_tracker):
        assert state_tracker.transition_history == []

    def test_not_terminal_at_start(self, state_tracker):
        assert not state_tracker.is_terminal


# ── INITIAL_EVIDENCE → PARTIAL_ALIGNMENT ─────────────────────────────────────

class TestInitialToPartial:
    def test_advances_when_enough_rules_active(
        self, state_tracker, psoriasis_evidence_result,
        no_conflict_result, stable_certainty
    ):
        # psoriasis_evidence_result has total_rules_active=4 >= min_partial=2
        state = state_tracker.advance(
            stage=0,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
        )
        assert state == DiagnosticState.PARTIAL_ALIGNMENT

    def test_stays_in_initial_when_too_few_rules(
        self, state_tracker, psoriasis_evidence_result,
        no_conflict_result, stable_certainty
    ):
        tracker = DiagnosticStateTracker(min_rules_partial=100)
        state = tracker.advance(
            stage=0,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
        )
        assert state == DiagnosticState.INITIAL_EVIDENCE


# ── PARTIAL_ALIGNMENT → REINFORCING_ALIGNMENT ────────────────────────────────

class TestPartialToReinforcing:
    def test_advances_when_leading_has_enough_rules(
        self, psoriasis_evidence_result, no_conflict_result, stable_certainty
    ):
        # Use min_rules_reinforcing=3; psoriasis has active_rule_count=3
        tracker = DiagnosticStateTracker(
            min_rules_partial=2,
            min_rules_reinforcing=3,
        )
        tracker.advance(0, psoriasis_evidence_result, no_conflict_result, stable_certainty)
        assert tracker.current_state == DiagnosticState.PARTIAL_ALIGNMENT

        state = tracker.advance(1, psoriasis_evidence_result, no_conflict_result, stable_certainty)
        assert state == DiagnosticState.REINFORCING_ALIGNMENT


# ── BIOPSY_ESCALATION override ────────────────────────────────────────────────

class TestBiopsyEscalationOverride:
    def test_high_contradiction_triggers_biopsy_escalation(
        self, state_tracker, psoriasis_evidence_result,
        high_conflict_result, stable_certainty
    ):
        # high_conflict_result has load=0.50 >= biopsy ceiling=0.40
        state = state_tracker.advance(
            stage=0,
            evidence=psoriasis_evidence_result,
            conflict=high_conflict_result,
            certainty=stable_certainty,
        )
        assert state == DiagnosticState.BIOPSY_ESCALATION

    def test_high_entropy_triggers_biopsy_escalation(
        self, state_tracker, psoriasis_evidence_result,
        no_conflict_result, ambiguous_certainty
    ):
        # ambiguous_certainty has entropy=2.30 >= biopsy entropy ceiling=1.50
        state = state_tracker.advance(
            stage=0,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=ambiguous_certainty,
        )
        assert state == DiagnosticState.BIOPSY_ESCALATION

    def test_biopsy_escalation_is_terminal(
        self, state_tracker, psoriasis_evidence_result,
        high_conflict_result, stable_certainty
    ):
        state_tracker.advance(0, psoriasis_evidence_result, high_conflict_result, stable_certainty)
        assert state_tracker.is_terminal

    def test_no_further_transitions_from_terminal(
        self, state_tracker, psoriasis_evidence_result,
        high_conflict_result, no_conflict_result, stable_certainty
    ):
        state_tracker.advance(0, psoriasis_evidence_result, high_conflict_result, stable_certainty)
        # Calling advance again should NOT change state
        state = state_tracker.advance(
            1, psoriasis_evidence_result, no_conflict_result, stable_certainty
        )
        assert state == DiagnosticState.BIOPSY_ESCALATION


# ── UNSTABLE_REASONING ────────────────────────────────────────────────────────

class TestUnstableReasoning:
    def test_high_instability_index_triggers_unstable(
        self, state_tracker, psoriasis_evidence_result,
        no_conflict_result, stable_certainty
    ):
        state = state_tracker.advance(
            stage=0,
            evidence=psoriasis_evidence_result,
            conflict=no_conflict_result,
            certainty=stable_certainty,
            instability_index=0.75,   # above threshold=0.60
        )
        assert state == DiagnosticState.UNSTABLE_REASONING

    def test_unstable_reasoning_is_terminal(
        self, state_tracker, psoriasis_evidence_result,
        no_conflict_result, stable_certainty
    ):
        state_tracker.advance(0, psoriasis_evidence_result, no_conflict_result,
                              stable_certainty, instability_index=0.80)
        assert state_tracker.is_terminal


# ── Reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_state_and_history(
        self, state_tracker, psoriasis_evidence_result,
        no_conflict_result, stable_certainty
    ):
        state_tracker.advance(0, psoriasis_evidence_result, no_conflict_result, stable_certainty)
        state_tracker.reset()
        assert state_tracker.current_state == DiagnosticState.INITIAL_EVIDENCE
        assert state_tracker.transition_history == []


# ── Transition records ────────────────────────────────────────────────────────

class TestTransitionRecords:
    def test_transition_history_records_each_change(
        self, state_tracker, psoriasis_evidence_result,
        no_conflict_result, stable_certainty
    ):
        state_tracker.advance(0, psoriasis_evidence_result, no_conflict_result, stable_certainty)
        assert len(state_tracker.transition_history) == 1
        t = state_tracker.transition_history[0]
        assert t.from_state == DiagnosticState.INITIAL_EVIDENCE
        assert t.to_state == DiagnosticState.PARTIAL_ALIGNMENT
        assert t.stage == 0

    def test_escalation_flag_set_for_biopsy_transition(
        self, state_tracker, psoriasis_evidence_result,
        high_conflict_result, stable_certainty
    ):
        state_tracker.advance(0, psoriasis_evidence_result, high_conflict_result, stable_certainty)
        t = state_tracker.transition_history[0]
        assert t.is_escalation
