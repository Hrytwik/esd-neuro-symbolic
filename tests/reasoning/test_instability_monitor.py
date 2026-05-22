"""
Tests for DiagnosticInstabilityMonitor — certainty volatility tracking.

Validates oscillation detection, hypothesis switch counting, entropy
trend analysis, instability index computation, and composite signal weighting.
"""

import pytest

from src.reasoning.instability_monitor import (
    DiagnosticInstabilityMonitor,
    InstabilityReport,
    InstabilitySignal,
)


# ── Empty / baseline ──────────────────────────────────────────────────────────

class TestEmptyAssessment:
    def test_assess_with_no_updates_returns_zero_index(self, instability_monitor):
        report = instability_monitor.assess()
        assert report.instability_index == pytest.approx(0.0)
        assert not report.is_unstable
        assert report.active_signals == []

    def test_single_update_produces_zero_volatility(self, instability_monitor):
        instability_monitor.update(
            stage=0, max_certainty=0.72, contradiction_load=0.0,
            ambiguity_index=0.50, leading_disease="psoriasis",
            state="PARTIAL_ALIGNMENT",
        )
        report = instability_monitor.assess()
        assert report.certainty_volatility == pytest.approx(0.0)


# ── Certainty oscillation signal ──────────────────────────────────────────────

class TestCertaintyOscillation:
    def test_oscillating_certainty_triggers_signal(self, instability_monitor):
        # High std → oscillation signal
        instability_monitor.update(0, 0.30, 0.0, 0.5, "psoriasis", "PARTIAL_ALIGNMENT")
        instability_monitor.update(1, 0.80, 0.0, 0.5, "psoriasis", "REINFORCING_ALIGNMENT")
        instability_monitor.update(2, 0.20, 0.0, 0.5, "psoriasis", "PARTIAL_ALIGNMENT")

        report = instability_monitor.assess()
        signal_types = [s.signal_type for s in report.active_signals]
        assert "certainty_oscillation" in signal_types

    def test_stable_certainty_no_oscillation_signal(self, instability_monitor):
        for i in range(4):
            instability_monitor.update(i, 0.75, 0.0, 0.5, "psoriasis", "CERTAINTY_STABILIZATION")

        report = instability_monitor.assess()
        signal_types = [s.signal_type for s in report.active_signals]
        assert "certainty_oscillation" not in signal_types


# ── Hypothesis switching ──────────────────────────────────────────────────────

class TestHypothesisSwitching:
    def test_hypothesis_switch_detected(self, instability_monitor):
        instability_monitor.update(0, 0.60, 0.0, 0.5, "psoriasis", "PARTIAL_ALIGNMENT")
        instability_monitor.update(1, 0.50, 0.0, 0.5, "lichen_planus", "CONTRADICTION_DETECTED")
        instability_monitor.update(2, 0.65, 0.0, 0.5, "psoriasis", "REINFORCING_ALIGNMENT")

        report = instability_monitor.assess()
        assert report.hypothesis_switches == 2
        signal_types = [s.signal_type for s in report.active_signals]
        assert "hypothesis_switch" in signal_types

    def test_no_switch_detected_with_constant_leader(self, instability_monitor):
        for i in range(4):
            instability_monitor.update(i, 0.70, 0.0, 0.5, "psoriasis", "CERTAINTY_STABILIZATION")
        report = instability_monitor.assess()
        assert report.hypothesis_switches == 0


# ── Contradiction oscillation ─────────────────────────────────────────────────

class TestContradictionOscillation:
    def test_contradiction_oscillation_detected(self, instability_monitor):
        instability_monitor.update(0, 0.65, 0.50, 0.5, "psoriasis", "CONTRADICTION_DETECTED")
        instability_monitor.update(1, 0.70, 0.05, 0.5, "psoriasis", "PARTIAL_ALIGNMENT")
        instability_monitor.update(2, 0.60, 0.45, 0.5, "psoriasis", "CONTRADICTION_DETECTED")

        report = instability_monitor.assess()
        signal_types = [s.signal_type for s in report.active_signals]
        assert "contradiction_oscillation" in signal_types

    def test_stable_contradiction_no_oscillation(self, instability_monitor):
        for i in range(4):
            instability_monitor.update(i, 0.70, 0.10, 0.5, "psoriasis", "PARTIAL_ALIGNMENT")
        report = instability_monitor.assess()
        signal_types = [s.signal_type for s in report.active_signals]
        assert "contradiction_oscillation" not in signal_types


# ── Entropy amplification ─────────────────────────────────────────────────────

class TestEntropyAmplification:
    def test_rising_entropy_triggers_signal(self, instability_monitor):
        instability_monitor.update(0, 0.65, 0.0, 0.20, "psoriasis", "PARTIAL_ALIGNMENT")
        instability_monitor.update(1, 0.60, 0.0, 0.60, "psoriasis", "AMBIGUITY_ESCALATION")
        instability_monitor.update(2, 0.50, 0.0, 1.10, "psoriasis", "AMBIGUITY_ESCALATION")

        report = instability_monitor.assess()
        signal_types = [s.signal_type for s in report.active_signals]
        assert "entropy_amplification" in signal_types

    def test_decreasing_entropy_no_amplification_signal(self, instability_monitor):
        instability_monitor.update(0, 0.50, 0.0, 1.50, "psoriasis", "AMBIGUITY_ESCALATION")
        instability_monitor.update(1, 0.65, 0.0, 0.80, "psoriasis", "CERTAINTY_STABILIZATION")
        instability_monitor.update(2, 0.80, 0.0, 0.40, "psoriasis", "CERTAINTY_STABILIZATION")

        report = instability_monitor.assess()
        signal_types = [s.signal_type for s in report.active_signals]
        assert "entropy_amplification" not in signal_types


# ── Instability index and threshold ──────────────────────────────────────────

class TestInstabilityIndex:
    def test_is_unstable_above_threshold(self):
        monitor = DiagnosticInstabilityMonitor(instability_threshold=0.60)
        # Create all signals to push index high
        for i, (cert, lead) in enumerate([
            (0.20, "psoriasis"), (0.80, "lichen_planus"),
            (0.10, "psoriasis"), (0.90, "lichen_planus"),
        ]):
            monitor.update(i, cert, 0.50 if i % 2 == 0 else 0.05, 0.50, lead, "PARTIAL_ALIGNMENT")
        report = monitor.assess()
        assert report.instability_index >= 0.0
        # is_unstable follows the threshold
        assert report.is_unstable == (report.instability_index >= 0.60)

    def test_instability_index_bounded_to_one(self, instability_monitor):
        for i in range(5):
            lead = "psoriasis" if i % 2 == 0 else "lichen_planus"
            cert = 0.10 if i % 2 == 0 else 0.90
            instability_monitor.update(i, cert, 0.60 - 0.1 * i, 1.0 + i * 0.2, lead, "X")
        report = instability_monitor.assess()
        assert report.instability_index <= 1.0

    def test_escalation_influence_is_half_index(self, instability_monitor):
        instability_monitor.update(0, 0.30, 0.0, 0.5, "psoriasis", "A")
        instability_monitor.update(1, 0.80, 0.0, 0.5, "psoriasis", "A")
        report = instability_monitor.assess()
        assert report.escalation_influence == pytest.approx(report.instability_index * 0.5)


# ── InstabilityReport properties ─────────────────────────────────────────────

class TestInstabilityReportProperties:
    def test_primary_signal_is_max_magnitude(self, instability_monitor):
        instability_monitor.update(0, 0.20, 0.0, 0.5, "psoriasis", "A")
        instability_monitor.update(1, 0.90, 0.5, 0.5, "lichen_planus", "B")
        instability_monitor.update(2, 0.10, 0.0, 0.5, "psoriasis", "C")

        report = instability_monitor.assess()
        if report.active_signals:
            primary = report.primary_signal
            assert primary is not None
            assert primary.magnitude == max(s.magnitude for s in report.active_signals)

    def test_signal_count_matches_active_signals(self, instability_monitor):
        instability_monitor.update(0, 0.30, 0.0, 0.5, "psoriasis", "A")
        instability_monitor.update(1, 0.80, 0.0, 0.5, "psoriasis", "A")
        report = instability_monitor.assess()
        assert report.signal_count == len(report.active_signals)

    def test_primary_signal_none_when_no_signals(self, instability_monitor):
        report = instability_monitor.assess()
        assert report.primary_signal is None


# ── Reset ─────────────────────────────────────────────────────────────────────

class TestMonitorReset:
    def test_reset_clears_all_series(self, instability_monitor):
        instability_monitor.update(0, 0.50, 0.10, 0.80, "psoriasis", "A")
        instability_monitor.reset()
        report = instability_monitor.assess()
        assert report.instability_index == 0.0
        assert report.active_signals == []


# ── Internal helpers ──────────────────────────────────────────────────────────

class TestInternalHelpers:
    def test_std_zero_for_single_value(self):
        assert DiagnosticInstabilityMonitor._std([0.5]) == 0.0

    def test_std_zero_for_empty(self):
        assert DiagnosticInstabilityMonitor._std([]) == 0.0

    def test_trend_positive_for_rising_series(self):
        slope = DiagnosticInstabilityMonitor._trend([0.1, 0.3, 0.5, 0.7])
        assert slope > 0

    def test_trend_negative_for_falling_series(self):
        slope = DiagnosticInstabilityMonitor._trend([0.7, 0.5, 0.3, 0.1])
        assert slope < 0

    def test_trend_zero_for_constant_series(self):
        slope = DiagnosticInstabilityMonitor._trend([0.5, 0.5, 0.5])
        assert slope == pytest.approx(0.0, abs=1e-10)
