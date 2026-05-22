"""
DiagnosticInstabilityMonitor — certainty volatility and oscillation tracking.

Monitors the temporal evolution of the certainty distribution across
reasoning stages to detect unstable or oscillatory reasoning trajectories.
An unstable trajectory indicates that the evidence is internally contradictory
or that the feature profile is positioned near a confusion boundary — both
conditions that warrant escalated triage caution.

Instability signals
-------------------
1. Certainty oscillation    — leading certainty rises then falls between stages
2. State reversal           — FSM state moves backward (contradiction resolves
                              then re-emerges)
3. Contradiction oscillation — contradiction load fluctuates > threshold
4. Hypothesis switching     — leading disease changes mid-trajectory
5. Entropy amplification    — ambiguity index increases rather than decreasing

Instability index
-----------------
  instability_index = weighted sum of active instability signals ∈ [0.0, 1.0]

  Threshold for UNSTABLE_REASONING state entry: 0.60 (configurable)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Instability record ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InstabilitySignal:
    """A single detected instability signal at a reasoning stage."""

    signal_type:   str      # "certainty_oscillation" | "hypothesis_switch" | etc.
    stage:         int
    magnitude:     float    # 0.0 – 1.0
    description:   str


@dataclass
class InstabilityReport:
    """
    Aggregate instability assessment across the reasoning trajectory.
    Produced by DiagnosticInstabilityMonitor.assess().
    """

    instability_index:    float                  # composite [0.0, 1.0]
    active_signals:       list[InstabilitySignal]
    certainty_volatility: float                  # std of certainty series
    contradiction_oscillation: float             # std of contradiction series
    hypothesis_switches:  int                    # count of leading disease changes
    entropy_trend:        float                  # positive = increasing entropy
    is_unstable:          bool                   # index >= threshold
    escalation_influence: float                  # instability → escalation pressure

    @property
    def primary_signal(self) -> InstabilitySignal | None:
        return (
            max(self.active_signals, key=lambda s: s.magnitude)
            if self.active_signals else None
        )

    @property
    def signal_count(self) -> int:
        return len(self.active_signals)


# ── Instability monitor ───────────────────────────────────────────────────────

class DiagnosticInstabilityMonitor:
    """
    Tracks and quantifies reasoning instability across pipeline stages.

    The monitor is updated incrementally as each stage completes.
    After all stages, it produces an InstabilityReport capturing the
    full volatility profile of the trajectory.

    Parameters
    ----------
    instability_threshold:
        Index above which UNSTABLE_REASONING is flagged. Default: 0.60.
    oscillation_window:
        Number of consecutive stages over which oscillation is checked. Default: 3.
    """

    _SIGNAL_WEIGHTS: dict[str, float] = {
        "certainty_oscillation":    0.30,
        "hypothesis_switch":        0.25,
        "contradiction_oscillation": 0.20,
        "entropy_amplification":    0.15,
        "state_reversal":           0.10,
    }

    def __init__(
        self,
        instability_threshold: float = 0.60,
        oscillation_window: int = 3,
    ) -> None:
        self._threshold       = instability_threshold
        self._window          = oscillation_window
        self._certainty_series:    list[float] = []
        self._contradiction_series: list[float] = []
        self._entropy_series:      list[float] = []
        self._leading_series:      list[str]   = []
        self._state_series:        list[str]   = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        stage: int,
        max_certainty: float,
        contradiction_load: float,
        ambiguity_index: float,
        leading_disease: str,
        state: str,
    ) -> None:
        """Register measurements from a completed reasoning stage."""
        self._certainty_series.append(max_certainty)
        self._contradiction_series.append(contradiction_load)
        self._entropy_series.append(ambiguity_index)
        self._leading_series.append(leading_disease)
        self._state_series.append(state)

    def assess(self) -> InstabilityReport:
        """
        Compute the full instability assessment from all recorded stages.
        Can be called at any point; returns the current assessment.
        """
        signals: list[InstabilitySignal] = []

        certainty_vol = self._std(self._certainty_series)
        contra_osc    = self._std(self._contradiction_series)
        entropy_trend = self._trend(self._entropy_series)
        switches      = self._hypothesis_switches()

        # ── Signal: certainty oscillation ─────────────────────────────────────
        if certainty_vol > 0.05:
            magnitude = min(certainty_vol * 3.0, 1.0)
            signals.append(InstabilitySignal(
                signal_type="certainty_oscillation",
                stage=len(self._certainty_series) - 1,
                magnitude=magnitude,
                description=(
                    f"Certainty std={certainty_vol:.3f} indicates oscillating "
                    f"leading hypothesis strength."
                ),
            ))

        # ── Signal: contradiction oscillation ─────────────────────────────────
        if contra_osc > 0.05:
            magnitude = min(contra_osc * 4.0, 1.0)
            signals.append(InstabilitySignal(
                signal_type="contradiction_oscillation",
                stage=len(self._contradiction_series) - 1,
                magnitude=magnitude,
                description=(
                    f"Contradiction load std={contra_osc:.3f} indicates "
                    f"oscillating cross-disease conflict."
                ),
            ))

        # ── Signal: hypothesis switching ──────────────────────────────────────
        if switches > 0:
            magnitude = min(switches * 0.35, 1.0)
            signals.append(InstabilitySignal(
                signal_type="hypothesis_switch",
                stage=len(self._leading_series) - 1,
                magnitude=magnitude,
                description=(
                    f"Leading disease changed {switches} time(s) during "
                    f"reasoning — indicative of hypothesis instability."
                ),
            ))

        # ── Signal: entropy amplification ─────────────────────────────────────
        if entropy_trend > 0.05:
            magnitude = min(entropy_trend * 2.0, 1.0)
            signals.append(InstabilitySignal(
                signal_type="entropy_amplification",
                stage=len(self._entropy_series) - 1,
                magnitude=magnitude,
                description=(
                    f"Ambiguity index trending upward (slope={entropy_trend:.3f}). "
                    f"Diagnostic uncertainty increasing rather than resolving."
                ),
            ))

        # ── Compute composite instability index ───────────────────────────────
        index = 0.0
        for signal in signals:
            w = self._SIGNAL_WEIGHTS.get(signal.signal_type, 0.10)
            index += w * signal.magnitude
        index = min(index, 1.0)

        return InstabilityReport(
            instability_index=index,
            active_signals=signals,
            certainty_volatility=certainty_vol,
            contradiction_oscillation=contra_osc,
            hypothesis_switches=switches,
            entropy_trend=entropy_trend,
            is_unstable=(index >= self._threshold),
            escalation_influence=index * 0.5,  # fractional escalation pressure
        )

    def reset(self) -> None:
        """Reset all recorded series for a new case."""
        self._certainty_series     = []
        self._contradiction_series = []
        self._entropy_series       = []
        self._leading_series       = []
        self._state_series         = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _hypothesis_switches(self) -> int:
        if len(self._leading_series) < 2:
            return 0
        return sum(
            1 for i in range(1, len(self._leading_series))
            if self._leading_series[i] != self._leading_series[i - 1]
        )

    @staticmethod
    def _std(series: list[float]) -> float:
        if len(series) < 2:
            return 0.0
        mean = sum(series) / len(series)
        variance = sum((x - mean) ** 2 for x in series) / len(series)
        return math.sqrt(variance)

    @staticmethod
    def _trend(series: list[float]) -> float:
        """
        Linear trend coefficient of a series.
        Positive value = increasing; negative = decreasing.
        """
        n = len(series)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(series) / n
        num = sum((i - x_mean) * (series[i] - y_mean) for i in range(n))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        return num / denom if denom > 1e-12 else 0.0
