"""
TrajectoryValidator — certainty evolution smoothness and trajectory stability.

Evaluates whether the reasoning trajectory produced by the pipeline evolves
in a clinically believable manner.  Good trajectories show:

  · Progressive evidence accumulation (certainty generally increases)
  · Smooth convergence toward a leading hypothesis
  · Contradiction-driven decay when conflict is detected
  · No abrupt certainty jumps exceeding a calibrated threshold

Abrupt jumps, sustained oscillation, or premature convergence are
flagged as trajectory anomalies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.pipeline.pipeline_runner import PipelineResult
from src.validation.behavioral_validator import Severity, ValidationSignal


# ── Trajectory validation result ──────────────────────────────────────────────

@dataclass
class TrajectoryValidationReport:
    """Per-case trajectory quality assessment."""

    case_id:             str
    passed:              bool
    stage_count:         int
    final_certainty:     float
    certainty_range:     tuple[float, float]    # (min, max) across series
    max_step_delta:      float                  # largest single-step change
    oscillation_count:   int                    # direction reversals in series
    convergence_label:   str                    # "smooth" | "oscillating" | "abrupt" | "flat"
    signals:             list[ValidationSignal] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Quality score [0, 1] — higher is better."""
        passed_count = sum(1 for s in self.signals if s.passed)
        total = max(len(self.signals), 1)
        return passed_count / total


# ── Validator ─────────────────────────────────────────────────────────────────

class TrajectoryValidator:
    """
    Validates the certainty evolution trajectory recorded in a PipelineResult.

    Parameters
    ----------
    max_allowed_step_delta:
        Largest acceptable single-step certainty change.  Steps larger than
        this are flagged as abrupt jumps.
    max_allowed_oscillations:
        Maximum direction reversals tolerated in the certainty series before
        it is classified as oscillating.
    min_final_certainty:
        Minimum acceptable final certainty value for any case that produces
        a leading hypothesis (excludes contradiction-dominated cases).
    """

    def __init__(
        self,
        max_allowed_step_delta:  float = 0.50,
        max_allowed_oscillations: int  = 2,
        min_final_certainty:     float = 0.10,
    ) -> None:
        self._max_delta      = max_allowed_step_delta
        self._max_osc        = max_allowed_oscillations
        self._min_final_cert = min_final_certainty

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        result: PipelineResult,
    ) -> tuple[list[ValidationSignal], float]:
        """
        Validate the trajectory embedded in result.

        Returns
        -------
        (signals, score):
            List of ValidationSignal objects and an aggregate quality score
            in [0, 1].
        """
        signals: list[ValidationSignal] = []

        traj = result.trajectory
        if traj is None:
            signals.append(ValidationSignal(
                validator="trajectory",
                signal_name="trajectory_present",
                passed=False,
                severity="critical",
                description="No trajectory recorded — cannot validate reasoning evolution.",
            ))
            return signals, 0.0

        series = traj.certainty_series()
        signals.append(ValidationSignal(
            validator="trajectory",
            signal_name="trajectory_present",
            passed=True,
            severity="info",
            description=f"Trajectory present with {len(series)} certainty snapshots.",
        ))

        if len(series) == 0:
            signals.append(ValidationSignal(
                validator="trajectory",
                signal_name="certainty_series_non_empty",
                passed=False,
                severity="critical",
                description="Certainty series is empty; no reasoning stages recorded.",
            ))
            return signals, 0.0

        signals.extend(self._validate_bounds(series, result.case_id))
        signals.extend(self._validate_step_deltas(series, result.case_id))
        signals.extend(self._validate_oscillation(series, result.case_id))
        signals.extend(self._validate_final_certainty(result))
        signals.extend(self._validate_state_sequence(traj, result))

        score = sum(1 for s in signals if s.passed) / max(len(signals), 1)
        return signals, score

    # ── Sub-checks ────────────────────────────────────────────────────────────

    def _validate_bounds(
        self,
        series: list[float],
        case_id: str,
    ) -> list[ValidationSignal]:
        """All certainty values must lie strictly in [0, 1]."""
        out_of_range = [v for v in series if not (0.0 <= v <= 1.0)]
        passed = len(out_of_range) == 0
        return [ValidationSignal(
            validator="trajectory",
            signal_name="certainty_bounds",
            passed=passed,
            severity="critical",
            description=(
                "All certainty values in [0, 1]." if passed
                else f"{len(out_of_range)} value(s) outside [0, 1]: {out_of_range[:3]}"
            ),
        )]

    def _validate_step_deltas(
        self,
        series: list[float],
        case_id: str,
    ) -> list[ValidationSignal]:
        """No single step should jump by more than max_allowed_step_delta."""
        if len(series) < 2:
            return []
        deltas = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
        max_delta = max(deltas)
        passed = max_delta <= self._max_delta
        return [ValidationSignal(
            validator="trajectory",
            signal_name="step_delta_bound",
            passed=passed,
            severity="warning",
            description=(
                f"Max step delta {max_delta:.4f} "
                + ("within" if passed else "exceeds")
                + f" threshold {self._max_delta:.4f}."
            ),
            measured_value=max_delta,
            expected_range=(0.0, self._max_delta),
        )]

    def _validate_oscillation(
        self,
        series: list[float],
        case_id: str,
    ) -> list[ValidationSignal]:
        """Count direction reversals in certainty series."""
        if len(series) < 3:
            return []
        directions = [
            1 if series[i] > series[i - 1] else (-1 if series[i] < series[i - 1] else 0)
            for i in range(1, len(series))
        ]
        reversals = sum(
            1 for i in range(1, len(directions))
            if directions[i] != 0 and directions[i - 1] != 0 and directions[i] != directions[i - 1]
        )
        passed = reversals <= self._max_osc
        return [ValidationSignal(
            validator="trajectory",
            signal_name="oscillation_count",
            passed=passed,
            severity="warning",
            description=(
                f"Certainty series has {reversals} direction reversal(s) "
                + ("(within tolerance)." if passed else f"(exceeds tolerance of {self._max_osc}).")
            ),
            measured_value=float(reversals),
            expected_range=(0.0, float(self._max_osc)),
        )]

    def _validate_final_certainty(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """Final certainty must be above a reasonable floor for any resolved case."""
        cert = result.max_certainty
        passed = cert >= self._min_final_cert
        sev: Severity = "warning" if not passed else "info"
        return [ValidationSignal(
            validator="trajectory",
            signal_name="final_certainty_floor",
            passed=passed,
            severity=sev,
            description=(
                f"Final certainty {cert:.4f} "
                + ("meets" if passed else f"is below")
                + f" minimum floor {self._min_final_cert:.4f}."
            ),
            measured_value=cert,
            expected_range=(self._min_final_cert, 1.0),
        )]

    def _validate_state_sequence(
        self,
        traj,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """State sequence must be non-empty and end at a recognised terminal state."""
        from src.reasoning.state_tracker import DiagnosticState
        valid_states = {s.value for s in DiagnosticState}
        seq = traj.state_sequence()
        signals: list[ValidationSignal] = []

        if not seq:
            signals.append(ValidationSignal(
                validator="trajectory",
                signal_name="state_sequence_present",
                passed=False,
                severity="critical",
                description="State sequence is empty; FSM produced no transitions.",
            ))
            return signals

        signals.append(ValidationSignal(
            validator="trajectory",
            signal_name="state_sequence_present",
            passed=True,
            severity="info",
            description=f"State sequence has {len(seq)} entry(ies): {seq[:3]}...",
        ))

        final_valid = result.final_state in valid_states
        signals.append(ValidationSignal(
            validator="trajectory",
            signal_name="terminal_state_valid",
            passed=final_valid,
            severity="critical" if not final_valid else "info",
            description=(
                f"Terminal state '{result.final_state}' "
                + ("is a" if final_valid else "is NOT a")
                + " recognised DiagnosticState."
            ),
        ))

        # Contradiction-detected state should appear for high-contradiction cases
        if result.contradiction_load >= 0.40:
            contradiction_in_seq = any(
                "CONTRADICTION" in s or "BIOPSY" in s for s in seq
            )
            signals.append(ValidationSignal(
                validator="trajectory",
                signal_name="contradiction_state_reached",
                passed=contradiction_in_seq,
                severity="warning",
                description=(
                    "High contradiction load should produce CONTRADICTION_DETECTED "
                    "or BIOPSY_ESCALATION in the FSM sequence. "
                    + ("Found." if contradiction_in_seq else "Not found.")
                ),
                measured_value=result.contradiction_load,
            ))

        return signals
