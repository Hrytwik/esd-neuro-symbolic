"""
StabilityValidator — deterministic output and replay consistency.

Validates that the symbolic reasoning pipeline produces deterministic,
reproducible outputs when run multiple times on identical inputs.

Stability is a fundamental property of a symbolic clinical system:
  · Same features must always produce the same recommendation
  · Certainty values must be numerically identical across runs
  · State sequences must be deterministically reproduced
  · Contradiction loads must not fluctuate between executions

The StabilityValidator supports two modes:

  1. Single-result audit — checks internal stability signals embedded
     in the trajectory (instability monitor signals, FSM oscillation).
  2. Multi-result comparison — given two PipelineResults for the same
     case, validates that outputs are byte-for-byte equivalent.

Cross-run comparison is the gold standard for stability validation.
Single-result audit provides a weaker but always-available check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.pipeline_runner import PipelineResult
from src.validation.behavioral_validator import Severity, ValidationSignal


# ── Stability report ──────────────────────────────────────────────────────────

@dataclass
class StabilityReport:
    """
    Per-pair stability assessment result.

    Produced when comparing two runs of the same case.
    """

    case_id:          str
    run_a_id:         str
    run_b_id:         str
    is_stable:        bool
    divergences:      list[str] = field(default_factory=list)
    signals:          list[ValidationSignal] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.signals:
            return 1.0
        return sum(1 for s in self.signals if s.passed) / len(self.signals)


# ── Validator ─────────────────────────────────────────────────────────────────

class StabilityValidator:
    """
    Validates deterministic output stability and replay consistency.

    Parameters
    ----------
    certainty_tolerance:
        Maximum acceptable floating-point difference between corresponding
        certainty values across two runs of the same case.
    load_tolerance:
        Maximum acceptable difference between contradiction loads.
    entropy_tolerance:
        Maximum acceptable difference between ambiguity indices.
    require_identical_recommendation:
        When True, both runs must produce identical recommendation strings.
    require_identical_state:
        When True, both runs must produce identical final FSM states.
    """

    def __init__(
        self,
        certainty_tolerance:              float = 1e-6,
        load_tolerance:                   float = 1e-6,
        entropy_tolerance:                float = 1e-6,
        require_identical_recommendation: bool  = True,
        require_identical_state:          bool  = True,
    ) -> None:
        self._cert_tol   = certainty_tolerance
        self._load_tol   = load_tolerance
        self._entr_tol   = entropy_tolerance
        self._req_rec    = require_identical_recommendation
        self._req_state  = require_identical_state

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        result: PipelineResult,
    ) -> tuple[list[ValidationSignal], float]:
        """
        Single-result audit mode.

        Validates internal stability indicators embedded in the trajectory
        without requiring a second run. This is the weaker but always-
        available stability check.

        Returns (signals, score) where score in [0, 1].
        """
        signals: list[ValidationSignal] = []

        signals.extend(self._check_internal_stability_signals(result))
        signals.extend(self._check_trajectory_oscillation(result))
        signals.extend(self._check_stage_completion_integrity(result))

        score = sum(1 for s in signals if s.passed) / max(len(signals), 1)
        return signals, score

    def compare(
        self,
        result_a: PipelineResult,
        result_b: PipelineResult,
    ) -> StabilityReport:
        """
        Cross-run comparison mode.

        Compares two PipelineResults for the same case_id and produces a
        StabilityReport documenting any divergences.

        Both results must share the same case_id.
        """
        if result_a.case_id != result_b.case_id:
            raise ValueError(
                f"Cannot compare results from different cases: "
                f"'{result_a.case_id}' vs '{result_b.case_id}'."
            )

        signals: list[ValidationSignal] = []
        divergences: list[str] = []

        signals.extend(self._compare_recommendations(result_a, result_b, divergences))
        signals.extend(self._compare_final_state(result_a, result_b, divergences))
        signals.extend(self._compare_certainty_metrics(result_a, result_b, divergences))
        signals.extend(self._compare_contradiction_load(result_a, result_b, divergences))
        signals.extend(self._compare_trajectory_series(result_a, result_b, divergences))

        is_stable = len(divergences) == 0
        return StabilityReport(
            case_id=result_a.case_id,
            run_a_id=result_a.run_id,
            run_b_id=result_b.run_id,
            is_stable=is_stable,
            divergences=divergences,
            signals=signals,
        )

    # ── Single-result internal checks ─────────────────────────────────────────

    def _check_internal_stability_signals(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Examine trajectory for internal instability markers:
        UNSTABLE_REASONING state and repeated direction reversals in certainty.
        """
        signals: list[ValidationSignal] = []

        # UNSTABLE_REASONING state in trajectory indicates the instability
        # monitor detected oscillation during inference
        traj = result.trajectory
        if traj is not None:
            seq = traj.state_sequence()
            unstable_detected = any("UNSTABLE" in s for s in seq)
            signals.append(ValidationSignal(
                validator="stability",
                signal_name="no_unstable_reasoning_state_in_trajectory",
                passed=not unstable_detected,
                severity="warning",
                description=(
                    "No UNSTABLE_REASONING state detected in FSM trajectory — "
                    "reasoning was internally stable."
                    if not unstable_detected else
                    "UNSTABLE_REASONING state detected in FSM trajectory — "
                    "the instability monitor flagged oscillatory evidence patterns."
                ),
            ))

        # Final state must not be UNSTABLE_REASONING (only an intermediate state)
        final_state = result.final_state or ""
        final_stable = "UNSTABLE" not in final_state
        signals.append(ValidationSignal(
            validator="stability",
            signal_name="final_state_not_unstable_reasoning",
            passed=final_stable,
            severity="warning",
            description=(
                f"Final state '{final_state}' is "
                + ("stable (not UNSTABLE_REASONING)."
                   if final_stable else
                   "UNSTABLE_REASONING — reasoning did not converge to a stable terminal state.")
            ),
        ))

        return signals

    def _check_trajectory_oscillation(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Examine the certainty series for oscillation patterns that indicate
        unstable evidence processing — more than two direction reversals
        in a trajectory constitutes a stability concern.
        """
        signals: list[ValidationSignal] = []
        traj = result.trajectory

        if traj is None:
            return signals

        series = traj.certainty_series()
        if len(series) < 3:
            return signals

        directions = [
            1 if series[i] > series[i - 1] else (-1 if series[i] < series[i - 1] else 0)
            for i in range(1, len(series))
        ]
        reversals = sum(
            1 for i in range(1, len(directions))
            if directions[i] != 0 and directions[i - 1] != 0
            and directions[i] != directions[i - 1]
        )

        low_oscillation = reversals <= 2
        signals.append(ValidationSignal(
            validator="stability",
            signal_name="certainty_trajectory_low_oscillation",
            passed=low_oscillation,
            severity="warning",
            description=(
                f"Certainty trajectory has {reversals} direction reversal(s). "
                + ("Stable within tolerance (<= 2 reversals)."
                   if low_oscillation else
                   "Exceeds stability tolerance — possible oscillatory evidence processing.")
            ),
            measured_value=float(reversals),
            expected_range=(0.0, 2.0),
        ))

        return signals

    def _check_stage_completion_integrity(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        A stable pipeline run must complete all expected stages.
        Incomplete stage sequences indicate premature termination.
        """
        signals: list[ValidationSignal] = []
        completed = set(result.completed_stages)

        # For a successful run, all 8 core stages (0–7) should appear
        expected_core = {
            "clinical_grading",
            "evidence_activation",
            "contradiction_analysis",
            "certainty_propagation",
        }
        missing_core = expected_core - completed
        all_core_present = len(missing_core) == 0

        signals.append(ValidationSignal(
            validator="stability",
            signal_name="core_stages_all_completed",
            passed=all_core_present,
            severity="warning",
            description=(
                "All core reasoning stages completed — execution was not truncated."
                if all_core_present else
                f"Missing core stages: {sorted(missing_core)}. "
                "Execution was truncated — stability cannot be guaranteed."
            ),
        ))

        # No stage errors
        no_errors = len(result.stage_errors) == 0
        signals.append(ValidationSignal(
            validator="stability",
            signal_name="no_stage_errors",
            passed=no_errors,
            severity="warning",
            description=(
                "No stage errors recorded — execution was error-free."
                if no_errors else
                f"{len(result.stage_errors)} stage error(s): {result.stage_errors[:2]}."
            ),
        ))

        return signals

    # ── Cross-run comparison checks ───────────────────────────────────────────

    def _compare_recommendations(
        self,
        a: PipelineResult,
        b: PipelineResult,
        divergences: list[str],
    ) -> list[ValidationSignal]:
        """Recommendations must match across runs."""
        if not self._req_rec:
            return []
        match = a.recommendation == b.recommendation
        if not match:
            divergences.append(
                f"recommendation: '{a.recommendation}' vs '{b.recommendation}'"
            )
        return [ValidationSignal(
            validator="stability",
            signal_name="recommendation_deterministic",
            passed=match,
            severity="critical" if not match else "info",
            description=(
                f"Recommendation: '{a.recommendation}' "
                + ("matches across both runs." if match else
                   f"DIVERGES from run B: '{b.recommendation}'.")
            ),
        )]

    def _compare_final_state(
        self,
        a: PipelineResult,
        b: PipelineResult,
        divergences: list[str],
    ) -> list[ValidationSignal]:
        """Final FSM states must match across runs."""
        if not self._req_state:
            return []
        match = a.final_state == b.final_state
        if not match:
            divergences.append(
                f"final_state: '{a.final_state}' vs '{b.final_state}'"
            )
        return [ValidationSignal(
            validator="stability",
            signal_name="final_state_deterministic",
            passed=match,
            severity="critical" if not match else "info",
            description=(
                f"Final state: '{a.final_state}' "
                + ("matches across both runs." if match else
                   f"DIVERGES from run B: '{b.final_state}'.")
            ),
        )]

    def _compare_certainty_metrics(
        self,
        a: PipelineResult,
        b: PipelineResult,
        divergences: list[str],
    ) -> list[ValidationSignal]:
        """max_certainty and certainty_gap must be numerically identical."""
        signals: list[ValidationSignal] = []

        cert_diff = abs(a.max_certainty - b.max_certainty)
        cert_match = cert_diff <= self._cert_tol
        if not cert_match:
            divergences.append(f"max_certainty: {a.max_certainty:.6f} vs {b.max_certainty:.6f}")
        signals.append(ValidationSignal(
            validator="stability",
            signal_name="max_certainty_deterministic",
            passed=cert_match,
            severity="critical" if not cert_match else "info",
            description=(
                f"max_certainty difference: {cert_diff:.2e} "
                + ("(within tolerance)." if cert_match else
                   f"(EXCEEDS tolerance {self._cert_tol:.2e}) — non-deterministic output.")
            ),
            measured_value=cert_diff,
            expected_range=(0.0, self._cert_tol),
        ))

        gap_diff = abs(a.certainty_gap - b.certainty_gap)
        gap_match = gap_diff <= self._cert_tol
        if not gap_match:
            divergences.append(f"certainty_gap: {a.certainty_gap:.6f} vs {b.certainty_gap:.6f}")
        signals.append(ValidationSignal(
            validator="stability",
            signal_name="certainty_gap_deterministic",
            passed=gap_match,
            severity="critical" if not gap_match else "info",
            description=(
                f"certainty_gap difference: {gap_diff:.2e} "
                + ("(within tolerance)." if gap_match else
                   f"(EXCEEDS tolerance {self._cert_tol:.2e}) — non-deterministic output.")
            ),
            measured_value=gap_diff,
            expected_range=(0.0, self._cert_tol),
        ))

        entropy_diff = abs(a.ambiguity_index - b.ambiguity_index)
        entropy_match = entropy_diff <= self._entr_tol
        if not entropy_match:
            divergences.append(f"ambiguity_index: {a.ambiguity_index:.6f} vs {b.ambiguity_index:.6f}")
        signals.append(ValidationSignal(
            validator="stability",
            signal_name="ambiguity_index_deterministic",
            passed=entropy_match,
            severity="critical" if not entropy_match else "info",
            description=(
                f"ambiguity_index difference: {entropy_diff:.2e} "
                + ("(within tolerance)." if entropy_match else
                   f"(EXCEEDS tolerance {self._entr_tol:.2e}) — non-deterministic output.")
            ),
            measured_value=entropy_diff,
            expected_range=(0.0, self._entr_tol),
        ))

        return signals

    def _compare_contradiction_load(
        self,
        a: PipelineResult,
        b: PipelineResult,
        divergences: list[str],
    ) -> list[ValidationSignal]:
        """Contradiction load must be identical across runs."""
        diff  = abs(a.contradiction_load - b.contradiction_load)
        match = diff <= self._load_tol
        if not match:
            divergences.append(
                f"contradiction_load: {a.contradiction_load:.6f} vs {b.contradiction_load:.6f}"
            )
        return [ValidationSignal(
            validator="stability",
            signal_name="contradiction_load_deterministic",
            passed=match,
            severity="critical" if not match else "info",
            description=(
                f"contradiction_load difference: {diff:.2e} "
                + ("(within tolerance)." if match else
                   f"(EXCEEDS tolerance {self._load_tol:.2e}) — non-deterministic output.")
            ),
            measured_value=diff,
            expected_range=(0.0, self._load_tol),
        )]

    def _compare_trajectory_series(
        self,
        a: PipelineResult,
        b: PipelineResult,
        divergences: list[str],
    ) -> list[ValidationSignal]:
        """Certainty series must be identical in length and values."""
        signals: list[ValidationSignal] = []

        if a.trajectory is None or b.trajectory is None:
            return signals

        series_a = a.trajectory.certainty_series()
        series_b = b.trajectory.certainty_series()

        length_match = len(series_a) == len(series_b)
        if not length_match:
            divergences.append(
                f"trajectory length: {len(series_a)} vs {len(series_b)}"
            )
        signals.append(ValidationSignal(
            validator="stability",
            signal_name="trajectory_length_deterministic",
            passed=length_match,
            severity="warning",
            description=(
                f"Trajectory lengths: {len(series_a)} vs {len(series_b)}. "
                + ("Match." if length_match else "DIVERGE — different stage counts across runs.")
            ),
        ))

        if length_match and series_a:
            max_point_diff = max(abs(a - b) for a, b in zip(series_a, series_b))
            values_match   = max_point_diff <= self._cert_tol
            if not values_match:
                divergences.append(
                    f"trajectory series max point difference: {max_point_diff:.2e}"
                )
            signals.append(ValidationSignal(
                validator="stability",
                signal_name="trajectory_series_deterministic",
                passed=values_match,
                severity="warning",
                description=(
                    f"Max trajectory point difference: {max_point_diff:.2e} "
                    + ("(within tolerance)." if values_match else
                       f"(EXCEEDS tolerance {self._cert_tol:.2e}) — trajectory is non-deterministic.")
                ),
                measured_value=max_point_diff,
                expected_range=(0.0, self._cert_tol),
            ))

        return signals
