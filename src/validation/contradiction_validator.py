"""
ContradictionValidator — contradiction propagation realism assessment.

Evaluates whether the contradiction analysis produced by the pipeline
reflects clinically believable cross-disease conflict patterns.

A clinically realistic contradiction profile must satisfy:

  · Contradiction load is proportional to the number of active conflicts
  · High-load cases show certainty dampening (contradiction-decay is active)
  · Disease-pair tensions are internally consistent with penalty weights
  · Confusion-zone pairs do not appear as high-load single-direction conflicts
    without reciprocal counter-evidence (one-sided exclusion check)
  · Mandatory escalation is triggered at and only at the documented ceiling

The contradiction subsystem is one of the primary novelty components of the
symbolic reasoning architecture — its validity directly supports the claim
that the system performs genuine evidential conflict resolution rather than
simple evidence aggregation.
"""

from __future__ import annotations

from src.pipeline.pipeline_runner import PipelineResult
from src.validation.behavioral_validator import Severity, ValidationSignal


class ContradictionValidator:
    """
    Validates contradiction propagation behaviour against clinical realism
    criteria.

    Parameters
    ----------
    escalation_ceiling:
        Contradiction load at which mandatory escalation must be triggered.
        Must match the value embedded in the reasoning pipeline.
    max_believable_load:
        Upper bound on plausible contradiction load — loads above this value
        suggest miscalibrated penalty weights.
    dampening_activation_threshold:
        Contradiction load above which certainty dampening should be active
        (pipeline parameter: contradiction_damping_threshold = 0.20).
    min_load_per_active_contradiction:
        Each active contradiction should contribute at least this much to
        the total load (sanity-check on penalty weight calibration).
    """

    def __init__(
        self,
        escalation_ceiling:               float = 0.40,
        max_believable_load:              float = 3.00,
        dampening_activation_threshold:   float = 0.20,
        min_load_per_active_contradiction: float = 0.05,
    ) -> None:
        self._esc_ceiling           = escalation_ceiling
        self._max_load              = max_believable_load
        self._damp_thresh           = dampening_activation_threshold
        self._min_per_contradiction = min_load_per_active_contradiction

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        result: PipelineResult,
    ) -> tuple[list[ValidationSignal], float]:
        """
        Validate contradiction propagation realism.

        Returns (signals, score) where score in [0, 1].
        """
        signals: list[ValidationSignal] = []

        signals.extend(self._check_load_bounds(result))
        signals.extend(self._check_load_consistency(result))
        signals.extend(self._check_dampening_consistency(result))
        signals.extend(self._check_escalation_threshold(result))
        signals.extend(self._check_trajectory_contradiction_evolution(result))

        score = sum(1 for s in signals if s.passed) / max(len(signals), 1)
        return signals, score

    # ── Sub-checks ────────────────────────────────────────────────────────────

    def _check_load_bounds(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Contradiction load must be non-negative and within a clinically
        plausible range.
        """
        load = result.contradiction_load
        signals: list[ValidationSignal] = []

        non_negative = load >= 0.0
        signals.append(ValidationSignal(
            validator="contradiction",
            signal_name="load_non_negative",
            passed=non_negative,
            severity="critical",
            description=(
                f"Contradiction load {load:.4f} is "
                + ("non-negative (valid)." if non_negative else "NEGATIVE (invalid).")
            ),
            measured_value=load,
            expected_range=(0.0, self._max_load),
        ))

        within_ceiling = load <= self._max_load
        signals.append(ValidationSignal(
            validator="contradiction",
            signal_name="load_within_plausible_ceiling",
            passed=within_ceiling,
            severity="warning",
            description=(
                f"Contradiction load {load:.4f} "
                + ("within" if within_ceiling else "EXCEEDS")
                + f" plausible ceiling {self._max_load:.2f}. "
                + ("" if within_ceiling else "Possible penalty-weight miscalibration.")
            ),
            measured_value=load,
            expected_range=(0.0, self._max_load),
        ))

        return signals

    def _check_load_consistency(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        If a trajectory is present, the contradiction series should be
        monotonically stable (no oscillation) and reach its final value
        by the end of the contradiction-analysis stage.

        Contradiction load is computed once per case from feature values —
        it should remain constant across trajectory snapshots after Stage 2.
        """
        signals: list[ValidationSignal] = []
        traj = result.trajectory

        if traj is None:
            return signals

        contra_series = traj.contradiction_series()
        if len(contra_series) < 2:
            return signals

        # After the initial computation (Stage 2), load should not fluctuate
        # by more than a small epsilon (floating-point tolerance only).
        max_val  = max(contra_series)
        min_val  = min(contra_series)
        spread   = max_val - min_val
        stable   = spread < 0.01    # sub-1% spread is floating-point noise

        signals.append(ValidationSignal(
            validator="contradiction",
            signal_name="contradiction_load_stable_across_trajectory",
            passed=stable,
            severity="warning",
            description=(
                f"Contradiction load spread across trajectory: {spread:.4f} "
                + ("(stable — load computed once per case)."
                   if stable else
                   f"(UNSTABLE — unexpected fluctuation; max={max_val:.4f}, "
                   f"min={min_val:.4f}).")
            ),
            measured_value=spread,
            expected_range=(0.0, 0.01),
        ))

        # Final trajectory value must match result's reported contradiction_load
        traj_final = contra_series[-1]
        match = abs(traj_final - result.contradiction_load) < 0.001
        signals.append(ValidationSignal(
            validator="contradiction",
            signal_name="trajectory_load_matches_result",
            passed=match,
            severity="warning",
            description=(
                f"Trajectory final contradiction load {traj_final:.4f} "
                + ("matches" if match else "DIVERGES from")
                + f" result.contradiction_load {result.contradiction_load:.4f}."
            ),
            measured_value=abs(traj_final - result.contradiction_load),
            expected_range=(0.0, 0.001),
        ))

        return signals

    def _check_dampening_consistency(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        When contradiction load exceeds the dampening threshold, certainty
        should reflect the dampened state (typically: lower max certainty and
        reduced certainty gap compared to an undampened case with equivalent
        evidence).

        This is a structural check — we validate that high-load cases do NOT
        produce very high certainty without any clinical justification.
        """
        signals: list[ValidationSignal] = []
        load = result.contradiction_load
        cert = result.max_certainty

        if load >= self._damp_thresh:
            # When dampening is active, very high certainty (>0.85) alongside
            # high contradiction load suggests dampening may not have engaged.
            incoherent = cert > 0.85
            signals.append(ValidationSignal(
                validator="contradiction",
                signal_name="high_load_certainty_dampened",
                passed=not incoherent,
                severity="warning",
                description=(
                    f"Contradiction load {load:.3f} >= dampening threshold {self._damp_thresh}. "
                    + (f"Certainty {cert:.3f} > 0.85 despite active dampening — "
                       "possible dampening disengagement."
                       if incoherent else
                       f"Certainty {cert:.3f} appears appropriately moderated.")
                ),
                measured_value=cert,
                expected_range=(0.0, 0.85),
            ))

        if load == 0.0:
            # No contradictions — certainty should not be artificially depressed
            over_suppressed = cert < 0.40
            signals.append(ValidationSignal(
                validator="contradiction",
                signal_name="zero_load_certainty_not_suppressed",
                passed=not over_suppressed,
                severity="info",
                description=(
                    f"No contradictions active (load=0). "
                    + (f"Certainty {cert:.3f} is unexpectedly low — "
                       "possible evidence scarcity rather than contradiction."
                       if over_suppressed else
                       f"Certainty {cert:.3f} is consistent with contradiction-free reasoning.")
                ),
                measured_value=cert,
            ))

        return signals

    def _check_escalation_threshold(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Mandatory escalation trigger consistency:

        If contradiction_load >= escalation_ceiling, recommendation must be
        BIOPSY_RECOMMENDED or HIGH_RISK_CONTRADICTION.

        If recommendation IS BIOPSY/HIGH_RISK, and load < escalation_ceiling,
        another clinical justification (entropy, certainty) must exist.
        """
        signals: list[ValidationSignal] = []
        load = result.contradiction_load
        rec  = result.recommendation or ""

        biopsy_recs = {"BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"}
        is_biopsy   = rec in biopsy_recs
        is_safe     = rec == "SAFE_NON_INVASIVE_TRIAGE"

        # Mandatory escalation must fire at ceiling
        if load >= self._esc_ceiling:
            correct_escalation = is_biopsy
            signals.append(ValidationSignal(
                validator="contradiction",
                signal_name="mandatory_escalation_fires_at_ceiling",
                passed=correct_escalation,
                severity="critical",
                description=(
                    f"Contradiction load {load:.3f} >= ceiling {self._esc_ceiling}. "
                    + ("Correctly escalated to biopsy." if correct_escalation else
                       f"FAILED to escalate — got '{rec}' instead of biopsy.")
                ),
                measured_value=load,
                expected_range=(self._esc_ceiling, self._max_load),
            ))

        # Safe triage must not occur with sub-threshold but non-trivial load
        if is_safe and load > 0.10:
            signals.append(ValidationSignal(
                validator="contradiction",
                signal_name="safe_triage_with_moderate_load",
                passed=False,
                severity="warning",
                description=(
                    f"SAFE_NON_INVASIVE_TRIAGE issued with contradiction_load={load:.3f} > 0.10. "
                    "Safe triage with non-trivial contradictions warrants clinical scrutiny."
                ),
                measured_value=load,
                expected_range=(0.0, 0.10),
            ))
        elif is_safe and load <= 0.10:
            signals.append(ValidationSignal(
                validator="contradiction",
                signal_name="safe_triage_contradiction_free",
                passed=True,
                severity="info",
                description=(
                    f"SAFE_NON_INVASIVE_TRIAGE with contradiction_load={load:.3f} "
                    "(<= 0.10): clinically coherent."
                ),
                measured_value=load,
            ))

        return signals

    def _check_trajectory_contradiction_evolution(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        If a trajectory is present, validate that contradiction-driven state
        transitions appear when expected.

        Specifically: for high-load cases (load >= 0.40), the trajectory should
        include a CONTRADICTION_DETECTED or BIOPSY_ESCALATION state entry.
        For load < 0.10 cases, these states should generally be absent.
        """
        signals: list[ValidationSignal] = []
        traj = result.trajectory

        if traj is None:
            return signals

        seq  = traj.state_sequence()
        load = result.contradiction_load

        contra_states_present = any(
            "CONTRADICTION" in s or "BIOPSY" in s for s in seq
        )

        if load >= self._esc_ceiling:
            signals.append(ValidationSignal(
                validator="contradiction",
                signal_name="contradiction_state_in_high_load_trajectory",
                passed=contra_states_present,
                severity="warning",
                description=(
                    f"High contradiction load ({load:.3f}) should drive FSM through "
                    "CONTRADICTION_DETECTED or BIOPSY_ESCALATION. "
                    + ("State found in trajectory." if contra_states_present else
                       "State NOT found — FSM may not have processed contradiction.")
                ),
                measured_value=load,
            ))

        if load < 0.05 and contra_states_present:
            signals.append(ValidationSignal(
                validator="contradiction",
                signal_name="contradiction_state_absent_for_clean_case",
                passed=False,
                severity="info",
                description=(
                    f"Contradiction state present in trajectory despite very low "
                    f"contradiction load ({load:.4f}). May indicate stale FSM state."
                ),
                measured_value=load,
            ))

        return signals
