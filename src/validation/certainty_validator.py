"""
CertaintyValidator — certainty stabilisation and entropy behavior assessment.

Evaluates whether the certainty distribution produced by the pipeline is
internally consistent, clinically calibrated, and mathematically sound.

A well-calibrated certainty profile must satisfy:

  · max_certainty in (0, 1] — never exactly 0 (softmax always distributes)
  · certainty_gap >= 0 and < max_certainty
  · ambiguity_index >= 0 bits (Shannon entropy is always non-negative)
  · Entropy and certainty are inversely consistent — high certainty cases
    should have low entropy, and vice versa
  · Certainty gap is meaningful relative to the leading certainty
  · SAFE triage certainty meets the floor required for non-invasive diagnosis
  · Certainty evolution (if trajectory present) trends toward stabilisation
    rather than oscillating or collapsing

Shannon entropy (ambiguity_index) is computed over the full 6-disease
softmax distribution. Uniform distribution yields log2(6) ≈ 2.585 bits.
"""

from __future__ import annotations

import math

from src.pipeline.pipeline_runner import PipelineResult
from src.validation.behavioral_validator import Severity, ValidationSignal


# Maximum possible entropy for 6 equally-likely diseases
_MAX_ENTROPY_6_CLASS: float = math.log2(6)   # ≈ 2.585 bits


class CertaintyValidator:
    """
    Validates certainty metric integrity and entropy behaviour.

    Parameters
    ----------
    min_certainty_floor:
        No resolved case should have max_certainty below this (softmax
        always produces non-zero values; floor reflects at least some
        evidence concentration).
    max_entropy_ceiling:
        Shannon entropy cannot exceed log2(n_classes). A small tolerance
        above the theoretical maximum flags numerical issues.
    high_certainty_threshold:
        Cases with max_certainty above this are considered high-certainty;
        their entropy should be correspondingly low.
    high_certainty_max_entropy:
        Maximum acceptable entropy for a high-certainty case.
    min_gap_fraction:
        Minimum fraction of max_certainty that the certainty gap should
        represent (gap / max_certainty >= this).
    """

    def __init__(
        self,
        min_certainty_floor:      float = 0.05,
        max_entropy_ceiling:      float = _MAX_ENTROPY_6_CLASS + 0.05,
        high_certainty_threshold: float = 0.65,
        high_certainty_max_entropy: float = 1.20,
        min_gap_fraction:         float = 0.05,
    ) -> None:
        self._cert_floor         = min_certainty_floor
        self._entropy_ceil       = max_entropy_ceiling
        self._high_cert_thresh   = high_certainty_threshold
        self._high_cert_entropy  = high_certainty_max_entropy
        self._min_gap_fraction   = min_gap_fraction

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        result: PipelineResult,
    ) -> tuple[list[ValidationSignal], float]:
        """
        Validate certainty metric integrity and entropy coherence.

        Returns (signals, score) where score in [0, 1].
        """
        signals: list[ValidationSignal] = []

        signals.extend(self._check_certainty_bounds(result))
        signals.extend(self._check_gap_validity(result))
        signals.extend(self._check_entropy_bounds(result))
        signals.extend(self._check_entropy_certainty_coherence(result))
        signals.extend(self._check_trajectory_certainty_evolution(result))

        score = sum(1 for s in signals if s.passed) / max(len(signals), 1)
        return signals, score

    # ── Sub-checks ────────────────────────────────────────────────────────────

    def _check_certainty_bounds(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """max_certainty must lie strictly in (0, 1]."""
        cert = result.max_certainty
        signals: list[ValidationSignal] = []

        above_floor = cert >= self._cert_floor
        signals.append(ValidationSignal(
            validator="certainty",
            signal_name="certainty_above_floor",
            passed=above_floor,
            severity="warning",
            description=(
                f"max_certainty={cert:.4f} "
                + ("meets" if above_floor else "is below")
                + f" minimum floor {self._cert_floor:.4f}. "
                + ("" if above_floor else
                   "Softmax over 6 classes always yields >0; "
                   "very low certainty indicates no supporting evidence.")
            ),
            measured_value=cert,
            expected_range=(self._cert_floor, 1.0),
        ))

        at_most_one = cert <= 1.0
        signals.append(ValidationSignal(
            validator="certainty",
            signal_name="certainty_at_most_one",
            passed=at_most_one,
            severity="critical",
            description=(
                f"max_certainty={cert:.4f} "
                + ("is valid (<= 1.0)." if at_most_one else "EXCEEDS 1.0 — invalid probability.")
            ),
            measured_value=cert,
            expected_range=(0.0, 1.0),
        ))

        return signals

    def _check_gap_validity(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        certainty_gap must be non-negative and less than max_certainty.
        As a fraction of max_certainty it must meet the minimum fraction floor.
        """
        gap  = result.certainty_gap
        cert = result.max_certainty
        signals: list[ValidationSignal] = []

        non_neg = gap >= 0.0
        signals.append(ValidationSignal(
            validator="certainty",
            signal_name="gap_non_negative",
            passed=non_neg,
            severity="critical",
            description=(
                f"certainty_gap={gap:.4f} is "
                + ("non-negative (valid)." if non_neg else "NEGATIVE — invalid.")
            ),
            measured_value=gap,
            expected_range=(0.0, 1.0),
        ))

        less_than_cert = gap < cert if cert > 0 else True
        signals.append(ValidationSignal(
            validator="certainty",
            signal_name="gap_less_than_certainty",
            passed=less_than_cert,
            severity="critical",
            description=(
                f"certainty_gap={gap:.4f} "
                + ("is less than" if less_than_cert else "EXCEEDS")
                + f" max_certainty={cert:.4f}. "
                + ("" if less_than_cert else "Gap cannot exceed leading certainty.")
            ),
            measured_value=gap,
        ))

        # Gap as fraction of max_certainty
        if cert > 0:
            gap_fraction = gap / cert
            fraction_ok  = gap_fraction >= self._min_gap_fraction
            signals.append(ValidationSignal(
                validator="certainty",
                signal_name="gap_fraction_meaningful",
                passed=fraction_ok,
                severity="info",
                description=(
                    f"certainty_gap/max_certainty = {gap_fraction:.4f} "
                    + ("meets" if fraction_ok else "is below")
                    + f" minimum fraction {self._min_gap_fraction:.4f}. "
                    + ("" if fraction_ok else
                       "Extremely small gap suggests near-tied competition.")
                ),
                measured_value=gap_fraction,
                expected_range=(self._min_gap_fraction, 1.0),
            ))

        return signals

    def _check_entropy_bounds(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Shannon entropy (ambiguity_index) must be in [0, log2(6)+tolerance].
        """
        entropy = result.ambiguity_index
        signals: list[ValidationSignal] = []

        non_neg = entropy >= 0.0
        signals.append(ValidationSignal(
            validator="certainty",
            signal_name="entropy_non_negative",
            passed=non_neg,
            severity="critical",
            description=(
                f"ambiguity_index={entropy:.4f} bits is "
                + ("non-negative (valid)." if non_neg else "NEGATIVE — Shannon entropy cannot be negative.")
            ),
            measured_value=entropy,
            expected_range=(0.0, _MAX_ENTROPY_6_CLASS),
        ))

        within_max = entropy <= self._entropy_ceil
        signals.append(ValidationSignal(
            validator="certainty",
            signal_name="entropy_within_theoretical_maximum",
            passed=within_max,
            severity="warning",
            description=(
                f"ambiguity_index={entropy:.4f} bits "
                + ("within" if within_max else "EXCEEDS")
                + f" theoretical maximum {_MAX_ENTROPY_6_CLASS:.4f} bits for 6 classes "
                + f"(ceiling with tolerance: {self._entropy_ceil:.4f})."
            ),
            measured_value=entropy,
            expected_range=(0.0, self._entropy_ceil),
        ))

        return signals

    def _check_entropy_certainty_coherence(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        High certainty must correlate with low entropy, and vice versa.

        Incoherent combinations:
          · High certainty (>= threshold) with high entropy (>= high_cert_max)
          · Recommendation = SAFE_NON_INVASIVE_TRIAGE with entropy >= 1.5 bits
        """
        cert    = result.max_certainty
        entropy = result.ambiguity_index
        signals: list[ValidationSignal] = []

        if cert >= self._high_cert_thresh:
            entropy_low = entropy <= self._high_cert_entropy
            signals.append(ValidationSignal(
                validator="certainty",
                signal_name="high_certainty_low_entropy",
                passed=entropy_low,
                severity="warning",
                description=(
                    f"max_certainty={cert:.4f} >= {self._high_cert_thresh} "
                    "(high-certainty case). "
                    + (f"Entropy {entropy:.4f} <= {self._high_cert_entropy} (coherent)."
                       if entropy_low else
                       f"Entropy {entropy:.4f} > {self._high_cert_entropy} — "
                       "high certainty should not coexist with high ambiguity.")
                ),
                measured_value=entropy,
                expected_range=(0.0, self._high_cert_entropy),
            ))

        # Safe triage coherence: I3 cross-check (mirrors escalation_validator I3)
        if result.recommendation == "SAFE_NON_INVASIVE_TRIAGE" and entropy > 1.50:
            signals.append(ValidationSignal(
                validator="certainty",
                signal_name="safe_triage_entropy_coherent",
                passed=False,
                severity="critical",
                description=(
                    f"SAFE_NON_INVASIVE_TRIAGE with ambiguity_index={entropy:.4f} > 1.50 bits. "
                    "High entropy invalidates safe triage — clinical invariant I3 violated."
                ),
                measured_value=entropy,
                expected_range=(0.0, 1.50),
            ))

        # BIOPSY with very low entropy and high certainty — over-escalation signal
        if (
            result.recommendation in ("BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION")
            and cert >= 0.85
            and entropy < 0.50
            and result.contradiction_load < 0.10
        ):
            signals.append(ValidationSignal(
                validator="certainty",
                signal_name="biopsy_not_entropy_driven",
                passed=False,
                severity="info",
                description=(
                    f"BIOPSY issued with high certainty ({cert:.3f}), "
                    f"low entropy ({entropy:.4f} bits), and low contradiction "
                    f"({result.contradiction_load:.3f}). "
                    "Biopsy appears over-conservative for this certainty profile."
                ),
                measured_value=entropy,
            ))

        return signals

    def _check_trajectory_certainty_evolution(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Certainty evolution should show progressive accumulation:
        the final certainty should generally exceed the initial certainty
        unless contradiction-driven decay occurred.
        """
        signals: list[ValidationSignal] = []
        traj = result.trajectory

        if traj is None:
            return signals

        series = traj.certainty_series()
        if len(series) < 2:
            return signals

        initial  = series[0]
        final    = series[-1]
        improved = final >= initial - 0.05   # allow minor decay from dampening

        signals.append(ValidationSignal(
            validator="certainty",
            signal_name="certainty_improves_or_stable_across_stages",
            passed=improved,
            severity="info",
            description=(
                f"Certainty trajectory: initial={initial:.4f}, final={final:.4f}. "
                + ("Progressed or stable — consistent with evidence accumulation."
                   if improved else
                   f"Declined by {initial - final:.4f} — "
                   "may reflect contradiction dampening or evidence scarcity.")
            ),
            measured_value=final - initial,
        ))

        # Certainty convergence: series should not end with a downward spike
        # from the peak (suggests instability or premature certainty collapse)
        if len(series) >= 3:
            peak = max(series)
            peak_drop = peak - final
            stable_convergence = peak_drop < 0.25
            signals.append(ValidationSignal(
                validator="certainty",
                signal_name="certainty_convergence_stable",
                passed=stable_convergence,
                severity="warning",
                description=(
                    f"Peak certainty {peak:.4f}, final certainty {final:.4f}. "
                    f"Drop from peak: {peak_drop:.4f}. "
                    + ("Convergence is stable."
                       if stable_convergence else
                       "Large drop from peak suggests certainty collapse — "
                       "possible late-stage contradiction override.")
                ),
                measured_value=peak_drop,
                expected_range=(0.0, 0.25),
            ))

        return signals
