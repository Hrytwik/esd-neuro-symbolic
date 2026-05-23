"""
BehavioralValidator — central coordinator for clinical reasoning validation.

Orchestrates all domain-specific validators against a PipelineResult and
produces a unified BehavioralValidationReport that characterises whether the
system is reasoning in a clinically believable, diagnostically coherent manner.

The coordinator does NOT re-implement validation logic — it delegates to:

  TrajectoryValidator    — certainty evolution smoothness
  EscalationValidator    — biopsy escalation appropriateness
  ContradictionValidator — contradiction propagation realism
  CertaintyValidator     — certainty metric integrity
  NarrativeValidator     — clinical narrative plausibility

A ValidationSignal is the atomic output unit: each represents a single
testable clinical reasoning property with a pass/fail status, severity
level, and measured value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.pipeline.pipeline_runner import PipelineResult
from src.pipeline.synthetic_case_library import SyntheticCase
from src.utils.logger import get_logger

log = get_logger(__name__, subsystem="BehavioralValidator")


# ── Shared validation types ───────────────────────────────────────────────────

Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class ValidationSignal:
    """
    Atomic unit of behavioral validation output.

    Represents one testable clinical reasoning property.
    """

    validator:     str          # originating validator name
    signal_name:   str          # short machine-readable identifier
    passed:        bool
    severity:      Severity     # "info" | "warning" | "critical"
    description:   str          # human-readable explanation
    measured_value: float | None = None
    expected_range: tuple[float, float] | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        meas = f" (measured={self.measured_value:.4f})" if self.measured_value is not None else ""
        return f"[{status}][{self.severity}] {self.validator}.{self.signal_name}{meas}: {self.description}"


@dataclass
class BehavioralValidationReport:
    """
    Complete behavioral validation result for a single pipeline execution.

    Aggregates all sub-validator signals into a unified clinical
    reasoning quality assessment.
    """

    case_id:    str
    run_id:     str
    passed:     bool

    signals:    list[ValidationSignal] = field(default_factory=list)

    # Domain-level quality scores [0.0 – 1.0]
    trajectory_score:     float = 0.0
    escalation_score:     float = 0.0
    contradiction_score:  float = 0.0
    certainty_score:      float = 0.0
    narrative_score:      float = 0.0

    @property
    def overall_score(self) -> float:
        """Weighted aggregate of all domain scores."""
        return (
            0.25 * self.trajectory_score
            + 0.25 * self.escalation_score
            + 0.20 * self.contradiction_score
            + 0.20 * self.certainty_score
            + 0.10 * self.narrative_score
        )

    @property
    def critical_failures(self) -> list[ValidationSignal]:
        return [s for s in self.signals if not s.passed and s.severity == "critical"]

    @property
    def warnings(self) -> list[ValidationSignal]:
        return [s for s in self.signals if not s.passed and s.severity == "warning"]

    @property
    def passed_count(self) -> int:
        return sum(1 for s in self.signals if s.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.signals if not s.passed)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] case={self.case_id} "
            f"score={self.overall_score:.3f} "
            f"signals={len(self.signals)} "
            f"passed={self.passed_count} "
            f"failed={self.failed_count} "
            f"critical={len(self.critical_failures)}"
        )


@dataclass
class BatchValidationReport:
    """
    Aggregated validation report across multiple cases.
    """

    label:          str
    case_reports:   list[BehavioralValidationReport] = field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.case_reports)

    @property
    def passed_cases(self) -> int:
        return sum(1 for r in self.case_reports if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / max(self.total_cases, 1)

    @property
    def mean_overall_score(self) -> float:
        if not self.case_reports:
            return 0.0
        return sum(r.overall_score for r in self.case_reports) / len(self.case_reports)

    @property
    def critical_failure_cases(self) -> list[str]:
        return [r.case_id for r in self.case_reports if r.critical_failures]

    def summary(self) -> str:
        return (
            f"BatchValidation[{self.label}] "
            f"cases={self.total_cases} "
            f"pass={self.passed_cases}/{self.total_cases} "
            f"({self.pass_rate:.1%}) "
            f"mean_score={self.mean_overall_score:.3f} "
            f"critical_failures={len(self.critical_failure_cases)}"
        )


# ── Behavioral validator (coordinator) ───────────────────────────────────────

class BehavioralValidator:
    """
    Coordinates all domain-specific validators and produces a unified
    BehavioralValidationReport for each pipeline execution.

    Parameters
    ----------
    escalation_false_positive_ceiling:
        Maximum acceptable biopsy recommendation rate for genuinely
        safe cases (prevents over-escalation detection).
    critical_failure_threshold:
        Minimum number of critical failures that causes the report to fail.
    """

    def __init__(
        self,
        escalation_false_positive_ceiling: float = 0.30,
        critical_failure_threshold: int = 1,
    ) -> None:
        self._fp_ceiling    = escalation_false_positive_ceiling
        self._crit_thresh   = critical_failure_threshold

        # Import here to avoid circular deps at module level
        from src.validation.trajectory_validator   import TrajectoryValidator
        from src.validation.escalation_validator   import EscalationValidator
        from src.validation.contradiction_validator import ContradictionValidator
        from src.validation.certainty_validator    import CertaintyValidator
        from src.validation.narrative_validator    import NarrativeValidator

        self._trajectory    = TrajectoryValidator()
        self._escalation    = EscalationValidator()
        self._contradiction = ContradictionValidator()
        self._certainty     = CertaintyValidator()
        self._narrative     = NarrativeValidator()

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        result: PipelineResult,
        case: SyntheticCase | None = None,
    ) -> BehavioralValidationReport:
        """
        Run all domain validators against a completed pipeline result.

        Parameters
        ----------
        result:
            The PipelineResult from a completed pipeline execution.
        case:
            Optional SyntheticCase providing expected-behavior ground truth.
            When provided, expected_outcome and expect_contradiction assertions
            are also evaluated.

        Returns
        -------
        BehavioralValidationReport:
            Unified report with per-domain scores and all validation signals.
        """
        all_signals: list[ValidationSignal] = []

        # Domain validators
        traj_signals,  traj_score  = self._trajectory.validate(result)
        esc_signals,   esc_score   = self._escalation.validate(result, case)
        contra_signals, contra_score = self._contradiction.validate(result)
        cert_signals,  cert_score  = self._certainty.validate(result)
        narr_signals,  narr_score  = self._narrative.validate(result)

        all_signals.extend(traj_signals)
        all_signals.extend(esc_signals)
        all_signals.extend(contra_signals)
        all_signals.extend(cert_signals)
        all_signals.extend(narr_signals)

        # Cross-validator coherence checks
        cross_signals = self._cross_validate(result, case)
        all_signals.extend(cross_signals)

        # Overall pass/fail
        critical_count = sum(
            1 for s in all_signals if not s.passed and s.severity == "critical"
        )
        passed = critical_count < self._crit_thresh

        report = BehavioralValidationReport(
            case_id=result.case_id,
            run_id=result.run_id,
            passed=passed,
            signals=all_signals,
            trajectory_score=traj_score,
            escalation_score=esc_score,
            contradiction_score=contra_score,
            certainty_score=cert_score,
            narrative_score=narr_score,
        )

        log.debug(
            "Behavioral validation complete",
            case_id=result.case_id,
            passed=passed,
            score=f"{report.overall_score:.3f}",
            critical=critical_count,
        )
        return report

    def validate_batch(
        self,
        results_and_cases: list[tuple[PipelineResult, SyntheticCase | None]],
        label: str = "batch",
    ) -> BatchValidationReport:
        """
        Validate multiple pipeline results and aggregate into a batch report.
        """
        batch = BatchValidationReport(label=label)
        for result, case in results_and_cases:
            report = self.validate(result, case)
            batch.case_reports.append(report)
        log.info(
            "Batch validation complete",
            label=label,
            **{k: str(v) for k, v in {
                "cases": batch.total_cases,
                "passed": batch.passed_cases,
                "score": f"{batch.mean_overall_score:.3f}",
            }.items()},
        )
        return batch

    # ── Cross-validator coherence ─────────────────────────────────────────────

    def _cross_validate(
        self,
        result: PipelineResult,
        case: SyntheticCase | None,
    ) -> list[ValidationSignal]:
        """
        Checks that require outputs from multiple domains simultaneously.
        """
        signals: list[ValidationSignal] = []

        # High contradiction load must not coexist with SAFE_NON_INVASIVE_TRIAGE
        if (
            result.recommendation == "SAFE_NON_INVASIVE_TRIAGE"
            and result.contradiction_load > 0.20
        ):
            signals.append(ValidationSignal(
                validator="behavioral",
                signal_name="safe_triage_with_high_contradiction",
                passed=False,
                severity="critical",
                description=(
                    f"SAFE_NON_INVASIVE_TRIAGE issued while contradiction_load="
                    f"{result.contradiction_load:.3f} > 0.20. "
                    "Safe triage with contradictions present is clinically incoherent."
                ),
                measured_value=result.contradiction_load,
                expected_range=(0.0, 0.20),
            ))

        # BIOPSY_RECOMMENDED must not be issued for very high certainty + zero contradiction
        if (
            result.recommendation == "BIOPSY_RECOMMENDED"
            and result.max_certainty >= 0.90
            and result.contradiction_load == 0.0
            and result.ambiguity_index < 1.0
        ):
            signals.append(ValidationSignal(
                validator="behavioral",
                signal_name="excessive_biopsy_at_high_certainty",
                passed=False,
                severity="warning",
                description=(
                    f"BIOPSY_RECOMMENDED with certainty={result.max_certainty:.3f} >= 0.90 "
                    "and no contradiction/ambiguity. Possible over-escalation."
                ),
                measured_value=result.max_certainty,
                expected_range=(0.0, 0.90),
            ))

        # Leading disease must be set on any successful run
        if result.success and result.leading_disease is None:
            signals.append(ValidationSignal(
                validator="behavioral",
                signal_name="missing_leading_disease",
                passed=False,
                severity="critical",
                description="Successful run produced no leading disease hypothesis.",
            ))

        # Expected outcome agreement (when case provided)
        if case is not None:
            outcome_match = result.recommendation == case.expected_outcome
            signals.append(ValidationSignal(
                validator="behavioral",
                signal_name="expected_outcome_match",
                passed=outcome_match,
                severity="warning",
                description=(
                    f"Recommendation '{result.recommendation}' "
                    + ("matches" if outcome_match else f"differs from expected '{case.expected_outcome}'")
                    + f" for case {case.case_id}."
                ),
            ))

            leader_match = result.leading_disease == case.expected_leader
            signals.append(ValidationSignal(
                validator="behavioral",
                signal_name="expected_leader_match",
                passed=leader_match,
                severity="info",
                description=(
                    f"Leading disease '{result.leading_disease}' "
                    + ("matches" if leader_match else f"differs from expected '{case.expected_leader}'")
                    + f" for case {case.case_id}."
                ),
            ))

        return signals
