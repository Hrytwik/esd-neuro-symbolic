"""
EscalationSensitivityAnalyzer — threshold sweep and calibration.

Performs a systematic sweep across the three escalation control parameters:

  · Ambiguity ceiling   (default 1.50 bits) — primary driver of 99.4% escalation
  · Certainty floor     (default 0.55)      — secondary safety gate
  · Contradiction ceiling (default 0.40)    — must NOT be relaxed (safety-critical)

For each (ambiguity, certainty) threshold combination, computes:
  · Projected escalation rate on the test set
  · Fraction of safe-triage decisions that would be correct (precision)
  · Fraction of true-safe cases correctly identified (recall)
  · Safety violation count — cases escalated as safe despite being misclassified

Clinical safety constraint
---------------------------
The contradiction ceiling (0.40) is NON-NEGOTIABLE.
Any threshold configuration that increases contradiction-triggered escalation
is rejected. The purpose of the sweep is to identify parameter pairs that:
  · Reduce ambiguity-triggered over-escalation
  · Maintain ≥ 100% safety (zero false safe-triage decisions)
  · Are compatible with the symbolic pipeline's clinical-only certainty range

Terminology
-----------
  "safe-triage" = system emits PROCEED_WITHOUT_BIOPSY recommendation
  "escalate"    = system emits REQUEST_BIOPSY recommendation
  "false safe"  = system says safe but classifier (B or C) is wrong
  "true safe"   = system says safe and classifier is correct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Constants ─────────────────────────────────────────────────────────────────

# Grid values for the parameter sweep
AMBIGUITY_GRID = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00]
CERTAINTY_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
CONTRADICTION_CEILING = 0.40   # Fixed — never swept


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThresholdConfig:
    """A specific (ambiguity, certainty) threshold pair."""
    ambiguity_ceiling: float
    certainty_floor:   float


@dataclass
class ThresholdSweepResult:
    """
    Outcome of evaluating a single threshold configuration.

    Attributes
    ----------
    config:
        The threshold configuration evaluated.
    escalation_rate:
        Fraction of records that would be escalated.
    safe_triage_rate:
        Fraction of records that would be triaged as safe.
    safe_triage_precision:
        Of cases declared safe, fraction where classifier is correct.
        If classifier is wrong, that is a false-safe (safety violation).
    safe_triage_recall:
        Of correctly-classified cases, fraction declared safe.
    false_safe_count:
        Number of safety violations (declared safe, classifier wrong).
    true_safe_count:
        Number of correctly safe-triaged cases.
    accuracy_delta_model_b:
        Change in effective accuracy when escalated cases are excluded
        (only non-escalated cases are classified; remainder go to biopsy).
    is_safe_configuration:
        True if false_safe_count == 0.
    """

    config:                  ThresholdConfig
    escalation_rate:         float = 0.0
    safe_triage_rate:        float = 0.0
    safe_triage_precision:   float = 0.0
    safe_triage_recall:      float = 0.0
    false_safe_count:        int   = 0
    true_safe_count:         int   = 0
    accuracy_delta_model_b:  float = 0.0
    is_safe_configuration:   bool  = True


@dataclass
class SensitivityReport:
    """
    Complete threshold sensitivity analysis output.

    Attributes
    ----------
    sweep_results:
        All evaluated (ambiguity, certainty) configurations.
    safe_configurations:
        Subset where false_safe_count == 0 (zero safety violations).
    recommended_config:
        Single recommended threshold pair balancing safety and coverage.
    default_config:
        The current default threshold configuration (1.50, 0.55).
    escalation_rate_at_default:
        Current escalation rate (≈ 0.994).
    projected_escalation_at_recommended:
        Projected escalation rate after threshold recalibration.
    biopsy_reduction_achievable:
        Fraction of currently-escalated cases that could safely be
        de-escalated with the recommended configuration.
    safety_constraint_satisfied:
        True if the recommended config has zero safety violations.
    summary_table:
        Pre-formatted text table of selected sweep points.
    """

    sweep_results:                       list[ThresholdSweepResult] = field(default_factory=list)
    safe_configurations:                 list[ThresholdSweepResult] = field(default_factory=list)
    recommended_config:                  ThresholdConfig | None = None
    default_config:                      ThresholdConfig = field(
        default_factory=lambda: ThresholdConfig(1.50, 0.55)
    )
    escalation_rate_at_default:          float = 0.0
    projected_escalation_at_recommended: float = 0.0
    biopsy_reduction_achievable:         float = 0.0
    safety_constraint_satisfied:         bool  = True
    summary_table:                       str   = ""

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "ESCALATION THRESHOLD SENSITIVITY REPORT",
            "=" * 72,
            f"  Current escalation rate  (amb=1.50, cert=0.55): "
            f"{self.escalation_rate_at_default:.1%}",
        ]
        if self.recommended_config:
            lines.append(
                f"  Recommended config       "
                f"(amb={self.recommended_config.ambiguity_ceiling:.2f}, "
                f"cert={self.recommended_config.certainty_floor:.2f}): "
                f"{self.projected_escalation_at_recommended:.1%}"
            )
        lines += [
            f"  Achievable biopsy reduction: {self.biopsy_reduction_achievable:.1%}",
            f"  Safety constraint satisfied: {self.safety_constraint_satisfied}",
            f"  Safe configurations found : {len(self.safe_configurations)}",
            "-" * 72,
            self.summary_table,
            "=" * 72,
        ]
        return "\n".join(lines)


# ── Analyser ──────────────────────────────────────────────────────────────────

class EscalationSensitivityAnalyzer:
    """
    Sweeps escalation thresholds and identifies safe-rebalancing candidates.

    Parameters
    ----------
    ambiguity_grid:
        Ambiguity ceiling values to sweep (bits).
    certainty_grid:
        Certainty floor values to sweep.
    """

    def __init__(
        self,
        ambiguity_grid: list[float] | None = None,
        certainty_grid: list[float] | None = None,
    ) -> None:
        self.amb_grid  = ambiguity_grid or AMBIGUITY_GRID
        self.cert_grid = certainty_grid or CERTAINTY_GRID

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        symbolic_vectors: list[SymbolicFeatureVector],
        y_pred_model_b:   np.ndarray,
        y_true:           np.ndarray,
    ) -> SensitivityReport:
        """
        Run the threshold sensitivity sweep.

        Parameters
        ----------
        symbolic_vectors:
            Test-set symbolic feature vectors (one per test record).
        y_pred_model_b:
            Model B predictions for the test set (0-based integer codes).
        y_true:
            True labels for the test set (0-based integer codes).
        """
        if len(symbolic_vectors) == 0:
            return SensitivityReport()

        sweep_results: list[ThresholdSweepResult] = []

        for amb_ceil, cert_floor in product(self.amb_grid, self.cert_grid):
            cfg    = ThresholdConfig(amb_ceil, cert_floor)
            result = self._evaluate_config(
                cfg, symbolic_vectors, y_pred_model_b, y_true
            )
            sweep_results.append(result)

        # Default config
        default_cfg = ThresholdConfig(1.50, 0.55)
        default_result = next(
            (r for r in sweep_results
             if abs(r.config.ambiguity_ceiling - 1.50) < 0.01
             and abs(r.config.certainty_floor - 0.55) < 0.01),
            self._evaluate_config(
                default_cfg, symbolic_vectors, y_pred_model_b, y_true
            ),
        )

        safe_configs = [r for r in sweep_results if r.is_safe_configuration]
        recommended  = self._pick_recommendation(safe_configs, default_result)

        proj_rate = (
            recommended.escalation_rate if recommended else default_result.escalation_rate
        )
        reduction = (
            max(0.0, default_result.escalation_rate - proj_rate)
            / max(default_result.escalation_rate, 1e-9)
        )

        table = self._build_table(sweep_results)

        return SensitivityReport(
            sweep_results=sweep_results,
            safe_configurations=safe_configs,
            recommended_config=(
                recommended.config if recommended else None
            ),
            default_config=default_cfg,
            escalation_rate_at_default=default_result.escalation_rate,
            projected_escalation_at_recommended=proj_rate,
            biopsy_reduction_achievable=reduction,
            safety_constraint_satisfied=recommended is not None,
            summary_table=table,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _evaluate_config(
        self,
        cfg: ThresholdConfig,
        vectors: list[SymbolicFeatureVector],
        y_pred_b: np.ndarray,
        y_true:   np.ndarray,
    ) -> ThresholdSweepResult:
        """Evaluate a single threshold configuration."""
        n = len(vectors)
        escalated   = np.zeros(n, dtype=bool)
        contradiction_viol = 0

        for i, v in enumerate(vectors):
            # Contradiction ceiling is fixed — always escalate if exceeded
            if v.contradiction_load > CONTRADICTION_CEILING:
                escalated[i] = True
                contradiction_viol += 1
                continue
            # Apply sweep thresholds
            if v.ambiguity_index > cfg.ambiguity_ceiling:
                escalated[i] = True
                continue
            if v.certainty < cfg.certainty_floor:
                escalated[i] = True
                continue

        n_escalated    = int(np.sum(escalated))
        n_safe         = n - n_escalated
        esc_rate       = n_escalated / n

        # Safety analysis for non-escalated cases
        false_safe = 0
        true_safe  = 0
        for i in range(n):
            if not escalated[i]:
                if y_pred_b[i] == y_true[i]:
                    true_safe += 1
                else:
                    false_safe += 1

        prec = true_safe / max(n_safe, 1)
        # Recall: fraction of correctly-classified cases that are declared safe
        n_correct = int(np.sum(y_pred_b == y_true))
        rec  = true_safe / max(n_correct, 1)

        # Effective accuracy: correct safe-triage / all non-escalated
        acc_delta = (
            (true_safe / max(n_safe, 1)) - 1.0
        )  # negative if precision < 1

        return ThresholdSweepResult(
            config=cfg,
            escalation_rate=esc_rate,
            safe_triage_rate=n_safe / n,
            safe_triage_precision=prec,
            safe_triage_recall=rec,
            false_safe_count=false_safe,
            true_safe_count=true_safe,
            accuracy_delta_model_b=acc_delta,
            is_safe_configuration=(false_safe == 0),
        )

    def _pick_recommendation(
        self,
        safe_configs: list[ThresholdSweepResult],
        default_result: ThresholdSweepResult,
    ) -> ThresholdSweepResult | None:
        """
        Select the recommended threshold configuration.

        Priority:
          1. Zero safety violations (is_safe_configuration=True)
          2. Maximum biopsy reduction (lowest escalation rate)
          3. Among ties, prefer higher ambiguity ceiling (more lenient)
             so the system flags fewer borderline cases.
        """
        if not safe_configs:
            return None

        # Among safe configs, pick the one with lowest escalation rate
        # that is still below the default (i.e., some reduction achieved)
        improving = [
            r for r in safe_configs
            if r.escalation_rate < default_result.escalation_rate - 0.001
        ]
        candidates = improving if improving else safe_configs

        # Sort by escalation rate ascending, then ambiguity ceiling descending
        candidates.sort(
            key=lambda r: (r.escalation_rate, -r.config.ambiguity_ceiling)
        )
        return candidates[0]

    def _build_table(
        self,
        results: list[ThresholdSweepResult],
    ) -> str:
        """Build a formatted text table of sweep results."""
        header = (
            f"{'Amb':>6} {'Cert':>6} {'EscRate':>8} "
            f"{'SafePrec':>9} {'FalseSafe':>10} {'Safe?':>6}"
        )
        lines = [header, "-" * len(header)]
        # Show a representative subset
        shown = sorted(results, key=lambda r: r.escalation_rate)
        step  = max(1, len(shown) // 16)
        for r in shown[::step]:
            safe_mark = "YES" if r.is_safe_configuration else "NO"
            lines.append(
                f"{r.config.ambiguity_ceiling:6.2f} "
                f"{r.config.certainty_floor:6.2f} "
                f"{r.escalation_rate:8.1%} "
                f"{r.safe_triage_precision:9.3f} "
                f"{r.false_safe_count:10d} "
                f"{safe_mark:>6}"
            )
        return "\n".join(lines)
