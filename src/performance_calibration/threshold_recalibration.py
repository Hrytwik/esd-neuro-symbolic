"""
ThresholdRecalibrator — empirically-derived escalation threshold calibration.

The Phase 5 Step 1 diagnostic audit established that the current thresholds
were calibrated for the full 34-feature biopsy-complete context, where
pipeline certainty regularly reaches 0.70–0.90. On 12 clinical features,
mean certainty is only 0.21 and mean ambiguity is 2.38 bits — rendering
the thresholds pathologically strict (99%+ escalation).

This module implements:

  1. Post-processing recalibration of SymbolicFeatureVectors — applies
     new (ambiguity_ceiling, certainty_floor) pairs to recompute the
     binary signals {requires_biopsy, is_safe_triage, certainty_sufficiency,
     recommendation_encoded, fsm_state_encoded} without re-executing the
     full pipeline.

  2. Calibration sweep — evaluates all (ambiguity, certainty) pairs on
     the training set to find the Pareto-optimal configuration satisfying:
       · zero false-safe violations (safety non-negotiable)
       · maximum biopsy reduction
       · preserved contradiction escalation (ceiling unchanged at 0.40)

  3. Calibration curves — escalation rate vs threshold for publication.

Safety guarantees
-----------------
  · The contradiction ceiling (0.40) is NEVER relaxed.
  · Any case with contradiction_load > 0.40 is escalated regardless of
    other threshold settings.
  · A "false safe" (declaring safe when the classifier is wrong) is a
    safety violation. Zero-violation configurations are the only candidates.
  · The calibration is performed on TRAINING data only; the recommended
    config is validated on held-out TEST data.

Usage
-----
  calibrator = ThresholdRecalibrator(
      ambiguity_ceiling=2.50,
      certainty_floor=0.40,
  )
  recalibrated_vecs = calibrator.recalibrate(original_vectors)
  report = calibrator.fit_and_report(train_vecs, y_pred_train, y_train)
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import (
    SymbolicFeatureVector,
    _FSM_STATE_ORDER,
    _RECOMMENDATION_ORDER,
    _MAX_ENTROPY_6CLASS,
)


# ── Constants ─────────────────────────────────────────────────────────────────

CONTRADICTION_CEILING:   float = 0.40   # Fixed — never swept
CERTAINTY_GAP_FLOOR:     float = 0.15   # Minimum gap for safe triage

# Default search grids
DEFAULT_AMBIGUITY_GRID:  list[float] = [1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
DEFAULT_CERTAINTY_GRID:  list[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


# ── FSM state mapping ─────────────────────────────────────────────────────────

def _encode_recalibrated_state(requires_biopsy: bool, is_safe: bool) -> tuple[int, str, int, str]:
    """
    Derive FSM state + recommendation codes for a recalibrated decision.

    Returns (fsm_state_encoded, final_state, recommendation_encoded, recommendation).
    """
    if requires_biopsy:
        return (
            _FSM_STATE_ORDER.get("BIOPSY_ESCALATION", 7),
            "BIOPSY_ESCALATION",
            _RECOMMENDATION_ORDER.get("BIOPSY_RECOMMENDED", 3),
            "BIOPSY_RECOMMENDED",
        )
    if is_safe:
        return (
            _FSM_STATE_ORDER.get("SAFE_TRIAGE", 6),
            "SAFE_TRIAGE",
            _RECOMMENDATION_ORDER.get("SAFE_NON_INVASIVE_TRIAGE", 0),
            "SAFE_NON_INVASIVE_TRIAGE",
        )
    return (
        _FSM_STATE_ORDER.get("AMBIGUITY_ESCALATION", 4),
        "AMBIGUITY_ESCALATION",
        _RECOMMENDATION_ORDER.get("AMBIGUOUS_PRESENTATION", 2),
        "AMBIGUOUS_PRESENTATION",
    )


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThresholdConfig:
    """A specific (ambiguity_ceiling, certainty_floor) threshold pair."""
    ambiguity_ceiling: float = 1.50
    certainty_floor:   float = 0.55
    gap_floor:         float = CERTAINTY_GAP_FLOOR

    def label(self) -> str:
        return f"amb={self.ambiguity_ceiling:.2f} cert={self.certainty_floor:.2f}"


@dataclass
class ThresholdEvaluationResult:
    """
    Outcome of evaluating a single threshold configuration on a labelled set.

    Attributes
    ----------
    config:
        Threshold configuration evaluated.
    escalation_rate:
        Fraction of records escalated under this configuration.
    safe_triage_rate:
        Fraction declared safe-triage.
    safe_precision:
        Precision of safe-triage decisions (1.0 = zero false-safes).
    false_safe_count:
        Count of safety violations.
    true_safe_count:
        Count of correctly safe-triaged records.
    contradiction_escalated:
        Count escalated due to contradiction_load > 0.40 (non-negotiable).
    is_zero_violation:
        True iff false_safe_count == 0.
    effective_accuracy_delta:
        Change in effective accuracy vs. always-escalate baseline.
        Positive = identifying safe cases correctly improves effective acc.
    """
    config:                   ThresholdConfig
    escalation_rate:          float = 0.0
    safe_triage_rate:         float = 0.0
    safe_precision:           float = 1.0
    false_safe_count:         int   = 0
    true_safe_count:          int   = 0
    contradiction_escalated:  int   = 0
    is_zero_violation:        bool  = True
    effective_accuracy_delta: float = 0.0

    def summary_line(self) -> str:
        safe_mark = "SAFE" if self.is_zero_violation else "VIOLATION"
        return (
            f"[{safe_mark:9s}] "
            f"{self.config.label()} "
            f"esc={self.escalation_rate:.1%} "
            f"safe_prec={self.safe_precision:.3f} "
            f"false_safe={self.false_safe_count}"
        )


@dataclass
class CalibrationReport:
    """
    Complete threshold calibration report.

    Attributes
    ----------
    best_config:
        Recommended threshold configuration (zero-violation, max safe-triage).
    default_config:
        Original default configuration (ambiguity=1.50, certainty=0.55).
    default_result:
        Evaluation of the default config on the training set.
    best_result:
        Evaluation of the recommended config on training set.
    all_results:
        All evaluated (ambiguity, certainty) configurations.
    zero_violation_configs:
        Subset with false_safe_count == 0.
    escalation_reduction:
        Reduction in escalation rate vs. default.
    safe_triage_gain:
        Gain in safe-triage rate vs. default.
    calibration_set_size:
        Number of records used for calibration.
    """
    best_config:               ThresholdConfig
    default_config:            ThresholdConfig = field(
        default_factory=lambda: ThresholdConfig(1.50, 0.55)
    )
    default_result:            ThresholdEvaluationResult | None = None
    best_result:               ThresholdEvaluationResult | None = None
    all_results:               list[ThresholdEvaluationResult] = field(default_factory=list)
    zero_violation_configs:    list[ThresholdEvaluationResult] = field(default_factory=list)
    escalation_reduction:      float = 0.0
    safe_triage_gain:          float = 0.0
    calibration_set_size:      int   = 0

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "THRESHOLD CALIBRATION REPORT",
            "=" * 72,
            f"  Calibration records    : {self.calibration_set_size}",
            f"  Configurations tested  : {len(self.all_results)}",
            f"  Zero-violation configs : {len(self.zero_violation_configs)}",
            "-" * 72,
            "  DEFAULT CONFIG:",
        ]
        if self.default_result:
            lines.append(f"    {self.default_result.summary_line()}")
        lines += [
            "  RECOMMENDED CONFIG:",
        ]
        if self.best_result:
            lines.append(f"    {self.best_result.summary_line()}")
        lines += [
            "-" * 72,
            f"  Escalation reduction   : {self.escalation_reduction:+.1%}",
            f"  Safe-triage gain       : {self.safe_triage_gain:+.1%}",
            f"  Contradiction ceiling  : {CONTRADICTION_CEILING:.2f} (unchanged)",
            "=" * 72,
        ]
        return "\n".join(lines)


# ── Recalibrator ──────────────────────────────────────────────────────────────

class ThresholdRecalibrator:
    """
    Empirically calibrates and applies revised escalation thresholds.

    Parameters
    ----------
    ambiguity_ceiling:
        Maximum ambiguity index (bits) for non-escalated triage.
    certainty_floor:
        Minimum certainty for safe-triage classification.
    gap_floor:
        Minimum certainty gap for safe-triage classification.
    """

    def __init__(
        self,
        ambiguity_ceiling: float = 2.50,
        certainty_floor:   float = 0.40,
        gap_floor:         float = CERTAINTY_GAP_FLOOR,
    ) -> None:
        self.config = ThresholdConfig(ambiguity_ceiling, certainty_floor, gap_floor)

    # ── Public API ────────────────────────────────────────────────────────────

    def recalibrate(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> list[SymbolicFeatureVector]:
        """
        Apply recalibrated thresholds to a list of SymbolicFeatureVectors.

        Returns new frozen SymbolicFeatureVector instances with updated
        {requires_biopsy, is_safe_triage, certainty_sufficiency,
        fsm_state_encoded, final_state, recommendation_encoded, recommendation}
        fields. All other fields are unchanged.

        The contradiction ceiling (0.40) is always enforced regardless of
        the configured thresholds.

        Parameters
        ----------
        vectors:
            Original symbolic feature vectors from the pipeline.

        Returns
        -------
        New list of SymbolicFeatureVectors with recalibrated decision fields.
        """
        return [self._recalibrate_one(v) for v in vectors]

    def fit_and_report(
        self,
        vectors:     list[SymbolicFeatureVector],
        y_pred:      np.ndarray,
        y_true:      np.ndarray,
        ambiguity_grid: list[float] | None = None,
        certainty_grid: list[float] | None = None,
    ) -> CalibrationReport:
        """
        Sweep thresholds on a labelled set and identify the optimal config.

        Parameters
        ----------
        vectors:
            Symbolic feature vectors (training or validation set).
        y_pred:
            Model B predicted labels (0-based integer codes).
        y_true:
            True labels (0-based integer codes).
        ambiguity_grid:
            Ambiguity ceiling values to sweep.
        certainty_grid:
            Certainty floor values to sweep.
        """
        amb_grid  = ambiguity_grid or DEFAULT_AMBIGUITY_GRID
        cert_grid = certainty_grid or DEFAULT_CERTAINTY_GRID

        default_cfg = ThresholdConfig(1.50, 0.55)
        all_results: list[ThresholdEvaluationResult] = []

        for amb, cert in product(amb_grid, cert_grid):
            cfg    = ThresholdConfig(amb, cert)
            result = self._evaluate_config(cfg, vectors, y_pred, y_true)
            all_results.append(result)

        default_result = next(
            (r for r in all_results
             if abs(r.config.ambiguity_ceiling - 1.50) < 0.01
             and abs(r.config.certainty_floor - 0.55) < 0.01),
            self._evaluate_config(default_cfg, vectors, y_pred, y_true),
        )

        zero_viol = [r for r in all_results if r.is_zero_violation]
        best      = self._select_best(zero_viol, default_result)

        esc_reduction  = (
            default_result.escalation_rate - best.escalation_rate
            if best else 0.0
        )
        safe_gain = (
            best.safe_triage_rate - default_result.safe_triage_rate
            if best else 0.0
        )

        return CalibrationReport(
            best_config=best.config if best else default_cfg,
            default_config=default_cfg,
            default_result=default_result,
            best_result=best,
            all_results=all_results,
            zero_violation_configs=zero_viol,
            escalation_reduction=esc_reduction,
            safe_triage_gain=safe_gain,
            calibration_set_size=len(vectors),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _recalibrate_one(
        self,
        v: SymbolicFeatureVector,
    ) -> SymbolicFeatureVector:
        """Apply recalibrated thresholds to a single SymbolicFeatureVector."""
        cfg = self.config

        # Contradiction escalation: non-negotiable
        if v.contradiction_load > CONTRADICTION_CEILING:
            requires_biopsy = True
            is_safe_triage  = False
        elif v.ambiguity_index > cfg.ambiguity_ceiling:
            requires_biopsy = True
            is_safe_triage  = False
        elif v.certainty < cfg.certainty_floor:
            requires_biopsy = True
            is_safe_triage  = False
        else:
            requires_biopsy = False
            is_safe_triage  = (
                v.certainty >= cfg.certainty_floor
                and v.certainty_gap >= cfg.gap_floor
            )

        # Recalibrated certainty sufficiency
        cert_suff = (
            1.0 if (
                not requires_biopsy
                and v.certainty >= cfg.certainty_floor
                and v.certainty_gap >= cfg.gap_floor
            ) else 0.0
        )

        # Derive FSM state and recommendation from new escalation decision
        fsm_enc, final_state, rec_enc, recommendation = _encode_recalibrated_state(
            requires_biopsy, is_safe_triage
        )

        return dataclasses.replace(
            v,
            requires_biopsy=requires_biopsy,
            is_safe_triage=is_safe_triage,
            certainty_sufficiency=cert_suff,
            fsm_state_encoded=fsm_enc,
            final_state=final_state,
            recommendation_encoded=rec_enc,
            recommendation=recommendation,
        )

    def _evaluate_config(
        self,
        cfg:     ThresholdConfig,
        vectors: list[SymbolicFeatureVector],
        y_pred:  np.ndarray,
        y_true:  np.ndarray,
    ) -> ThresholdEvaluationResult:
        """Evaluate a single threshold config on labelled data."""
        recalibrator = ThresholdRecalibrator(
            cfg.ambiguity_ceiling, cfg.certainty_floor, cfg.gap_floor
        )
        recal_vecs  = recalibrator.recalibrate(vectors)
        n           = len(recal_vecs)

        n_esc           = sum(1 for v in recal_vecs if v.requires_biopsy)
        n_safe          = n - n_esc
        n_contr         = sum(
            1 for v in vectors if v.contradiction_load > CONTRADICTION_CEILING
        )

        false_safe = 0
        true_safe  = 0
        for i, v in enumerate(recal_vecs):
            if not v.requires_biopsy:
                if y_pred[i] == y_true[i]:
                    true_safe += 1
                else:
                    false_safe += 1

        esc_rate      = n_esc / n
        safe_rate     = n_safe / n
        precision     = true_safe / max(n_safe, 1)
        n_correct     = int(np.sum(y_pred == y_true))
        eff_acc_delta = true_safe / max(n_correct, 1) - 0.0

        return ThresholdEvaluationResult(
            config=cfg,
            escalation_rate=esc_rate,
            safe_triage_rate=safe_rate,
            safe_precision=precision,
            false_safe_count=false_safe,
            true_safe_count=true_safe,
            contradiction_escalated=n_contr,
            is_zero_violation=(false_safe == 0),
            effective_accuracy_delta=eff_acc_delta,
        )

    def _select_best(
        self,
        zero_viol: list[ThresholdEvaluationResult],
        default:   ThresholdEvaluationResult,
    ) -> ThresholdEvaluationResult | None:
        """
        Select the best zero-violation config:
        minimum escalation rate, breaking ties with higher ambiguity ceiling.
        """
        improving = [
            r for r in zero_viol
            if r.escalation_rate < default.escalation_rate - 0.001
        ]
        candidates = improving if improving else zero_viol
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda r: (r.escalation_rate, -r.config.ambiguity_ceiling),
        )
