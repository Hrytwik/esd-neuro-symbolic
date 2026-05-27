"""
SymbolicRecoveryAnalyzer — Model B failure vs Model C recovery attribution.

This module is the primary evidence layer for the symbolic reasoning
contribution. It answers:

  Q: When Model B misclassifies a patient, does Model C correct it?
     And if so, WHICH symbolic reasoning signal drove the correction?

Recovery mechanism taxonomy
---------------------------
  CONTRADICTION_RECOVERY    — high contradiction_load in the failing case;
                              symbolic reasoning detected the conflict that
                              the clinical classifier missed.
  AMBIGUITY_RECOVERY        — high ambiguity_index or entropy; symbolic
                              system's entropy signals helped disambiguate.
  TRAJECTORY_RECOVERY       — oscillation or convergence signals guided
                              Model C toward the correct class.
  LEADERSHIP_RECOVERY       — leading_disease_encoded matched the true
                              label even when the clinical model failed.
  COMPETITION_RECOVERY      — certainty_gap or competition dynamics helped
                              separate the correct hypothesis.
  ESCALATION_RECOVERY       — recalibrated requires_biopsy prevented a
                              wrong confident prediction.
  UNEXPLAINED_RECOVERY      — Model C is correct but no dominant symbolic
                              signal explains the correction.

Output
------
RecoveryReport provides:
  · Per-record recovery attribution table
  · Per-mechanism recovery count and fraction
  · Disease-specific recovery analysis
  · Symbolic contribution evidence (primary publication contribution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Recovery mechanism taxonomy ───────────────────────────────────────────────

class RecoveryMechanism(str, Enum):
    CONTRADICTION_RECOVERY = "contradiction_recovery"
    AMBIGUITY_RECOVERY     = "ambiguity_recovery"
    TRAJECTORY_RECOVERY    = "trajectory_recovery"
    LEADERSHIP_RECOVERY    = "leadership_recovery"
    COMPETITION_RECOVERY   = "competition_recovery"
    ESCALATION_RECOVERY    = "escalation_recovery"
    UNEXPLAINED_RECOVERY   = "unexplained_recovery"
    NO_RECOVERY            = "no_recovery"   # B wrong AND C wrong
    BOTH_CORRECT           = "both_correct"  # B correct (no recovery needed)


# ── Thresholds for mechanism attribution ─────────────────────────────────────

_CONTR_RECOVERY_THRESHOLD:    float = 0.15   # contradiction_load
_AMBIGUITY_RECOVERY_THRESHOLD: float = 2.00  # ambiguity_index (bits)
_OSCILLATION_RECOVERY_THRESHOLD: int = 1     # oscillation_count
_GAP_RECOVERY_THRESHOLD:      float = 0.10   # certainty_gap
_LEADERSHIP_MATCH_THRESHOLD:  int   = 0      # leading_disease_encoded == true label


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RecoveryRecord:
    """
    Attribution record for a single test patient.

    Attributes
    ----------
    patient_id:
        Source patient identifier.
    true_disease:
        Ground-truth disease label.
    true_class_int:
        Ground-truth 0-based integer class.
    pred_model_b:
        Model B predicted class.
    pred_model_c:
        Model C predicted class.
    b_correct:
        True if Model B prediction is correct.
    c_correct:
        True if Model C prediction is correct.
    is_recovery:
        True if B is wrong AND C is correct.
    is_regression:
        True if B is correct AND C is wrong.
    recovery_mechanism:
        Primary symbolic recovery mechanism (if is_recovery).
    mechanism_signals:
        Key signal values that triggered the recovery mechanism.
    """

    patient_id:        str
    true_disease:      str
    true_class_int:    int
    pred_model_b:      int
    pred_model_c:      int
    b_correct:         bool
    c_correct:         bool
    is_recovery:       bool
    is_regression:     bool
    recovery_mechanism: RecoveryMechanism
    mechanism_signals: dict[str, float] = field(default_factory=dict)


@dataclass
class MechanismStats:
    """Statistics for a single recovery mechanism."""
    mechanism:    RecoveryMechanism
    count:        int
    fraction:     float   # of total recoveries
    per_disease:  dict[str, int] = field(default_factory=dict)
    mean_signals: dict[str, float] = field(default_factory=dict)


@dataclass
class RecoveryReport:
    """
    Complete symbolic recovery analysis output.

    Attributes
    ----------
    records:
        Per-patient recovery attribution records.
    n_b_correct:
        Model B correct predictions.
    n_c_correct:
        Model C correct predictions.
    n_recoveries:
        Cases where B is wrong and C is correct.
    n_regressions:
        Cases where B is correct and C is wrong.
    n_both_wrong:
        Cases where both B and C are wrong.
    recovery_rate:
        n_recoveries / n_b_errors.
    net_symbolic_gain:
        n_recoveries - n_regressions.
    mechanism_stats:
        Per-mechanism recovery statistics.
    per_disease_recovery_rate:
        Fraction of B-errors corrected by C, per disease.
    symbolic_contribution_index:
        Composite score quantifying symbolic reasoning contribution.
        = (recoveries - regressions) / total_test_size.
    """

    records:                    list[RecoveryRecord] = field(default_factory=list)
    n_b_correct:                int = 0
    n_c_correct:                int = 0
    n_recoveries:               int = 0
    n_regressions:              int = 0
    n_both_wrong:               int = 0
    recovery_rate:              float = 0.0
    net_symbolic_gain:          int = 0
    mechanism_stats:            list[MechanismStats] = field(default_factory=list)
    per_disease_recovery_rate:  dict[str, float] = field(default_factory=dict)
    symbolic_contribution_index: float = 0.0

    def summary(self) -> str:
        n_total = len(self.records)
        lines = [
            "=" * 72,
            "SYMBOLIC RECOVERY ANALYSIS",
            "=" * 72,
            f"  Test records          : {n_total}",
            f"  Model B correct       : {self.n_b_correct} ({self.n_b_correct/max(n_total,1):.1%})",
            f"  Model C correct       : {self.n_c_correct} ({self.n_c_correct/max(n_total,1):.1%})",
            f"  Recoveries (B-wrong, C-right) : {self.n_recoveries}",
            f"  Regressions (B-right, C-wrong): {self.n_regressions}",
            f"  Both wrong            : {self.n_both_wrong}",
            f"  Recovery rate         : {self.recovery_rate:.1%}",
            f"  Net symbolic gain     : {self.net_symbolic_gain:+d}",
            f"  Symbolic contribution index: {self.symbolic_contribution_index:+.4f}",
            "-" * 72,
            "  RECOVERY MECHANISM BREAKDOWN:",
        ]
        for ms in sorted(
            self.mechanism_stats, key=lambda x: -x.count
        ):
            if ms.mechanism == RecoveryMechanism.NO_RECOVERY:
                continue
            if ms.mechanism == RecoveryMechanism.BOTH_CORRECT:
                continue
            lines.append(
                f"    {ms.mechanism.value:30s} "
                f"{ms.count:3d} ({ms.fraction:.1%} of recoveries)"
            )
        lines += [
            "-" * 72,
            "  PER-DISEASE RECOVERY RATE:",
        ]
        for dis, rate in sorted(
            self.per_disease_recovery_rate.items(), key=lambda x: -x[1]
        ):
            lines.append(f"    {dis:35s} {rate:.1%}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Analyser ──────────────────────────────────────────────────────────────────

class SymbolicRecoveryAnalyzer:
    """
    Attributes symbolic recovery mechanisms to Model C improvements over B.

    Parameters
    ----------
    class_labels:
        Ordered canonical disease names.
    """

    def __init__(self, class_labels: list[str]) -> None:
        self.class_labels = class_labels

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        symbolic_vectors: list[SymbolicFeatureVector],
        y_pred_model_b:   np.ndarray,
        y_pred_model_c:   np.ndarray,
        y_true:           np.ndarray,
    ) -> RecoveryReport:
        """
        Run recovery attribution analysis.

        Parameters
        ----------
        symbolic_vectors:
            Test-set symbolic feature vectors (one per record).
        y_pred_model_b:
            Model B predicted labels (0-based).
        y_pred_model_c:
            Model C predicted labels (0-based).
        y_true:
            True labels (0-based).
        """
        if not symbolic_vectors:
            return RecoveryReport()

        records: list[RecoveryRecord] = []
        for i, v in enumerate(symbolic_vectors):
            pred_b  = int(y_pred_model_b[i])
            pred_c  = int(y_pred_model_c[i])
            true_y  = int(y_true[i])
            b_ok    = pred_b == true_y
            c_ok    = pred_c == true_y
            recover = not b_ok and c_ok
            regress = b_ok and not c_ok
            mech, signals = self._attribute_mechanism(v, recover, true_y)

            true_dis = (
                self.class_labels[true_y]
                if 0 <= true_y < len(self.class_labels) else str(true_y)
            )
            records.append(RecoveryRecord(
                patient_id=v.patient_id,
                true_disease=true_dis,
                true_class_int=true_y,
                pred_model_b=pred_b,
                pred_model_c=pred_c,
                b_correct=b_ok,
                c_correct=c_ok,
                is_recovery=recover,
                is_regression=regress,
                recovery_mechanism=mech,
                mechanism_signals=signals,
            ))

        return self._build_report(records, y_pred_model_b, y_pred_model_c, y_true)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _attribute_mechanism(
        self,
        v:        SymbolicFeatureVector,
        recover:  bool,
        true_cls: int,
    ) -> tuple[RecoveryMechanism, dict[str, float]]:
        """Attribute the primary recovery mechanism for a single record."""
        if not recover:
            if v.requires_biopsy:
                return RecoveryMechanism.NO_RECOVERY, {}
            return RecoveryMechanism.NO_RECOVERY, {}

        signals: dict[str, float] = {
            "contradiction_load":  v.contradiction_load,
            "ambiguity_index":     v.ambiguity_index,
            "oscillation_count":   float(v.oscillation_count),
            "certainty_gap":       v.certainty_gap,
            "leading_disease_enc": float(v.leading_disease_encoded),
            "requires_biopsy":     float(v.requires_biopsy),
        }

        # Priority order: most specific signal drives attribution
        if v.contradiction_load >= _CONTR_RECOVERY_THRESHOLD:
            return RecoveryMechanism.CONTRADICTION_RECOVERY, signals

        if v.leading_disease_encoded == true_cls:
            return RecoveryMechanism.LEADERSHIP_RECOVERY, signals

        if v.ambiguity_index >= _AMBIGUITY_RECOVERY_THRESHOLD:
            return RecoveryMechanism.AMBIGUITY_RECOVERY, signals

        if v.oscillation_count >= _OSCILLATION_RECOVERY_THRESHOLD:
            return RecoveryMechanism.TRAJECTORY_RECOVERY, signals

        if v.certainty_gap >= _GAP_RECOVERY_THRESHOLD:
            return RecoveryMechanism.COMPETITION_RECOVERY, signals

        if v.requires_biopsy:
            return RecoveryMechanism.ESCALATION_RECOVERY, signals

        return RecoveryMechanism.UNEXPLAINED_RECOVERY, signals

    def _build_report(
        self,
        records:    list[RecoveryRecord],
        y_pred_b:   np.ndarray,
        y_pred_c:   np.ndarray,
        y_true:     np.ndarray,
    ) -> RecoveryReport:
        """Aggregate all records into a report."""
        n_total    = len(records)
        n_b_ok     = sum(1 for r in records if r.b_correct)
        n_c_ok     = sum(1 for r in records if r.c_correct)
        n_recover  = sum(1 for r in records if r.is_recovery)
        n_regress  = sum(1 for r in records if r.is_regression)
        n_both_wr  = sum(1 for r in records if not r.b_correct and not r.c_correct)
        n_b_errors = n_total - n_b_ok
        rec_rate   = n_recover / max(n_b_errors, 1)
        sci        = (n_recover - n_regress) / max(n_total, 1)

        # Mechanism stats
        mech_stats: list[MechanismStats] = []
        for mech in RecoveryMechanism:
            mech_recs = [r for r in records if r.recovery_mechanism == mech]
            per_dis: dict[str, int] = {}
            for r in mech_recs:
                per_dis[r.true_disease] = per_dis.get(r.true_disease, 0) + 1

            # Mean signals for recoveries in this mechanism
            sig_vals: dict[str, list[float]] = {}
            for r in mech_recs:
                for sig, val in r.mechanism_signals.items():
                    sig_vals.setdefault(sig, []).append(val)
            mean_sigs = {sig: float(np.mean(vals)) for sig, vals in sig_vals.items()}

            mech_stats.append(MechanismStats(
                mechanism=mech,
                count=len(mech_recs),
                fraction=len(mech_recs) / max(n_recover, 1),
                per_disease=per_dis,
                mean_signals=mean_sigs,
            ))

        # Per-disease recovery rate
        diseases = list({r.true_disease for r in records})
        per_dis_rate: dict[str, float] = {}
        for dis in diseases:
            dis_recs  = [r for r in records if r.true_disease == dis]
            dis_b_err = sum(1 for r in dis_recs if not r.b_correct)
            dis_rec   = sum(1 for r in dis_recs if r.is_recovery)
            per_dis_rate[dis] = dis_rec / max(dis_b_err, 1)

        return RecoveryReport(
            records=records,
            n_b_correct=n_b_ok,
            n_c_correct=n_c_ok,
            n_recoveries=n_recover,
            n_regressions=n_regress,
            n_both_wrong=n_both_wr,
            recovery_rate=rec_rate,
            net_symbolic_gain=n_recover - n_regress,
            mechanism_stats=mech_stats,
            per_disease_recovery_rate=per_dis_rate,
            symbolic_contribution_index=sci,
        )
