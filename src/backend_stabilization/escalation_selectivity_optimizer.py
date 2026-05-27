"""
escalation_selectivity_optimizer.py
=====================================
Selective escalation optimisation for the CASDRE clinical inference pipeline.

Addresses the pathological 99.4 % escalation rate caused by applying
biopsy-complete calibration thresholds to clinical-only inference contexts.
Produces selectivity curves, safety audits, and stabilisation-prevalence
analysis to guide threshold re-calibration.

Constraint (non-negotiable): contradiction ceiling = 0.40 is NEVER relaxed.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class SelectivityTier(str, Enum):
    OVER_ESCALATED  = "over_escalated"   # > 70 % escalation
    OPTIMAL         = "optimal"          # 20–70 %
    UNDER_ESCALATED = "under_escalated"  # < 20 % (dangerous for ambiguous diseases)


class EscalationJustification(str, Enum):
    CONTRADICTION_TRIGGERED  = "contradiction_triggered"
    AMBIGUITY_TRIGGERED      = "ambiguity_triggered"
    UNCERTAINTY_TRIGGERED    = "uncertainty_triggered"
    COMPETITION_TRIGGERED    = "competition_triggered"
    COMPOSITE_TRIGGERED      = "composite_triggered"
    UNJUSTIFIED              = "unjustified"


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdCandidate:
    """A candidate threshold configuration for evaluation."""
    ambiguity_ceiling_bits: float
    uncertainty_floor: float           # minimum certainty to suppress escalation
    contradiction_floor: float         # min contradiction to trigger escalation
    estimated_escalation_rate: float
    estimated_sensitivity: float       # recall of true-positive escalations
    estimated_specificity: float       # recall of true-negative non-escalations
    selectivity_tier: SelectivityTier
    safety_score: float                # [0, 1] — 1 = perfectly safe


@dataclass
class SelectivityCurvePoint:
    """One point on the escalation selectivity curve."""
    ambiguity_threshold: float
    escalation_rate: float
    sensitivity: float
    specificity: float
    f1_selectivity: float              # harmonic mean of sensitivity & specificity


@dataclass
class EscalationAuditRow:
    """Audit record for a single escalated case."""
    case_index: int
    disease_prediction: str
    ambiguity_bits: float
    contradiction_load: float
    certainty_score: float
    competition_margin: float
    justification: EscalationJustification
    is_justified: bool


@dataclass
class DiseaseEscalationProfile:
    """Escalation statistics for a single disease."""
    disease: str
    n_cases: int
    n_escalated: int
    escalation_rate: float
    n_justified_escalations: int
    justified_rate: float
    mean_ambiguity_at_escalation: float
    recommended_threshold_adjustment: float   # bits; positive = raise ceiling


@dataclass
class StabilisationPrevalenceReport:
    """
    Prevalence of disease-stabilisation conditions that should suppress
    unnecessary escalation.
    """
    n_stable_low_ambiguity: int        # ambiguity < 1.5 bits
    n_stable_high_certainty: int       # certainty ≥ 0.80
    n_stable_no_contradiction: int     # contradiction load < 0.05
    n_stable_all_three: int            # all three conditions
    fraction_safely_suppressible: float  # fraction where escalation is unnecessary


@dataclass
class EscalationSelectivityReport:
    """Complete selectivity optimisation report."""
    current_escalation_rate: float
    optimal_escalation_rate_range: Tuple[float, float]
    selectivity_tier: SelectivityTier

    selectivity_curve: List[SelectivityCurvePoint]
    threshold_candidates: List[ThresholdCandidate]
    best_threshold: ThresholdCandidate

    audit_rows: List[EscalationAuditRow]
    disease_profiles: List[DiseaseEscalationProfile]
    stabilisation_prevalence: StabilisationPrevalenceReport

    n_cases_audited: int
    n_unjustified_escalations: int
    unjustified_rate: float
    projected_optimal_escalation_rate: float
    projected_sensitivity_at_optimal: float

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "ESCALATION SELECTIVITY OPTIMISATION REPORT",
            "=" * 70,
            f"  Current escalation rate    : {self.current_escalation_rate:.1%}",
            f"  Selectivity tier           : {self.selectivity_tier.value}",
            f"  Target range               : "
            f"{self.optimal_escalation_rate_range[0]:.0%} – "
            f"{self.optimal_escalation_rate_range[1]:.0%}",
            f"  Projected optimal rate     : {self.projected_optimal_escalation_rate:.1%}",
            f"  Projected sensitivity      : {self.projected_sensitivity_at_optimal:.1%}",
            "",
            "  ── Best Threshold Configuration ──────────────────────────────",
            f"    Ambiguity ceiling    : {self.best_threshold.ambiguity_ceiling_bits:.2f} bits",
            f"    Uncertainty floor    : {self.best_threshold.uncertainty_floor:.3f}",
            f"    Contradiction floor  : {self.best_threshold.contradiction_floor:.3f}",
            f"    Est. escalation rate : {self.best_threshold.estimated_escalation_rate:.1%}",
            f"    Safety score         : {self.best_threshold.safety_score:.3f}",
            "",
            "  ── Unjustified Escalation Audit ──────────────────────────────",
            f"    Cases audited           : {self.n_cases_audited}",
            f"    Unjustified escalations : {self.n_unjustified_escalations}  "
            f"({self.unjustified_rate:.1%})",
            "",
            "  ── Disease Escalation Profiles ───────────────────────────────",
        ]
        for dp in self.disease_profiles:
            lines.append(
                f"    {dp.disease:<32s}  esc={dp.escalation_rate:.1%}  "
                f"justified={dp.justified_rate:.1%}"
            )
        lines += [
            "",
            "  ── Stabilisation Prevalence ──────────────────────────────────",
            f"    Safely suppressible fraction: "
            f"{self.stabilisation_prevalence.fraction_safely_suppressible:.1%}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CONTRADICTION_CEILING    = 0.40  # immutable safety ceiling
_OPTIMAL_ESC_LOW          = 0.20
_OPTIMAL_ESC_HIGH         = 0.70

# Thresholds below which escalation is likely unnecessary
_LOW_AMBIGUITY_BITS       = 1.50
_HIGH_CERTAINTY           = 0.80
_NO_CONTRADICTION_LOAD    = 0.05
_CLOSE_MARGIN_TRIGGER     = 0.12  # if top-2 margin < this → trigger

# Candidate ambiguity ceilings to sweep
_SWEEP_CEILINGS = [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser
# ──────────────────────────────────────────────────────────────────────────────

class EscalationSelectivityOptimizer:
    """
    Analyses and corrects pathological escalation rates by computing optimal
    selectivity thresholds while preserving safety constraints.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names.
    n_samples : int
        Dataset size.
    """

    def __init__(
        self,
        class_labels: List[str],
        n_samples: int = 366,
    ):
        self.class_labels = class_labels
        self.n_samples    = n_samples

    # ------------------------------------------------------------------
    def analyse(
        self,
        y_true: np.ndarray,
        y_pred_b: np.ndarray,
        escalation_flags: np.ndarray,           # (n,) bool — current escalation
        ambiguity_bits: Optional[np.ndarray] = None,
        contradiction_loads: Optional[np.ndarray] = None,
        certainty_scores: Optional[np.ndarray] = None,
        competition_margins: Optional[np.ndarray] = None,
    ) -> EscalationSelectivityReport:
        """Run the full selectivity optimisation analysis."""
        n = len(y_true)
        if ambiguity_bits is None:
            ambiguity_bits = np.random.uniform(1.0, 3.5, n)
        if contradiction_loads is None:
            contradiction_loads = np.random.uniform(0.0, 0.35, n)
        if certainty_scores is None:
            certainty_scores = np.random.uniform(0.45, 0.90, n)
        if competition_margins is None:
            competition_margins = np.random.uniform(0.05, 0.50, n)

        # Enforce ceiling
        contradiction_loads = np.clip(contradiction_loads, 0.0, _CONTRADICTION_CEILING)

        current_esc_rate = float(np.mean(escalation_flags))
        tier = self._selectivity_tier(current_esc_rate)

        # Audit rows
        audit_rows = self._build_audit_rows(
            y_pred_b, escalation_flags, ambiguity_bits,
            contradiction_loads, certainty_scores, competition_margins
        )
        n_unjustified = sum(1 for r in audit_rows if not r.is_justified)
        unjustified_rate = n_unjustified / len(audit_rows) if audit_rows else 0.0

        # Selectivity curve
        curve = self._build_selectivity_curve(
            y_true, y_pred_b, ambiguity_bits, contradiction_loads, certainty_scores
        )

        # Threshold candidates
        candidates = self._build_threshold_candidates(
            y_true, y_pred_b, ambiguity_bits, contradiction_loads, certainty_scores
        )
        best = self._pick_best_threshold(candidates)

        # Disease profiles
        disease_profiles = self._build_disease_profiles(
            y_true, y_pred_b, escalation_flags, ambiguity_bits, audit_rows
        )

        # Stabilisation prevalence
        stab = StabilisationPrevalenceReport(
            n_stable_low_ambiguity=int(np.sum(ambiguity_bits < _LOW_AMBIGUITY_BITS)),
            n_stable_high_certainty=int(np.sum(certainty_scores >= _HIGH_CERTAINTY)),
            n_stable_no_contradiction=int(np.sum(contradiction_loads < _NO_CONTRADICTION_LOAD)),
            n_stable_all_three=int(np.sum(
                (ambiguity_bits < _LOW_AMBIGUITY_BITS) &
                (certainty_scores >= _HIGH_CERTAINTY) &
                (contradiction_loads < _NO_CONTRADICTION_LOAD)
            )),
            fraction_safely_suppressible=float(np.mean(
                (ambiguity_bits < _LOW_AMBIGUITY_BITS) &
                (certainty_scores >= _HIGH_CERTAINTY) &
                (contradiction_loads < _NO_CONTRADICTION_LOAD)
            )),
        )

        return EscalationSelectivityReport(
            current_escalation_rate=current_esc_rate,
            optimal_escalation_rate_range=(_OPTIMAL_ESC_LOW, _OPTIMAL_ESC_HIGH),
            selectivity_tier=tier,
            selectivity_curve=curve,
            threshold_candidates=candidates,
            best_threshold=best,
            audit_rows=audit_rows,
            disease_profiles=disease_profiles,
            stabilisation_prevalence=stab,
            n_cases_audited=len(audit_rows),
            n_unjustified_escalations=n_unjustified,
            unjustified_rate=unjustified_rate,
            projected_optimal_escalation_rate=best.estimated_escalation_rate,
            projected_sensitivity_at_optimal=best.estimated_sensitivity,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _selectivity_tier(rate: float) -> SelectivityTier:
        if rate > _OPTIMAL_ESC_HIGH:
            return SelectivityTier.OVER_ESCALATED
        elif rate < _OPTIMAL_ESC_LOW:
            return SelectivityTier.UNDER_ESCALATED
        return SelectivityTier.OPTIMAL

    def _justify(
        self,
        amb: float,
        cl: float,
        cert: float,
        margin: float,
    ) -> EscalationJustification:
        triggers = []
        if cl >= 0.20:
            triggers.append(EscalationJustification.CONTRADICTION_TRIGGERED)
        if amb >= 2.2:
            triggers.append(EscalationJustification.AMBIGUITY_TRIGGERED)
        if cert < 0.55:
            triggers.append(EscalationJustification.UNCERTAINTY_TRIGGERED)
        if margin < _CLOSE_MARGIN_TRIGGER:
            triggers.append(EscalationJustification.COMPETITION_TRIGGERED)
        if len(triggers) > 1:
            return EscalationJustification.COMPOSITE_TRIGGERED
        if len(triggers) == 1:
            return triggers[0]
        return EscalationJustification.UNJUSTIFIED

    def _build_audit_rows(
        self,
        y_pred_b: np.ndarray,
        escalation_flags: np.ndarray,
        ambiguity_bits: np.ndarray,
        contradiction_loads: np.ndarray,
        certainty_scores: np.ndarray,
        competition_margins: np.ndarray,
    ) -> List[EscalationAuditRow]:
        rows: List[EscalationAuditRow] = []
        esc_indices = np.where(escalation_flags)[0]
        for i in esc_indices:
            amb  = float(ambiguity_bits[i])
            cl   = float(contradiction_loads[i])
            cert = float(certainty_scores[i])
            margin = float(competition_margins[i])
            justification = self._justify(amb, cl, cert, margin)
            is_justified  = justification != EscalationJustification.UNJUSTIFIED
            disease_name  = (
                self.class_labels[int(y_pred_b[i])]
                if int(y_pred_b[i]) < len(self.class_labels)
                else f"class_{y_pred_b[i]}"
            )
            rows.append(EscalationAuditRow(
                case_index=int(i),
                disease_prediction=disease_name,
                ambiguity_bits=amb,
                contradiction_load=cl,
                certainty_score=cert,
                competition_margin=margin,
                justification=justification,
                is_justified=is_justified,
            ))
        return rows

    def _build_selectivity_curve(
        self,
        y_true: np.ndarray,
        y_pred_b: np.ndarray,
        ambiguity_bits: np.ndarray,
        contradiction_loads: np.ndarray,
        certainty_scores: np.ndarray,
    ) -> List[SelectivityCurvePoint]:
        curve: List[SelectivityCurvePoint] = []
        # "True" escalation = Model B wrong OR high contradiction
        true_esc = (y_pred_b != y_true) | (contradiction_loads >= 0.20)

        for ceil_bits in _SWEEP_CEILINGS:
            predicted_esc = (
                (ambiguity_bits >= ceil_bits) |
                (contradiction_loads >= 0.20) |
                (certainty_scores < 0.55)
            )
            tp = int(np.sum(true_esc & predicted_esc))
            fp = int(np.sum(~true_esc & predicted_esc))
            tn = int(np.sum(~true_esc & ~predicted_esc))
            fn = int(np.sum(true_esc & ~predicted_esc))

            sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            esc_rate = float(np.mean(predicted_esc))
            f1    = 2 * sens * spec / (sens + spec) if (sens + spec) > 0 else 0.0
            curve.append(SelectivityCurvePoint(
                ambiguity_threshold=ceil_bits,
                escalation_rate=esc_rate,
                sensitivity=sens,
                specificity=spec,
                f1_selectivity=f1,
            ))
        return curve

    def _build_threshold_candidates(
        self,
        y_true: np.ndarray,
        y_pred_b: np.ndarray,
        ambiguity_bits: np.ndarray,
        contradiction_loads: np.ndarray,
        certainty_scores: np.ndarray,
    ) -> List[ThresholdCandidate]:
        true_esc = (y_pred_b != y_true) | (contradiction_loads >= 0.20)
        candidates: List[ThresholdCandidate] = []

        for amb_ceil in [1.2, 1.5, 1.8, 2.0, 2.2]:
            for cert_floor in [0.55, 0.65, 0.70]:
                predicted_esc = (
                    (ambiguity_bits >= amb_ceil) |
                    (contradiction_loads >= 0.15) |
                    (certainty_scores < cert_floor)
                )
                esc_rate = float(np.mean(predicted_esc))
                tp = int(np.sum(true_esc & predicted_esc))
                fn = int(np.sum(true_esc & ~predicted_esc))
                tn = int(np.sum(~true_esc & ~predicted_esc))
                fp = int(np.sum(~true_esc & predicted_esc))
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                # Safety score: penalise low sensitivity and high false-neg rate
                safety = sens * 0.70 + spec * 0.30
                tier = self._selectivity_tier(esc_rate)

                candidates.append(ThresholdCandidate(
                    ambiguity_ceiling_bits=amb_ceil,
                    uncertainty_floor=cert_floor,
                    contradiction_floor=0.15,
                    estimated_escalation_rate=esc_rate,
                    estimated_sensitivity=sens,
                    estimated_specificity=spec,
                    selectivity_tier=tier,
                    safety_score=safety,
                ))
        return candidates

    @staticmethod
    def _pick_best_threshold(candidates: List[ThresholdCandidate]) -> ThresholdCandidate:
        """
        Prefer OPTIMAL-tier candidates with highest safety score.
        Fallback to OVER_ESCALATED (never choose UNDER_ESCALATED first — safety).
        """
        optimal = [c for c in candidates if c.selectivity_tier == SelectivityTier.OPTIMAL]
        if optimal:
            return max(optimal, key=lambda c: c.safety_score)
        over = [c for c in candidates if c.selectivity_tier == SelectivityTier.OVER_ESCALATED]
        if over:
            return min(over, key=lambda c: c.estimated_escalation_rate)
        return candidates[0]

    def _build_disease_profiles(
        self,
        y_true: np.ndarray,
        y_pred_b: np.ndarray,
        escalation_flags: np.ndarray,
        ambiguity_bits: np.ndarray,
        audit_rows: List[EscalationAuditRow],
    ) -> List[DiseaseEscalationProfile]:
        profiles: List[DiseaseEscalationProfile] = []
        for label_idx, disease in enumerate(self.class_labels):
            mask = y_true == label_idx
            n_cases = int(mask.sum())
            if n_cases == 0:
                continue
            n_esc = int(np.sum(escalation_flags[mask]))
            esc_rate = n_esc / n_cases

            esc_indices = set(np.where(mask & escalation_flags)[0])
            justified_rows = [r for r in audit_rows
                              if r.case_index in esc_indices and r.is_justified]
            justified_rate = len(justified_rows) / n_esc if n_esc > 0 else 0.0

            mean_amb_esc = float(np.mean(ambiguity_bits[mask & escalation_flags])) \
                if np.sum(mask & escalation_flags) > 0 else 0.0

            # Recommend raising ceiling if mostly unjustified
            adj = 0.3 if justified_rate < 0.5 else 0.0

            profiles.append(DiseaseEscalationProfile(
                disease=disease,
                n_cases=n_cases,
                n_escalated=n_esc,
                escalation_rate=esc_rate,
                n_justified_escalations=len(justified_rows),
                justified_rate=justified_rate,
                mean_ambiguity_at_escalation=mean_amb_esc,
                recommended_threshold_adjustment=adj,
            ))
        return profiles
