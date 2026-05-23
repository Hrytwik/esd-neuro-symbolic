"""
ReasoningMetrics — aggregated symbolic reasoning quality metrics.

Consolidates per-case symbolic signals into dataset-level clinical reasoning
quality indicators. These metrics characterise how the symbolic reasoning
system behaves across the full patient cohort — independently of whether
the terminal classification label is correct.

Reasoning quality is measured on dimensions that are purely the responsibility
of the symbolic system (not the downstream classifier):
  · Certainty calibration: does certainty correlate with diagnostic difficulty?
  · Escalation appropriateness: is biopsy recommendation clinically warranted?
  · Contradiction sensitivity: does contradiction load reflect real confusion zones?
  · Trajectory stability: does reasoning converge reliably?
  · Sufficiency detection: does the system know when it knows enough?

These metrics support two audiences:
  1. Clinical evaluation: is the system behaviourally safe?
  2. Publication reporting: what is the measurable symbolic contribution?

Usage
-----
  from src.evaluation_pipeline.reasoning_metrics import ReasoningMetricsAggregator

  metrics = ReasoningMetricsAggregator.aggregate(all_vectors, disease_labels)
  print(metrics.summary())
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Per-case reasoning metrics ────────────────────────────────────────────────

@dataclass(frozen=True)
class CaseReasoningMetrics:
    """
    Symbolic reasoning quality indicators for a single patient case.

    Attributes
    ----------
    patient_id:
        Source patient identifier.
    disease_label:
        Ground-truth disease label.
    certainty:
        Terminal leading hypothesis certainty.
    certainty_gap:
        Terminal certainty gap.
    contradiction_load:
        Terminal bilateral contradiction load.
    ambiguity_index:
        Terminal Shannon entropy (bits).
    normalised_entropy:
        Entropy normalised by log2(6) → [0, 1].
    certainty_sufficiency:
        1.0 if certainty ≥ 0.55 and gap ≥ 0.20; else 0.0.
    convergence_index:
        Final / peak certainty.
    oscillation_count:
        Number of trajectory oscillations.
    was_dampened:
        Whether certainty was suppressed by contradiction.
    requires_biopsy:
        Whether biopsy was recommended.
    is_safe_triage:
        Whether safe non-invasive triage was recommended.
    recommendation:
        Terminal triage recommendation.
    reasoning_is_certain:
        True if certainty ≥ 0.65 and gap ≥ 0.20 (strong differential).
    reasoning_is_ambiguous:
        True if entropy > 1.50 bits.
    reasoning_is_contradicted:
        True if contradiction_load > 0.0.
    """

    patient_id:               str
    disease_label:            str
    certainty:                float
    certainty_gap:            float
    contradiction_load:       float
    ambiguity_index:          float
    normalised_entropy:       float
    certainty_sufficiency:    float
    convergence_index:        float
    oscillation_count:        int
    was_dampened:             bool
    requires_biopsy:          bool
    is_safe_triage:           bool
    recommendation:           str
    reasoning_is_certain:     bool
    reasoning_is_ambiguous:   bool
    reasoning_is_contradicted: bool


# ── Aggregated reasoning metrics ──────────────────────────────────────────────

@dataclass
class AggregatedReasoningMetrics:
    """
    Dataset-level symbolic reasoning quality summary.

    Attributes
    ----------
    total_cases:
        Total patients evaluated.
    certainty_metrics:
        Certainty summary statistics.
    gap_metrics:
        Certainty gap summary statistics.
    contradiction_metrics:
        Contradiction load summary statistics.
    entropy_metrics:
        Shannon entropy summary statistics.
    escalation_rate:
        Fraction of cases receiving biopsy recommendation.
    safe_triage_rate:
        Fraction of cases receiving safe non-invasive triage.
    certain_reasoning_rate:
        Fraction of cases meeting strong certainty criteria.
    ambiguous_reasoning_rate:
        Fraction of cases with entropy > 1.50 bits.
    contradiction_prevalence:
        Fraction of cases with any contradiction.
    mean_convergence_index:
        Mean trajectory convergence across all cases.
    mean_oscillation_count:
        Mean oscillation count across all cases.
    dampened_case_rate:
        Fraction of cases where dampening was active.
    per_disease_certainty:
        Mean certainty per disease.
    per_disease_contradiction:
        Mean contradiction load per disease.
    per_disease_safe_rate:
        Fraction of safe triage recommendations per disease.
    per_disease_biopsy_rate:
        Fraction of biopsy recommendations per disease.
    sufficiency_rate:
        Fraction of cases meeting certainty sufficiency criterion.
    case_metrics:
        All per-case CaseReasoningMetrics.
    """

    total_cases:               int
    certainty_metrics:         dict[str, float] = field(default_factory=dict)
    gap_metrics:               dict[str, float] = field(default_factory=dict)
    contradiction_metrics:     dict[str, float] = field(default_factory=dict)
    entropy_metrics:           dict[str, float] = field(default_factory=dict)
    escalation_rate:           float = 0.0
    safe_triage_rate:          float = 0.0
    certain_reasoning_rate:    float = 0.0
    ambiguous_reasoning_rate:  float = 0.0
    contradiction_prevalence:  float = 0.0
    mean_convergence_index:    float = 0.0
    mean_oscillation_count:    float = 0.0
    dampened_case_rate:        float = 0.0
    per_disease_certainty:     dict[str, float] = field(default_factory=dict)
    per_disease_contradiction: dict[str, float] = field(default_factory=dict)
    per_disease_safe_rate:     dict[str, float] = field(default_factory=dict)
    per_disease_biopsy_rate:   dict[str, float] = field(default_factory=dict)
    sufficiency_rate:          float = 0.0
    case_metrics:              list[CaseReasoningMetrics] = field(default_factory=list)

    def summary(self) -> str:
        n = max(self.total_cases, 1)
        c = self.certainty_metrics
        lines = [
            "REASONING QUALITY METRICS",
            f"  Total cases             : {self.total_cases}",
            f"  Certainty (mean/std)    : "
            f"{c.get('mean', 0):.4f} / {c.get('std', 0):.4f}",
            f"  Certainty (min/max)     : "
            f"{c.get('min', 0):.4f} / {c.get('max', 0):.4f}",
            f"  Safe triage rate        : {self.safe_triage_rate:.1%}",
            f"  Escalation rate         : {self.escalation_rate:.1%}",
            f"  Sufficiency rate        : {self.sufficiency_rate:.1%}",
            f"  Certain reasoning       : {self.certain_reasoning_rate:.1%}",
            f"  Ambiguous reasoning     : {self.ambiguous_reasoning_rate:.1%}",
            f"  Contradiction prevalence: {self.contradiction_prevalence:.1%}",
            f"  Dampened cases          : {self.dampened_case_rate:.1%}",
            f"  Mean convergence index  : {self.mean_convergence_index:.4f}",
            f"  Mean oscillations       : {self.mean_oscillation_count:.2f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_cases":              self.total_cases,
            "certainty":                self.certainty_metrics,
            "gap":                      self.gap_metrics,
            "contradiction":            self.contradiction_metrics,
            "entropy":                  self.entropy_metrics,
            "escalation_rate":          self.escalation_rate,
            "safe_triage_rate":         self.safe_triage_rate,
            "certain_reasoning_rate":   self.certain_reasoning_rate,
            "ambiguous_reasoning_rate": self.ambiguous_reasoning_rate,
            "contradiction_prevalence": self.contradiction_prevalence,
            "mean_convergence_index":   self.mean_convergence_index,
            "mean_oscillation_count":   self.mean_oscillation_count,
            "dampened_case_rate":       self.dampened_case_rate,
            "per_disease_certainty":    self.per_disease_certainty,
            "per_disease_contradiction": self.per_disease_contradiction,
            "per_disease_safe_rate":    self.per_disease_safe_rate,
            "per_disease_biopsy_rate":  self.per_disease_biopsy_rate,
            "sufficiency_rate":         self.sufficiency_rate,
        }


# ── Aggregator ────────────────────────────────────────────────────────────────

class ReasoningMetricsAggregator:
    """
    Stateless aggregator: takes a list of SymbolicFeatureVectors and
    returns an AggregatedReasoningMetrics summary.
    """

    # Certainty threshold for "reasoning is certain" criterion
    _CERTAIN_CERTAINTY_FLOOR: float = 0.65
    _CERTAIN_GAP_FLOOR:       float = 0.20
    _AMBIGUITY_BITS:          float = 1.50

    @classmethod
    def aggregate(
        cls,
        vectors: list[SymbolicFeatureVector],
        disease_labels: list[str] | None = None,
    ) -> AggregatedReasoningMetrics:
        """
        Compute dataset-level reasoning quality metrics.

        Parameters
        ----------
        vectors:
            Symbolic reasoning outputs, one per patient.
        disease_labels:
            Ground-truth disease labels (optional override).
        """
        labels      = disease_labels or [v.disease_label for v in vectors]
        case_mets   = [
            cls._build_case_metrics(v, lbl)
            for v, lbl in zip(vectors, labels)
        ]
        return cls._aggregate_cases(case_mets)

    @classmethod
    def _build_case_metrics(
        cls,
        v: SymbolicFeatureVector,
        disease_label: str,
    ) -> CaseReasoningMetrics:
        return CaseReasoningMetrics(
            patient_id=v.patient_id,
            disease_label=disease_label,
            certainty=v.certainty,
            certainty_gap=v.certainty_gap,
            contradiction_load=v.contradiction_load,
            ambiguity_index=v.ambiguity_index,
            normalised_entropy=v.normalised_entropy,
            certainty_sufficiency=v.certainty_sufficiency,
            convergence_index=v.convergence_index,
            oscillation_count=v.oscillation_count,
            was_dampened=v.was_dampened,
            requires_biopsy=v.requires_biopsy,
            is_safe_triage=v.is_safe_triage,
            recommendation=v.recommendation,
            reasoning_is_certain=(
                v.certainty >= cls._CERTAIN_CERTAINTY_FLOOR
                and v.certainty_gap >= cls._CERTAIN_GAP_FLOOR
            ),
            reasoning_is_ambiguous=v.ambiguity_index > cls._AMBIGUITY_BITS,
            reasoning_is_contradicted=v.contradiction_load > 0.0,
        )

    @classmethod
    def _stats(cls, values: list[float]) -> dict[str, float]:
        """Return descriptive statistics dict for a list of floats."""
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        return {
            "mean":   statistics.mean(values),
            "std":    statistics.stdev(values) if len(values) > 1 else 0.0,
            "min":    min(values),
            "max":    max(values),
            "median": statistics.median(values),
        }

    @classmethod
    def _aggregate_cases(
        cls,
        case_mets: list[CaseReasoningMetrics],
    ) -> AggregatedReasoningMetrics:
        n = len(case_mets)
        if n == 0:
            return AggregatedReasoningMetrics(total_cases=0)

        by_disease: dict[str, list[CaseReasoningMetrics]] = {}
        for cm in case_mets:
            by_disease.setdefault(cm.disease_label, []).append(cm)

        per_cert:  dict[str, float] = {
            d: statistics.mean(c.certainty for c in dm)
            for d, dm in by_disease.items()
        }
        per_contra: dict[str, float] = {
            d: statistics.mean(c.contradiction_load for c in dm)
            for d, dm in by_disease.items()
        }
        per_safe: dict[str, float] = {
            d: sum(1 for c in dm if c.is_safe_triage) / len(dm)
            for d, dm in by_disease.items()
        }
        per_bx: dict[str, float] = {
            d: sum(1 for c in dm if c.requires_biopsy) / len(dm)
            for d, dm in by_disease.items()
        }

        return AggregatedReasoningMetrics(
            total_cases=n,
            certainty_metrics=cls._stats([c.certainty for c in case_mets]),
            gap_metrics=cls._stats([c.certainty_gap for c in case_mets]),
            contradiction_metrics=cls._stats([c.contradiction_load for c in case_mets]),
            entropy_metrics=cls._stats([c.ambiguity_index for c in case_mets]),
            escalation_rate=sum(1 for c in case_mets if c.requires_biopsy) / n,
            safe_triage_rate=sum(1 for c in case_mets if c.is_safe_triage) / n,
            certain_reasoning_rate=sum(1 for c in case_mets if c.reasoning_is_certain) / n,
            ambiguous_reasoning_rate=sum(1 for c in case_mets if c.reasoning_is_ambiguous) / n,
            contradiction_prevalence=sum(
                1 for c in case_mets if c.reasoning_is_contradicted
            ) / n,
            mean_convergence_index=statistics.mean(
                c.convergence_index for c in case_mets
            ),
            mean_oscillation_count=statistics.mean(
                float(c.oscillation_count) for c in case_mets
            ),
            dampened_case_rate=sum(1 for c in case_mets if c.was_dampened) / n,
            per_disease_certainty=per_cert,
            per_disease_contradiction=per_contra,
            per_disease_safe_rate=per_safe,
            per_disease_biopsy_rate=per_bx,
            sufficiency_rate=sum(
                c.certainty_sufficiency for c in case_mets
            ) / n,
            case_metrics=case_mets,
        )
