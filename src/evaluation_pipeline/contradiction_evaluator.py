"""
ContradictionEvaluator — contradiction prevalence and certainty decay analysis.

Measures how contradiction load distributes across the dataset and how it
affects diagnostic certainty. This is the quantitative basis for the claim
that the symbolic reasoning system detects clinically meaningful conflicts
between competing disease hypotheses.

Key clinical insights
---------------------
  · Disease pairs in known confusion zones (psoriasis ↔ PRP, etc.) should
    show systematically higher contradiction loads.
  · Contradiction load ≥ 0.40 (the escalation ceiling) should correlate with
    genuine diagnostic uncertainty (cases where even biopsy-equipped classifiers
    struggle).
  · Certainty should be lower (dampened) when contradiction load is elevated.

Contradiction severity tiers
-----------------------------
  NONE:     load = 0.0
  LOW:      0.0 < load < 0.15
  MODERATE: 0.15 ≤ load < 0.30
  HIGH:     0.30 ≤ load < 0.40
  CRITICAL: load ≥ 0.40 (mandatory biopsy escalation triggered)

Usage
-----
  from src.evaluation_pipeline.contradiction_evaluator import ContradictionEvaluator

  result = ContradictionEvaluator.evaluate_vectors(test_vectors)
  print(result.summary())
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Thresholds ────────────────────────────────────────────────────────────────

_CONTRADICTION_ESCALATION_CEILING: float = 0.40
_DAMPENING_THRESHOLD:              float = 0.20
_CERTAINTY_DAMPENING_THRESHOLD:    float = 0.85  # cert > this with high load is suspicious


# ── Contradiction profile ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContradictionProfile:
    """
    Contradiction analysis for a single patient.

    Attributes
    ----------
    patient_id:
        Source patient identifier.
    disease_label:
        Ground-truth disease label.
    contradiction_load:
        Bilateral contradiction load from terminal reasoning stage.
    certainty:
        Leading hypothesis certainty.
    ambiguity_index:
        Shannon entropy (bits).
    is_contradicted:
        True if contradiction_load > 0.
    is_dampened:
        True if contradiction_load > dampening threshold (0.20).
    is_critical:
        True if contradiction_load ≥ escalation ceiling (0.40).
    severity:
        String severity tier: "none" | "low" | "moderate" | "high" | "critical".
    recommendation:
        Terminal triage recommendation.
    contradiction_emerged:
        True if contradiction was observed at any trajectory stage.
    was_dampened:
        True if certainty dampening was active at any stage.
    """

    patient_id:             str
    disease_label:          str
    contradiction_load:     float
    certainty:              float
    ambiguity_index:        float
    is_contradicted:        bool
    is_dampened:            bool
    is_critical:            bool
    severity:               str
    recommendation:         str
    contradiction_emerged:  bool
    was_dampened:           bool

    @property
    def certainty_decay(self) -> float:
        """
        Proxy for certainty suppression: 1 - certainty when contradicted,
        0 when no contradiction. Higher = more certainty lost to contradiction.
        """
        if self.is_contradicted:
            return 1.0 - self.certainty
        return 0.0


def _severity_label(load: float) -> str:
    if load == 0.0:
        return "none"
    if load < 0.15:
        return "low"
    if load < 0.30:
        return "moderate"
    if load < _CONTRADICTION_ESCALATION_CEILING:
        return "high"
    return "critical"


# ── Contradiction evaluation result ───────────────────────────────────────────

@dataclass
class ContradictionEvaluationResult:
    """
    Dataset-level contradiction prevalence and impact analysis.

    Attributes
    ----------
    total_cases:
        Total number of evaluated patients.
    contradiction_cases:
        Patients with any contradiction (load > 0).
    critical_cases:
        Patients with contradiction load ≥ 0.40.
    dampened_cases:
        Patients with contradiction load ≥ 0.20 (dampening active).
    contradiction_prevalence:
        Fraction of cases with any contradiction.
    critical_prevalence:
        Fraction of cases at critical threshold.
    mean_contradiction_load:
        Mean contradiction load across all cases.
    std_contradiction_load:
        Standard deviation of contradiction load.
    mean_certainty_with_contradiction:
        Mean certainty for cases with any contradiction.
    mean_certainty_without_contradiction:
        Mean certainty for cases with zero contradiction.
    certainty_decay_under_contradiction:
        Mean certainty decay (1 - certainty) in contradicted cases.
    per_disease_contradiction_prevalence:
        Fraction of contradicted cases per disease.
    per_disease_mean_load:
        Mean contradiction load per disease.
    per_disease_critical_prevalence:
        Fraction of critical cases per disease.
    severity_distribution:
        Count of cases at each severity tier.
    escalation_under_contradiction_rate:
        Fraction of contradicted cases that received biopsy recommendation.
    profiles:
        All per-patient ContradictionProfile instances.
    """

    total_cases:                         int
    contradiction_cases:                 int   = 0
    critical_cases:                      int   = 0
    dampened_cases:                      int   = 0
    contradiction_prevalence:            float = 0.0
    critical_prevalence:                 float = 0.0
    mean_contradiction_load:             float = 0.0
    std_contradiction_load:              float = 0.0
    mean_certainty_with_contradiction:   float = 0.0
    mean_certainty_without_contradiction: float = 0.0
    certainty_decay_under_contradiction: float = 0.0
    per_disease_contradiction_prevalence: dict[str, float] = field(default_factory=dict)
    per_disease_mean_load:               dict[str, float] = field(default_factory=dict)
    per_disease_critical_prevalence:     dict[str, float] = field(default_factory=dict)
    severity_distribution:               dict[str, int]   = field(default_factory=dict)
    escalation_under_contradiction_rate: float = 0.0
    profiles:                            list[ContradictionProfile] = field(default_factory=list)

    def summary(self) -> str:
        n = max(self.total_cases, 1)
        lines = [
            "CONTRADICTION EVALUATION",
            f"  Total cases             : {self.total_cases}",
            f"  Contradicted            : {self.contradiction_cases} "
            f"({self.contradiction_prevalence:.1%})",
            f"  Critical (load >= 0.40) : {self.critical_cases} "
            f"({self.critical_prevalence:.1%})",
            f"  Dampened (load >= 0.20) : {self.dampened_cases} "
            f"({self.dampened_cases/n:.1%})",
            f"  Mean load               : {self.mean_contradiction_load:.4f} "
            f"(std={self.std_contradiction_load:.4f})",
            f"  Certainty (contra)      : {self.mean_certainty_with_contradiction:.4f}",
            f"  Certainty (no contra)   : {self.mean_certainty_without_contradiction:.4f}",
            f"  Certainty decay         : {self.certainty_decay_under_contradiction:.4f}",
            f"  Escalation under contra : {self.escalation_under_contradiction_rate:.1%}",
            "  Severity: " + " | ".join(
                f"{k}={v}" for k, v in self.severity_distribution.items()
            ),
        ]
        return "\n".join(lines)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class ContradictionEvaluator:
    """
    Stateless contradiction prevalence and certainty decay analyser.
    """

    @classmethod
    def evaluate_vectors(
        cls,
        vectors: list[SymbolicFeatureVector],
        disease_labels: list[str] | None = None,
    ) -> ContradictionEvaluationResult:
        """
        Analyse contradiction behaviour across all symbolic reasoning vectors.

        Parameters
        ----------
        vectors:
            Symbolic reasoning outputs, one per patient.
        disease_labels:
            Ground-truth labels (optional override).
        """
        labels   = disease_labels or [v.disease_label for v in vectors]
        profiles = [
            cls._build_profile(v, lbl)
            for v, lbl in zip(vectors, labels)
        ]
        return cls._aggregate(profiles)

    @classmethod
    def _build_profile(
        cls,
        v: SymbolicFeatureVector,
        disease_label: str,
    ) -> ContradictionProfile:
        load = v.contradiction_load
        return ContradictionProfile(
            patient_id=v.patient_id,
            disease_label=disease_label,
            contradiction_load=load,
            certainty=v.certainty,
            ambiguity_index=v.ambiguity_index,
            is_contradicted=load > 0.0,
            is_dampened=load >= _DAMPENING_THRESHOLD,
            is_critical=load >= _CONTRADICTION_ESCALATION_CEILING,
            severity=_severity_label(load),
            recommendation=v.recommendation,
            contradiction_emerged=v.contradiction_emerged,
            was_dampened=v.was_dampened,
        )

    @classmethod
    def _aggregate(
        cls,
        profiles: list[ContradictionProfile],
    ) -> ContradictionEvaluationResult:
        n = len(profiles)
        if n == 0:
            return ContradictionEvaluationResult(total_cases=0)

        loads  = [p.contradiction_load for p in profiles]
        contra = [p for p in profiles if p.is_contradicted]
        no_con = [p for p in profiles if not p.is_contradicted]

        mean_load = statistics.mean(loads)
        std_load  = statistics.stdev(loads) if len(loads) > 1 else 0.0

        cert_with  = (statistics.mean(p.certainty for p in contra)
                      if contra else 0.0)
        cert_witho = (statistics.mean(p.certainty for p in no_con)
                      if no_con else 0.0)
        decay = (statistics.mean(p.certainty_decay for p in contra)
                 if contra else 0.0)

        # Severity distribution
        sev_dist: dict[str, int] = {
            "none": 0, "low": 0, "moderate": 0, "high": 0, "critical": 0,
        }
        for p in profiles:
            sev_dist[p.severity] = sev_dist.get(p.severity, 0) + 1

        # Escalation under contradiction
        esc_under = (
            sum(1 for p in contra if p.recommendation in
                ("BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"))
            / max(len(contra), 1)
        )

        # Per-disease
        by_disease: dict[str, list[ContradictionProfile]] = {}
        for p in profiles:
            by_disease.setdefault(p.disease_label, []).append(p)

        per_prev:  dict[str, float] = {}
        per_load:  dict[str, float] = {}
        per_crit:  dict[str, float] = {}
        for dis, dp in by_disease.items():
            nd = len(dp)
            per_prev[dis] = sum(1 for p in dp if p.is_contradicted) / nd
            per_load[dis] = statistics.mean(p.contradiction_load for p in dp)
            per_crit[dis] = sum(1 for p in dp if p.is_critical) / nd

        return ContradictionEvaluationResult(
            total_cases=n,
            contradiction_cases=len(contra),
            critical_cases=sum(1 for p in profiles if p.is_critical),
            dampened_cases=sum(1 for p in profiles if p.is_dampened),
            contradiction_prevalence=len(contra) / n,
            critical_prevalence=sum(1 for p in profiles if p.is_critical) / n,
            mean_contradiction_load=mean_load,
            std_contradiction_load=std_load,
            mean_certainty_with_contradiction=cert_with,
            mean_certainty_without_contradiction=cert_witho,
            certainty_decay_under_contradiction=decay,
            per_disease_contradiction_prevalence=per_prev,
            per_disease_mean_load=per_load,
            per_disease_critical_prevalence=per_crit,
            severity_distribution=sev_dist,
            escalation_under_contradiction_rate=esc_under,
            profiles=profiles,
        )
