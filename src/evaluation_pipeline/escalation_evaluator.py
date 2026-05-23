"""
EscalationEvaluator — safe biopsy avoidance and escalation appropriateness.

The escalation evaluation is one of the strongest novelty claims of this
project. This module measures whether the symbolic reasoning system makes
CLINICALLY APPROPRIATE escalation decisions — not just whether it classifies
correctly.

Clinical safety framing
-----------------------
  SAFE_NON_INVASIVE_TRIAGE    — biopsy avoided; appropriate if certainty is high
  MODERATE_CERTAINTY          — follow-up recommended
  AMBIGUOUS_PRESENTATION      — ambiguity acknowledged; monitoring indicated
  BIOPSY_RECOMMENDED          — invasive procedure mandated by evidence
  HIGH_RISK_CONTRADICTION     — immediate escalation; contradictory evidence

A "safe biopsy avoidance" is a case that:
  · Received SAFE_NON_INVASIVE_TRIAGE AND
  · Would have been correctly diagnosed clinically (model B or C correct)

An "unsafe avoidance" is a case that:
  · Received SAFE_NON_INVASIVE_TRIAGE BUT
  · Subsequent analysis suggests escalation was clinically warranted

Escalation appropriateness
--------------------------
For cases with high contradiction load (≥ 0.40): mandatory biopsy escalation
  → correct if recommendation is BIOPSY_RECOMMENDED or HIGH_RISK_CONTRADICTION

For cases with high entropy (> 1.50 bits): ambiguity escalation
  → correct if recommendation is NOT SAFE_NON_INVASIVE_TRIAGE

The symbolic system's strength is that it can identify WHICH cases need
escalation and WHY (contradiction-driven vs ambiguity-driven vs safety-gate).

Usage
-----
  from src.evaluation_pipeline.escalation_evaluator import EscalationEvaluator

  result = EscalationEvaluator.evaluate_vectors(test_vectors, test_labels)
  print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Thresholds mirroring reasoning pipeline ───────────────────────────────────

_BIOPSY_ESCALATION_CEILING: float = 0.40
_AMBIGUITY_ESCALATION_BITS: float = 1.50
_SAFE_CERTAINTY_FLOOR:      float = 0.55
_SAFE_GAP_FLOOR:            float = 0.20

# Recommendations requiring biopsy
_BIOPSY_RECOMMENDATIONS: frozenset[str] = frozenset({
    "BIOPSY_RECOMMENDED",
    "HIGH_RISK_CONTRADICTION",
})

# Recommendations indicating safe non-invasive triage
_SAFE_RECOMMENDATIONS: frozenset[str] = frozenset({
    "SAFE_NON_INVASIVE_TRIAGE",
})


# ── Per-patient escalation profile ───────────────────────────────────────────

@dataclass(frozen=True)
class EscalationProfile:
    """
    Escalation classification for a single patient.

    Attributes
    ----------
    patient_id:
        Source patient identifier.
    disease_label:
        Ground-truth disease label.
    recommendation:
        Terminal triage recommendation from symbolic reasoning.
    requires_biopsy:
        True if the recommendation mandates biopsy.
    is_safe_triage:
        True if the recommendation is SAFE_NON_INVASIVE_TRIAGE.
    contradiction_load:
        Bilateral contradiction load at terminal stage.
    certainty:
        Leading hypothesis certainty at terminal stage.
    certainty_gap:
        Certainty gap at terminal stage.
    ambiguity_index:
        Shannon entropy (bits) at terminal stage.
    safety_triggered:
        True if the safety gate fired (contradiction ≥ 0.40 or entropy > 1.50).
    contradiction_triggered:
        True if escalation was driven by contradiction load ≥ 0.40.
    ambiguity_triggered:
        True if escalation was driven by ambiguity > 1.50 bits.
    """

    patient_id:              str
    disease_label:           str
    recommendation:          str
    requires_biopsy:         bool
    is_safe_triage:          bool
    contradiction_load:      float
    certainty:               float
    certainty_gap:           float
    ambiguity_index:         float
    safety_triggered:        bool
    contradiction_triggered: bool
    ambiguity_triggered:     bool

    @property
    def is_certain_safe(self) -> bool:
        """True if certainty criteria for safe triage are met."""
        return (
            self.certainty >= _SAFE_CERTAINTY_FLOOR
            and self.certainty_gap >= _SAFE_GAP_FLOOR
        )

    @property
    def escalation_justified(self) -> bool:
        """
        True if the biopsy recommendation is clinically justified
        by contradiction or ambiguity thresholds.
        """
        return (
            self.contradiction_load >= _BIOPSY_ESCALATION_CEILING
            or self.ambiguity_index > _AMBIGUITY_ESCALATION_BITS
        )

    @property
    def safe_triage_justified(self) -> bool:
        """True if safe triage was warranted by certainty criteria."""
        return (
            self.is_safe_triage
            and self.is_certain_safe
            and self.contradiction_load < _BIOPSY_ESCALATION_CEILING
        )


# ── Escalation evaluation result ──────────────────────────────────────────────

@dataclass
class EscalationEvaluationResult:
    """
    Dataset-level escalation analysis from the symbolic reasoning system.

    Attributes
    ----------
    total_cases:
        Total number of evaluated patients.
    biopsy_recommended_count:
        Cases receiving BIOPSY_RECOMMENDED or HIGH_RISK_CONTRADICTION.
    safe_triage_count:
        Cases receiving SAFE_NON_INVASIVE_TRIAGE.
    moderate_count:
        Cases receiving MODERATE_CERTAINTY.
    ambiguous_count:
        Cases receiving AMBIGUOUS_PRESENTATION.
    high_risk_count:
        Cases receiving HIGH_RISK_CONTRADICTION specifically.
    biopsy_rate:
        Proportion of cases escalated to biopsy.
    safe_rate:
        Proportion of cases receiving safe non-invasive triage.
    contradiction_driven_escalation_count:
        Cases where contradiction load ≥ 0.40 drove escalation.
    ambiguity_driven_escalation_count:
        Cases where entropy > 1.50 bits drove escalation.
    safety_gate_activation_count:
        Cases where the safety gate fired.
    justified_biopsy_count:
        Biopsy recommendations that were clinically justified.
    justified_safe_triage_count:
        Safe triage decisions that were clinically justified.
    per_disease_biopsy_rate:
        Per-disease biopsy escalation rate.
    per_disease_safe_rate:
        Per-disease safe triage rate.
    per_disease_mean_certainty:
        Mean certainty per disease.
    per_disease_mean_contradiction:
        Mean contradiction load per disease.
    profiles:
        All per-patient EscalationProfile instances.
    """

    total_cases:                        int
    biopsy_recommended_count:           int   = 0
    safe_triage_count:                  int   = 0
    moderate_count:                     int   = 0
    ambiguous_count:                    int   = 0
    high_risk_count:                    int   = 0
    biopsy_rate:                        float = 0.0
    safe_rate:                          float = 0.0
    contradiction_driven_escalation_count: int = 0
    ambiguity_driven_escalation_count:  int   = 0
    safety_gate_activation_count:       int   = 0
    justified_biopsy_count:             int   = 0
    justified_safe_triage_count:        int   = 0
    per_disease_biopsy_rate:            dict[str, float] = field(default_factory=dict)
    per_disease_safe_rate:              dict[str, float] = field(default_factory=dict)
    per_disease_mean_certainty:         dict[str, float] = field(default_factory=dict)
    per_disease_mean_contradiction:     dict[str, float] = field(default_factory=dict)
    profiles:                           list[EscalationProfile] = field(default_factory=list)

    def summary(self) -> str:
        n = max(self.total_cases, 1)
        lines = [
            "ESCALATION EVALUATION",
            f"  Total cases         : {self.total_cases}",
            f"  Biopsy recommended  : {self.biopsy_recommended_count} "
            f"({self.biopsy_rate:.1%})",
            f"  Safe non-invasive   : {self.safe_triage_count} "
            f"({self.safe_rate:.1%})",
            f"  Moderate certainty  : {self.moderate_count} "
            f"({self.moderate_count/n:.1%})",
            f"  Ambiguous           : {self.ambiguous_count} "
            f"({self.ambiguous_count/n:.1%})",
            f"  High-risk contra    : {self.high_risk_count} "
            f"({self.high_risk_count/n:.1%})",
            "  ---",
            f"  Contradiction-driven: {self.contradiction_driven_escalation_count}",
            f"  Ambiguity-driven    : {self.ambiguity_driven_escalation_count}",
            f"  Safety gate fired   : {self.safety_gate_activation_count}",
            f"  Justified biopsies  : {self.justified_biopsy_count}",
            f"  Justified safe triage: {self.justified_safe_triage_count}",
        ]
        return "\n".join(lines)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class EscalationEvaluator:
    """
    Evaluates escalation behaviour of the symbolic reasoning system.

    Stateless — all methods are classmethods operating on
    lists of SymbolicFeatureVector.
    """

    @classmethod
    def evaluate_vectors(
        cls,
        vectors: list[SymbolicFeatureVector],
        disease_labels: list[str] | None = None,
    ) -> EscalationEvaluationResult:
        """
        Evaluate escalation behaviour across all symbolic reasoning vectors.

        Parameters
        ----------
        vectors:
            Symbolic reasoning outputs, one per patient.
        disease_labels:
            Ground-truth disease labels in vector order.
            If None, labels are taken from vector.disease_label.
        """
        labels = disease_labels or [v.disease_label for v in vectors]
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
    ) -> EscalationProfile:
        contra_triggered = v.contradiction_load >= _BIOPSY_ESCALATION_CEILING
        ambig_triggered  = v.ambiguity_index > _AMBIGUITY_ESCALATION_BITS
        safety_triggered = contra_triggered or ambig_triggered

        return EscalationProfile(
            patient_id=v.patient_id,
            disease_label=disease_label,
            recommendation=v.recommendation,
            requires_biopsy=v.requires_biopsy,
            is_safe_triage=v.is_safe_triage,
            contradiction_load=v.contradiction_load,
            certainty=v.certainty,
            certainty_gap=v.certainty_gap,
            ambiguity_index=v.ambiguity_index,
            safety_triggered=safety_triggered,
            contradiction_triggered=contra_triggered,
            ambiguity_triggered=ambig_triggered,
        )

    @classmethod
    def _aggregate(
        cls,
        profiles: list[EscalationProfile],
    ) -> EscalationEvaluationResult:
        n = len(profiles)
        if n == 0:
            return EscalationEvaluationResult(total_cases=0)

        biopsy_n    = sum(1 for p in profiles if p.requires_biopsy)
        safe_n      = sum(1 for p in profiles if p.is_safe_triage)
        moderate_n  = sum(
            1 for p in profiles
            if p.recommendation == "MODERATE_CERTAINTY"
        )
        ambiguous_n = sum(
            1 for p in profiles
            if p.recommendation == "AMBIGUOUS_PRESENTATION"
        )
        high_risk_n = sum(
            1 for p in profiles
            if p.recommendation == "HIGH_RISK_CONTRADICTION"
        )
        contra_n    = sum(1 for p in profiles if p.contradiction_triggered)
        ambig_n     = sum(1 for p in profiles if p.ambiguity_triggered)
        safety_n    = sum(1 for p in profiles if p.safety_triggered)
        just_bx     = sum(
            1 for p in profiles
            if p.requires_biopsy and p.escalation_justified
        )
        just_safe   = sum(
            1 for p in profiles
            if p.safe_triage_justified
        )

        # Per-disease aggregation
        by_disease: dict[str, list[EscalationProfile]] = {}
        for p in profiles:
            by_disease.setdefault(p.disease_label, []).append(p)

        per_biopsy:  dict[str, float] = {}
        per_safe:    dict[str, float] = {}
        per_cert:    dict[str, float] = {}
        per_contra:  dict[str, float] = {}

        for dis, dp in by_disease.items():
            nd = len(dp)
            per_biopsy[dis] = sum(1 for p in dp if p.requires_biopsy) / nd
            per_safe[dis]   = sum(1 for p in dp if p.is_safe_triage) / nd
            per_cert[dis]   = sum(p.certainty for p in dp) / nd
            per_contra[dis] = sum(p.contradiction_load for p in dp) / nd

        return EscalationEvaluationResult(
            total_cases=n,
            biopsy_recommended_count=biopsy_n,
            safe_triage_count=safe_n,
            moderate_count=moderate_n,
            ambiguous_count=ambiguous_n,
            high_risk_count=high_risk_n,
            biopsy_rate=biopsy_n / n,
            safe_rate=safe_n / n,
            contradiction_driven_escalation_count=contra_n,
            ambiguity_driven_escalation_count=ambig_n,
            safety_gate_activation_count=safety_n,
            justified_biopsy_count=just_bx,
            justified_safe_triage_count=just_safe,
            per_disease_biopsy_rate=per_biopsy,
            per_disease_safe_rate=per_safe,
            per_disease_mean_certainty=per_cert,
            per_disease_mean_contradiction=per_contra,
            profiles=profiles,
        )
