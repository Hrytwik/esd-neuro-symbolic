"""
Inter-subsystem pipeline contracts for the Certainty-Aware Symbolic
Dermatological Reasoning Engine.

All data flowing between pipeline stages must conform to these typed
contracts. Pydantic v2 enforces field constraints at construction time,
making contract violations explicit rather than silent.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class DiagnosticState(str, Enum):
    """9-state diagnostic state machine encoding the current epistemic status."""
    S0_EVIDENCE_SPARSE       = "S0_EVIDENCE_SPARSE"
    S1_HYPOTHESIS_FORMING    = "S1_HYPOTHESIS_FORMING"
    S2_REINFORCING           = "S2_REINFORCING"
    S3_CONTRADICTION_EMERGED = "S3_CONTRADICTION_EMERGED"
    S4_DIAGNOSTIC_TENSION    = "S4_DIAGNOSTIC_TENSION"
    S5_AMBIGUITY_ESCALATED   = "S5_AMBIGUITY_ESCALATED"
    S6_CERTAINTY_STABILIZING = "S6_CERTAINTY_STABILIZING"
    S7_CERTAINTY_STABILIZED  = "S7_CERTAINTY_STABILIZED"
    S8_BIOPSY_ESCALATED      = "S8_BIOPSY_ESCALATED"


class TriageRecommendation(str, Enum):
    """Primary output of the Clinical Escalation Engine."""
    SAFE_BIOPSY_FREE    = "SAFE_BIOPSY_FREE"
    MODERATE_CERTAINTY  = "MODERATE_CERTAINTY"
    AMBIGUOUS_CASE      = "AMBIGUOUS_CASE"
    BIOPSY_ADVISED      = "BIOPSY_ADVISED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceTier(str, Enum):
    """Rule evidence tier per the Diagnostic Evidence Evaluator taxonomy."""
    A = "A"  # pathognomonic
    B = "B"  # supportive
    C = "C"  # auxiliary
    D = "D"  # cross-disease discriminating


class RuleStatus(str, Enum):
    """Activation status of a diagnostic rule after evaluation."""
    ACTIVE   = "ACTIVE"    # fully activated (activation_score >= confidence_weight)
    PARTIAL  = "PARTIAL"   # partially activated (min_partial > score < confidence_weight)
    DORMANT  = "DORMANT"   # below dormant threshold
    PENALISED = "PENALISED"  # contradiction penalty applied


class DiseaseLabel(str, Enum):
    """Canonical disease labels for the six erythemato-squamous conditions."""
    PSORIASIS               = "psoriasis"
    SEBORRHEIC_DERMATITIS   = "seborrheic_dermatitis"
    LICHEN_PLANUS           = "lichen_planus"
    PITYRIASIS_ROSEA        = "pityriasis_rosea"
    CHRONIC_DERMATITIS      = "chronic_dermatitis"
    PITYRIASIS_RUBRA_PILARIS = "pityriasis_rubra_pilaris"


# ── Feature Vector ────────────────────────────────────────────────────────────

class ClinicalFeatureVector(BaseModel):
    """
    Input feature vector for clinical-only inference (Model B / Model C).
    Contains only non-invasive, non-biopsy-derived features.
    Missing values encoded as None (not 0 — absence must be explicit).
    """
    # Ordinal clinical features (0–3)
    erythema:          int | None = Field(None, ge=0, le=3)
    scaling:           int | None = Field(None, ge=0, le=3)
    definite_borders:  int | None = Field(None, ge=0, le=3)
    itching:           int | None = Field(None, ge=0, le=3)

    # Binary clinical features (0 or 1)
    koebner_phenomenon:          int | None = Field(None, ge=0, le=1)
    polygonal_papules:           int | None = Field(None, ge=0, le=1)
    follicular_papules:          int | None = Field(None, ge=0, le=1)
    oral_mucosal_involvement:    int | None = Field(None, ge=0, le=1)
    knee_and_elbow_involvement:  int | None = Field(None, ge=0, le=1)
    scalp_involvement:           int | None = Field(None, ge=0, le=1)
    family_history:              int | None = Field(None, ge=0, le=1)

    # Continuous clinical feature
    age: float | None = Field(None, ge=0.0, le=120.0)

    # Metadata
    case_id:      str | None = None
    ground_truth: int | None = Field(None, ge=1, le=6)

    @field_validator("erythema", "scaling", "definite_borders", "itching", mode="before")
    @classmethod
    def coerce_ordinal_none(cls, v: Any) -> int | None:
        if v is None:
            return None
        val = int(v)
        return val

    def observed_features(self) -> list[str]:
        """Return names of features with non-None values."""
        exclude = {"case_id", "ground_truth"}
        return [
            k for k, v in self.model_dump().items()
            if k not in exclude and v is not None
        ]

    def missing_features(self) -> list[str]:
        """Return names of features with None values."""
        exclude = {"case_id", "ground_truth"}
        return [
            k for k, v in self.model_dump().items()
            if k not in exclude and v is None
        ]

    def completeness_score(self) -> float:
        """Fraction of the 12 clinical features that are observed."""
        total = 12  # fixed clinical feature count
        return len(self.observed_features()) / total


# ── Rule Activation ───────────────────────────────────────────────────────────

class ContradictionEvent(BaseModel):
    """Records a single contradiction detected during Stage 3 analysis."""
    contradiction_id:  str
    trigger_feature:   str
    trigger_value:     int | float
    source_disease:    str
    target_disease:    str
    penalty_weight:    float = Field(ge=0.0, le=1.0)
    is_active:         bool = True


class RuleActivation(BaseModel):
    """Result of evaluating a single diagnostic rule against a feature vector."""
    rule_id:           str
    disease_target:    str
    evidence_tier:     EvidenceTier
    status:            RuleStatus
    raw_score:         float = Field(ge=0.0, le=1.0)
    penalised_score:   float = Field(ge=0.0, le=1.0)
    confidence_weight: float = Field(ge=0.0, le=1.0)
    activated_features: list[str] = Field(default_factory=list)
    applied_penalties:  list[ContradictionEvent] = Field(default_factory=list)


# ── Certainty State ───────────────────────────────────────────────────────────

class HypothesisScore(BaseModel):
    """Certainty score for a single disease hypothesis after propagation."""
    disease:           str
    raw_evidence:      float = Field(ge=0.0)
    penalised_score:   float = Field(ge=0.0)
    certainty:         float = Field(ge=0.0, le=1.0)
    rank:              int = Field(ge=1)
    activated_rule_count: int = Field(ge=0)
    tier_a_count:      int = Field(ge=0)


class CertaintyVector(BaseModel):
    """Full certainty distribution across all six disease hypotheses."""
    hypotheses:           list[HypothesisScore]
    leading_disease:      str
    second_disease:       str
    max_certainty:        float = Field(ge=0.0, le=1.0)
    certainty_gap:        float = Field(ge=0.0, le=1.0)
    contradiction_load:   float = Field(ge=0.0)
    ambiguity_index:      float = Field(ge=0.0)  # Shannon entropy in bits

    @model_validator(mode="after")
    def validate_certainty_sum(self) -> "CertaintyVector":
        total = sum(h.certainty for h in self.hypotheses)
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Certainty scores must sum to 1.0; got {total:.4f}"
            )
        return self


# ── Safety Gate ───────────────────────────────────────────────────────────────

class SafetyGateResult(BaseModel):
    """Result of the Clinical Safety Gate evaluation."""
    gate_name:        str
    triggered:        bool
    applied_cap:      TriageRecommendation | None = None
    rationale:        str
    measured_value:   float | None = None
    threshold_value:  float | None = None


class SafetyGateReport(BaseModel):
    """Aggregate report of all safety gate evaluations for a single case."""
    invariant_results: list[SafetyGateResult] = Field(default_factory=list)
    gate_results:      list[SafetyGateResult] = Field(default_factory=list)
    effective_cap:     TriageRecommendation | None = None
    any_triggered:     bool = False

    @model_validator(mode="after")
    def compute_effective_cap(self) -> "SafetyGateReport":
        triggered = [
            r for r in (self.invariant_results + self.gate_results)
            if r.triggered and r.applied_cap is not None
        ]
        if triggered:
            self.any_triggered = True
            # Highest severity cap wins (BIOPSY_ADVISED > AMBIGUOUS > MODERATE)
            severity = {
                TriageRecommendation.BIOPSY_ADVISED: 3,
                TriageRecommendation.AMBIGUOUS_CASE: 2,
                TriageRecommendation.MODERATE_CERTAINTY: 1,
            }
            self.effective_cap = max(
                triggered, key=lambda r: severity.get(r.applied_cap, 0)
            ).applied_cap
        return self


# ── Graph Snapshot ────────────────────────────────────────────────────────────

class GraphSnapshot(BaseModel):
    """
    A point-in-time capture of the reasoning graph state after a pipeline stage.
    Ordered sequence of snapshots constitutes the CaseTrajectory.
    """
    stage:             int = Field(ge=0, le=6)
    stage_name:        str
    state:             DiagnosticState
    activated_rules:   list[RuleActivation] = Field(default_factory=list)
    contradictions:    list[ContradictionEvent] = Field(default_factory=list)
    certainty_vector:  CertaintyVector | None = None
    safety_report:     SafetyGateReport | None = None
    triage:            TriageRecommendation | None = None
    narrative_fragment: str | None = None


# ── Stage Update ─────────────────────────────────────────────────────────────

class StageUpdate(BaseModel):
    """
    Lightweight event emitted at the completion of each pipeline stage.
    Used for streaming inference updates and trajectory construction.
    """
    stage:          int
    stage_name:     str
    state:          DiagnosticState
    triage_so_far:  TriageRecommendation | None = None
    delta_rules:    list[str] = Field(default_factory=list)
    snapshot:       GraphSnapshot


# ── Case Trajectory ───────────────────────────────────────────────────────────

class CaseTrajectory(BaseModel):
    """
    Complete ordered reasoning trajectory for a single case.
    Captures the full sequence of state transitions from Stage 0 to Stage 6.
    """
    case_id:        str
    run_id:         str
    snapshots:      list[GraphSnapshot] = Field(default_factory=list)
    total_stages:   int = Field(ge=0, le=7)
    final_state:    DiagnosticState | None = None

    def add_snapshot(self, snapshot: GraphSnapshot) -> None:
        self.snapshots.append(snapshot)
        self.total_stages = len(self.snapshots)
        self.final_state = snapshot.state

    def get_stage(self, stage: int) -> GraphSnapshot | None:
        for s in self.snapshots:
            if s.stage == stage:
                return s
        return None


# ── Inference Result ──────────────────────────────────────────────────────────

class InferenceResult(BaseModel):
    """
    Terminal output contract for a completed inference run on a single case.
    Produced by the Clinical Escalation Engine at Stage 6.
    """
    case_id:                  str
    run_id:                   str
    triage_recommendation:    TriageRecommendation
    leading_disease:          str
    second_disease:           str
    max_certainty:            float = Field(ge=0.0, le=1.0)
    certainty_gap:            float = Field(ge=0.0, le=1.0)
    contradiction_load:       float = Field(ge=0.0)
    ambiguity_index:          float = Field(ge=0.0)
    final_state:              DiagnosticState
    activated_rule_count:     int = Field(ge=0)
    tier_a_rule_count:        int = Field(ge=0)
    safety_gate_triggered:    bool
    applied_safety_cap:       TriageRecommendation | None = None
    narrative:                str
    certainty_vector:         CertaintyVector
    trajectory:               CaseTrajectory
    ground_truth:             int | None = Field(None, ge=1, le=6)
    ground_truth_label:       str | None = None
    prediction_correct:       bool | None = None
    completeness_score:       float = Field(ge=0.0, le=1.0)
    timestamp:                datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def resolve_prediction_correct(self) -> "InferenceResult":
        if self.ground_truth is not None and self.leading_disease is not None:
            disease_map = {
                1: "psoriasis",
                2: "seborrheic_dermatitis",
                3: "lichen_planus",
                4: "pityriasis_rosea",
                5: "chronic_dermatitis",
                6: "pityriasis_rubra_pilaris",
            }
            self.ground_truth_label = disease_map.get(self.ground_truth)
            self.prediction_correct = (
                self.leading_disease == self.ground_truth_label
            )
        return self
