"""
Shared fixtures for Phase 2 reasoning subsystem tests.

Provides pre-built instances, sample rules, feature vectors, and
result objects used across all 13 reasoning subsystem test modules.
"""

from __future__ import annotations

import pytest

from src.reasoning.certainty_propagator import (
    CertaintyDistribution,
    HypothesisCertainty,
    HypothesisCertaintyPropagator,
)
from src.reasoning.clinical_grading import ClinicalGradingModule, GradingResult
from src.reasoning.conflict_analyzer import (
    ActiveContradiction,
    ConflictAnalysisResult,
    DiagnosticConflictAnalyzer,
    DiseasePairTension,
)
from src.reasoning.differential_competition import DifferentialCompetitionEngine
from src.reasoning.escalation_engine import ClinicalEscalationEngine, TriageDecision
from src.reasoning.evidence_evaluator import (
    DiagnosticEvidenceEvaluator,
    DiseaseEvidenceVector,
    EvidenceEvaluationResult,
    RuleEvaluationResult,
)
from src.reasoning.evidence_sufficiency import EvidenceSufficiencyAnalyzer
from src.reasoning.instability_monitor import DiagnosticInstabilityMonitor
from src.reasoning.narrative_generator import DiagnosticNarrativeGenerator
from src.reasoning.safety_gate import (
    ClinicalSafetyGate,
    SafetyGateReport,
    TriageRecommendation,
)
from src.reasoning.state_tracker import DiagnosticState, DiagnosticStateTracker
from src.reasoning.trajectory_memory import DiagnosticTrajectoryMemory


# ── Subsystem instances ───────────────────────────────────────────────────────

@pytest.fixture
def grading_module() -> ClinicalGradingModule:
    return ClinicalGradingModule()


@pytest.fixture
def evidence_evaluator() -> DiagnosticEvidenceEvaluator:
    return DiagnosticEvidenceEvaluator()


@pytest.fixture
def certainty_propagator() -> HypothesisCertaintyPropagator:
    return HypothesisCertaintyPropagator()


@pytest.fixture
def state_tracker() -> DiagnosticStateTracker:
    return DiagnosticStateTracker()


@pytest.fixture
def safety_gate() -> ClinicalSafetyGate:
    return ClinicalSafetyGate()


@pytest.fixture
def escalation_engine() -> ClinicalEscalationEngine:
    return ClinicalEscalationEngine()


@pytest.fixture
def narrative_generator() -> DiagnosticNarrativeGenerator:
    return DiagnosticNarrativeGenerator()


@pytest.fixture
def instability_monitor() -> DiagnosticInstabilityMonitor:
    return DiagnosticInstabilityMonitor()


@pytest.fixture
def sufficiency_analyzer() -> EvidenceSufficiencyAnalyzer:
    return EvidenceSufficiencyAnalyzer()


@pytest.fixture
def competition_engine() -> DifferentialCompetitionEngine:
    return DifferentialCompetitionEngine()


# ── Sample diagnostic rules ───────────────────────────────────────────────────

@pytest.fixture
def psoriasis_binary_rule() -> dict:
    """PSO_001 equivalent — binary Tier-A rule for koebner phenomenon."""
    return {
        "rule_id": "PSO_001",
        "disease_target": "psoriasis",
        "evidence_tier": "A",
        "activation_logic": "binary",
        "confidence_weight": 0.85,
        "min_activation_threshold": 0.10,
        "supporting_features": [
            {"feature": "koebner_phenomenon", "condition": "eq", "threshold": 1},
        ],
    }


@pytest.fixture
def psoriasis_composite_rule() -> dict:
    """PSO_005 equivalent — composite Tier-B rule for erythema + scaling."""
    return {
        "rule_id": "PSO_005",
        "disease_target": "psoriasis",
        "evidence_tier": "B",
        "activation_logic": "composite",
        "confidence_weight": 0.65,
        "min_activation_threshold": 0.10,
        "supporting_features": [
            {"feature": "erythema",  "condition": "gte", "threshold": 2, "partial_weight": 0.55},
            {"feature": "scaling",   "condition": "gte", "threshold": 2, "partial_weight": 0.45},
        ],
    }


@pytest.fixture
def lp_binary_rule() -> dict:
    """LP_001 equivalent — Tier-A binary rule for polygonal papules."""
    return {
        "rule_id": "LP_001",
        "disease_target": "lichen_planus",
        "evidence_tier": "A",
        "activation_logic": "binary",
        "confidence_weight": 0.90,
        "min_activation_threshold": 0.10,
        "supporting_features": [
            {"feature": "polygonal_papules", "condition": "eq", "threshold": 1},
        ],
    }


@pytest.fixture
def psoriasis_threshold_rule() -> dict:
    """Threshold Tier-B rule for scalp involvement."""
    return {
        "rule_id": "PSO_003",
        "disease_target": "psoriasis",
        "evidence_tier": "B",
        "activation_logic": "threshold",
        "confidence_weight": 0.75,
        "min_activation_threshold": 0.10,
        "supporting_features": [
            {"feature": "scalp_involvement", "condition": "eq", "threshold": 1, "partial_weight": 1.0},
        ],
    }


@pytest.fixture
def minimal_rules(psoriasis_binary_rule, psoriasis_composite_rule, lp_binary_rule) -> list[dict]:
    """A minimal set of rules covering two diseases."""
    return [psoriasis_binary_rule, psoriasis_composite_rule, lp_binary_rule]


# ── Sample feature vectors ────────────────────────────────────────────────────

@pytest.fixture
def psoriasis_features() -> dict:
    """
    Canonical psoriasis feature profile — strong pathognomonic + supportive.
    koebner_phenomenon=1, erythema=3, scaling=3, scalp_involvement=1,
    knee_and_elbow_involvement=1, family_history=1.
    """
    return {
        "koebner_phenomenon":        1,
        "erythema":                  3,
        "scaling":                   3,
        "definite_borders":          2,
        "itching":                   2,
        "scalp_involvement":         1,
        "knee_and_elbow_involvement": 1,
        "family_history":            1,
        "polygonal_papules":         0,
        "follicular_papules":        0,
        "oral_mucosal_involvement":  0,
    }


@pytest.fixture
def lichen_planus_features() -> dict:
    """Canonical lichen planus feature profile."""
    return {
        "koebner_phenomenon":        0,
        "erythema":                  2,
        "scaling":                   1,
        "definite_borders":          2,
        "itching":                   3,
        "scalp_involvement":         0,
        "knee_and_elbow_involvement": 0,
        "family_history":            0,
        "polygonal_papules":         1,
        "follicular_papules":        0,
        "oral_mucosal_involvement":  1,
    }


@pytest.fixture
def sparse_features() -> dict:
    """Sparse feature profile — most features absent."""
    return {
        "koebner_phenomenon":        0,
        "erythema":                  0,
        "scaling":                   0,
        "definite_borders":          0,
        "itching":                   0,
        "scalp_involvement":         0,
        "knee_and_elbow_involvement": 0,
        "family_history":            0,
        "polygonal_papules":         0,
        "follicular_papules":        0,
        "oral_mucosal_involvement":  0,
    }


@pytest.fixture
def missing_features() -> dict:
    """Feature profile with several None values."""
    return {
        "koebner_phenomenon":        None,
        "erythema":                  2,
        "scaling":                   2,
        "definite_borders":          None,
        "itching":                   1,
        "scalp_involvement":         None,
        "knee_and_elbow_involvement": 1,
        "family_history":            None,
        "polygonal_papules":         0,
        "follicular_papules":        0,
        "oral_mucosal_involvement":  0,
    }


BINARY_FEATURES = {
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history",
}


# ── Pre-built grading results ─────────────────────────────────────────────────

@pytest.fixture
def psoriasis_grading(grading_module, psoriasis_features) -> GradingResult:
    return grading_module.grade_vector(psoriasis_features, binary_features=BINARY_FEATURES)


@pytest.fixture
def sparse_grading(grading_module, sparse_features) -> GradingResult:
    return grading_module.grade_vector(sparse_features, binary_features=BINARY_FEATURES)


# ── Pre-built evidence results ────────────────────────────────────────────────

def _make_rule_result(rule_id, disease, tier, score, status="active", features=None) -> RuleEvaluationResult:
    return RuleEvaluationResult(
        rule_id=rule_id,
        disease_target=disease,
        evidence_tier=tier,
        activation_logic="binary",
        activation_score=score,
        raw_score=score,
        confidence_weight=score,
        status=status,
        contributing_features=features or [],
        failed_features=[],
        is_tier_a=(tier == "A"),
    )


@pytest.fixture
def psoriasis_evidence_result() -> EvidenceEvaluationResult:
    """Strong psoriasis evidence — pathognomonic rule active + 2 supportive."""
    all_diseases = [
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris",
    ]
    vectors = {}
    for d in all_diseases:
        vectors[d] = DiseaseEvidenceVector(
            disease=d, raw_evidence_score=0.0,
            tier_a_score=0.0, tier_b_score=0.0, tier_d_score=0.0,
        )

    # Psoriasis — 1 Tier-A + 2 Tier-B active
    r_a = _make_rule_result("PSO_001", "psoriasis", "A", 0.85,
                            features=["koebner_phenomenon"])
    r_b1 = _make_rule_result("PSO_005", "psoriasis", "B", 0.55,
                             features=["erythema", "scaling"])
    r_b2 = _make_rule_result("PSO_003", "psoriasis", "B", 0.60,
                             features=["scalp_involvement"])

    pso = vectors["psoriasis"]
    pso.activated_rules = [r_a, r_b1, r_b2]
    pso.tier_a_score = 0.85
    pso.tier_b_score = 1.15
    pso.raw_evidence_score = 2.00
    pso.active_rule_count = 3
    pso.tier_a_count = 1
    pso.tier_b_count = 2
    pso.has_pathognomonic = True
    pso.coverage_fraction = 0.75

    # Lichen planus — 1 weak Tier-B only
    r_lp = _make_rule_result("LP_002", "lichen_planus", "B", 0.15,
                              status="partial", features=["oral_mucosal_involvement"])
    lp = vectors["lichen_planus"]
    lp.activated_rules = [r_lp]
    lp.tier_b_score = 0.15
    lp.raw_evidence_score = 0.15
    lp.active_rule_count = 1
    lp.tier_b_count = 1
    lp.coverage_fraction = 0.20

    return EvidenceEvaluationResult(
        disease_vectors=vectors,
        evaluated_tiers=["A", "B"],
        total_rules_checked=10,
        total_rules_active=4,
        leading_disease="psoriasis",
        second_disease="lichen_planus",
    )


# ── Pre-built conflict results ────────────────────────────────────────────────

@pytest.fixture
def no_conflict_result() -> ConflictAnalysisResult:
    return ConflictAnalysisResult(
        active_contradictions=[],
        pair_tensions=[],
        penalty_by_disease={},
        contradiction_load=0.0,
        confusion_zone_active=[],
        instability_contribution=0.0,
        mandatory_escalation=False,
    )


@pytest.fixture
def moderate_conflict_result() -> ConflictAnalysisResult:
    """Contradiction load 0.30 — below mandatory escalation ceiling."""
    c = ActiveContradiction(
        contradiction_id="CONTRA_001",
        trigger_feature="koebner_phenomenon",
        trigger_value=1.0,
        source_disease="psoriasis",
        target_disease="lichen_planus",
        penalty_weight=0.30,
        clinical_rationale="Koebner phenomenon favours psoriasis over LP.",
    )
    tension = DiseasePairTension(
        source_disease="psoriasis",
        target_disease="lichen_planus",
        cumulative_penalty=0.30,
        active_contradictions=[c],
    )
    return ConflictAnalysisResult(
        active_contradictions=[c],
        pair_tensions=[tension],
        penalty_by_disease={"lichen_planus": 0.30},
        contradiction_load=0.30,
        confusion_zone_active=[],
        instability_contribution=0.30,
        mandatory_escalation=False,
    )


@pytest.fixture
def high_conflict_result() -> ConflictAnalysisResult:
    """Contradiction load 0.50 — above mandatory escalation ceiling."""
    c1 = ActiveContradiction(
        contradiction_id="CONTRA_001",
        trigger_feature="koebner_phenomenon",
        trigger_value=1.0,
        source_disease="psoriasis",
        target_disease="lichen_planus",
        penalty_weight=0.30,
    )
    c2 = ActiveContradiction(
        contradiction_id="CONTRA_005",
        trigger_feature="follicular_papules",
        trigger_value=1.0,
        source_disease="pityriasis_rubra_pilaris",
        target_disease="psoriasis",
        penalty_weight=0.20,
    )
    return ConflictAnalysisResult(
        active_contradictions=[c1, c2],
        pair_tensions=[],
        penalty_by_disease={"lichen_planus": 0.30, "psoriasis": 0.20},
        contradiction_load=0.50,
        confusion_zone_active=[],
        instability_contribution=0.50,
        mandatory_escalation=True,
    )


# ── Pre-built certainty distributions ────────────────────────────────────────

def _make_hyp(disease, certainty, rank, rules=3, tier_a=1, patho=True) -> HypothesisCertainty:
    return HypothesisCertainty(
        disease=disease,
        raw_evidence=certainty * 2.0,
        penalised_score=certainty * 1.9,
        certainty=certainty,
        rank=rank,
        active_rule_count=rules,
        tier_a_count=tier_a,
        has_pathognomonic=patho,
    )


@pytest.fixture
def stable_certainty() -> CertaintyDistribution:
    """Psoriasis leading with good separation — stable, not highly certain."""
    hyps = [
        _make_hyp("psoriasis",               0.72, 1),
        _make_hyp("seborrheic_dermatitis",    0.10, 2, rules=1, tier_a=0, patho=False),
        _make_hyp("lichen_planus",            0.08, 3, rules=1, tier_a=0, patho=False),
        _make_hyp("pityriasis_rosea",         0.05, 4, rules=0, tier_a=0, patho=False),
        _make_hyp("chronic_dermatitis",       0.03, 5, rules=0, tier_a=0, patho=False),
        _make_hyp("pityriasis_rubra_pilaris", 0.02, 6, rules=0, tier_a=0, patho=False),
    ]
    return CertaintyDistribution(
        hypotheses=hyps,
        leading_disease="psoriasis",
        second_disease="seborrheic_dermatitis",
        max_certainty=0.72,
        certainty_gap=0.62,
        ambiguity_index=0.85,
        contradiction_load=0.0,
        contradiction_dampened=False,
        is_stable=True,
        is_highly_certain=True,
        is_ambiguous=False,
    )


@pytest.fixture
def high_certainty_dist() -> CertaintyDistribution:
    """Psoriasis with very high certainty (>0.82 + gap>0.40)."""
    hyps = [
        _make_hyp("psoriasis",               0.87, 1),
        _make_hyp("seborrheic_dermatitis",    0.05, 2, rules=0, tier_a=0, patho=False),
        _make_hyp("lichen_planus",            0.03, 3, rules=0, tier_a=0, patho=False),
        _make_hyp("pityriasis_rosea",         0.02, 4, rules=0, tier_a=0, patho=False),
        _make_hyp("chronic_dermatitis",       0.02, 5, rules=0, tier_a=0, patho=False),
        _make_hyp("pityriasis_rubra_pilaris", 0.01, 6, rules=0, tier_a=0, patho=False),
    ]
    return CertaintyDistribution(
        hypotheses=hyps,
        leading_disease="psoriasis",
        second_disease="seborrheic_dermatitis",
        max_certainty=0.87,
        certainty_gap=0.82,
        ambiguity_index=0.42,
        contradiction_load=0.0,
        contradiction_dampened=False,
        is_stable=True,
        is_highly_certain=True,
        is_ambiguous=False,
    )


@pytest.fixture
def ambiguous_certainty() -> CertaintyDistribution:
    """High-entropy distribution — ambiguous presentation."""
    hyps = [
        _make_hyp("psoriasis",               0.30, 1, rules=2, tier_a=0, patho=False),
        _make_hyp("seborrheic_dermatitis",    0.25, 2, rules=1, tier_a=0, patho=False),
        _make_hyp("lichen_planus",            0.20, 3, rules=1, tier_a=0, patho=False),
        _make_hyp("pityriasis_rosea",         0.12, 4, rules=0, tier_a=0, patho=False),
        _make_hyp("chronic_dermatitis",       0.08, 5, rules=0, tier_a=0, patho=False),
        _make_hyp("pityriasis_rubra_pilaris", 0.05, 6, rules=0, tier_a=0, patho=False),
    ]
    return CertaintyDistribution(
        hypotheses=hyps,
        leading_disease="psoriasis",
        second_disease="seborrheic_dermatitis",
        max_certainty=0.30,
        certainty_gap=0.05,
        ambiguity_index=2.30,
        contradiction_load=0.0,
        contradiction_dampened=False,
        is_stable=False,
        is_highly_certain=False,
        is_ambiguous=True,
    )


# ── Pre-built safety reports ──────────────────────────────────────────────────

@pytest.fixture
def safe_safety_report() -> SafetyGateReport:
    """No safety gates triggered."""
    return SafetyGateReport(
        invariant_results=[],
        gate_results=[],
        effective_cap=None,
        any_triggered=False,
        certainty_penalty=0.0,
    )


# ── Pre-built triage decision ─────────────────────────────────────────────────

@pytest.fixture
def safe_triage_decision() -> TriageDecision:
    return TriageDecision(
        recommendation=TriageRecommendation.SAFE_NON_INVASIVE_TRIAGE,
        leading_disease="psoriasis",
        second_disease="seborrheic_dermatitis",
        max_certainty=0.87,
        certainty_gap=0.82,
        contradiction_load=0.0,
        ambiguity_index=0.42,
        final_state=DiagnosticState.SAFE_TRIAGE,
        safety_gate_applied=False,
        applied_gate_ids=[],
        decision_rationale=(
            "High certainty (0.87) with large certainty gap (0.82). "
            "No contradictions. Evidence sufficient for non-invasive triage."
        ),
    )


@pytest.fixture
def biopsy_triage_decision() -> TriageDecision:
    return TriageDecision(
        recommendation=TriageRecommendation.BIOPSY_RECOMMENDED,
        leading_disease="psoriasis",
        second_disease="seborrheic_dermatitis",
        max_certainty=0.42,
        certainty_gap=0.05,
        contradiction_load=0.50,
        ambiguity_index=2.30,
        final_state=DiagnosticState.BIOPSY_ESCALATION,
        safety_gate_applied=True,
        applied_gate_ids=["I1"],
        decision_rationale=(
            "Contradiction load (0.50) exceeds mandatory escalation ceiling. "
            "Histological confirmation required."
        ),
    )
