"""
Tests for DiagnosticConflictAnalyzer — cross-disease contradiction detection.

Validates contradiction firing logic, penalty aggregation, pair tension
computation, contradiction load, confusion zone detection, and mandatory
escalation thresholds.
"""

import pytest

from src.reasoning.conflict_analyzer import (
    ActiveContradiction,
    ConflictAnalysisResult,
    DiagnosticConflictAnalyzer,
)


# ── Fixture helpers ───────────────────────────────────────────────────────────

@pytest.fixture
def contradiction_entries():
    return [
        {
            "contradiction_id": "CONTRA_001",
            "trigger_feature":  "koebner_phenomenon",
            "trigger_value":    1,
            "supports_disease": "psoriasis",
            "contradicts_disease": "lichen_planus",
            "penalty_weight":   0.30,
            "clinical_rationale": "Koebner favours psoriasis.",
        },
        {
            "contradiction_id": "CONTRA_002",
            "trigger_feature":  "follicular_papules",
            "trigger_value":    1,
            "supports_disease": "pityriasis_rubra_pilaris",
            "contradicts_disease": "psoriasis",
            "penalty_weight":   0.45,
            "clinical_rationale": "Follicular papules pathognomonic for PRP.",
        },
    ]


@pytest.fixture
def confusion_zones():
    return [
        {"pair": ["psoriasis", "seborrheic_dermatitis"]},
        {"pair": ["lichen_planus", "pityriasis_rosea"]},
    ]


@pytest.fixture
def analyzer(contradiction_entries, confusion_zones):
    return DiagnosticConflictAnalyzer(
        contradiction_entries=contradiction_entries,
        confusion_zones=confusion_zones,
        escalation_ceiling=0.40,
    )


# ── Contradiction firing ──────────────────────────────────────────────────────

class TestContradictionFiring:
    def test_contradiction_fires_on_exact_trigger(self, analyzer):
        result = analyzer.analyze({"koebner_phenomenon": 1})
        assert len(result.active_contradictions) == 1
        assert result.active_contradictions[0].contradiction_id == "CONTRA_001"

    def test_contradiction_fires_within_tolerance(self, analyzer):
        # trigger_value=1, observed=1 → abs(1-1)=0 < 0.5 → fires
        result = analyzer.analyze({"koebner_phenomenon": 1})
        assert len(result.active_contradictions) == 1

    def test_contradiction_does_not_fire_on_wrong_value(self, analyzer):
        result = analyzer.analyze({"koebner_phenomenon": 0})
        koebner_fires = [
            c for c in result.active_contradictions
            if c.contradiction_id == "CONTRA_001"
        ]
        assert len(koebner_fires) == 0

    def test_multiple_contradictions_fire_simultaneously(self, analyzer):
        result = analyzer.analyze({
            "koebner_phenomenon": 1,
            "follicular_papules": 1,
        })
        assert len(result.active_contradictions) == 2

    def test_missing_feature_does_not_fire(self, analyzer):
        result = analyzer.analyze({"koebner_phenomenon": None})
        assert len(result.active_contradictions) == 0


# ── Penalty aggregation ───────────────────────────────────────────────────────

class TestPenaltyAggregation:
    def test_penalty_applied_to_correct_target_disease(self, analyzer):
        result = analyzer.analyze({"koebner_phenomenon": 1})
        assert "lichen_planus" in result.penalty_by_disease
        assert result.penalty_by_disease["lichen_planus"] == pytest.approx(0.30)

    def test_contradiction_load_is_sum_of_penalties(self, analyzer):
        result = analyzer.analyze({
            "koebner_phenomenon": 1,
            "follicular_papules": 1,
        })
        assert result.contradiction_load == pytest.approx(0.30 + 0.45)

    def test_is_contradiction_free_when_no_features(self, analyzer):
        result = analyzer.analyze({})
        assert result.is_contradiction_free


# ── Mandatory escalation ──────────────────────────────────────────────────────

class TestMandatoryEscalation:
    def test_mandatory_escalation_when_bilateral(self, analyzer):
        # Both koebner (PSO→LP) and follicular (PRP→PSO) are present.
        # PSO is both a source and a target → bilateral conflict.
        # Bilateral load = 0.30 + 0.45 = 0.75 ≥ ceiling 0.40.
        result = analyzer.analyze({
            "koebner_phenomenon": 1,
            "follicular_papules": 1,
        })
        assert result.mandatory_escalation

    def test_unidirectional_contradiction_does_not_escalate(self, analyzer):
        # CONTRA_002 (PRP→PSO, 0.45) fires alone; PSO is never contradicted
        # back, and PRP is not an active source-of-contradiction target.
        # → unidirectional exclusion only; bilateral load = 0 < ceiling.
        result = analyzer.analyze({"follicular_papules": 1})
        assert not result.mandatory_escalation

    def test_no_mandatory_escalation_below_ceiling(self, analyzer):
        # CONTRA_001 (PSO→LP, 0.30) fires alone — unidirectional, load = 0.
        result = analyzer.analyze({"koebner_phenomenon": 1})
        assert not result.mandatory_escalation

    def test_custom_ceiling_raises_escalation_threshold(self, contradiction_entries):
        high_ceiling_analyzer = DiagnosticConflictAnalyzer(
            contradiction_entries=contradiction_entries,
            escalation_ceiling=0.80,
        )
        result = high_ceiling_analyzer.analyze({
            "koebner_phenomenon": 1,
            "follicular_papules": 1,
        })
        assert not result.mandatory_escalation


# ── Disease pair tensions ─────────────────────────────────────────────────────

class TestPairTensions:
    def test_pair_tension_created_for_active_contradiction(self, analyzer):
        result = analyzer.analyze({"koebner_phenomenon": 1})
        assert len(result.pair_tensions) == 1
        tension = result.pair_tensions[0]
        assert tension.source_disease == "psoriasis"
        assert tension.target_disease == "lichen_planus"

    def test_highest_tension_pair_returns_max_penalty(self, analyzer):
        result = analyzer.analyze({
            "koebner_phenomenon": 1,
            "follicular_papules": 1,
        })
        highest = result.highest_tension_pair
        assert highest is not None
        assert highest.cumulative_penalty == pytest.approx(0.45)

    def test_pair_tension_severity_labels(self):
        from src.reasoning.conflict_analyzer import DiseasePairTension
        assert DiseasePairTension("a", "b", 0.10).severity_label == "low"
        assert DiseasePairTension("a", "b", 0.25).severity_label == "moderate"
        assert DiseasePairTension("a", "b", 0.40).severity_label == "high"
        assert DiseasePairTension("a", "b", 0.50).severity_label == "critical"


# ── from_matrix constructor ───────────────────────────────────────────────────

class TestFromMatrix:
    def test_from_matrix_constructor(self, contradiction_entries, confusion_zones):
        matrix = {
            "contradictions":  contradiction_entries,
            "confusion_zones": confusion_zones,
        }
        analyzer = DiagnosticConflictAnalyzer.from_matrix(matrix)
        result = analyzer.analyze({"koebner_phenomenon": 1})
        assert len(result.active_contradictions) == 1

    def test_from_matrix_empty_dict(self):
        analyzer = DiagnosticConflictAnalyzer.from_matrix({})
        result = analyzer.analyze({"koebner_phenomenon": 1})
        assert result.is_contradiction_free


# ── Penalty lookup ────────────────────────────────────────────────────────────

class TestPenaltyLookup:
    def test_penalty_for_returns_correct_value(self, analyzer):
        result = analyzer.analyze({"koebner_phenomenon": 1})
        assert result.penalty_for("lichen_planus") == pytest.approx(0.30)
        assert result.penalty_for("psoriasis") == 0.0
        assert result.penalty_for("nonexistent") == 0.0
