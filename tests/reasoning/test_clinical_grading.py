"""
Tests for ClinicalGradingModule — Stage 0 fuzzy grading.

Validates ordinal-to-fuzzy mapping, binary feature handling,
partial contribution, missing value semantics, and evidence
strength labelling.
"""

import pytest

from src.reasoning.clinical_grading import ClinicalGradingModule, GradingResult


# ── Ordinal grading ────────────────────────────────────────────────────────────

class TestOrdinalGrading:
    def test_grade_zero_maps_to_zero(self, grading_module):
        result = grading_module.grade_feature("erythema", 0)
        assert result.fuzzy_grade == 0.0
        assert not result.is_present
        assert result.evidence_strength == "absent"

    def test_grade_one_maps_to_033(self, grading_module):
        result = grading_module.grade_feature("erythema", 1)
        assert result.fuzzy_grade == pytest.approx(0.33)
        assert result.is_present
        assert result.evidence_strength == "weak"

    def test_grade_two_maps_to_067(self, grading_module):
        result = grading_module.grade_feature("scaling", 2)
        assert result.fuzzy_grade == pytest.approx(0.67)
        assert result.is_clinically_significant
        assert result.evidence_strength == "moderate"

    def test_grade_three_maps_to_one(self, grading_module):
        result = grading_module.grade_feature("scaling", 3)
        assert result.fuzzy_grade == 1.0
        assert result.evidence_strength == "strong"

    def test_grade_clamps_above_three(self, grading_module):
        result = grading_module.grade_feature("erythema", 5)
        assert result.fuzzy_grade == 1.0  # clamped to 3→1.0

    def test_clinical_significance_threshold_at_two(self, grading_module):
        below = grading_module.grade_feature("erythema", 1)
        at    = grading_module.grade_feature("erythema", 2)
        assert not below.is_clinically_significant
        assert at.is_clinically_significant


# ── Binary grading ────────────────────────────────────────────────────────────

class TestBinaryGrading:
    def test_binary_zero_is_absent(self, grading_module):
        result = grading_module.grade_feature("koebner_phenomenon", 0, is_binary=True)
        assert result.fuzzy_grade == 0.0
        assert not result.is_present
        assert result.evidence_strength == "absent"

    def test_binary_one_is_full_present(self, grading_module):
        result = grading_module.grade_feature("koebner_phenomenon", 1, is_binary=True)
        assert result.fuzzy_grade == 1.0
        assert result.is_present
        assert result.evidence_strength == "strong"

    def test_binary_one_is_clinically_significant(self, grading_module):
        result = grading_module.grade_feature("polygonal_papules", 1, is_binary=True)
        assert result.is_clinically_significant


# ── Missing value handling ────────────────────────────────────────────────────

class TestMissingValues:
    def test_none_produces_zero_fuzzy(self, grading_module):
        result = grading_module.grade_feature("erythema", None)
        assert result.fuzzy_grade == 0.0
        assert result.is_missing
        assert not result.is_present

    def test_missing_feature_not_clinically_significant(self, grading_module):
        result = grading_module.grade_feature("erythema", None)
        assert not result.is_clinically_significant


# ── grade_vector ──────────────────────────────────────────────────────────────

class TestGradeVector:
    def test_grade_vector_produces_grading_result(self, grading_module, psoriasis_features):
        from tests.reasoning.conftest import BINARY_FEATURES
        result = grading_module.grade_vector(psoriasis_features, binary_features=BINARY_FEATURES)
        assert isinstance(result, GradingResult)
        assert len(result.graded_features) == len(psoriasis_features)

    def test_present_features_excludes_absent(self, grading_module, psoriasis_features):
        from tests.reasoning.conftest import BINARY_FEATURES
        result = grading_module.grade_vector(psoriasis_features, binary_features=BINARY_FEATURES)
        for feat in result.present_features:
            assert feat.fuzzy_grade > 0.05

    def test_significant_features_at_threshold(self, grading_module, psoriasis_features):
        from tests.reasoning.conftest import BINARY_FEATURES
        result = grading_module.grade_vector(psoriasis_features, binary_features=BINARY_FEATURES)
        for feat in result.significant_features:
            assert feat.is_clinically_significant

    def test_completeness_score_all_present(self, grading_module, psoriasis_features):
        from tests.reasoning.conftest import BINARY_FEATURES
        result = grading_module.grade_vector(psoriasis_features, binary_features=BINARY_FEATURES)
        assert result.completeness_score == 1.0

    def test_completeness_score_with_missing(self, grading_module, missing_features):
        from tests.reasoning.conftest import BINARY_FEATURES
        result = grading_module.grade_vector(missing_features, binary_features=BINARY_FEATURES)
        assert result.completeness_score < 1.0

    def test_fuzzy_value_lookup(self, grading_module, psoriasis_features):
        from tests.reasoning.conftest import BINARY_FEATURES
        result = grading_module.grade_vector(psoriasis_features, binary_features=BINARY_FEATURES)
        assert result.fuzzy_value("koebner_phenomenon") == 1.0
        assert result.fuzzy_value("nonexistent", default=0.5) == 0.5

    def test_sparse_vector_has_no_present_features(self, grading_module, sparse_features):
        from tests.reasoning.conftest import BINARY_FEATURES
        result = grading_module.grade_vector(sparse_features, binary_features=BINARY_FEATURES)
        assert result.present_features == []


# ── partial_activation ────────────────────────────────────────────────────────

class TestPartialActivation:
    def test_partial_activation_binary_present(self, grading_module):
        val = grading_module.partial_activation("koebner_phenomenon", 1, 0.85, is_binary=True)
        assert val == pytest.approx(0.85)

    def test_partial_activation_ordinal_grade_two(self, grading_module):
        val = grading_module.partial_activation("erythema", 2, 0.55)
        assert val == pytest.approx(0.67 * 0.55, rel=1e-3)

    def test_partial_activation_missing_returns_zero(self, grading_module):
        val = grading_module.partial_activation("erythema", None, 0.80)
        assert val == 0.0


# ── Custom grade map ──────────────────────────────────────────────────────────

class TestCustomGradeMap:
    def test_custom_grade_map_applied(self):
        custom_map = {0: 0.0, 1: 0.25, 2: 0.75, 3: 1.0}
        module = ClinicalGradingModule(grade_map=custom_map)
        result = module.grade_feature("erythema", 1)
        assert result.fuzzy_grade == pytest.approx(0.25)

    def test_frozen_graded_feature(self, grading_module):
        result = grading_module.grade_feature("erythema", 2)
        with pytest.raises(Exception):
            result.fuzzy_grade = 0.5   # type: ignore[misc]
