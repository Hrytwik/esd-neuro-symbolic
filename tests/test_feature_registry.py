"""
Tests for FeatureRegistry — authoritative metadata for all 34 features.
"""

from __future__ import annotations

import pytest

from src.data.feature_registry import (
    FeatureGroup,
    FeatureRegistry,
    FeatureType,
)

EXPECTED_TOTAL       = 34
EXPECTED_CLINICAL    = 12
EXPECTED_HISTOPATH   = 22
EXPECTED_SYMBOLIC    = 11  # all clinical except 'age'
EXPECTED_PATHOGNOMONIC_CLINICAL = 4  # koebner, polygonal, follicular, oral


class TestFeatureRegistryCompleteness:
    def test_total_feature_count(self, feature_registry):
        assert len(feature_registry.all_features()) == EXPECTED_TOTAL

    def test_clinical_feature_count(self, feature_registry):
        assert len(feature_registry.clinical_features()) == EXPECTED_CLINICAL

    def test_histopathological_feature_count(self, feature_registry):
        assert len(feature_registry.histopathological_features()) == EXPECTED_HISTOPATH

    def test_symbolic_features_exclude_age(self, feature_registry):
        symbolic = feature_registry.symbolic_features()
        names = [f.canonical_name for f in symbolic]
        assert "age" not in names
        assert len(symbolic) == EXPECTED_SYMBOLIC

    def test_pathognomonic_clinical_count(self, feature_registry):
        patho = [
            f for f in feature_registry.pathognomonic_features()
            if f.feature_group == FeatureGroup.CLINICAL
        ]
        assert len(patho) == EXPECTED_PATHOGNOMONIC_CLINICAL

    def test_critical_discriminators_present(self, feature_registry):
        critical = feature_registry.critical_discriminators()
        names = {f.canonical_name for f in critical}
        expected = {
            "koebner_phenomenon", "polygonal_papules",
            "follicular_papules", "oral_mucosal_involvement",
        }
        assert expected.issubset(names)


class TestFeatureRegistryLookup:
    def test_get_known_feature(self, feature_registry):
        meta = feature_registry.get("koebner_phenomenon")
        assert meta.canonical_name == "koebner_phenomenon"
        assert meta.feature_type == FeatureType.BINARY
        assert meta.feature_group == FeatureGroup.CLINICAL
        assert meta.is_pathognomonic_indicator is True
        assert meta.is_critical_discriminator is True

    def test_get_ordinal_feature(self, feature_registry):
        meta = feature_registry.get("erythema")
        assert meta.feature_type == FeatureType.ORDINAL
        assert meta.ordinal_min == 0
        assert meta.ordinal_max == 3

    def test_get_continuous_feature(self, feature_registry):
        meta = feature_registry.get("age")
        assert meta.feature_type == FeatureType.CONTINUOUS
        assert meta.ordinal_min is None
        assert meta.eligible_for_symbolic_reasoning is False

    def test_get_histopath_feature(self, feature_registry):
        meta = feature_registry.get("band_like_infiltrate")
        assert meta.feature_group == FeatureGroup.HISTOPATHOLOGICAL
        assert meta.eligible_for_symbolic_reasoning is False
        assert meta.eligible_for_biopsy_baseline is True

    def test_get_unknown_feature_raises(self, feature_registry):
        with pytest.raises(KeyError):
            feature_registry.get("nonexistent_feature")

    def test_names_clinical_group(self, feature_registry):
        names = feature_registry.names(group=FeatureGroup.CLINICAL)
        assert len(names) == EXPECTED_CLINICAL
        assert "koebner_phenomenon" in names

    def test_ordinal_features_by_type(self, feature_registry):
        ordinals = feature_registry.by_type(FeatureType.ORDINAL)
        names = {f.canonical_name for f in ordinals}
        assert "erythema" in names
        assert "age" not in names


class TestFeatureRegistryFrozen:
    def test_metadata_is_immutable(self, feature_registry):
        meta = feature_registry.get("erythema")
        with pytest.raises((AttributeError, TypeError)):
            meta.canonical_name = "changed"  # type: ignore[misc]
