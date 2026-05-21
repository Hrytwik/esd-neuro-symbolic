"""
Tests for ClinicalDataLoader — schema, feature split, and caching.
These tests use the synthetic dataset and do not require network access.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    CLINICAL_FEATURE_NAMES,
    HISTOPATHOLOGICAL_FEATURE_NAMES,
    TARGET_COLUMN,
    ClinicalDataLoader,
    DataLoadError,
)
from src.data.preprocessing import BINARY_FEATURES


class TestClinicalDataLoaderWithSyntheticCache:
    """
    Tests that operate by injecting the synthetic DataFrame into the
    loader's cache directory, bypassing the network fetch.
    """

    @pytest.fixture
    def loader_with_cache(self, synthetic_df, tmp_path) -> ClinicalDataLoader:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "dermatology_uci33.pkl"
        with open(cache_path, "wb") as fh:
            pickle.dump(synthetic_df, fh)
        loader = ClinicalDataLoader(cache_dir=cache_dir, use_cache=True)
        loader.load()
        return loader

    def test_load_returns_self(self, synthetic_df, tmp_path):
        cache_dir = tmp_path / "cache2"
        cache_dir.mkdir()
        cache_path = cache_dir / "dermatology_uci33.pkl"
        with open(cache_path, "wb") as fh:
            pickle.dump(synthetic_df, fh)
        loader = ClinicalDataLoader(cache_dir=cache_dir, use_cache=True)
        result = loader.load()
        assert result is loader

    def test_n_samples(self, loader_with_cache):
        assert loader_with_cache.n_samples == 366

    def test_all_features_shape(self, loader_with_cache):
        X = loader_with_cache.all_features
        assert X.shape == (366, 34)

    def test_clinical_features_shape(self, loader_with_cache):
        X = loader_with_cache.clinical_features
        assert X.shape == (366, 12)

    def test_clinical_feature_column_names(self, loader_with_cache):
        cols = list(loader_with_cache.clinical_features.columns)
        assert cols == CLINICAL_FEATURE_NAMES

    def test_histopathological_features_shape(self, loader_with_cache):
        X = loader_with_cache.histopathological_features
        assert X.shape == (366, 22)

    def test_target_series_length(self, loader_with_cache):
        y = loader_with_cache.target
        assert len(y) == 366

    def test_target_class_range(self, loader_with_cache):
        y = loader_with_cache.target
        assert set(y.unique()).issubset({1, 2, 3, 4, 5, 6})

    def test_class_distribution_keys(self, loader_with_cache):
        dist = loader_with_cache.class_distribution()
        assert set(dist.keys()) == {1, 2, 3, 4, 5, 6}

    def test_features_do_not_contain_target(self, loader_with_cache):
        assert TARGET_COLUMN not in loader_with_cache.all_features.columns
        assert TARGET_COLUMN not in loader_with_cache.clinical_features.columns

    def test_assert_loaded_raises_before_load(self, tmp_path):
        loader = ClinicalDataLoader(cache_dir=tmp_path, use_cache=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = loader.all_features

    def test_corrupt_cache_raises(self, tmp_path):
        cache_dir = tmp_path / "cache3"
        cache_dir.mkdir()
        bad_cache = cache_dir / "dermatology_uci33.pkl"
        with open(bad_cache, "wb") as fh:
            pickle.dump("not_a_dataframe", fh)
        loader = ClinicalDataLoader(cache_dir=cache_dir, use_cache=True)
        with pytest.raises(DataLoadError):
            loader.load()
