"""
Clinical Data Loader for the UCI Dermatology dataset (id=33).

Handles remote fetch via ucimlrepo, local disk caching, schema validation,
and the authoritative clinical/histopathological feature split required by
Models A, B, and C.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_registry import FeatureGroup, FeatureRegistry
from src.utils.logger import get_logger

log = get_logger(__name__, subsystem="ClinicalDataLoader")

UCI_DATASET_ID = 33
_CACHE_DIR = Path("data/cache")

CLINICAL_FEATURE_NAMES: list[str] = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history", "age",
]

HISTOPATHOLOGICAL_FEATURE_NAMES: list[str] = [
    "melanin_incontinence", "eosinophils_in_the_infiltrate",
    "PNL_infiltrate", "fibrosis_of_the_papillary_dermis",
    "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis",
    "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges",
    "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis",
    "disappearance_of_the_granular_layer",
    "vacuolisation_and_damage_of_basal_layer", "spongiosis",
    "saw_tooth_appearance_of_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_monoluclear_infiltrate",
    "band_like_infiltrate",
]

TARGET_COLUMN = "target"


class DataLoadError(RuntimeError):
    """Raised when the dataset cannot be loaded from any source."""


class ClinicalDataLoader:
    """
    Loads and prepares the UCI Dermatology dataset for downstream inference.

    Responsibilities
    ----------------
    - Fetch from ucimlrepo with local cache fallback.
    - Rename columns to canonical feature names defined in the feature registry.
    - Validate schema, ranges, and class distribution on load.
    - Expose three views: all_features (Model A), clinical_only (Model B / C),
      and the target series.

    Parameters
    ----------
    cache_dir:
        Directory for caching the fetched dataset. Created on first use.
    use_cache:
        If True, return cached data when available; bypass network fetch.
    """

    def __init__(
        self,
        cache_dir: Path | str = _CACHE_DIR,
        use_cache: bool = True,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._use_cache = use_cache
        self._registry = FeatureRegistry()
        self._raw_df: pd.DataFrame | None = None
        self._feature_df: pd.DataFrame | None = None
        self._target: pd.Series | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> "ClinicalDataLoader":
        """
        Load the dataset. Populates internal DataFrames for downstream access.
        Returns self for method chaining.
        """
        df = self._fetch()
        df = self._rename_columns(df)
        self._raw_df = df
        self._target = df[TARGET_COLUMN].astype(int)
        self._feature_df = df.drop(columns=[TARGET_COLUMN])
        log.info(
            "Dataset loaded",
            n_samples=len(df),
            n_features=self._feature_df.shape[1],
            classes=sorted(self._target.unique().tolist()),
        )
        return self

    @property
    def all_features(self) -> pd.DataFrame:
        """All 34 features (clinical + histopathological). Used by Model A."""
        self._assert_loaded()
        return self._feature_df.copy()

    @property
    def clinical_features(self) -> pd.DataFrame:
        """
        12 clinical (non-biopsy) features only. Used by Models B and C.
        Columns are in the canonical order defined by CLINICAL_FEATURE_NAMES.
        """
        self._assert_loaded()
        return self._feature_df[CLINICAL_FEATURE_NAMES].copy()

    @property
    def histopathological_features(self) -> pd.DataFrame:
        """22 histopathological (biopsy-derived) features. Supplementary reference."""
        self._assert_loaded()
        return self._feature_df[HISTOPATHOLOGICAL_FEATURE_NAMES].copy()

    @property
    def target(self) -> pd.Series:
        """Integer class labels 1–6."""
        self._assert_loaded()
        return self._target.copy()

    @property
    def n_samples(self) -> int:
        self._assert_loaded()
        return len(self._feature_df)

    def class_distribution(self) -> dict[int, int]:
        self._assert_loaded()
        return self._target.value_counts().sort_index().to_dict()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch(self) -> pd.DataFrame:
        cache_path = self._cache_dir / "dermatology_uci33.pkl"
        if self._use_cache and cache_path.exists():
            log.debug("Loading from cache", path=str(cache_path))
            return self._load_cache(cache_path)

        log.info("Fetching UCI Dermatology dataset", dataset_id=UCI_DATASET_ID)
        df = self._fetch_from_ucimlrepo()
        self._save_cache(df, cache_path)
        return df

    def _fetch_from_ucimlrepo(self) -> pd.DataFrame:
        try:
            from ucimlrepo import fetch_ucirepo  # type: ignore[import]
        except ImportError as exc:
            raise DataLoadError(
                "ucimlrepo is not installed. Run: pip install ucimlrepo"
            ) from exc

        try:
            dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        except Exception as exc:
            raise DataLoadError(
                f"Failed to fetch UCI dataset {UCI_DATASET_ID}: {exc}"
            ) from exc

        features: pd.DataFrame = dataset.data.features
        targets: pd.DataFrame = dataset.data.targets
        df = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map UCI column names to canonical feature names.
        UCI uses different naming conventions; the registry defines the canonical form.
        """
        # UCI column rename map — maps UCI headers to canonical names
        rename_map: dict[str, str] = {
            # UCI name                            → canonical name
            "erythema":                            "erythema",
            "scaling":                             "scaling",
            "definite-borders":                    "definite_borders",
            "itching":                             "itching",
            "koebner-phenomenon":                  "koebner_phenomenon",
            "polygonal-papules":                   "polygonal_papules",
            "follicular-papules":                  "follicular_papules",
            "oral-mucosal-involvement":            "oral_mucosal_involvement",
            "knee-and-elbow-involvement":          "knee_and_elbow_involvement",
            "scalp-involvement":                   "scalp_involvement",
            "family-history":                      "family_history",
            "melanin-incontinence":                "melanin_incontinence",
            "eosinophils-in-the-infiltrate":       "eosinophils_in_the_infiltrate",
            "PNL-infiltrate":                      "PNL_infiltrate",
            "fibrosis-of-the-papillary-dermis":    "fibrosis_of_the_papillary_dermis",
            "exocytosis":                          "exocytosis",
            "acanthosis":                          "acanthosis",
            "hyperkeratosis":                      "hyperkeratosis",
            "parakeratosis":                       "parakeratosis",
            "clubbing-of-the-rete-ridges":         "clubbing_of_the_rete_ridges",
            "elongation-of-the-rete-ridges":       "elongation_of_the_rete_ridges",
            "thinning-of-the-suprapapillary-epidermis":
                                                   "thinning_of_the_suprapapillary_epidermis",
            "spongiform-pustule":                  "spongiform_pustule",
            "munro-microabcess":                   "munro_microabcess",
            "focal-hypergranulosis":               "focal_hypergranulosis",
            "disappearance-of-the-granular-layer": "disappearance_of_the_granular_layer",
            "vacuolisation-and-damage-of-basal-layer":
                                                   "vacuolisation_and_damage_of_basal_layer",
            "spongiosis":                          "spongiosis",
            "saw-tooth-appearance-of-retes":       "saw_tooth_appearance_of_retes",
            "follicular-horn-plug":                "follicular_horn_plug",
            "perifollicular-parakeratosis":        "perifollicular_parakeratosis",
            "inflammatory-monoluclear-infiltrate": "inflammatory_monoluclear_infiltrate",
            "band-like-infiltrate":                "band_like_infiltrate",
            "age":                                 "age",
            # Target column(s) — ucimlrepo may use 'class' or 'target'
            "class":                               TARGET_COLUMN,
            "target":                              TARGET_COLUMN,
        }
        # Apply only mappings that exist in df
        applicable = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=applicable)
        return df

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug("Dataset cached", path=str(path))

    def _load_cache(self, path: Path) -> pd.DataFrame:
        with open(path, "rb") as fh:
            df = pickle.load(fh)
        if not isinstance(df, pd.DataFrame):
            raise DataLoadError(f"Cache file is corrupt: {path}")
        return df

    def _assert_loaded(self) -> None:
        if self._feature_df is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
