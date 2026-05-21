"""
Clinical Data Preprocessor.

Handles ordinal integrity, missing value treatment, and feature normalisation
for the three modelling contexts:
  - Model A: all 34 features (includes histopathological)
  - Model B: 12 clinical features (statistical baseline)
  - Model C: 12 clinical features (symbolic engine input)

Design principle: Model C inputs are kept as integers (0–3 / 0–1) to
preserve ordinal semantics for fuzzy grading in Stage 0 of the reasoning
pipeline. Normalisation is applied only for statistical models (A and B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold  # type: ignore[import]

from src.utils.logger import get_logger

log = get_logger(__name__, subsystem="ClinicalDataPreprocessor")

ORDINAL_FEATURES = [
    "erythema", "scaling", "definite_borders", "itching",
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

BINARY_FEATURES = [
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history",
]

CONTINUOUS_FEATURES = ["age"]


class MissingValueStrategy(str):
    ORDINAL_MODE   = "ordinal_mode"   # replace NaN with per-column mode
    MEDIAN         = "median"         # replace NaN with median (continuous)
    ZERO           = "zero"           # treat missing as grade 0


class ClinicalDataPreprocessor:
    """
    Preprocesses clinical and histopathological feature DataFrames.

    Parameters
    ----------
    missing_strategy:
        How to handle missing values in ordinal/binary features.
    normalise_continuous:
        If True, standardise the age feature (zero mean, unit variance).
    random_state:
        Seed for train/test splitting operations.
    """

    def __init__(
        self,
        missing_strategy: str = MissingValueStrategy.ORDINAL_MODE,
        normalise_continuous: bool = True,
        random_state: int = 42,
    ) -> None:
        self.missing_strategy = missing_strategy
        self.normalise_continuous = normalise_continuous
        self.random_state = random_state
        self._fit_stats: dict[str, float] = {}

    # ── Fit / Transform ───────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame) -> "ClinicalDataPreprocessor":
        """
        Compute imputation statistics and normalisation parameters from training data.
        Must be called before transform().
        """
        for col in ORDINAL_FEATURES + BINARY_FEATURES:
            if col in X.columns:
                self._fit_stats[f"{col}_mode"] = float(
                    X[col].mode(dropna=True).iloc[0] if not X[col].mode(dropna=True).empty else 0
                )
        if "age" in X.columns:
            self._fit_stats["age_mean"] = float(X["age"].mean())
            self._fit_stats["age_std"]  = float(X["age"].std(ddof=1))
        log.debug("Preprocessor fitted", n_samples=len(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation and optional normalisation. Returns a new DataFrame.
        Does NOT modify the input in-place.
        """
        X = X.copy()
        X = self._impute_missing(X)
        X = self._enforce_ordinal_bounds(X)
        if self.normalise_continuous:
            X = self._normalise_age(X)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ── Cross-Validation Split ────────────────────────────────────────────────

    def stratified_kfold_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 10,
    ) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Return a list of (X_train, y_train, X_val, y_val) tuples for
        stratified k-fold cross-validation.

        Preprocessing is fit on training folds and applied to validation folds
        to prevent data leakage.
        """
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )
        folds = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr = X.iloc[train_idx].reset_index(drop=True)
            y_tr = y.iloc[train_idx].reset_index(drop=True)
            X_vl = X.iloc[val_idx].reset_index(drop=True)
            y_vl = y.iloc[val_idx].reset_index(drop=True)

            # Fit preprocessor on training fold only
            preprocessor = ClinicalDataPreprocessor(
                missing_strategy=self.missing_strategy,
                normalise_continuous=self.normalise_continuous,
                random_state=self.random_state,
            )
            X_tr_proc = preprocessor.fit_transform(X_tr)
            X_vl_proc = preprocessor.transform(X_vl)
            folds.append((X_tr_proc, y_tr, X_vl_proc, y_vl))

        log.info("K-fold splits created", n_folds=n_folds)
        return folds

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _impute_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in ORDINAL_FEATURES + BINARY_FEATURES:
            if col not in X.columns:
                continue
            if X[col].isna().any():
                if self.missing_strategy == MissingValueStrategy.ZERO:
                    X[col] = X[col].fillna(0)
                else:
                    fill = self._fit_stats.get(f"{col}_mode", 0)
                    X[col] = X[col].fillna(fill)
        if "age" in X.columns and X["age"].isna().any():
            fill = self._fit_stats.get("age_mean", 40.0)
            X["age"] = X["age"].fillna(fill)
        return X

    def _enforce_ordinal_bounds(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in ORDINAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].clip(0, 3).astype(int)
        for col in BINARY_FEATURES:
            if col in X.columns:
                X[col] = X[col].clip(0, 1).astype(int)
        return X

    def _normalise_age(self, X: pd.DataFrame) -> pd.DataFrame:
        if "age" not in X.columns:
            return X
        mean = self._fit_stats.get("age_mean", 0.0)
        std  = self._fit_stats.get("age_std", 1.0)
        if std == 0.0:
            std = 1.0
        X["age"] = (X["age"] - mean) / std
        return X
