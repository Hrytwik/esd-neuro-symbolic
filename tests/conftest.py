"""
Pytest fixtures for the Phase 1 test suite.

Provides a synthetic UCI Dermatology-compatible dataset for offline testing
(no network access required), plus pre-built instances of loaders,
preprocessors, validators, and the rule repository.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    CLINICAL_FEATURE_NAMES,
    HISTOPATHOLOGICAL_FEATURE_NAMES,
    TARGET_COLUMN,
)
from src.data.preprocessing import BINARY_FEATURES
from src.data.feature_registry import FeatureRegistry


# ── Seed all randomness in fixtures ──────────────────────────────────────────
random.seed(42)
np.random.seed(42)


# ── Constants ─────────────────────────────────────────────────────────────────
RULES_DIR = Path(__file__).parent.parent / "rules"
N_SYNTHETIC = 366
CLASS_COUNTS = {1: 112, 2: 61, 3: 72, 4: 49, 5: 52, 6: 20}


# ── Synthetic dataset builder ─────────────────────────────────────────────────

def _build_synthetic_df(n: int = N_SYNTHETIC, seed: int = 42) -> pd.DataFrame:
    """
    Construct a synthetic DataFrame that mirrors the UCI Dermatology schema.

    Ordinal features are sampled uniformly from {0, 1, 2, 3}.
    Binary features are sampled from {0, 1} with P(1)=0.3.
    Age is sampled from a realistic clinical range.
    Target labels are generated to reproduce the expected class distribution.
    """
    rng = np.random.default_rng(seed)

    ordinal_cols = [
        f for f in CLINICAL_FEATURE_NAMES
        if f not in BINARY_FEATURES and f != "age"
    ] + HISTOPATHOLOGICAL_FEATURE_NAMES

    data: dict[str, np.ndarray] = {}

    for col in ordinal_cols:
        data[col] = rng.integers(0, 4, size=n)  # [0, 3]

    for col in BINARY_FEATURES:
        data[col] = rng.choice([0, 1], size=n, p=[0.7, 0.3])

    data["age"] = rng.uniform(5, 75, size=n)

    # Build labels matching expected class distribution
    labels: list[int] = []
    for cls, count in CLASS_COUNTS.items():
        labels.extend([cls] * count)
    rng.shuffle(labels)
    data[TARGET_COLUMN] = np.array(labels[:n])

    return pd.DataFrame(data)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Full synthetic DataFrame with all 34 features + target."""
    return _build_synthetic_df()


@pytest.fixture(scope="session")
def synthetic_X(synthetic_df) -> pd.DataFrame:
    return synthetic_df.drop(columns=[TARGET_COLUMN])


@pytest.fixture(scope="session")
def synthetic_y(synthetic_df) -> pd.Series:
    return synthetic_df[TARGET_COLUMN].astype(int)


@pytest.fixture(scope="session")
def synthetic_clinical_X(synthetic_X) -> pd.DataFrame:
    return synthetic_X[CLINICAL_FEATURE_NAMES].copy()


@pytest.fixture(scope="session")
def feature_registry() -> FeatureRegistry:
    return FeatureRegistry()


@pytest.fixture(scope="session")
def rules_dir() -> Path:
    return RULES_DIR


@pytest.fixture(scope="session")
def rule_repository(rules_dir):
    from src.symbolic_engine.rule_registry import DiagnosticRuleRepository
    return DiagnosticRuleRepository(rules_dir=rules_dir, validate=True)


@pytest.fixture(scope="session")
def schema_validator():
    from src.symbolic_engine.schema_validator import RuleSchemaValidator
    return RuleSchemaValidator()
