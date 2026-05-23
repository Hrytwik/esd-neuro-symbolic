"""
DatasetSplitter — stratified train / validation / test partitioning.

Produces deterministic, reproducible splits of the DermatologyDataset
with disease distribution preserved across all partitions. This is
essential because pityriasis_rubra_pilaris has only 20 records — random
splits without stratification can easily produce zero-record classes in
the test set.

All splits are seeded for reproducibility. The same seed always produces
the same partition.

Splitting strategies
--------------------
  HOLDOUT   — single train / validation / test split (default 70/15/15)
  K_FOLD    — stratified k-fold cross-validation (default k=5)

Usage
-----
  from src.dataset_integration.dataset_splitter import DatasetSplitter
  splitter = DatasetSplitter(seed=42)

  # Standard holdout
  split = splitter.split(dataset)
  print(len(split.train), len(split.test))

  # Cross-validation
  folds = splitter.cross_validate(dataset, n_folds=5)
  for fold in folds:
      model.fit(fold.train_records)
      model.evaluate(fold.validation_records)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.dataset_integration.dataset_loader import DermatologyRecord, DermatologyDataset
from src.dataset_integration.feature_partitioning import (
    CLINICAL_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
)


# ── Split data model ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataSplit:
    """
    A single stratified train / validation / test partition.

    Attributes
    ----------
    train_records:
        Training records (largest partition).
    validation_records:
        Validation records (used for hyperparameter selection).
    test_records:
        Test records (held-out evaluation set, never touched during training).
    seed:
        Random seed used to produce this split.
    train_ratio:
        Fraction of data allocated to training.
    validation_ratio:
        Fraction of data allocated to validation.
    test_ratio:
        Fraction of data allocated to testing.
    """

    train_records:      tuple[DermatologyRecord, ...]
    validation_records: tuple[DermatologyRecord, ...]
    test_records:       tuple[DermatologyRecord, ...]
    seed:               int
    train_ratio:        float
    validation_ratio:   float
    test_ratio:         float

    @property
    def train_labels(self) -> list[str]:
        return [r.disease_label for r in self.train_records]

    @property
    def validation_labels(self) -> list[str]:
        return [r.disease_label for r in self.validation_records]

    @property
    def test_labels(self) -> list[str]:
        return [r.disease_label for r in self.test_records]

    @property
    def n_train(self) -> int:
        return len(self.train_records)

    @property
    def n_validation(self) -> int:
        return len(self.validation_records)

    @property
    def n_test(self) -> int:
        return len(self.test_records)

    def train_feature_matrix(
        self,
        feature_names: tuple[str, ...] = CLINICAL_FEATURE_NAMES,
    ) -> list[list[float]]:
        """Return training feature matrix (patient × feature)."""
        return [r.feature_vector(feature_names) for r in self.train_records]

    def validation_feature_matrix(
        self,
        feature_names: tuple[str, ...] = CLINICAL_FEATURE_NAMES,
    ) -> list[list[float]]:
        return [r.feature_vector(feature_names) for r in self.validation_records]

    def test_feature_matrix(
        self,
        feature_names: tuple[str, ...] = CLINICAL_FEATURE_NAMES,
    ) -> list[list[float]]:
        return [r.feature_vector(feature_names) for r in self.test_records]

    def label_distribution_summary(self) -> dict[str, dict[str, int]]:
        """Return per-split label counts for balance verification."""
        def _counts(records: tuple[DermatologyRecord, ...]) -> dict[str, int]:
            out: dict[str, int] = {}
            for r in records:
                out[r.disease_label] = out.get(r.disease_label, 0) + 1
            return out

        return {
            "train":      _counts(self.train_records),
            "validation": _counts(self.validation_records),
            "test":       _counts(self.test_records),
        }

    def __str__(self) -> str:
        return (
            f"DataSplit(seed={self.seed}, "
            f"train={self.n_train}, "
            f"val={self.n_validation}, "
            f"test={self.n_test})"
        )


# ── Cross-validation fold ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class CrossValidationFold:
    """
    A single fold from stratified k-fold cross-validation.

    Attributes
    ----------
    fold_index:
        Zero-based fold index [0, k-1].
    n_folds:
        Total number of folds in this cross-validation run.
    train_records:
        Training records for this fold (k-1 partitions).
    validation_records:
        Held-out validation records for this fold (1 partition).
    seed:
        Random seed used for the full cross-validation.
    """

    fold_index:         int
    n_folds:            int
    train_records:      tuple[DermatologyRecord, ...]
    validation_records: tuple[DermatologyRecord, ...]
    seed:               int

    @property
    def train_labels(self) -> list[str]:
        return [r.disease_label for r in self.train_records]

    @property
    def validation_labels(self) -> list[str]:
        return [r.disease_label for r in self.validation_records]

    @property
    def n_train(self) -> int:
        return len(self.train_records)

    @property
    def n_validation(self) -> int:
        return len(self.validation_records)

    def __str__(self) -> str:
        return (
            f"CrossValidationFold({self.fold_index + 1}/{self.n_folds}, "
            f"train={self.n_train}, val={self.n_validation})"
        )


# ── Splitter ──────────────────────────────────────────────────────────────────

class DatasetSplitter:
    """
    Stratified dataset splitter for the UCI Dermatology dataset.

    All splits preserve the class distribution of the full dataset
    within each partition. This is critical for rare classes like
    pityriasis_rubra_pilaris (only 20 records).

    Parameters
    ----------
    seed:
        Master random seed. All splits derived from this seed are
        fully deterministic.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    # ── Public API ────────────────────────────────────────────────────────────

    def split(
        self,
        dataset: DermatologyDataset,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> DataSplit:
        """
        Produce a single stratified train / validation / test split.

        The test partition is assembled first (held-out), then the
        remainder is split into train and validation.

        Parameters
        ----------
        dataset:
            Loaded DermatologyDataset.
        train_ratio:
            Fraction of records for training. Default 0.70.
        validation_ratio:
            Fraction of records for validation. Default 0.15.
        test_ratio:
            Fraction of records for testing. Default 0.15.

        Raises
        ------
        ValueError:
            If ratios do not sum to approximately 1.0.
        """
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0; "
                f"got {train_ratio + validation_ratio + test_ratio:.4f}."
            )

        by_class = self._group_by_class(dataset.records)
        train_r, val_r, test_r = [], [], []

        for disease, records in by_class.items():
            shuffled = self._shuffle(records, salt=disease)
            n        = len(shuffled)
            n_test   = max(1, round(n * test_ratio))
            n_val    = max(1, round(n * validation_ratio))
            n_train  = n - n_test - n_val

            if n_train < 1:
                # Very small class: put all but 1 in train
                n_train = max(1, n - 2)
                n_val   = max(0, n - n_train - 1)
                n_test  = n - n_train - n_val

            test_r.extend(shuffled[:n_test])
            val_r.extend(shuffled[n_test: n_test + n_val])
            train_r.extend(shuffled[n_test + n_val:])

        return DataSplit(
            train_records=tuple(self._shuffle(train_r, salt="train")),
            validation_records=tuple(self._shuffle(val_r, salt="val")),
            test_records=tuple(self._shuffle(test_r, salt="test")),
            seed=self._seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

    def cross_validate(
        self,
        dataset: DermatologyDataset,
        n_folds: int = 5,
    ) -> list[CrossValidationFold]:
        """
        Produce stratified k-fold cross-validation partitions.

        Parameters
        ----------
        dataset:
            Loaded DermatologyDataset.
        n_folds:
            Number of folds. Default 5. Must be ≥ 2.

        Returns
        -------
        list[CrossValidationFold]:
            n_folds CrossValidationFold instances, one per fold.
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be ≥ 2; got {n_folds}.")

        by_class  = self._group_by_class(dataset.records)
        # Build per-class folds
        class_folds: dict[str, list[list[DermatologyRecord]]] = {}
        for disease, records in by_class.items():
            shuffled = self._shuffle(records, salt=disease)
            folds_   = [[] for _ in range(n_folds)]
            for i, r in enumerate(shuffled):
                folds_[i % n_folds].append(r)
            class_folds[disease] = folds_

        result: list[CrossValidationFold] = []
        for fold_idx in range(n_folds):
            val_records:   list[DermatologyRecord] = []
            train_records: list[DermatologyRecord] = []
            for disease, folds_ in class_folds.items():
                for i, fold_group in enumerate(folds_):
                    if i == fold_idx:
                        val_records.extend(fold_group)
                    else:
                        train_records.extend(fold_group)

            result.append(CrossValidationFold(
                fold_index=fold_idx,
                n_folds=n_folds,
                train_records=tuple(self._shuffle(train_records, salt=f"fold_{fold_idx}_train")),
                validation_records=tuple(self._shuffle(val_records, salt=f"fold_{fold_idx}_val")),
                seed=self._seed,
            ))

        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _group_by_class(
        records: list[DermatologyRecord],
    ) -> dict[str, list[DermatologyRecord]]:
        """Group records by disease_label."""
        groups: dict[str, list[DermatologyRecord]] = {}
        for r in records:
            groups.setdefault(r.disease_label, []).append(r)
        return groups

    def _shuffle(
        self,
        records: list[DermatologyRecord],
        salt: str = "",
    ) -> list[DermatologyRecord]:
        """Shuffle records deterministically using seeded RNG."""
        rng = random.Random(self._seed + hash(salt) % (2 ** 31))
        shuffled = list(records)
        rng.shuffle(shuffled)
        return shuffled
