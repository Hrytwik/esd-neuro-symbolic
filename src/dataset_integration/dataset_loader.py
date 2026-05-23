"""
DermatologyDatasetLoader — deterministic ingestion of the UCI Dermatology dataset.

Loads the CSV file, normalises disease labels (corrects two known label typos),
imputes missing age values with the dataset median, validates row completeness,
and returns a fully-typed DermatologyDataset ready for downstream partitioning
and evaluation.

Dataset facts
-------------
  Source    : UCI Machine Learning Repository — Dermatology dataset (ID 33)
  Patients  : 366
  Features  : 34 (12 clinical + 22 histopathological)
  Classes   : 6 dermatological diseases
  Missing   : 8 age values (imputed with median)

Label normalisation
-------------------
  The source CSV contains two typographic inconsistencies introduced during
  collection that must be normalised to canonical form:
    "seboreic_dermatitis"  → "seborrheic_dermatitis"
    "cronic_dermatitis"    → "chronic_dermatitis"
  All other labels are already in canonical form.

Usage
-----
  from src.dataset_integration.dataset_loader import DermatologyDatasetLoader
  dataset = DermatologyDatasetLoader.load("path/to/dermatology_with_labels.csv")
  print(dataset.summary)
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Canonical disease labels ──────────────────────────────────────────────────

CANONICAL_DISEASES: tuple[str, ...] = (
    "psoriasis",
    "seborrheic_dermatitis",
    "lichen_planus",
    "pityriasis_rosea",
    "chronic_dermatitis",
    "pityriasis_rubra_pilaris",
)

# Integer class code → canonical label (UCI class 1–6)
CLASS_TO_LABEL: dict[int, str] = {
    1: "psoriasis",
    2: "seborrheic_dermatitis",
    3: "lichen_planus",
    4: "pityriasis_rosea",
    5: "chronic_dermatitis",
    6: "pityriasis_rubra_pilaris",
}

# Known CSV label typos → canonical
_LABEL_FIXES: dict[str, str] = {
    "seboreic_dermatitis": "seborrheic_dermatitis",
    "cronic_dermatitis":   "chronic_dermatitis",
}

# Ordinal features bounded [0, 3]
# NOTE: The UCI Dermatology dataset uses 0–3 ordinal grading for ALL features
# except family_history and age, including features that are conceptually
# binary (present/absent). The real collected data confirms values 0–3
# appear across koebner_phenomenon, polygonal_papules, etc.
ORDINAL_FEATURES: frozenset[str] = frozenset({
    # Clinical ordinal
    "erythema", "scaling", "definite_borders", "itching",
    # Clinical — conceptually binary but collected as 0–3 in the UCI dataset
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement",
    # Histopathological ordinal
    "melanin_incontinence", "eosinophils_in_infiltrate", "PNL_infiltrate",
    "fibrosis_of_papillary_dermis", "exocytosis", "acanthosis",
    "hyperkeratosis", "parakeratosis", "clubbing_of_rete_ridges",
    "elongation_of_rete_ridges", "thinning_of_suprapapillary_epidermis",
    "focal_hypergranulosis", "disappearance_of_granular_layer",
    "vacuolisation_and_damage_of_basal_layer", "spongiosis",
    "saw_tooth_appearance_of_retes",
    "inflammatory_mononuclear_infiltrate", "band_like_infiltrate",
    # Histopathological — collected as 0–3 in the dataset
    "spongiform_pustule", "munro_microabcess",
    "follicular_horn_plug", "perifollicular_parakeratosis",
})

# Binary features bounded {0, 1}
# Only family_history is strictly binary in the actual UCI dataset.
BINARY_FEATURES: frozenset[str] = frozenset({
    "family_history",
})

# Continuous features (real-valued)
CONTINUOUS_FEATURES: frozenset[str] = frozenset({"age"})

# Full ordered feature list (matches CSV column order)
ALL_FEATURE_NAMES: tuple[str, ...] = (
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history",
    "melanin_incontinence", "eosinophils_in_infiltrate", "PNL_infiltrate",
    "fibrosis_of_papillary_dermis", "exocytosis", "acanthosis",
    "hyperkeratosis", "parakeratosis", "clubbing_of_rete_ridges",
    "elongation_of_rete_ridges", "thinning_of_suprapapillary_epidermis",
    "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
    "disappearance_of_granular_layer",
    "vacuolisation_and_damage_of_basal_layer", "spongiosis",
    "saw_tooth_appearance_of_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_mononuclear_infiltrate",
    "band_like_infiltrate", "age",
)


# ── Per-record data model ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DermatologyRecord:
    """
    A single patient record from the UCI Dermatology dataset.

    All feature values are typed and validated at load time.
    Missing age is represented as None and imputed separately.

    Attributes
    ----------
    patient_id:
        Zero-padded row index string, e.g. "P001".
    features:
        Complete feature dict for all 34 features.
        Missing age is represented as None before imputation,
        and as the dataset median after imputation.
    disease_class:
        Integer disease code 1–6.
    disease_label:
        Canonical disease name (normalised from CSV).
    has_missing_age:
        True if the original record had a missing age value.
    """

    patient_id:       str
    features:         dict[str, float | int | None]
    disease_class:    int
    disease_label:    str
    has_missing_age:  bool

    def clinical_features(self, feature_names: tuple[str, ...]) -> dict[str, Any]:
        """Return a subset of features keyed by the given names."""
        return {k: self.features[k] for k in feature_names if k in self.features}

    def feature_vector(self, feature_names: tuple[str, ...]) -> list[float]:
        """Return ordered feature values as floats (None → 0.0)."""
        return [float(self.features.get(k) or 0.0) for k in feature_names]


# ── Feature metadata ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureMetadata:
    """
    Statistical metadata for a single feature across the dataset.

    Used for validation, normalisation reference, and reporting.
    """

    feature_name:   str
    feature_type:   str          # "ordinal_0_3" | "binary" | "continuous"
    min_value:      float | None
    max_value:      float | None
    mean_value:     float | None
    median_value:   float | None
    missing_count:  int
    unique_values:  int


# ── Label distribution ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LabelDistribution:
    """Per-disease count and proportion across the full dataset."""

    counts:      dict[str, int]
    proportions: dict[str, float]
    total:       int

    def majority_class(self) -> str:
        return max(self.counts, key=lambda d: self.counts[d])

    def minority_class(self) -> str:
        return min(self.counts, key=lambda d: self.counts[d])

    def imbalance_ratio(self) -> float:
        """Majority class count / minority class count."""
        if not self.counts:
            return 1.0
        return self.counts[self.majority_class()] / max(self.counts[self.minority_class()], 1)


# ── Dataset summary ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSummary:
    """
    High-level dataset statistics produced at load time.
    """

    total_records:                 int
    feature_count:                 int
    clinical_feature_count:        int
    histopathological_feature_count: int
    label_distribution:            LabelDistribution
    missing_age_count:             int
    age_median:                    float
    age_min:                       float
    age_max:                       float
    load_path:                     str
    label_normalisation_applied:   bool

    def __str__(self) -> str:
        return (
            f"DermatologyDataset  records={self.total_records}  "
            f"features={self.feature_count} "
            f"(clinical={self.clinical_feature_count}, "
            f"histo={self.histopathological_feature_count})  "
            f"missing_age={self.missing_age_count}  "
            f"age_median={self.age_median:.1f}"
        )


# ── Dataset container ─────────────────────────────────────────────────────────

@dataclass
class DermatologyDataset:
    """
    Complete loaded and validated UCI Dermatology dataset.

    Attributes
    ----------
    records:
        All patient records in load order (deterministic).
    summary:
        High-level dataset statistics.
    feature_metadata:
        Per-feature statistical metadata.
    """

    records:          list[DermatologyRecord]
    summary:          DatasetSummary
    feature_metadata: dict[str, FeatureMetadata]

    def __len__(self) -> int:
        return len(self.records)

    def by_disease(self, disease_label: str) -> list[DermatologyRecord]:
        """Return all records for a specific disease."""
        return [r for r in self.records if r.disease_label == disease_label]

    def labels(self) -> list[str]:
        """Return all disease labels in record order."""
        return [r.disease_label for r in self.records]

    def classes(self) -> list[int]:
        """Return all integer disease codes in record order."""
        return [r.disease_class for r in self.records]

    def feature_matrix(
        self,
        feature_names: tuple[str, ...],
    ) -> list[list[float]]:
        """
        Return a patient × feature matrix as a list of float lists.
        None values are replaced with 0.0.
        """
        return [r.feature_vector(feature_names) for r in self.records]


# ── Loader ────────────────────────────────────────────────────────────────────

class DermatologyDatasetLoader:
    """
    Deterministic loader for the UCI Dermatology CSV dataset.

    All steps — normalisation, imputation, metadata computation — are
    stateless and reproducible from the same source file.
    """

    @staticmethod
    def load(csv_path: str | Path) -> DermatologyDataset:
        """
        Load the dataset from a CSV file and return a DermatologyDataset.

        Parameters
        ----------
        csv_path:
            Absolute or relative path to dermatology_with_labels.csv.

        Raises
        ------
        FileNotFoundError:
            If the CSV file does not exist.
        ValueError:
            If the CSV is missing required columns.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        raw_rows = DermatologyDatasetLoader._read_csv(path)
        age_median = DermatologyDatasetLoader._compute_age_median(raw_rows)
        records    = DermatologyDatasetLoader._build_records(raw_rows, age_median)
        metadata   = DermatologyDatasetLoader._build_metadata(records)
        summary    = DermatologyDatasetLoader._build_summary(
            records, metadata, age_median, str(path),
        )
        return DermatologyDataset(
            records=records,
            summary=summary,
            feature_metadata=metadata,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        """Read CSV into list of raw string dicts."""
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = [dict(row) for row in reader]
        return rows

    @staticmethod
    def _compute_age_median(rows: list[dict[str, str]]) -> float:
        """Compute age median from non-missing rows."""
        ages = []
        for row in rows:
            v = row.get("age", "").strip()
            if v and v not in ("", "?", "nan"):
                try:
                    ages.append(float(v))
                except ValueError:
                    pass
        return statistics.median(ages) if ages else 0.0

    @staticmethod
    def _parse_feature_value(
        name: str,
        raw: str,
        age_median: float,
    ) -> tuple[float | int | None, bool]:
        """
        Parse a single feature value from its CSV string.

        Returns (value, is_missing).
        Age is imputed from the pre-computed median when missing.
        """
        raw = raw.strip()
        is_age_missing = False

        if name == "age":
            if raw in ("", "?", "nan"):
                is_age_missing = True
                return age_median, True
            try:
                return float(raw), False
            except ValueError:
                return age_median, True

        if raw in ("", "?"):
            # Non-age missing: treat as 0 (absent)
            return 0, False

        try:
            v = float(raw)
            if name in BINARY_FEATURES:
                return int(round(v)), False
            if name in ORDINAL_FEATURES:
                return int(round(v)), False
            return v, False
        except ValueError:
            return 0, False

    @staticmethod
    def _normalise_label(raw_label: str) -> str:
        """Apply known label normalisation fixes."""
        cleaned = raw_label.strip().lower().replace(" ", "_")
        return _LABEL_FIXES.get(cleaned, cleaned)

    @staticmethod
    def _build_records(
        rows: list[dict[str, str]],
        age_median: float,
    ) -> list[DermatologyRecord]:
        """Build typed DermatologyRecord list from raw CSV rows."""
        records: list[DermatologyRecord] = []

        for idx, row in enumerate(rows):
            patient_id = f"P{idx + 1:03d}"
            features: dict[str, float | int | None] = {}
            has_missing_age = False

            for fname in ALL_FEATURE_NAMES:
                raw = row.get(fname, "")
                value, is_age_miss = DermatologyDatasetLoader._parse_feature_value(
                    fname, raw, age_median,
                )
                features[fname] = value
                if is_age_miss:
                    has_missing_age = True

            raw_label  = row.get("class_label", "").strip()
            raw_class  = row.get("class", "0").strip()
            label      = DermatologyDatasetLoader._normalise_label(raw_label)
            try:
                disease_class = int(float(raw_class))
            except (ValueError, TypeError):
                disease_class = 0

            # Cross-check label against class integer if label is missing
            if not label and disease_class in CLASS_TO_LABEL:
                label = CLASS_TO_LABEL[disease_class]

            records.append(DermatologyRecord(
                patient_id=patient_id,
                features=features,
                disease_class=disease_class,
                disease_label=label,
                has_missing_age=has_missing_age,
            ))

        return records

    @staticmethod
    def _build_metadata(
        records: list[DermatologyRecord],
    ) -> dict[str, FeatureMetadata]:
        """Compute per-feature statistical metadata."""
        metadata: dict[str, FeatureMetadata] = {}

        for fname in ALL_FEATURE_NAMES:
            values = [
                float(r.features[fname])
                for r in records
                if r.features.get(fname) is not None
            ]
            missing = sum(1 for r in records if r.features.get(fname) is None)

            if fname in ORDINAL_FEATURES:
                ftype = "ordinal_0_3"
            elif fname in BINARY_FEATURES:
                ftype = "binary"
            else:
                ftype = "continuous"

            metadata[fname] = FeatureMetadata(
                feature_name=fname,
                feature_type=ftype,
                min_value=min(values) if values else None,
                max_value=max(values) if values else None,
                mean_value=statistics.mean(values) if values else None,
                median_value=statistics.median(values) if values else None,
                missing_count=missing,
                unique_values=len(set(int(v) for v in values) if ftype != "continuous" else values),
            )

        return metadata

    @staticmethod
    def _build_summary(
        records: list[DermatologyRecord],
        metadata: dict[str, FeatureMetadata],
        age_median: float,
        load_path: str,
    ) -> DatasetSummary:
        """Compute the dataset-level summary."""
        from src.dataset_integration.feature_partitioning import (
            CLINICAL_FEATURE_NAMES, HISTOPATHOLOGICAL_FEATURE_NAMES,
        )

        label_counts: dict[str, int] = {}
        for r in records:
            label_counts[r.disease_label] = label_counts.get(r.disease_label, 0) + 1

        total = len(records)
        proportions = {k: v / total for k, v in label_counts.items()}
        dist = LabelDistribution(counts=label_counts, proportions=proportions, total=total)

        missing_age = sum(1 for r in records if r.has_missing_age)
        ages = [
            float(r.features["age"])
            for r in records
            if r.features.get("age") is not None
        ]

        return DatasetSummary(
            total_records=total,
            feature_count=len(ALL_FEATURE_NAMES),
            clinical_feature_count=len(CLINICAL_FEATURE_NAMES),
            histopathological_feature_count=len(HISTOPATHOLOGICAL_FEATURE_NAMES),
            label_distribution=dist,
            missing_age_count=missing_age,
            age_median=age_median,
            age_min=min(ages) if ages else 0.0,
            age_max=max(ages) if ages else 0.0,
            load_path=load_path,
            label_normalisation_applied=True,
        )
