"""
ClinicalFeatureMapper — mapping dataset records to symbolic pipeline inputs.

Translates DermatologyRecord instances into the feature_values dict format
expected by PipelineRunner.run(). This is the bridge between the structured
dataset representation and the symbolic reasoning engine.

The mapping preserves ordinal grades exactly as-is (the pipeline's
ClinicalGradingModule handles the 0–3 → fuzzy conversion internally).
Age is optionally normalised for the reasoning engine's age gate.

Mapping rules
-------------
  · Ordinal features (0–3): pass raw integer value unchanged
  · Binary features (0/1):  pass raw integer value unchanged
  · Age: pass raw float (the reasoning engine contains its own age gate)
  · Missing values: replaced with feature-appropriate neutral value (0 or median)

Usage
-----
  from src.dataset_integration.clinical_feature_mapper import ClinicalFeatureMapper
  from src.dataset_integration.dataset_loader import DermatologyDatasetLoader

  dataset = DermatologyDatasetLoader.load("dermatology_with_labels.csv")
  mapper  = ClinicalFeatureMapper()
  inputs  = [mapper.map_clinical(r) for r in dataset.records]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.dataset_integration.dataset_loader import DermatologyRecord
from src.dataset_integration.feature_partitioning import (
    CLINICAL_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    BINARY_CLINICAL,
    BINARY_HISTOPATHOLOGICAL,
    FeaturePartition,
)


# ── Clinical input record ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClinicalInputRecord:
    """
    A mapped patient record ready for symbolic pipeline ingestion.

    Attributes
    ----------
    patient_id:
        Original patient identifier from the dataset.
    feature_values:
        Feature dict passed directly to PipelineRunner.run().
    disease_label:
        Ground-truth disease label (canonical form).
    disease_class:
        Ground-truth integer class code 1–6.
    has_missing_age:
        Whether the original record had a missing age value.
    partition:
        Which feature partition was used to build this input.
    feature_count:
        Number of features included in feature_values.
    """

    patient_id:       str
    feature_values:   dict[str, Any]
    disease_label:    str
    disease_class:    int
    has_missing_age:  bool
    partition:        FeaturePartition
    feature_count:    int

    def __repr__(self) -> str:
        return (
            f"ClinicalInputRecord(patient={self.patient_id}, "
            f"disease={self.disease_label}, "
            f"partition={self.partition.value}, "
            f"features={self.feature_count})"
        )


# ── Mapper ────────────────────────────────────────────────────────────────────

class ClinicalFeatureMapper:
    """
    Maps DermatologyRecord instances to symbolic pipeline feature dicts.

    The mapper is stateless. Create one instance and reuse across the
    full dataset.

    Parameters
    ----------
    age_imputation_value:
        Value used when age is missing. Default: dataset-level median
        (set externally, e.g. from DatasetSummary.age_median).
    """

    # Features the pipeline's symbolic engine explicitly gates on by name
    _PIPELINE_RELEVANT_FEATURES: frozenset[str] = frozenset({
        # Clinical ordinal — drive certainty propagation
        "erythema", "scaling", "definite_borders", "itching",
        # Clinical binary — drive rule activation
        "koebner_phenomenon", "polygonal_papules", "follicular_papules",
        "oral_mucosal_involvement", "knee_and_elbow_involvement",
        "scalp_involvement", "family_history",
        # Age — used in some rule conditions
        "age",
    })

    def __init__(self, age_imputation_value: float = 33.0) -> None:
        self._age_imputation = age_imputation_value

    # ── Public API ────────────────────────────────────────────────────────────

    def map_clinical(self, record: DermatologyRecord) -> ClinicalInputRecord:
        """
        Map only the 12 clinical (biopsy-free) features.

        This is the primary mapping used for Model B and Model C
        symbolic reasoning runs.

        Parameters
        ----------
        record:
            A loaded DermatologyRecord from DermatologyDatasetLoader.

        Returns
        -------
        ClinicalInputRecord:
            Ready for PipelineRunner.run(case_id, feature_values).
        """
        fv = self._extract_features(record, CLINICAL_FEATURE_NAMES)
        return ClinicalInputRecord(
            patient_id=record.patient_id,
            feature_values=fv,
            disease_label=record.disease_label,
            disease_class=record.disease_class,
            has_missing_age=record.has_missing_age,
            partition=FeaturePartition.CLINICAL,
            feature_count=len(fv),
        )

    def map_combined(self, record: DermatologyRecord) -> ClinicalInputRecord:
        """
        Map all 34 features (clinical + histopathological).

        Used to prepare inputs for Model A (full biopsy reference).

        Parameters
        ----------
        record:
            A loaded DermatologyRecord from DermatologyDatasetLoader.
        """
        fv = self._extract_features(record, ALL_FEATURE_NAMES)
        return ClinicalInputRecord(
            patient_id=record.patient_id,
            feature_values=fv,
            disease_label=record.disease_label,
            disease_class=record.disease_class,
            has_missing_age=record.has_missing_age,
            partition=FeaturePartition.COMBINED,
            feature_count=len(fv),
        )

    def map_batch_clinical(
        self,
        records: list[DermatologyRecord],
    ) -> list[ClinicalInputRecord]:
        """Map a list of records to clinical inputs."""
        return [self.map_clinical(r) for r in records]

    def map_batch_combined(
        self,
        records: list[DermatologyRecord],
    ) -> list[ClinicalInputRecord]:
        """Map a list of records to combined inputs."""
        return [self.map_combined(r) for r in records]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_features(
        self,
        record: DermatologyRecord,
        feature_names: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        Build a feature dict from the record for the given feature names.

        All binary and ordinal features are extracted as-is.
        Age is replaced with the imputation value if None.
        """
        fv: dict[str, Any] = {}
        all_binary = BINARY_CLINICAL | BINARY_HISTOPATHOLOGICAL

        for name in feature_names:
            raw = record.features.get(name)

            if raw is None:
                # Should not happen after loader imputation, but guard
                if name == "age":
                    fv[name] = self._age_imputation
                elif name in all_binary:
                    fv[name] = 0
                else:
                    fv[name] = 0
            else:
                fv[name] = raw

        return fv

    @staticmethod
    def feature_names_for_partition(partition: FeaturePartition) -> tuple[str, ...]:
        """Return the canonical feature name tuple for a partition."""
        if partition == FeaturePartition.CLINICAL:
            return CLINICAL_FEATURE_NAMES
        if partition == FeaturePartition.COMBINED:
            return ALL_FEATURE_NAMES
        from src.dataset_integration.feature_partitioning import HISTOPATHOLOGICAL_FEATURE_NAMES
        return HISTOPATHOLOGICAL_FEATURE_NAMES
