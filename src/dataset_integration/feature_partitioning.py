"""
FeaturePartitioning — explicit clinical vs histopathological feature separation.

This module defines the foundational partition that drives the entire
three-model comparative evaluation:

  CLINICAL PARTITION (12 features)
  ---------------------------------
  Features assessable in a standard outpatient dermatology consultation
  without any invasive procedure. A clinician can score all 12 features
  from patient history, visual inspection, and physical examination alone.

  HISTOPATHOLOGICAL PARTITION (22 features)
  ------------------------------------------
  Features derivable only from laboratory analysis of a tissue biopsy sample.
  Includes all microscopic dermatopathological findings: epidermal architecture,
  immune infiltrate patterns, and cytological markers.

  COMBINED PARTITION (34 features)
  ----------------------------------
  Full feature set used by Model A (biopsy reference upper bound).

This separation is the clinical research question:
  Can symbolic diagnostic reasoning compensate for missing biopsy information
  while preserving escalation safety and interpretability?

Usage
-----
  from src.dataset_integration.feature_partitioning import (
      clinical_partition, histopathological_partition, combined_partition,
      CLINICAL_FEATURE_NAMES, HISTOPATHOLOGICAL_FEATURE_NAMES,
  )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ── Clinical features (12) — biopsy-free ─────────────────────────────────────
#
# These 12 features can all be assessed in a standard outpatient consultation:
#   · Ordinal (0–3): erythema, scaling, definite_borders, itching
#   · Binary  (0/1): koebner_phenomenon, polygonal_papules, follicular_papules,
#                    oral_mucosal_involvement, knee_and_elbow_involvement,
#                    scalp_involvement, family_history
#   · Continuous:    age

CLINICAL_FEATURE_NAMES: tuple[str, ...] = (
    "erythema",
    "scaling",
    "definite_borders",
    "itching",
    "koebner_phenomenon",
    "polygonal_papules",
    "follicular_papules",
    "oral_mucosal_involvement",
    "knee_and_elbow_involvement",
    "scalp_involvement",
    "family_history",
    "age",
)

# ── Histopathological features (22) — biopsy-required ────────────────────────
#
# These 22 features require laboratory analysis of a tissue specimen.
# They encode the microscopic architecture of the skin biopsy.

HISTOPATHOLOGICAL_FEATURE_NAMES: tuple[str, ...] = (
    "melanin_incontinence",
    "eosinophils_in_infiltrate",
    "PNL_infiltrate",
    "fibrosis_of_papillary_dermis",
    "exocytosis",
    "acanthosis",
    "hyperkeratosis",
    "parakeratosis",
    "clubbing_of_rete_ridges",
    "elongation_of_rete_ridges",
    "thinning_of_suprapapillary_epidermis",
    "spongiform_pustule",
    "munro_microabcess",
    "focal_hypergranulosis",
    "disappearance_of_granular_layer",
    "vacuolisation_and_damage_of_basal_layer",
    "spongiosis",
    "saw_tooth_appearance_of_retes",
    "follicular_horn_plug",
    "perifollicular_parakeratosis",
    "inflammatory_mononuclear_infiltrate",
    "band_like_infiltrate",
)

# ── Combined (34) ─────────────────────────────────────────────────────────────

ALL_FEATURE_NAMES: tuple[str, ...] = (
    CLINICAL_FEATURE_NAMES + HISTOPATHOLOGICAL_FEATURE_NAMES
)

# ── Feature type sets ─────────────────────────────────────────────────────────

# Clinical ordinal: erythema/scaling/borders/itching AND the presence-absence
# features (koebner etc.) which use a 0–3 grade in the actual UCI dataset.
ORDINAL_CLINICAL: frozenset[str] = frozenset({
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement",
})

# Strictly binary (only {0, 1} appear in the real UCI dataset)
BINARY_CLINICAL: frozenset[str] = frozenset({"family_history"})

CONTINUOUS_CLINICAL: frozenset[str] = frozenset({"age"})

# All histopathological features use 0–3 ordinal grading in the dataset
ORDINAL_HISTOPATHOLOGICAL: frozenset[str] = frozenset({
    "melanin_incontinence", "eosinophils_in_infiltrate", "PNL_infiltrate",
    "fibrosis_of_papillary_dermis", "exocytosis", "acanthosis",
    "hyperkeratosis", "parakeratosis", "clubbing_of_rete_ridges",
    "elongation_of_rete_ridges", "thinning_of_suprapapillary_epidermis",
    "focal_hypergranulosis", "disappearance_of_granular_layer",
    "vacuolisation_and_damage_of_basal_layer", "spongiosis",
    "saw_tooth_appearance_of_retes",
    "inflammatory_mononuclear_infiltrate", "band_like_infiltrate",
    "spongiform_pustule", "munro_microabcess",
    "follicular_horn_plug", "perifollicular_parakeratosis",
})

BINARY_HISTOPATHOLOGICAL: frozenset[str] = frozenset()


# ── Partition enum ────────────────────────────────────────────────────────────

class FeaturePartition(str, Enum):
    """The three evaluation partitions used across all three models."""

    CLINICAL          = "clinical"           # 12 features — Model B and C
    HISTOPATHOLOGICAL = "histopathological"  # 22 features — analysis only
    COMBINED          = "combined"           # 34 features — Model A


# ── Partitioned feature set ───────────────────────────────────────────────────

@dataclass(frozen=True)
class PartitionedFeatureSet:
    """
    A named, typed partition of the full feature space.

    Attributes
    ----------
    partition:
        Which partition this represents.
    feature_names:
        Ordered tuple of feature names included in this partition.
    ordinal_features:
        Feature names that are ordinal 0–3 within this partition.
    binary_features:
        Feature names that are binary 0/1 within this partition.
    continuous_features:
        Feature names that are continuous (real-valued) within this partition.
    """

    partition:           FeaturePartition
    feature_names:       tuple[str, ...]
    ordinal_features:    frozenset[str]
    binary_features:     frozenset[str]
    continuous_features: frozenset[str]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def clinical_count(self) -> int:
        return sum(1 for f in self.feature_names if f in CLINICAL_FEATURE_NAMES)

    @property
    def histopathological_count(self) -> int:
        return sum(1 for f in self.feature_names if f in HISTOPATHOLOGICAL_FEATURE_NAMES)

    def index_of(self, feature_name: str) -> int | None:
        """Return the index of a feature within this partition, or None."""
        try:
            return self.feature_names.index(feature_name)
        except ValueError:
            return None

    def __contains__(self, feature_name: object) -> bool:
        return feature_name in self.feature_names

    def __str__(self) -> str:
        return (
            f"PartitionedFeatureSet[{self.partition.value}] "
            f"features={self.feature_count} "
            f"(ordinal={len(self.ordinal_features)}, "
            f"binary={len(self.binary_features)}, "
            f"continuous={len(self.continuous_features)})"
        )


# ── Factory functions ─────────────────────────────────────────────────────────

def clinical_partition() -> PartitionedFeatureSet:
    """
    Return the 12-feature biopsy-free clinical partition.

    This is the input space for Model B (biopsy-free baseline) and
    Model C (symbolic reasoning augmentation).
    """
    return PartitionedFeatureSet(
        partition=FeaturePartition.CLINICAL,
        feature_names=CLINICAL_FEATURE_NAMES,
        ordinal_features=ORDINAL_CLINICAL,
        binary_features=BINARY_CLINICAL,
        continuous_features=CONTINUOUS_CLINICAL,
    )


def histopathological_partition() -> PartitionedFeatureSet:
    """
    Return the 22-feature biopsy-required histopathological partition.

    Used for analytical reference and ablation studies — not for
    standalone model training in the primary evaluation.
    """
    return PartitionedFeatureSet(
        partition=FeaturePartition.HISTOPATHOLOGICAL,
        feature_names=HISTOPATHOLOGICAL_FEATURE_NAMES,
        ordinal_features=ORDINAL_HISTOPATHOLOGICAL,
        binary_features=BINARY_HISTOPATHOLOGICAL,
        continuous_features=frozenset(),
    )


def combined_partition() -> PartitionedFeatureSet:
    """
    Return the full 34-feature combined partition.

    This is the input space for Model A (full biopsy reference upper bound).
    """
    return PartitionedFeatureSet(
        partition=FeaturePartition.COMBINED,
        feature_names=ALL_FEATURE_NAMES,
        ordinal_features=ORDINAL_CLINICAL | ORDINAL_HISTOPATHOLOGICAL,
        binary_features=BINARY_CLINICAL | BINARY_HISTOPATHOLOGICAL,
        continuous_features=CONTINUOUS_CLINICAL,
    )


def get_partition(partition: FeaturePartition) -> PartitionedFeatureSet:
    """Return the PartitionedFeatureSet for the given FeaturePartition."""
    if partition == FeaturePartition.CLINICAL:
        return clinical_partition()
    if partition == FeaturePartition.HISTOPATHOLOGICAL:
        return histopathological_partition()
    return combined_partition()
