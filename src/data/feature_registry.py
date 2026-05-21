"""
Feature Registry — authoritative metadata for all 34 clinical and
histopathological features in the UCI Dermatology dataset.

This is the single source of truth for feature semantics. All pipeline
subsystems must query the registry rather than hardcoding feature names
or types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class FeatureType(str, Enum):
    ORDINAL    = "ordinal"
    BINARY     = "binary"
    CONTINUOUS = "continuous"


class FeatureGroup(str, Enum):
    CLINICAL           = "clinical"
    HISTOPATHOLOGICAL  = "histopathological"


class AnatomicalCategory(str, Enum):
    SKIN_SURFACE    = "skin_surface"
    SKIN_APPENDAGE  = "skin_appendage"
    MUCOSAL         = "mucosal"
    JOINT_SURFACE   = "joint_surface"
    SCALP           = "scalp"
    SYSTEMIC        = "systemic"
    DEMOGRAPHIC     = "demographic"
    DERMAL          = "dermal"
    EPIDERMAL       = "epidermal"
    VASCULAR        = "vascular"
    CELLULAR        = "cellular"


@dataclass(frozen=True)
class FeatureMetadata:
    """Complete metadata record for a single clinical or histopathological feature."""

    # Identity
    canonical_name:     str
    feature_type:       FeatureType
    feature_group:      FeatureGroup
    abbreviation:       str
    visualization_label: str

    # Range and semantics
    ordinal_min:        int | None
    ordinal_max:        int | None
    semantic_meaning:   str
    diagnostic_relevance: str
    anatomical_category: AnatomicalCategory

    # Eligibility flags
    eligible_for_symbolic_reasoning:  bool = True
    eligible_for_biopsy_baseline:     bool = True  # Model A uses all features
    is_pathognomonic_indicator:       bool = False
    is_critical_discriminator:        bool = False

    # UCI column index (0-based, pre-drop of target column)
    uci_column_index: int | None = None


class FeatureRegistry:
    """
    Authoritative registry of all 34 features in the UCI Dermatology dataset.

    Usage
    -----
    registry = FeatureRegistry()
    meta = registry.get("koebner_phenomenon")
    clinical = registry.clinical_features()
    """

    _records: ClassVar[list[FeatureMetadata]] = [
        # ── Clinical — Ordinal (4) ────────────────────────────────────────────
        FeatureMetadata(
            canonical_name="erythema",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="ERYT",
            visualization_label="Erythema",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning=(
                "Degree of redness reflecting superficial dermal vasodilation. "
                "Grade 0 = none; 3 = severe vivid erythema."
            ),
            diagnostic_relevance=(
                "Non-specific but contributes composite evidence in psoriasis (bright red), "
                "PRP (salmon-orange), and chronic dermatitis (dull erythema)."
            ),
            anatomical_category=AnatomicalCategory.SKIN_SURFACE,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=False,
            is_critical_discriminator=False,
            uci_column_index=0,
        ),
        FeatureMetadata(
            canonical_name="scaling",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="SCAL",
            visualization_label="Scaling",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning=(
                "Degree of surface scale (hyperkeratosis visible at clinical examination). "
                "Grade 0 = none; 3 = severe adherent scale."
            ),
            diagnostic_relevance=(
                "Silvery adherent scale (PSO), greasy yellowish scale (SD), "
                "fine crinkled scale (PR), lamellar scale (PRP)."
            ),
            anatomical_category=AnatomicalCategory.SKIN_SURFACE,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            uci_column_index=1,
        ),
        FeatureMetadata(
            canonical_name="definite_borders",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="BORD",
            visualization_label="Definite Borders",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning=(
                "Sharpness of lesion margin. Grade 0 = indistinct; 3 = sharply demarcated."
            ),
            diagnostic_relevance=(
                "Sharp borders: psoriatic plaque, lichen planus, pityriasis rosea. "
                "Indistinct: seborrheic dermatitis, chronic dermatitis."
            ),
            anatomical_category=AnatomicalCategory.SKIN_SURFACE,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            uci_column_index=2,
        ),
        FeatureMetadata(
            canonical_name="itching",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="ITCH",
            visualization_label="Itching",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning=(
                "Severity of pruritus reported by patient. Grade 0 = none; 3 = severe."
            ),
            diagnostic_relevance=(
                "Severe pruritus: lichen planus, chronic dermatitis, seborrheic dermatitis. "
                "Mild: pityriasis rosea. Variable: psoriasis."
            ),
            anatomical_category=AnatomicalCategory.SYSTEMIC,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            uci_column_index=3,
        ),

        # ── Clinical — Binary (7) ─────────────────────────────────────────────
        FeatureMetadata(
            canonical_name="koebner_phenomenon",
            feature_type=FeatureType.BINARY,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="KOEB",
            visualization_label="Koebner Phenomenon",
            ordinal_min=0, ordinal_max=1,
            semantic_meaning=(
                "Appearance of new lesions at sites of cutaneous trauma (isomorphic response)."
            ),
            diagnostic_relevance=(
                "Pathognomonic for psoriasis among erythemato-squamous diseases. "
                "Absent in seborrheic dermatitis, PRP, and chronic dermatitis."
            ),
            anatomical_category=AnatomicalCategory.SKIN_SURFACE,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            is_critical_discriminator=True,
            uci_column_index=4,
        ),
        FeatureMetadata(
            canonical_name="polygonal_papules",
            feature_type=FeatureType.BINARY,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="POLY",
            visualization_label="Polygonal Papules",
            ordinal_min=0, ordinal_max=1,
            semantic_meaning=(
                "Presence of flat-topped, angulated (polygonal) papules — the '6 Ps' morphology."
            ),
            diagnostic_relevance=(
                "Pathognomonic for lichen planus. Absent in all other erythemato-squamous diseases."
            ),
            anatomical_category=AnatomicalCategory.SKIN_SURFACE,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            is_critical_discriminator=True,
            uci_column_index=5,
        ),
        FeatureMetadata(
            canonical_name="follicular_papules",
            feature_type=FeatureType.BINARY,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="FOLL",
            visualization_label="Follicular Papules",
            ordinal_min=0, ordinal_max=1,
            semantic_meaning=(
                "Perifollicular keratotic papules producing a nutmeg-grater texture."
            ),
            diagnostic_relevance=(
                "Pathognomonic for pityriasis rubra pilaris. Absent in all other differentials."
            ),
            anatomical_category=AnatomicalCategory.SKIN_APPENDAGE,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            is_critical_discriminator=True,
            uci_column_index=6,
        ),
        FeatureMetadata(
            canonical_name="oral_mucosal_involvement",
            feature_type=FeatureType.BINARY,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="ORAL",
            visualization_label="Oral Mucosal Involvement",
            ordinal_min=0, ordinal_max=1,
            semantic_meaning=(
                "Presence of oral mucosal lesions: Wickham's striae, reticular white patches, "
                "or erosive lesions on buccal mucosa or tongue."
            ),
            diagnostic_relevance=(
                "Tier-A evidence for lichen planus. Strongly contradicts psoriasis, "
                "seborrheic dermatitis, and PRP."
            ),
            anatomical_category=AnatomicalCategory.MUCOSAL,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            is_critical_discriminator=True,
            uci_column_index=7,
        ),
        FeatureMetadata(
            canonical_name="knee_and_elbow_involvement",
            feature_type=FeatureType.BINARY,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="KNEE",
            visualization_label="Knee/Elbow Involvement",
            ordinal_min=0, ordinal_max=1,
            semantic_meaning=(
                "Bilateral symmetric involvement of extensor surfaces of knees and elbows."
            ),
            diagnostic_relevance=(
                "Cardinal topographic feature of chronic plaque psoriasis. "
                "Distinguishes from seborrheic dermatitis (sebaceous distribution) "
                "and pityriasis rosea (truncal Christmas-tree)."
            ),
            anatomical_category=AnatomicalCategory.JOINT_SURFACE,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            uci_column_index=8,
        ),
        FeatureMetadata(
            canonical_name="scalp_involvement",
            feature_type=FeatureType.BINARY,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="SCAP",
            visualization_label="Scalp Involvement",
            ordinal_min=0, ordinal_max=1,
            semantic_meaning=(
                "Presence of lesions on the scalp, particularly at the hairline."
            ),
            diagnostic_relevance=(
                "Present in ~80% of psoriasis and very common in seborrheic dermatitis. "
                "Helps distinguish scalp-predominant conditions."
            ),
            anatomical_category=AnatomicalCategory.SCALP,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            uci_column_index=9,
        ),
        FeatureMetadata(
            canonical_name="family_history",
            feature_type=FeatureType.BINARY,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="FAMH",
            visualization_label="Family History",
            ordinal_min=0, ordinal_max=1,
            semantic_meaning=(
                "Positive family history of similar skin disease in a first-degree relative."
            ),
            diagnostic_relevance=(
                "Independent Bayesian prior for psoriasis (polygenic inheritance). "
                "Less specific but contributes composite evidence."
            ),
            anatomical_category=AnatomicalCategory.SYSTEMIC,
            eligible_for_symbolic_reasoning=True,
            eligible_for_biopsy_baseline=True,
            uci_column_index=10,
        ),

        # ── Clinical — Continuous (1) ─────────────────────────────────────────
        FeatureMetadata(
            canonical_name="age",
            feature_type=FeatureType.CONTINUOUS,
            feature_group=FeatureGroup.CLINICAL,
            abbreviation="AGE",
            visualization_label="Patient Age",
            ordinal_min=None, ordinal_max=None,
            semantic_meaning="Patient age in years at presentation.",
            diagnostic_relevance=(
                "Age of onset provides context: psoriasis bimodal (20s, 50s); "
                "seborrheic dermatitis common in adults; lichen planus: 30–60 years."
            ),
            anatomical_category=AnatomicalCategory.DEMOGRAPHIC,
            eligible_for_symbolic_reasoning=False,  # not used in clinical rules
            eligible_for_biopsy_baseline=True,
            uci_column_index=33,
        ),

        # ── Histopathological — Ordinal (22) ─────────────────────────────────
        FeatureMetadata(
            canonical_name="melanin_incontinence",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="MELI",
            visualization_label="Melanin Incontinence",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Melanin pigment within dermal macrophages due to basal layer damage.",
            diagnostic_relevance="Strong histological indicator of lichen planus (interface dermatitis).",
            anatomical_category=AnatomicalCategory.DERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=11,
        ),
        FeatureMetadata(
            canonical_name="eosinophils_in_the_infiltrate",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="EOSI",
            visualization_label="Eosinophils in Infiltrate",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Eosinophil count in the dermal inflammatory infiltrate.",
            diagnostic_relevance="Suggests allergic/drug reactions; relatively non-specific in this set.",
            anatomical_category=AnatomicalCategory.CELLULAR,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=12,
        ),
        FeatureMetadata(
            canonical_name="PNL_infiltrate",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="PNLI",
            visualization_label="PNL Infiltrate",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Polymorphonuclear leukocyte infiltrate density.",
            diagnostic_relevance="Elevated in psoriasis (Munro microabscesses); reduced in LP.",
            anatomical_category=AnatomicalCategory.CELLULAR,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=13,
        ),
        FeatureMetadata(
            canonical_name="fibrosis_of_the_papillary_dermis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="FIBR",
            visualization_label="Papillary Dermal Fibrosis",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Degree of fibrosis in the papillary dermis.",
            diagnostic_relevance="Seen in chronic lichen planus and hypertrophic lichen planus.",
            anatomical_category=AnatomicalCategory.DERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=14,
        ),
        FeatureMetadata(
            canonical_name="exocytosis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="EXOC",
            visualization_label="Exocytosis",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Migration of inflammatory cells into the epidermis.",
            diagnostic_relevance="Prominent in eczematous conditions (chronic dermatitis, seborrheic).",
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=15,
        ),
        FeatureMetadata(
            canonical_name="acanthosis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="ACAN",
            visualization_label="Acanthosis",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Epidermal hyperplasia (thickening of the stratum spinosum).",
            diagnostic_relevance="Prominent in psoriasis, PRP, and chronic lichen planus.",
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=16,
        ),
        FeatureMetadata(
            canonical_name="hyperkeratosis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="HKEY",
            visualization_label="Hyperkeratosis",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Thickening of the stratum corneum.",
            diagnostic_relevance="Prominent in psoriasis, PRP, and chronic dermatitis.",
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=17,
        ),
        FeatureMetadata(
            canonical_name="parakeratosis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="PARA",
            visualization_label="Parakeratosis",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning=(
                "Retention of nuclei in the stratum corneum (incomplete keratinisation)."
            ),
            diagnostic_relevance=(
                "Pathological hallmark of psoriasis; also seen in PRP and seborrheic dermatitis."
            ),
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=18,
        ),
        FeatureMetadata(
            canonical_name="clubbing_of_the_rete_ridges",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="RECC",
            visualization_label="Clubbing of Rete Ridges",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Club-shaped widening of rete ridge tips.",
            diagnostic_relevance="Characteristic of psoriasis in the psoriasiform reaction pattern.",
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=19,
        ),
        FeatureMetadata(
            canonical_name="elongation_of_the_rete_ridges",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="REEL",
            visualization_label="Elongation of Rete Ridges",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Downward elongation of rete ridges into the dermis.",
            diagnostic_relevance="Part of psoriasiform hyperplasia pattern; seen in psoriasis and PRP.",
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=20,
        ),
        FeatureMetadata(
            canonical_name="thinning_of_the_suprapapillary_epidermis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="SUPT",
            visualization_label="Suprapapillary Thinning",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Thinning of epidermis overlying the dermal papillae.",
            diagnostic_relevance=(
                "Hallmark of psoriasis; supports the Auspitz sign (pinpoint bleeding on scale removal)."
            ),
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=21,
        ),
        FeatureMetadata(
            canonical_name="spongiform_pustule",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="SPPU",
            visualization_label="Spongiform Pustule",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Spongiform collection of neutrophils in the spinous layer.",
            diagnostic_relevance="Characteristic of pustular variants of psoriasis.",
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=22,
        ),
        FeatureMetadata(
            canonical_name="munro_microabcess",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="MUNR",
            visualization_label="Munro Microabscess",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Neutrophil collections in the parakeratotic stratum corneum.",
            diagnostic_relevance="Pathognomonic histological feature of psoriasis.",
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            uci_column_index=23,
        ),
        FeatureMetadata(
            canonical_name="focal_hypergranulosis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="FHYP",
            visualization_label="Focal Hypergranulosis",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Focal thickening of the granular cell layer.",
            diagnostic_relevance="Characteristic of lichen planus (wedge-shaped hypergranulosis).",
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=24,
        ),
        FeatureMetadata(
            canonical_name="disappearance_of_the_granular_layer",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="DGRL",
            visualization_label="Granular Layer Loss",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Reduction or absence of the stratum granulosum.",
            diagnostic_relevance=(
                "Seen in psoriasis (parakeratosis-associated) and PRP. "
                "Contrasts with hypergranulosis in lichen planus."
            ),
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=25,
        ),
        FeatureMetadata(
            canonical_name="vacuolisation_and_damage_of_basal_layer",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="VACB",
            visualization_label="Basal Layer Vacuolisation",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Vacuolar degeneration of basal keratinocytes at the DEJ.",
            diagnostic_relevance=(
                "Hallmark of interface dermatitis — characteristic of lichen planus. "
                "Produces melanin incontinence as collateral damage."
            ),
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            uci_column_index=26,
        ),
        FeatureMetadata(
            canonical_name="spongiosis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="SPON",
            visualization_label="Spongiosis",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Intercellular oedema of the epidermis.",
            diagnostic_relevance=(
                "Characteristic of eczematous pattern: seborrheic dermatitis and chronic dermatitis. "
                "Absent or minimal in psoriasis."
            ),
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=27,
        ),
        FeatureMetadata(
            canonical_name="saw_tooth_appearance_of_retes",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="SAWT",
            visualization_label="Saw-tooth Rete Pattern",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Irregular saw-tooth pattern of the rete ridges.",
            diagnostic_relevance=(
                "Characteristic of lichen planus — the irregular rete architecture reflects "
                "the band-like inflammatory infiltrate remodelling the DEJ."
            ),
            anatomical_category=AnatomicalCategory.EPIDERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            uci_column_index=28,
        ),
        FeatureMetadata(
            canonical_name="follicular_horn_plug",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="FHPL",
            visualization_label="Follicular Horn Plug",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Keratin plugs within follicular infundibula.",
            diagnostic_relevance=(
                "Histological counterpart of the clinical follicular papule; "
                "characteristic of pityriasis rubra pilaris."
            ),
            anatomical_category=AnatomicalCategory.SKIN_APPENDAGE,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            uci_column_index=29,
        ),
        FeatureMetadata(
            canonical_name="perifollicular_parakeratosis",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="PFPK",
            visualization_label="Perifollicular Parakeratosis",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Parakeratosis localised around the follicular unit.",
            diagnostic_relevance=(
                "Highly characteristic of PRP; rarely seen in other conditions in this set."
            ),
            anatomical_category=AnatomicalCategory.SKIN_APPENDAGE,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            uci_column_index=30,
        ),
        FeatureMetadata(
            canonical_name="inflammatory_monoluclear_infiltrate",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="MONI",
            visualization_label="Mononuclear Infiltrate",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning="Density of mononuclear (lymphocytic) inflammatory infiltrate.",
            diagnostic_relevance=(
                "Present in most inflammatory dermatoses; density and pattern guide subtyping."
            ),
            anatomical_category=AnatomicalCategory.DERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            uci_column_index=31,
        ),
        FeatureMetadata(
            canonical_name="band_like_infiltrate",
            feature_type=FeatureType.ORDINAL,
            feature_group=FeatureGroup.HISTOPATHOLOGICAL,
            abbreviation="BAND",
            visualization_label="Band-like Infiltrate",
            ordinal_min=0, ordinal_max=3,
            semantic_meaning=(
                "Dense band-like lymphocytic infiltrate hugging the dermo-epidermal junction."
            ),
            diagnostic_relevance=(
                "Pathognomonic histological hallmark of lichen planus (lichenoid reaction pattern)."
            ),
            anatomical_category=AnatomicalCategory.DERMAL,
            eligible_for_symbolic_reasoning=False,
            eligible_for_biopsy_baseline=True,
            is_pathognomonic_indicator=True,
            uci_column_index=32,
        ),
    ]

    def __init__(self) -> None:
        self._index: dict[str, FeatureMetadata] = {
            r.canonical_name: r for r in self._records
        }

    def get(self, name: str) -> FeatureMetadata:
        if name not in self._index:
            raise KeyError(f"Unknown feature: '{name}'. Check canonical_name spelling.")
        return self._index[name]

    def all_features(self) -> list[FeatureMetadata]:
        return list(self._records)

    def clinical_features(self) -> list[FeatureMetadata]:
        return [r for r in self._records if r.feature_group == FeatureGroup.CLINICAL]

    def histopathological_features(self) -> list[FeatureMetadata]:
        return [r for r in self._records if r.feature_group == FeatureGroup.HISTOPATHOLOGICAL]

    def symbolic_features(self) -> list[FeatureMetadata]:
        return [r for r in self._records if r.eligible_for_symbolic_reasoning]

    def pathognomonic_features(self) -> list[FeatureMetadata]:
        return [r for r in self._records if r.is_pathognomonic_indicator]

    def critical_discriminators(self) -> list[FeatureMetadata]:
        return [r for r in self._records if r.is_critical_discriminator]

    def by_type(self, feature_type: FeatureType) -> list[FeatureMetadata]:
        return [r for r in self._records if r.feature_type == feature_type]

    def names(self, group: FeatureGroup | None = None) -> list[str]:
        if group is None:
            return [r.canonical_name for r in self._records]
        return [r.canonical_name for r in self._records if r.feature_group == group]
