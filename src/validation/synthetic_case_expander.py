"""
SyntheticCaseExpander — expanded clinical scenario library (30–50 cases).

Extends the eight curated base cases in SyntheticCaseLibrary to a comprehensive
40-case test corpus that covers:

  · All six disease presentations across severity gradients
  · Pathognomonic-present vs Tier-B-only evidence profiles
  · Contradiction-free, moderate, and high-load profiles
  · Confusion-zone disease pairs (known clinical mimicry)
  · Edge cases: zero features, maximum saturation, near-threshold certainty
  · Multi-disease overlaps producing mandatory escalation
  · Safe-triage cases (high certainty, low entropy, zero contradiction)
  · Age-stratified variants (paediatric, elderly) where clinically relevant

Case ID convention
------------------
  SYN_001–SYN_008  — original curated base library (SyntheticCaseLibrary)
  EXP_001–EXP_040  — expanded cases from this module (SyntheticCaseExpander)

Clinical domain groupings
-------------------------
  PSORIASIS         — EXP_001–EXP_007  (7 cases)
  SEBORRHEIC        — EXP_008–EXP_013  (6 cases)
  LICHEN_PLANUS     — EXP_014–EXP_019  (6 cases)
  PITYRIASIS_ROSEA  — EXP_020–EXP_024  (5 cases)
  CHRONIC_DERM      — EXP_025–EXP_029  (5 cases)
  PRP               — EXP_030–EXP_034  (5 cases)
  EDGE_CASES        — EXP_035–EXP_040  (6 cases)
"""

from __future__ import annotations

from typing import Iterator

from src.pipeline.synthetic_case_library import SyntheticCase, SyntheticCaseLibrary


# ── Base feature builder ──────────────────────────────────────────────────────

def _f(**overrides) -> dict[str, int | float | None]:
    """
    Produce a complete 11-feature clinical profile, defaulting all features
    to absent (0) and applying any named overrides.
    """
    base: dict[str, int | float | None] = {
        "erythema":                  0,
        "scaling":                   0,
        "definite_borders":          0,
        "itching":                   0,
        "koebner_phenomenon":        0,
        "polygonal_papules":         0,
        "follicular_papules":        0,
        "oral_mucosal_involvement":  0,
        "knee_and_elbow_involvement": 0,
        "scalp_involvement":         0,
        "family_history":            0,
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# PSORIASIS VARIANTS  (EXP_001 – EXP_007)
# ═══════════════════════════════════════════════════════════════════════════════

# EXP_001: Maximum-feature psoriasis — all supportive features at ceiling
_EXP_001 = SyntheticCase(
    case_id="EXP_001",
    description=(
        "Textbook psoriasis at maximum feature saturation: koebner phenomenon, "
        "knee and elbow involvement, scalp involvement, family history, erythema 3, "
        "scaling 3, definite borders 2, itching 2. The strongest achievable psoriasis "
        "profile; expect highest certainty and clean safe-triage path."
    ),
    feature_values=_f(
        erythema=3, scaling=3, definite_borders=2, itching=2,
        koebner_phenomenon=1, knee_and_elbow_involvement=1,
        scalp_involvement=1, family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="SAFE_NON_INVASIVE_TRIAGE",
    expect_contradiction=False,
    expect_biopsy_escalation=False,
    expect_stable=True,
    min_expected_certainty=0.65,
    tags=("psoriasis", "max_features", "safe_triage", "pathognomonic"),
)


# EXP_002: Psoriasis — koebner only, no anatomical supportive features
_EXP_002 = SyntheticCase(
    case_id="EXP_002",
    description=(
        "Psoriasis with isolated koebner phenomenon — pathognomonic present but "
        "no anatomical supportive features (knee/elbow, scalp, family history absent). "
        "Single Tier-A feature without Tier-B convergence; expect biopsy escalation "
        "due to insufficient evidence breadth despite pathognomonic presence."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=1, itching=1,
        koebner_phenomenon=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    expect_biopsy_escalation=False,
    tags=("psoriasis", "isolated_pathognomonic", "insufficient_evidence"),
)


# EXP_003: Psoriasis — anatomical features only, no koebner phenomenon
_EXP_003 = SyntheticCase(
    case_id="EXP_003",
    description=(
        "Psoriasis Tier-B profile: knee and elbow involvement, scalp involvement, "
        "family history present — but no koebner phenomenon. Good anatomical "
        "distribution without pathognomonic evidence. Expect moderate to biopsy "
        "outcome depending on certainty floor."
    ),
    feature_values=_f(
        erythema=2, scaling=3, definite_borders=2, itching=2,
        knee_and_elbow_involvement=1, scalp_involvement=1, family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    expect_stable=False,
    tags=("psoriasis", "tier_b_only", "no_pathognomonic"),
)


# EXP_004: Psoriasis with scalp + koebner — moderate certainty profile
_EXP_004 = SyntheticCase(
    case_id="EXP_004",
    description=(
        "Psoriasis with koebner phenomenon and scalp involvement only — "
        "two Tier-A/anatomical features without knee/elbow or family history. "
        "Intermediate evidence strength. Biopsy expected due to incomplete profile."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=1, itching=1,
        koebner_phenomenon=1, scalp_involvement=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("psoriasis", "partial_features"),
)


# EXP_005: Psoriasis + PRP confusion — follicular papules co-present
_EXP_005 = SyntheticCase(
    case_id="EXP_005",
    description=(
        "Psoriasis-PRP confusion zone: koebner phenomenon (psoriasis pathognomonic) "
        "co-present with follicular papules (PRP pathognomonic). High contradiction "
        "load from bilateral pathognomonic conflict. Mandatory biopsy escalation."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=1, itching=2,
        koebner_phenomenon=1, follicular_papules=1, scalp_involvement=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    tags=("psoriasis", "prp", "confusion_zone", "contradiction_heavy", "biopsy_required"),
)


# EXP_006: Psoriasis — family history + erythema, sparse profile
_EXP_006 = SyntheticCase(
    case_id="EXP_006",
    description=(
        "Psoriasis with family history and significant erythema/scaling only — "
        "no pathognomonic features, no anatomical distribution features beyond family history. "
        "Very sparse evidence; expect biopsy due to insufficient discriminative power."
    ),
    feature_values=_f(
        erythema=3, scaling=2, itching=1,
        family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("psoriasis", "sparse_evidence", "family_history_only"),
)


# EXP_007: Psoriasis + lichen planus overlap — triple pathognomonic conflict
_EXP_007 = SyntheticCase(
    case_id="EXP_007",
    description=(
        "Extreme cross-disease conflict: koebner phenomenon (psoriasis), polygonal "
        "papules + oral mucosal involvement (lichen planus), follicular papules (PRP) — "
        "four pathognomonic features from three different diseases simultaneously. "
        "Maximum contradiction load; HIGH_RISK_CONTRADICTION expected."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=2, itching=3,
        koebner_phenomenon=1, polygonal_papules=1,
        oral_mucosal_involvement=1, follicular_papules=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    max_expected_entropy=2.60,
    tags=("contradiction_heavy", "multi_disease", "biopsy_required", "maximum_conflict"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# SEBORRHEIC DERMATITIS VARIANTS  (EXP_008 – EXP_013)
# ═══════════════════════════════════════════════════════════════════════════════

# EXP_008: Seborrheic dermatitis — scalp + erythema, moderate profile
_EXP_008 = SyntheticCase(
    case_id="EXP_008",
    description=(
        "Seborrheic dermatitis: scalp involvement + moderate erythema and scaling. "
        "Characteristic sebaceous distribution without other pathognomonic features. "
        "Expect biopsy; sebaceous distribution alone is insufficient for safe triage."
    ),
    feature_values=_f(
        erythema=2, scaling=2, itching=2,
        scalp_involvement=1,
    ),
    expected_leader="seborrheic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("seborrheic_dermatitis", "scalp_distribution"),
)


# EXP_009: Seborrheic dermatitis — severe with high erythema and scaling
_EXP_009 = SyntheticCase(
    case_id="EXP_009",
    description=(
        "Severe seborrheic dermatitis: erythema 3, scaling 3, itching 2, scalp "
        "involvement. Maximum inflammation without pathognomonic markers. "
        "High evidence load for seborrheic profile; biopsy expected without "
        "discriminating Tier-A feature."
    ),
    feature_values=_f(
        erythema=3, scaling=3, itching=2, scalp_involvement=1, definite_borders=1,
    ),
    expected_leader="seborrheic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("seborrheic_dermatitis", "severe", "high_inflammation"),
)


# EXP_010: Seborrheic dermatitis — minimal scalp-only presentation
_EXP_010 = SyntheticCase(
    case_id="EXP_010",
    description=(
        "Minimal seborrheic dermatitis: scalp involvement alone with only mild "
        "erythema. Very sparse evidence — not enough to identify a clear leader "
        "with confidence. Biopsy expected for insufficient evidence."
    ),
    feature_values=_f(
        erythema=1, scaling=1,
        scalp_involvement=1,
    ),
    expected_leader="seborrheic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("seborrheic_dermatitis", "minimal_presentation", "sparse_evidence"),
)


# EXP_011: Seborrheic vs chronic dermatitis — overlapping profile
_EXP_011 = SyntheticCase(
    case_id="EXP_011",
    description=(
        "Seborrheic/chronic dermatitis overlap: moderate erythema, scaling, "
        "itching without site-specific localisation. Differential competition "
        "expected between seborrheic and chronic dermatitis. Biopsy warranted."
    ),
    feature_values=_f(
        erythema=2, scaling=2, itching=3,
    ),
    expected_leader="chronic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("seborrheic_dermatitis", "chronic_dermatitis", "differential_overlap"),
)


# EXP_012: Seborrheic dermatitis + koebner — psoriasis confusion
_EXP_012 = SyntheticCase(
    case_id="EXP_012",
    description=(
        "Seborrheic dermatitis with koebner phenomenon (psoriasis pathognomonic): "
        "scalp involvement, moderate erythema and scaling, itching, plus koebner "
        "phenomenon. Introduces psoriasis competition. Contradiction load expected "
        "from seborrheic + koebner combination. Biopsy required."
    ),
    feature_values=_f(
        erythema=2, scaling=2, itching=2,
        scalp_involvement=1, koebner_phenomenon=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    tags=("seborrheic_dermatitis", "psoriasis", "koebner", "confusion_zone"),
)


# EXP_013: Seborrheic dermatitis — high scaling sebaceous distribution
_EXP_013 = SyntheticCase(
    case_id="EXP_013",
    description=(
        "Seborrheic dermatitis with prominent scaling in sebaceous distribution: "
        "scaling 3, scalp involvement, moderate erythema, family history absent. "
        "Strong scaling signal but no discriminating pathognomonic. Biopsy expected."
    ),
    feature_values=_f(
        erythema=2, scaling=3, definite_borders=1, itching=1,
        scalp_involvement=1,
    ),
    expected_leader="seborrheic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("seborrheic_dermatitis", "prominent_scaling"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# LICHEN PLANUS VARIANTS  (EXP_014 – EXP_019)
# ═══════════════════════════════════════════════════════════════════════════════

# EXP_014: Classic lichen planus — pathognomonic dual + severe itch
_EXP_014 = SyntheticCase(
    case_id="EXP_014",
    description=(
        "Classic lichen planus: polygonal papules (pathognomonic) + oral mucosal "
        "involvement (pathognomonic) + severe itching. Dual Tier-A feature "
        "convergence. Expect biopsy (insufficient certainty floor with 6-disease "
        "competition) despite strong LP features."
    ),
    feature_values=_f(
        erythema=1, scaling=1, definite_borders=2, itching=3,
        polygonal_papules=1, oral_mucosal_involvement=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    expect_stable=False,
    tags=("lichen_planus", "dual_pathognomonic", "classic"),
)


# EXP_015: Lichen planus — polygonal papules only, no oral involvement
_EXP_015 = SyntheticCase(
    case_id="EXP_015",
    description=(
        "Lichen planus with cutaneous features only: polygonal papules without "
        "oral mucosal involvement. Single Tier-A pathognomonic plus severe itching. "
        "Biopsy expected — single pathognomonic insufficient for safe triage "
        "when only 11 features available."
    ),
    feature_values=_f(
        erythema=1, scaling=1, definite_borders=2, itching=3,
        polygonal_papules=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("lichen_planus", "cutaneous_only", "single_pathognomonic"),
)


# EXP_016: Lichen planus — oral mucosal only, no polygonal papules
_EXP_016 = SyntheticCase(
    case_id="EXP_016",
    description=(
        "Lichen planus with isolated oral mucosal involvement — no cutaneous "
        "polygonal papules. Oral LP variant. Incomplete feature profile for "
        "cutaneous LP. Biopsy expected."
    ),
    feature_values=_f(
        erythema=1, itching=2,
        oral_mucosal_involvement=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("lichen_planus", "oral_variant"),
)


# EXP_017: Lichen planus + psoriasis — koebner + polygonal papules conflict
_EXP_017 = SyntheticCase(
    case_id="EXP_017",
    description=(
        "Lichen planus-psoriasis overlap: polygonal papules (LP pathognomonic) "
        "co-occurring with koebner phenomenon (psoriasis pathognomonic). Known "
        "confusion zone. High contradiction load; mandatory biopsy."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=2, itching=3,
        koebner_phenomenon=1, polygonal_papules=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    tags=("lichen_planus", "psoriasis", "confusion_zone", "contradiction_heavy", "biopsy_required"),
)


# EXP_018: Lichen planus — moderate features, borderline certainty
_EXP_018 = SyntheticCase(
    case_id="EXP_018",
    description=(
        "Lichen planus with moderate feature intensity: polygonal papules, moderate "
        "itching and erythema, mild scaling. Intermediate evidence profile. "
        "Certainty below safe-triage floor; biopsy expected."
    ),
    feature_values=_f(
        erythema=2, scaling=1, definite_borders=1, itching=2,
        polygonal_papules=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("lichen_planus", "moderate_features", "borderline"),
)


# EXP_019: Lichen planus — high severity, all LP features present
_EXP_019 = SyntheticCase(
    case_id="EXP_019",
    description=(
        "High-severity lichen planus: polygonal papules, oral mucosal involvement, "
        "maximum itching (3), moderate erythema, definite borders — the strongest "
        "achievable LP profile within available feature set. Biopsy path expected "
        "due to 6-disease certainty dilution."
    ),
    feature_values=_f(
        erythema=2, scaling=1, definite_borders=2, itching=3,
        polygonal_papules=1, oral_mucosal_involvement=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("lichen_planus", "high_severity", "all_lp_features"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# PITYRIASIS ROSEA VARIANTS  (EXP_020 – EXP_024)
# ═══════════════════════════════════════════════════════════════════════════════

# EXP_020: Classic pityriasis rosea — definite borders + erythema + scaling
_EXP_020 = SyntheticCase(
    case_id="EXP_020",
    description=(
        "Classic pityriasis rosea presentation: definite borders (2), moderate "
        "erythema and scaling, mild itching — characteristic oval plaque morphology "
        "simulated through border definition. No pathognomonic features. Biopsy expected."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=2, itching=1,
    ),
    expected_leader="pityriasis_rosea",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("pityriasis_rosea", "classic", "defined_borders"),
)


# EXP_021: Pityriasis rosea — minimal presentation (herald patch equivalent)
_EXP_021 = SyntheticCase(
    case_id="EXP_021",
    description=(
        "Early pityriasis rosea: subtle erythema and scaling with moderate definite "
        "borders, very mild itching — simulating the herald patch phase before "
        "secondary eruption. Very sparse; biopsy expected."
    ),
    feature_values=_f(
        erythema=1, scaling=1, definite_borders=2,
    ),
    expected_leader="pityriasis_rosea",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("pityriasis_rosea", "early", "herald_patch"),
)


# EXP_022: Pityriasis rosea — secondary eruption (diffuse, moderate features)
_EXP_022 = SyntheticCase(
    case_id="EXP_022",
    description=(
        "Secondary eruption phase of pityriasis rosea: erythema 2, scaling 2, "
        "definite borders 2, itching 2 — diffuse spread with moderate intensity. "
        "Competitive with chronic dermatitis. Biopsy expected."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=2, itching=2,
    ),
    expected_leader="pityriasis_rosea",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("pityriasis_rosea", "secondary_eruption", "diffuse"),
)


# EXP_023: Pityriasis rosea + psoriasis confusion — scaling + borders + koebner
_EXP_023 = SyntheticCase(
    case_id="EXP_023",
    description=(
        "Pityriasis rosea with superimposed koebner phenomenon — known confusion "
        "zone with psoriasis. Scaling and definite borders characteristic of PR "
        "but koebner pathognomonic challenges leadership. Contradiction expected."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=2, itching=1,
        koebner_phenomenon=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    tags=("pityriasis_rosea", "psoriasis", "confusion_zone", "contradiction"),
)


# EXP_024: Pityriasis rosea — high border definition, low inflammatory features
_EXP_024 = SyntheticCase(
    case_id="EXP_024",
    description=(
        "Pityriasis rosea with strong border definition (3) but minimal erythema "
        "and scaling: atypical presentation where morphology (borders) outweighs "
        "inflammation markers. Unusual feature weighting; biopsy expected."
    ),
    feature_values=_f(
        erythema=1, scaling=1, definite_borders=3,
    ),
    expected_leader="pityriasis_rosea",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("pityriasis_rosea", "border_dominant", "atypical"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# CHRONIC DERMATITIS VARIANTS  (EXP_025 – EXP_029)
# ═══════════════════════════════════════════════════════════════════════════════

# EXP_025: Classic chronic dermatitis — erythema + itching + no specific markers
_EXP_025 = SyntheticCase(
    case_id="EXP_025",
    description=(
        "Classic chronic dermatitis presentation: severe erythema, severe itching, "
        "no pathognomonic markers, no anatomical distribution features. Non-specific "
        "but high-intensity inflammatory profile. Biopsy expected; no discriminating "
        "Tier-A evidence."
    ),
    feature_values=_f(
        erythema=3, itching=3, scaling=1,
    ),
    expected_leader="chronic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("chronic_dermatitis", "classic", "high_inflammation"),
)


# EXP_026: Chronic dermatitis — moderate across all ordinal features
_EXP_026 = SyntheticCase(
    case_id="EXP_026",
    description=(
        "Moderate chronic dermatitis: all four ordinal features at 2 — erythema, "
        "scaling, definite borders, itching all moderate. Balanced generic profile "
        "without any discriminating binary feature. Biopsy expected."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=2, itching=2,
    ),
    expected_leader="chronic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("chronic_dermatitis", "balanced_moderate"),
)


# EXP_027: Chronic dermatitis — minimal features (single symptom)
_EXP_027 = SyntheticCase(
    case_id="EXP_027",
    description=(
        "Minimal chronic dermatitis: only severe itching (3) present, all other "
        "features absent. Extremely sparse presentation — itching alone is "
        "insufficient to drive any confident diagnosis. Biopsy expected."
    ),
    feature_values=_f(
        itching=3,
    ),
    expected_leader="chronic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("chronic_dermatitis", "minimal", "single_feature"),
)


# EXP_028: Chronic dermatitis + scalp — seborrheic overlap
_EXP_028 = SyntheticCase(
    case_id="EXP_028",
    description=(
        "Chronic dermatitis with scalp involvement: erythema 2, itching 3, "
        "scalp involvement — creates seborrheic dermatitis confusion. Competition "
        "between chronic and seborrheic. Biopsy expected."
    ),
    feature_values=_f(
        erythema=2, itching=3,
        scalp_involvement=1,
    ),
    expected_leader="seborrheic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("chronic_dermatitis", "seborrheic_dermatitis", "overlap", "scalp"),
)


# EXP_029: Chronic dermatitis — erythema + itching + family history
_EXP_029 = SyntheticCase(
    case_id="EXP_029",
    description=(
        "Atopic-pattern chronic dermatitis: erythema, itching, family history "
        "present. Family history provides psoriasis Tier-B signal creating "
        "mild competition. Non-zero but sub-threshold contradiction. Biopsy expected."
    ),
    feature_values=_f(
        erythema=2, scaling=1, itching=3,
        family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("chronic_dermatitis", "psoriasis", "family_history", "atopic"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# PITYRIASIS RUBRA PILARIS VARIANTS  (EXP_030 – EXP_034)
# ═══════════════════════════════════════════════════════════════════════════════

# EXP_030: Classic PRP — follicular papules + scalp, no contradictions
_EXP_030 = SyntheticCase(
    case_id="EXP_030",
    description=(
        "Classic pityriasis rubra pilaris: follicular papules (pathognomonic) + "
        "scalp involvement + erythema + scaling. No competing pathognomonic features. "
        "Clean PRP profile; biopsy expected due to 6-disease certainty dilution."
    ),
    feature_values=_f(
        erythema=2, scaling=2, definite_borders=1, itching=1,
        follicular_papules=1, scalp_involvement=1,
    ),
    expected_leader="pityriasis_rubra_pilaris",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("pityriasis_rubra_pilaris", "classic", "pathognomonic"),
)


# EXP_031: PRP — follicular papules only, minimal inflammatory markers
_EXP_031 = SyntheticCase(
    case_id="EXP_031",
    description=(
        "Minimal PRP: follicular papules alone with mild erythema — isolated "
        "pathognomonic feature without supporting inflammatory or anatomical markers. "
        "Biopsy expected; single Tier-A insufficient for safe triage."
    ),
    feature_values=_f(
        erythema=1, scaling=1,
        follicular_papules=1,
    ),
    expected_leader="pityriasis_rubra_pilaris",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("pityriasis_rubra_pilaris", "minimal", "isolated_pathognomonic"),
)


# EXP_032: PRP — maximum features, all PRP-supportive
_EXP_032 = SyntheticCase(
    case_id="EXP_032",
    description=(
        "Severe PRP: follicular papules, scalp involvement, erythema 3, scaling 3, "
        "definite borders 2. High-severity PRP without competing pathognomonic features. "
        "Strongest achievable PRP profile. Biopsy expected."
    ),
    feature_values=_f(
        erythema=3, scaling=3, definite_borders=2, itching=2,
        follicular_papules=1, scalp_involvement=1,
    ),
    expected_leader="pityriasis_rubra_pilaris",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    min_expected_certainty=0.30,
    tags=("pityriasis_rubra_pilaris", "severe", "high_severity"),
)


# EXP_033: PRP + LP contradiction — follicular papules + polygonal papules
_EXP_033 = SyntheticCase(
    case_id="EXP_033",
    description=(
        "PRP-lichen planus overlap: follicular papules (PRP pathognomonic) + "
        "polygonal papules (LP pathognomonic). Cross-pathognomonic contradiction. "
        "High bilateral contradiction load; mandatory biopsy."
    ),
    feature_values=_f(
        erythema=2, scaling=1, itching=3,
        follicular_papules=1, polygonal_papules=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    tags=("pityriasis_rubra_pilaris", "lichen_planus", "contradiction_heavy", "biopsy_required"),
)


# EXP_034: PRP + oral mucosal — LP mimicry with follicular
_EXP_034 = SyntheticCase(
    case_id="EXP_034",
    description=(
        "PRP with oral mucosal involvement: follicular papules + oral mucosal "
        "involvement — the PRP pathognomonic alongside LP's oral pathognomonic. "
        "Creates LP-PRP confusion. High contradiction load; biopsy mandatory."
    ),
    feature_values=_f(
        erythema=2, scaling=2, itching=2,
        follicular_papules=1, oral_mucosal_involvement=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    tags=("pityriasis_rubra_pilaris", "lichen_planus", "oral_mucosal", "confusion_zone"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES  (EXP_035 – EXP_040)
# ═══════════════════════════════════════════════════════════════════════════════

# EXP_035: Zero-feature case — all features absent
_EXP_035 = SyntheticCase(
    case_id="EXP_035",
    description=(
        "Null presentation: all 11 features absent. No clinical evidence for any "
        "disease. The pipeline must handle this gracefully — certainty should be "
        "uniformly distributed (maximum entropy). Biopsy expected; no safe triage "
        "without any evidence."
    ),
    feature_values=_f(),
    expected_leader="chronic_dermatitis",   # default-mass disease under zero evidence
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    max_expected_entropy=2.60,
    tags=("edge_case", "zero_features", "null_presentation"),
)


# EXP_036: Maximum ordinal saturation — erythema 3, scaling 3, borders 3, itch 3
_EXP_036 = SyntheticCase(
    case_id="EXP_036",
    description=(
        "Maximum ordinal saturation: all four ordinal features at level 3. No binary "
        "features. Maximum non-specific inflammatory evidence without any "
        "discriminating markers. Expect high entropy; biopsy required."
    ),
    feature_values=_f(
        erythema=3, scaling=3, definite_borders=3, itching=3,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("edge_case", "max_ordinal", "non_specific"),
)


# EXP_037: All binary features present — maximum contradiction load scenario
_EXP_037 = SyntheticCase(
    case_id="EXP_037",
    description=(
        "All seven binary features simultaneously present: koebner, polygonal papules, "
        "follicular papules, oral mucosal, knee/elbow, scalp, family history — every "
        "pathognomonic activated at once. Maximum possible cross-disease contradiction; "
        "a stress test for contradiction propagation and mandatory escalation."
    ),
    feature_values=_f(
        erythema=2, scaling=2, itching=2,
        koebner_phenomenon=1, polygonal_papules=1, follicular_papules=1,
        oral_mucosal_involvement=1, knee_and_elbow_involvement=1,
        scalp_involvement=1, family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    max_expected_entropy=2.60,
    tags=("edge_case", "all_binary", "maximum_contradiction", "stress_test", "biopsy_required"),
)


# EXP_038: Near-threshold safe triage — strong psoriasis, just above floor
_EXP_038 = SyntheticCase(
    case_id="EXP_038",
    description=(
        "Near-threshold psoriasis: koebner phenomenon + knee/elbow involvement + "
        "family history + erythema 3 + scaling 3. Strong profile approaching the "
        "safe-triage certainty boundary. Outcome depends on certainty floor configuration; "
        "validated against the documented 0.72 threshold."
    ),
    feature_values=_f(
        erythema=3, scaling=3, definite_borders=2, itching=2,
        koebner_phenomenon=1, knee_and_elbow_involvement=1, family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="SAFE_NON_INVASIVE_TRIAGE",
    expect_contradiction=False,
    expect_biopsy_escalation=False,
    expect_stable=True,
    min_expected_certainty=0.65,
    tags=("edge_case", "psoriasis", "near_threshold", "safe_triage"),
)


# EXP_039: Contradiction at exactly the escalation ceiling
_EXP_039 = SyntheticCase(
    case_id="EXP_039",
    description=(
        "Designed to produce contradiction load at or near the 0.40 escalation "
        "ceiling: koebner phenomenon (psoriasis) + follicular papules (PRP) — "
        "bilateral pathognomonic conflict. Validates that I1 triggers correctly "
        "at the boundary condition."
    ),
    feature_values=_f(
        erythema=2, scaling=2, itching=2,
        koebner_phenomenon=1, follicular_papules=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    tags=("edge_case", "contradiction_boundary", "invariant_I1", "biopsy_required"),
)


# EXP_040: Borderline safe triage — psoriasis without koebner, strong Tier-B
_EXP_040 = SyntheticCase(
    case_id="EXP_040",
    description=(
        "Psoriasis profile without pathognomonic feature: knee/elbow involvement, "
        "scalp involvement, family history, erythema 3, scaling 3. Strong Tier-B "
        "convergence without Tier-A pathognomonic. Likely biopsy; validates that "
        "Tier-B alone cannot achieve safe triage in a 6-disease softmax setting."
    ),
    feature_values=_f(
        erythema=3, scaling=3, definite_borders=2, itching=2,
        knee_and_elbow_involvement=1, scalp_involvement=1, family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    expect_stable=False,
    tags=("edge_case", "psoriasis", "tier_b_only", "no_pathognomonic", "borderline"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Expanded case registry
# ═══════════════════════════════════════════════════════════════════════════════

_EXPANDED_CASES: tuple[SyntheticCase, ...] = (
    _EXP_001, _EXP_002, _EXP_003, _EXP_004, _EXP_005, _EXP_006, _EXP_007,
    _EXP_008, _EXP_009, _EXP_010, _EXP_011, _EXP_012, _EXP_013,
    _EXP_014, _EXP_015, _EXP_016, _EXP_017, _EXP_018, _EXP_019,
    _EXP_020, _EXP_021, _EXP_022, _EXP_023, _EXP_024,
    _EXP_025, _EXP_026, _EXP_027, _EXP_028, _EXP_029,
    _EXP_030, _EXP_031, _EXP_032, _EXP_033, _EXP_034,
    _EXP_035, _EXP_036, _EXP_037, _EXP_038, _EXP_039, _EXP_040,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SyntheticCaseExpander — registry interface
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticCaseExpander:
    """
    Expanded clinical case registry combining the original 8-case library
    with 40 additional scenario cases for a total of 48 clinical profiles.

    Provides the same interface as SyntheticCaseLibrary plus disease-group
    and tag-based filtering specific to the expanded set.

    Usage
    -----
    expander = SyntheticCaseExpander()
    all_cases = expander.all()                    # 48 cases
    psoriasis = expander.by_disease("psoriasis")  # cases tagged with psoriasis
    biopsy    = expander.biopsy_cases()           # expect_biopsy_escalation=True
    """

    def __init__(self) -> None:
        base = SyntheticCaseLibrary.all()
        self._all: tuple[SyntheticCase, ...] = base + _EXPANDED_CASES
        self._index: dict[str, SyntheticCase] = {c.case_id: c for c in self._all}

    # ── Full collection ───────────────────────────────────────────────────────

    def all(self) -> tuple[SyntheticCase, ...]:
        """Return all 48 synthetic cases (8 base + 40 expanded)."""
        return self._all

    def expanded_only(self) -> tuple[SyntheticCase, ...]:
        """Return only the 40 expanded cases (EXP_001–EXP_040)."""
        return _EXPANDED_CASES

    def base_only(self) -> tuple[SyntheticCase, ...]:
        """Return only the 8 original base library cases (SYN_001–SYN_008)."""
        return SyntheticCaseLibrary.all()

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, case_id: str) -> SyntheticCase:
        """Return a case by its ID. Raises KeyError if not found."""
        try:
            return self._index[case_id]
        except KeyError:
            raise KeyError(f"No synthetic case with case_id='{case_id}'.")

    def __iter__(self) -> Iterator[SyntheticCase]:
        return iter(self._all)

    def __len__(self) -> int:
        return len(self._all)

    # ── Filtering ─────────────────────────────────────────────────────────────

    def by_tag(self, tag: str) -> list[SyntheticCase]:
        """Return all cases containing the given tag."""
        return [c for c in self._all if tag in c.tags]

    def by_disease(self, disease: str) -> list[SyntheticCase]:
        """
        Return all cases tagged with the given disease name.

        Parameters
        ----------
        disease:
            One of: psoriasis, seborrheic_dermatitis, lichen_planus,
            pityriasis_rosea, chronic_dermatitis, pityriasis_rubra_pilaris.
        """
        return [c for c in self._all if disease in c.tags]

    def biopsy_cases(self) -> list[SyntheticCase]:
        """Return cases where expect_biopsy_escalation is True."""
        return [c for c in self._all if c.expect_biopsy_escalation]

    def safe_cases(self) -> list[SyntheticCase]:
        """Return cases with expected_outcome == SAFE_NON_INVASIVE_TRIAGE."""
        return [c for c in self._all if c.expected_outcome == "SAFE_NON_INVASIVE_TRIAGE"]

    def contradiction_cases(self) -> list[SyntheticCase]:
        """Return cases where expect_contradiction is True."""
        return [c for c in self._all if c.expect_contradiction]

    def stable_cases(self) -> list[SyntheticCase]:
        """Return cases where expect_stable is True."""
        return [c for c in self._all if c.expect_stable]

    def edge_cases(self) -> list[SyntheticCase]:
        """Return cases tagged as edge cases."""
        return self.by_tag("edge_case")

    def confusion_zone_cases(self) -> list[SyntheticCase]:
        """Return cases tagged as confusion zone encounters."""
        return self.by_tag("confusion_zone")

    # ── Statistics ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Return a brief descriptive summary of the expanded case corpus.
        """
        total         = len(self._all)
        biopsy_n      = len(self.biopsy_cases())
        safe_n        = len(self.safe_cases())
        contra_n      = len(self.contradiction_cases())
        edge_n        = len(self.edge_cases())
        confusion_n   = len(self.confusion_zone_cases())
        return (
            f"SyntheticCaseExpander: {total} cases total "
            f"({len(self.base_only())} base + {len(self.expanded_only())} expanded) | "
            f"biopsy_escalation={biopsy_n} | safe_triage={safe_n} | "
            f"contradiction={contra_n} | edge={edge_n} | confusion_zone={confusion_n}"
        )
