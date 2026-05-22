"""
SyntheticCaseLibrary — curated clinical cases for validation and demonstration.

Provides eight carefully constructed synthetic cases covering the full range
of diagnostic reasoning trajectories:

  1. stable_psoriasis              — strong pathognomonic evidence, clean profile
  2. stable_seborrheic_dermatitis  — convergent Tier-B evidence, no contradictions
  3. contradiction_heavy_psoriasis — simultaneous contradictory feature activation
  4. ambiguous_dermatitis          — sparse, non-discriminating feature profile
  5. competing_differential_lp_pr  — close lichen planus / pityriasis rosea contest
  6. weak_evidence_pityriasis_rosea — minimal feature presence, fragile hypothesis
  7. biopsy_escalation_prp         — follicular papules + contradictions → mandatory biopsy
  8. instability_trajectory        — oscillating certainty over multi-stage profile

Each case specifies:
  · feature_values   — raw clinical feature dict (ordinal 0–3, binary 0/1)
  · expected_outcome — expected triage recommendation string
  · expected_leader  — expected leading disease
  · expect_contradiction — whether contradiction load > 0
  · description      — clinical rationale for this profile

These cases drive:
  · Integration testing (deterministic assertion)
  · Standalone pipeline demonstration (synthetic patient simulation)
  · Publication figure generation (reasoning trajectory visualisation)
  · Frontend replay demos
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Case record ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SyntheticCase:
    """A curated synthetic clinical case for pipeline validation."""

    case_id:              str
    description:          str
    feature_values:       dict[str, int | float | None]

    # Expected reasoning outcomes (used for validation assertions)
    expected_leader:         str                    # leading disease
    expected_outcome:        str                    # TriageRecommendation value
    expect_contradiction:    bool = False           # contradiction_load > 0
    expect_biopsy_escalation: bool = False          # triage requires biopsy
    expect_stable:           bool = False           # is_stable certainty
    max_expected_entropy:    float | None = None    # optional entropy ceiling
    min_expected_certainty:  float | None = None    # optional certainty floor

    tags: tuple[str, ...] = field(default_factory=tuple)


# ── Complete feature template ─────────────────────────────────────────────────

def _base_features(**overrides) -> dict[str, int | float | None]:
    """
    Generate a complete 11-feature clinical profile with all features absent
    (ordinal = 0, binary = 0), then apply named overrides.
    """
    base = {
        # Ordinal (0–3)
        "erythema":       0,
        "scaling":        0,
        "definite_borders": 0,
        "itching":        0,
        # Binary (0/1)
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


# ── Synthetic case definitions ────────────────────────────────────────────────

# ── Case 1: Strong psoriasis — classic pathognomonic + supportive profile ──────

STABLE_PSORIASIS = SyntheticCase(
    case_id="SYN_001",
    description=(
        "Classic psoriasis presentation: koebner phenomenon (pathognomonic), "
        "symmetric knee and elbow involvement, scalp involvement, family history, "
        "and clinically significant erythema with scaling. No confounding features."
    ),
    feature_values=_base_features(
        erythema=3,
        scaling=3,
        definite_borders=2,
        itching=2,
        koebner_phenomenon=1,
        knee_and_elbow_involvement=1,
        scalp_involvement=1,
        family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="SAFE_NON_INVASIVE_TRIAGE",
    expect_contradiction=False,
    expect_biopsy_escalation=False,
    expect_stable=True,
    min_expected_certainty=0.65,
    tags=("psoriasis", "pathognomonic", "stable", "safe_triage"),
)


# ── Case 2: Seborrheic dermatitis — convergent Tier-B evidence ────────────────

STABLE_SEBORRHEIC_DERMATITIS = SyntheticCase(
    case_id="SYN_002",
    description=(
        "Seborrheic dermatitis with convergent Tier-B evidence: moderate erythema "
        "and scaling in sebaceous distribution (scalp involvement), itching, no "
        "pathognomonic features. Without a Tier-A discriminating feature the "
        "certainty cannot collapse below the evidence-sufficiency floor; the "
        "low-certainty safety path triggers biopsy escalation despite zero "
        "contradiction load. Leading disease is correctly identified."
    ),
    feature_values=_base_features(
        erythema=2,
        scaling=2,
        definite_borders=1,
        itching=2,
        scalp_involvement=1,
    ),
    expected_leader="seborrheic_dermatitis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    expect_biopsy_escalation=False,
    expect_stable=False,
    tags=("seborrheic_dermatitis", "weak_certainty", "biopsy_path"),
)


# ── Case 3: Contradiction-heavy psoriasis ─────────────────────────────────────

CONTRADICTION_HEAVY_PSORIASIS = SyntheticCase(
    case_id="SYN_003",
    description=(
        "Psoriasis profile with simultaneous lichen planus and PRP features: "
        "koebner phenomenon (psoriasis pathognomonic) co-occurs with polygonal papules "
        "and oral mucosal involvement (LP pathognomonic features), plus follicular "
        "papules (PRP pathognomonic). High cross-disease contradiction load. "
        "Mandatory biopsy escalation expected."
    ),
    feature_values=_base_features(
        erythema=2,
        scaling=2,
        definite_borders=2,
        itching=3,
        koebner_phenomenon=1,
        polygonal_papules=1,
        follicular_papules=1,
        oral_mucosal_involvement=1,
    ),
    expected_leader="lichen_planus",   # LP Tier-A features dominate under heavy contradiction
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    max_expected_entropy=2.50,
    tags=("contradiction_heavy", "biopsy_required", "cross_disease"),
)


# ── Case 4: Ambiguous presentation ───────────────────────────────────────────

AMBIGUOUS_DERMATITIS = SyntheticCase(
    case_id="SYN_004",
    description=(
        "Sparse, non-discriminating feature profile: mild erythema and scaling only, "
        "no pathognomonic features, no binary features present. Evidence is insufficient "
        "to favour any single hypothesis. Expected to reach ambiguous presentation "
        "due to low certainty and inadequate evidence."
    ),
    feature_values=_base_features(
        erythema=1,
        scaling=1,
        itching=1,
    ),
    expected_leader="chronic_dermatitis",  # may vary with sparse evidence
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    expect_biopsy_escalation=False,
    tags=("ambiguous", "sparse_evidence", "weak_features"),
)


# ── Case 5: Competing differential — lichen planus vs pityriasis rosea ─────

COMPETING_DIFFERENTIAL_LP_PR = SyntheticCase(
    case_id="SYN_005",
    description=(
        "Close competition between lichen planus and pityriasis rosea: "
        "polygonal papules (LP Tier-A) co-present with only moderate scaling and "
        "erythema. Itching is severe (LP supportive). No koebner phenomenon, "
        "no follicular papules. The Tier-A signal for LP is insufficient to "
        "suppress the ambiguity mass across the remaining diseases; certainty "
        "remains below the resolution floor, warranting biopsy escalation. "
        "Lichen planus correctly identified as leading hypothesis."
    ),
    feature_values=_base_features(
        erythema=2,
        scaling=1,
        definite_borders=2,
        itching=3,
        polygonal_papules=1,
    ),
    expected_leader="lichen_planus",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    expect_stable=False,
    tags=("lichen_planus", "competing_differential", "confusion_zone"),
)


# ── Case 6: Weak evidence pityriasis rosea ────────────────────────────────────

WEAK_EVIDENCE_PITYRIASIS_ROSEA = SyntheticCase(
    case_id="SYN_006",
    description=(
        "Minimal pityriasis rosea feature presence: moderate scaling, moderate erythema, "
        "mild definite borders. No pathognomonic features for any disease. Evidence "
        "insufficient for safe non-invasive triage. Fragility expected to be high."
    ),
    feature_values=_base_features(
        erythema=2,
        scaling=2,
        definite_borders=1,
        itching=1,
    ),
    expected_leader="pityriasis_rosea",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    tags=("pityriasis_rosea", "weak_evidence", "fragile"),
)


# ── Case 7: Biopsy escalation — pityriasis rubra pilaris ─────────────────────

BIOPSY_ESCALATION_PRP = SyntheticCase(
    case_id="SYN_007",
    description=(
        "Pityriasis rubra pilaris with high contradiction load: follicular papules "
        "(PRP pathognomonic) present alongside koebner phenomenon (psoriasis "
        "pathognomonic) and scalp involvement. The PRP→psoriasis contradiction "
        "and PSO→PRP contradiction together breach the mandatory escalation ceiling. "
        "Biopsy recommended by safety invariant I1."
    ),
    feature_values=_base_features(
        erythema=2,
        scaling=2,
        definite_borders=2,
        itching=2,
        koebner_phenomenon=1,
        follicular_papules=1,
        scalp_involvement=1,
    ),
    expected_leader="pityriasis_rubra_pilaris",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=True,
    expect_biopsy_escalation=True,
    tags=("pityriasis_rubra_pilaris", "biopsy_escalation", "pathognomonic_conflict"),
)


# ── Case 8: Instability trajectory ───────────────────────────────────────────

INSTABILITY_TRAJECTORY = SyntheticCase(
    case_id="SYN_008",
    description=(
        "Deliberately mixed evidence profile producing an unstable reasoning "
        "trajectory: moderate erythema and scaling with all binary features absent "
        "except itching, creating a flat differential with low discriminative power. "
        "This profile is expected to produce high ambiguity and divergent hypotheses."
    ),
    feature_values=_base_features(
        erythema=2,
        scaling=2,
        definite_borders=1,
        itching=3,
        family_history=1,
    ),
    expected_leader="psoriasis",
    expected_outcome="BIOPSY_RECOMMENDED",
    expect_contradiction=False,
    max_expected_entropy=2.40,
    tags=("instability", "ambiguous", "non_discriminating"),
)


# ── Case registry ─────────────────────────────────────────────────────────────

ALL_CASES: tuple[SyntheticCase, ...] = (
    STABLE_PSORIASIS,
    STABLE_SEBORRHEIC_DERMATITIS,
    CONTRADICTION_HEAVY_PSORIASIS,
    AMBIGUOUS_DERMATITIS,
    COMPETING_DIFFERENTIAL_LP_PR,
    WEAK_EVIDENCE_PITYRIASIS_ROSEA,
    BIOPSY_ESCALATION_PRP,
    INSTABILITY_TRAJECTORY,
)


class SyntheticCaseLibrary:
    """Registry interface for the curated synthetic case collection."""

    @staticmethod
    def all() -> tuple[SyntheticCase, ...]:
        """Return all curated synthetic cases."""
        return ALL_CASES

    @staticmethod
    def get(case_id: str) -> SyntheticCase:
        """Return a case by its ID."""
        for case in ALL_CASES:
            if case.case_id == case_id:
                return case
        raise KeyError(f"No synthetic case with case_id='{case_id}'.")

    @staticmethod
    def filter_by_tag(tag: str) -> list[SyntheticCase]:
        """Return all cases containing the given tag."""
        return [c for c in ALL_CASES if tag in c.tags]

    @staticmethod
    def biopsy_cases() -> list[SyntheticCase]:
        return [c for c in ALL_CASES if c.expect_biopsy_escalation]

    @staticmethod
    def safe_cases() -> list[SyntheticCase]:
        return [c for c in ALL_CASES if c.expected_outcome == "SAFE_NON_INVASIVE_TRIAGE"]
