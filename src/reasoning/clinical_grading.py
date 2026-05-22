"""
ClinicalGradingModule — Stage 0 of the progressive reasoning pipeline.

Converts raw ordinal clinical grades (0–3) and binary feature values (0/1)
to fuzzy membership grades on [0.0, 1.0]. The resulting fuzzy grades drive
all downstream rule activation and certainty propagation.

Grading philosophy
------------------
Ordinal grades encode degree of clinical manifestation, not binary presence.
A grade of 1 (weak signal) should contribute partial evidence rather than
being silently excluded. This preserves clinically meaningful weak signals
that may compound with other evidence.

Default mapping (configurable):
  0 → 0.00  (absent)
  1 → 0.33  (weak)
  2 → 0.67  (moderate — clinically significant threshold)
  3 → 1.00  (severe)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


# ── Default grade map ─────────────────────────────────────────────────────────
_DEFAULT_GRADE_MAP: dict[int, float] = {
    0: 0.00,
    1: 0.33,
    2: 0.67,
    3: 1.00,
}

_CLINICAL_SIGNIFICANCE_THRESHOLD = 2  # grade >= 2 is clinically significant
_DORMANT_THRESHOLD = 0.05              # fuzzy grades below this are treated as absent
_MIN_PARTIAL_ACTIVATION = 0.10        # minimum contribution for partial rule activation


# ── Per-feature grading result ────────────────────────────────────────────────

@dataclass(frozen=True)
class GradedFeature:
    """Fuzzy-graded result for a single clinical feature."""

    feature_name:              str
    raw_value:                 int | float | None
    fuzzy_grade:               float   # [0.0, 1.0]
    is_present:                bool    # fuzzy_grade > DORMANT_THRESHOLD
    is_clinically_significant: bool    # raw_value >= significance_threshold
    evidence_strength:         str     # "absent" | "weak" | "moderate" | "strong"
    is_missing:                bool    # raw_value is None

    def partial_contribution(self, weight: float) -> float:
        """Weighted activation contribution: fuzzy_grade × weight."""
        return self.fuzzy_grade * weight


# ── Aggregate grading result ──────────────────────────────────────────────────

@dataclass
class GradingResult:
    """
    Aggregate graded feature vector for a complete clinical case.
    Produced by ClinicalGradingModule.grade_vector().
    """

    graded_features: list[GradedFeature] = field(default_factory=list)

    # ── Query helpers ─────────────────────────────────────────────────────────

    @property
    def present_features(self) -> list[GradedFeature]:
        """Features with fuzzy_grade above the dormant threshold."""
        return [f for f in self.graded_features if f.is_present]

    @property
    def significant_features(self) -> list[GradedFeature]:
        """Features at or above the clinical significance threshold."""
        return [f for f in self.graded_features if f.is_clinically_significant]

    @property
    def missing_features(self) -> list[GradedFeature]:
        """Features with None raw values."""
        return [f for f in self.graded_features if f.is_missing]

    @property
    def completeness_score(self) -> float:
        """Fraction of features that are not missing."""
        total = len(self.graded_features)
        if total == 0:
            return 0.0
        return (total - len(self.missing_features)) / total

    def get(self, feature_name: str) -> GradedFeature | None:
        """Return the GradedFeature for a given feature name, or None."""
        for f in self.graded_features:
            if f.feature_name == feature_name:
                return f
        return None

    def fuzzy_value(self, feature_name: str, default: float = 0.0) -> float:
        """Return the fuzzy grade for a feature, or default if absent."""
        f = self.get(feature_name)
        return f.fuzzy_grade if f is not None else default

    def raw_value(self, feature_name: str) -> int | float | None:
        """Return the raw ordinal/binary value for a feature."""
        f = self.get(feature_name)
        return f.raw_value if f is not None else None

    def names(self) -> list[str]:
        return [f.feature_name for f in self.graded_features]


# ── Grading module ────────────────────────────────────────────────────────────

class ClinicalGradingModule:
    """
    Converts raw ordinal and binary clinical feature values to fuzzy
    membership grades for downstream symbolic rule activation.

    Parameters
    ----------
    grade_map:
        Custom ordinal-to-fuzzy mapping. Defaults to the four-level
        standard: {0: 0.00, 1: 0.33, 2: 0.67, 3: 1.00}.
    significance_threshold:
        Ordinal grade at or above which a feature is clinically significant.
    dormant_threshold:
        Fuzzy grades at or below this value are considered absent.
    """

    _STRENGTH_BOUNDS: ClassVar[list[tuple[float, float, str]]] = [
        (0.00, 0.00, "absent"),
        (0.01, 0.34, "weak"),
        (0.34, 0.68, "moderate"),
        (0.68, 1.01, "strong"),
    ]

    def __init__(
        self,
        grade_map: dict[int, float] | None = None,
        significance_threshold: int = _CLINICAL_SIGNIFICANCE_THRESHOLD,
        dormant_threshold: float = _DORMANT_THRESHOLD,
    ) -> None:
        self._grade_map = grade_map or dict(_DEFAULT_GRADE_MAP)
        self._significance_threshold = significance_threshold
        self._dormant_threshold = dormant_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def grade_feature(
        self,
        feature_name: str,
        raw_value: int | float | None,
        is_binary: bool = False,
    ) -> GradedFeature:
        """
        Grade a single feature value.

        Missing values (None) produce fuzzy_grade = 0.0 with is_missing = True.
        Binary features map 0→0.0, 1→1.0 directly.
        Ordinal features are mapped through the grade_map.
        """
        if raw_value is None:
            return GradedFeature(
                feature_name=feature_name,
                raw_value=None,
                fuzzy_grade=0.0,
                is_present=False,
                is_clinically_significant=False,
                evidence_strength="absent",
                is_missing=True,
            )

        if is_binary:
            v = int(raw_value)
            fuzzy = 1.0 if v == 1 else 0.0
            return GradedFeature(
                feature_name=feature_name,
                raw_value=v,
                fuzzy_grade=fuzzy,
                is_present=fuzzy > self._dormant_threshold,
                is_clinically_significant=(v == 1),
                evidence_strength="strong" if fuzzy > 0.0 else "absent",
                is_missing=False,
            )

        ordinal = max(0, min(3, int(raw_value)))
        fuzzy = self._grade_map.get(ordinal, 0.0)
        return GradedFeature(
            feature_name=feature_name,
            raw_value=ordinal,
            fuzzy_grade=fuzzy,
            is_present=fuzzy > self._dormant_threshold,
            is_clinically_significant=(ordinal >= self._significance_threshold),
            evidence_strength=self._strength_label(fuzzy),
            is_missing=False,
        )

    def grade_vector(
        self,
        features: dict[str, int | float | None],
        binary_features: set[str] | None = None,
    ) -> GradingResult:
        """
        Grade a complete feature dictionary.

        Parameters
        ----------
        features:
            Mapping of feature_name → raw_value.
        binary_features:
            Set of names to treat as binary (0/1) rather than ordinal.
        """
        binary_set = binary_features or set()
        result = GradingResult()
        for name, value in features.items():
            graded = self.grade_feature(name, value, is_binary=(name in binary_set))
            result.graded_features.append(graded)
        return result

    def partial_activation(
        self,
        feature_name: str,
        raw_value: int | float | None,
        partial_weight: float,
        is_binary: bool = False,
    ) -> float:
        """
        Compute the weighted partial activation of a single feature.
        Returns fuzzy_grade × partial_weight.
        """
        graded = self.grade_feature(feature_name, raw_value, is_binary)
        return graded.fuzzy_grade * partial_weight

    def grade_severity_profile(
        self, features: dict[str, int | float | None]
    ) -> dict[str, str]:
        """
        Return a mapping of feature_name → evidence_strength label for all
        features. Useful for clinical narrative generation.
        """
        profile: dict[str, str] = {}
        for name, value in features.items():
            graded = self.grade_feature(name, value)
            profile[name] = graded.evidence_strength
        return profile

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _strength_label(self, fuzzy: float) -> str:
        if fuzzy <= 0.0:
            return "absent"
        if fuzzy <= 0.34:
            return "weak"
        if fuzzy <= 0.68:
            return "moderate"
        return "strong"
