"""
Dataset Validation Layer.

Validates the UCI Dermatology dataset against expected schema, value ranges,
class distribution, and missingness constraints before any inference or
training proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__, subsystem="DatasetValidation")

EXPECTED_N_SAMPLES = 366
EXPECTED_N_FEATURES = 34
EXPECTED_CLASSES = {1, 2, 3, 4, 5, 6}
EXPECTED_CLASS_COUNTS = {
    1: 112,  # psoriasis
    2: 61,   # seborrheic_dermatitis
    3: 72,   # lichen_planus
    4: 49,   # pityriasis_rosea
    5: 52,   # chronic_dermatitis
    6: 20,   # pityriasis_rubra_pilaris
}

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


@dataclass
class ValidationIssue:
    severity:  str  # "error" | "warning"
    check:     str
    message:   str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, check: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, check, message))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Validation {'PASSED' if self.is_valid else 'FAILED'}: "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        ]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.upper()}] {issue.check}: {issue.message}")
        return "\n".join(lines)


class DatasetValidator:
    """
    Validates the loaded UCI Dermatology DataFrame against the expected schema.

    Parameters
    ----------
    strict:
        If True, raise ValueError on any validation error. If False, return
        the ValidationReport and let the caller decide.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ValidationReport:
        """
        Run all validation checks. Returns a ValidationReport.
        Raises ValueError if strict=True and any errors are found.
        """
        report = ValidationReport()

        self._check_sample_count(X, report)
        self._check_feature_count(X, report)
        self._check_class_labels(y, report)
        self._check_class_distribution(y, report)
        self._check_ordinal_ranges(X, report)
        self._check_binary_ranges(X, report)
        self._check_age_range(X, report)
        self._check_missingness(X, report)

        summary = report.summary()
        if report.is_valid:
            log.info("Dataset validation passed", **self._counts(report))
        else:
            log.error("Dataset validation failed", **self._counts(report))

        if self.strict and not report.is_valid:
            raise ValueError(
                f"Dataset validation failed with {len(report.errors)} error(s).\n"
                + summary
            )

        return report

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_sample_count(self, X: pd.DataFrame, report: ValidationReport) -> None:
        n = len(X)
        if n != EXPECTED_N_SAMPLES:
            report.add(
                "warning", "sample_count",
                f"Expected {EXPECTED_N_SAMPLES} samples; got {n}."
            )
        else:
            log.debug("Sample count OK", n=n)

    def _check_feature_count(self, X: pd.DataFrame, report: ValidationReport) -> None:
        n = X.shape[1]
        if n != EXPECTED_N_FEATURES:
            report.add(
                "error", "feature_count",
                f"Expected {EXPECTED_N_FEATURES} features; got {n}."
            )

    def _check_class_labels(self, y: pd.Series, report: ValidationReport) -> None:
        observed = set(y.unique())
        unexpected = observed - EXPECTED_CLASSES
        if unexpected:
            report.add(
                "error", "class_labels",
                f"Unexpected class labels: {sorted(unexpected)}. "
                f"Expected: {sorted(EXPECTED_CLASSES)}."
            )

    def _check_class_distribution(self, y: pd.Series, report: ValidationReport) -> None:
        counts = y.value_counts().to_dict()
        for cls, expected_count in EXPECTED_CLASS_COUNTS.items():
            actual = counts.get(cls, 0)
            if actual != expected_count:
                report.add(
                    "warning", "class_distribution",
                    f"Class {cls}: expected {expected_count} samples, got {actual}."
                )

    def _check_ordinal_ranges(self, X: pd.DataFrame, report: ValidationReport) -> None:
        for col in ORDINAL_FEATURES:
            if col not in X.columns:
                report.add("error", "ordinal_range", f"Missing ordinal feature: '{col}'.")
                continue
            series = X[col].dropna()
            out_of_range = series[(series < 0) | (series > 3)]
            if len(out_of_range) > 0:
                report.add(
                    "error", "ordinal_range",
                    f"Feature '{col}' has {len(out_of_range)} value(s) outside [0, 3]. "
                    f"Invalid values: {sorted(out_of_range.unique().tolist())}."
                )

    def _check_binary_ranges(self, X: pd.DataFrame, report: ValidationReport) -> None:
        for col in BINARY_FEATURES:
            if col not in X.columns:
                report.add("error", "binary_range", f"Missing binary feature: '{col}'.")
                continue
            series = X[col].dropna()
            out_of_range = series[~series.isin([0, 1])]
            if len(out_of_range) > 0:
                report.add(
                    "error", "binary_range",
                    f"Feature '{col}' has {len(out_of_range)} value(s) outside {{0, 1}}. "
                    f"Invalid values: {sorted(out_of_range.unique().tolist())}."
                )

    def _check_age_range(self, X: pd.DataFrame, report: ValidationReport) -> None:
        if "age" not in X.columns:
            report.add("error", "age_range", "Missing continuous feature: 'age'.")
            return
        series = X["age"].dropna()
        invalid = series[(series < 0) | (series > 120)]
        if len(invalid) > 0:
            report.add(
                "warning", "age_range",
                f"Feature 'age' has {len(invalid)} value(s) outside [0, 120]."
            )

    def _check_missingness(self, X: pd.DataFrame, report: ValidationReport) -> None:
        missing_counts = X.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                pct = 100.0 * count / len(X)
                severity = "error" if pct > 10.0 else "warning"
                report.add(
                    severity, "missingness",
                    f"Feature '{col}' has {count} missing values ({pct:.1f}%)."
                )

    @staticmethod
    def _counts(report: ValidationReport) -> dict[str, int]:
        return {"errors": len(report.errors), "warnings": len(report.warnings)}
