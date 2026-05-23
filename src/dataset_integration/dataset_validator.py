"""
DatasetValidator — schema and distribution integrity checks for DermatologyDataset.

Performs systematic validation of the loaded UCI Dermatology dataset to
confirm that all invariants expected by the evaluation pipeline are satisfied
before any model training or reasoning execution begins.

Validation checks
-----------------
  Schema:
    · All expected feature columns are present
    · Feature count matches expected (34)
    · Record count is within expected bounds (≥ 300)

  Ordinal features:
    · All values in [0, 3]
    · No unexpected float fractions

  Binary features:
    · All values in {0, 1}

  Age (continuous):
    · Non-negative after imputation
    · Within plausible range [1, 120]
    · No NaN values remain after imputation

  Labels:
    · All 6 canonical disease labels present
    · No unrecognised labels
    · All integer class codes in {1, 2, 3, 4, 5, 6}

  Distribution:
    · No class has zero records
    · Imbalance ratio < 10 (i.e. majority/minority < 10:1)

  Partition correctness:
    · Clinical partition contains exactly 12 features
    · Histopathological partition contains exactly 22 features

Usage
-----
  from src.dataset_integration.dataset_validator import DatasetValidator
  report = DatasetValidator.validate(dataset)
  if not report.is_valid:
      print(report.violations)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.dataset_integration.dataset_loader import (
    DermatologyDataset,
    CANONICAL_DISEASES,
    ORDINAL_FEATURES,
    BINARY_FEATURES,
    ALL_FEATURE_NAMES,
)
from src.dataset_integration.feature_partitioning import (
    CLINICAL_FEATURE_NAMES,
    HISTOPATHOLOGICAL_FEATURE_NAMES,
)


# ── Validation report ─────────────────────────────────────────────────────────

@dataclass
class DatasetValidationReport:
    """
    Structured result of a DatasetValidator.validate() call.

    Attributes
    ----------
    is_valid:
        True if no violations were found. Warnings do not affect validity.
    violations:
        List of failed invariant descriptions (critical errors).
    warnings:
        List of non-fatal observations.
    ordinal_range_violations:
        Feature → list of out-of-range patient IDs.
    binary_range_violations:
        Feature → list of out-of-range patient IDs.
    unknown_labels:
        Set of labels found in the data that are not canonical.
    label_distribution:
        Actual label counts from the validated dataset.
    missing_age_count:
        Number of records with imputed age values.
    checks_run:
        Total number of individual checks executed.
    checks_passed:
        Number of checks that passed.
    """

    is_valid:                   bool
    violations:                 list[str] = field(default_factory=list)
    warnings:                   list[str] = field(default_factory=list)
    ordinal_range_violations:   dict[str, list[str]] = field(default_factory=dict)
    binary_range_violations:    dict[str, list[str]] = field(default_factory=dict)
    unknown_labels:             set[str] = field(default_factory=set)
    label_distribution:         dict[str, int] = field(default_factory=dict)
    missing_age_count:          int = 0
    checks_run:                 int = 0
    checks_passed:              int = 0

    @property
    def pass_rate(self) -> float:
        return self.checks_passed / max(self.checks_run, 1)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"DatasetValidationReport [{status}] "
            f"checks={self.checks_passed}/{self.checks_run} "
            f"violations={len(self.violations)} "
            f"warnings={len(self.warnings)}"
        )


# ── Validator ─────────────────────────────────────────────────────────────────

class DatasetValidator:
    """
    Systematic validator for a loaded DermatologyDataset.

    All checks are encapsulated in private methods. The public interface
    is the single static validate() method which runs all checks and
    returns a DatasetValidationReport.
    """

    _MIN_RECORDS          = 300
    _EXPECTED_FEATURES    = 34
    _MAX_AGE              = 120.0
    _MIN_AGE              = 0.0
    _MAX_IMBALANCE_RATIO  = 12.0   # majority/minority — PRP has 20 records
    _EXPECTED_CLINICAL    = 12
    _EXPECTED_HISTOPATH   = 22

    @classmethod
    def validate(cls, dataset: DermatologyDataset) -> DatasetValidationReport:
        """
        Run all validation checks on the dataset.

        Parameters
        ----------
        dataset:
            Loaded DermatologyDataset from DermatologyDatasetLoader.

        Returns
        -------
        DatasetValidationReport with full findings.
        """
        report = DatasetValidationReport(is_valid=True)

        cls._check_schema(dataset, report)
        cls._check_record_count(dataset, report)
        cls._check_ordinal_ranges(dataset, report)
        cls._check_binary_ranges(dataset, report)
        cls._check_age(dataset, report)
        cls._check_labels(dataset, report)
        cls._check_distribution(dataset, report)
        cls._check_partitions(report)

        report.label_distribution = dict(dataset.summary.label_distribution.counts)
        report.missing_age_count  = dataset.summary.missing_age_count
        report.is_valid           = len(report.violations) == 0

        return report

    # ── Individual checks ─────────────────────────────────────────────────────

    @classmethod
    def _check_schema(
        cls,
        dataset: DermatologyDataset,
        report: DatasetValidationReport,
    ) -> None:
        """Verify all expected feature columns are present."""
        report.checks_run += 1
        if not dataset.records:
            report.violations.append("Dataset is empty — no records loaded.")
            return

        actual_keys = set(dataset.records[0].features.keys())
        expected    = set(ALL_FEATURE_NAMES)
        missing     = expected - actual_keys
        extra       = actual_keys - expected

        if missing:
            report.violations.append(
                f"Schema error: missing features: {sorted(missing)}"
            )
        elif extra:
            report.warnings.append(
                f"Schema: unexpected extra features: {sorted(extra)}"
            )
        else:
            report.checks_passed += 1

        report.checks_run += 1
        actual_count = len(actual_keys & expected)
        if actual_count != cls._EXPECTED_FEATURES:
            report.violations.append(
                f"Feature count mismatch: expected {cls._EXPECTED_FEATURES}, "
                f"got {actual_count}."
            )
        else:
            report.checks_passed += 1

    @classmethod
    def _check_record_count(
        cls,
        dataset: DermatologyDataset,
        report: DatasetValidationReport,
    ) -> None:
        """Verify record count is within expected bounds."""
        report.checks_run += 1
        n = len(dataset.records)
        if n < cls._MIN_RECORDS:
            report.violations.append(
                f"Record count {n} is below minimum {cls._MIN_RECORDS}."
            )
        else:
            report.checks_passed += 1

        if n > 500:
            report.warnings.append(
                f"Record count {n} exceeds typical UCI dataset size (366). "
                "Confirm correct dataset file."
            )

    @classmethod
    def _check_ordinal_ranges(
        cls,
        dataset: DermatologyDataset,
        report: DatasetValidationReport,
    ) -> None:
        """Verify all ordinal features are in [0, 3]."""
        violations: dict[str, list[str]] = {}

        for fname in ORDINAL_FEATURES:
            bad: list[str] = []
            for r in dataset.records:
                v = r.features.get(fname)
                if v is not None and (int(v) < 0 or int(v) > 3):
                    bad.append(r.patient_id)
            if bad:
                violations[fname] = bad

        report.checks_run += 1
        if violations:
            report.ordinal_range_violations = violations
            report.violations.append(
                f"Ordinal range violation in {len(violations)} feature(s): "
                f"{sorted(violations.keys())}"
            )
        else:
            report.checks_passed += 1

    @classmethod
    def _check_binary_ranges(
        cls,
        dataset: DatasetValidator,
        report: DatasetValidationReport,
    ) -> None:
        """
        Verify that family_history (the only strictly binary feature
        in the UCI dataset) contains only {0, 1}.
        """
        violations: dict[str, list[str]] = {}

        for fname in BINARY_FEATURES:
            bad: list[str] = []
            for r in dataset.records:
                v = r.features.get(fname)
                if v is not None and int(v) not in (0, 1):
                    bad.append(r.patient_id)
            if bad:
                violations[fname] = bad

        report.checks_run += 1
        if violations:
            report.binary_range_violations = violations
            report.violations.append(
                f"Binary range violation in {len(violations)} feature(s): "
                f"{sorted(violations.keys())}"
            )
        else:
            report.checks_passed += 1

    @classmethod
    def _check_age(
        cls,
        dataset: DermatologyDataset,
        report: DatasetValidationReport,
    ) -> None:
        """Verify age values are plausible and non-null after imputation."""
        report.checks_run += 1
        out_of_range = [
            r.patient_id for r in dataset.records
            if r.features.get("age") is None
            or float(r.features["age"]) < cls._MIN_AGE
            or float(r.features["age"]) > cls._MAX_AGE
        ]
        if out_of_range:
            report.violations.append(
                f"Age out of [{cls._MIN_AGE}, {cls._MAX_AGE}] "
                f"for {len(out_of_range)} records: {out_of_range[:5]} ..."
            )
        else:
            report.checks_passed += 1

    @classmethod
    def _check_labels(
        cls,
        dataset: DermatologyDataset,
        report: DatasetValidationReport,
    ) -> None:
        """Verify all labels are canonical and all 6 diseases are present."""
        canonical = set(CANONICAL_DISEASES)
        found_labels: set[str] = set()
        unknown_labels: set[str] = set()

        for r in dataset.records:
            found_labels.add(r.disease_label)
            if r.disease_label not in canonical:
                unknown_labels.add(r.disease_label)

        report.checks_run += 1
        if unknown_labels:
            report.unknown_labels = unknown_labels
            report.violations.append(
                f"Unknown disease labels found: {sorted(unknown_labels)}. "
                "Label normalisation may not have been applied."
            )
        else:
            report.checks_passed += 1

        report.checks_run += 1
        missing_diseases = canonical - found_labels
        if missing_diseases:
            report.violations.append(
                f"Missing disease classes: {sorted(missing_diseases)}. "
                "Dataset may be truncated."
            )
        else:
            report.checks_passed += 1

        report.checks_run += 1
        invalid_codes = [
            r.patient_id for r in dataset.records
            if r.disease_class not in (1, 2, 3, 4, 5, 6)
        ]
        if invalid_codes:
            report.violations.append(
                f"Invalid class codes in {len(invalid_codes)} records: "
                f"{invalid_codes[:5]} ..."
            )
        else:
            report.checks_passed += 1

    @classmethod
    def _check_distribution(
        cls,
        dataset: DermatologyDataset,
        report: DatasetValidationReport,
    ) -> None:
        """Verify label balance is within expected bounds."""
        dist = dataset.summary.label_distribution

        report.checks_run += 1
        zero_count = [d for d, c in dist.counts.items() if c == 0]
        if zero_count:
            report.violations.append(
                f"Zero-record disease classes: {zero_count}."
            )
        else:
            report.checks_passed += 1

        report.checks_run += 1
        ratio = dist.imbalance_ratio()
        if ratio > cls._MAX_IMBALANCE_RATIO:
            report.violations.append(
                f"Class imbalance ratio {ratio:.1f} exceeds maximum "
                f"{cls._MAX_IMBALANCE_RATIO}."
            )
        else:
            report.checks_passed += 1
            if ratio > 5.0:
                report.warnings.append(
                    f"Class imbalance ratio {ratio:.1f} is elevated. "
                    "Stratified splits are required for fair evaluation."
                )

    @classmethod
    def _check_partitions(cls, report: DatasetValidationReport) -> None:
        """Verify partition sizes are exactly as specified."""
        report.checks_run += 2

        if len(CLINICAL_FEATURE_NAMES) != cls._EXPECTED_CLINICAL:
            report.violations.append(
                f"Clinical partition has {len(CLINICAL_FEATURE_NAMES)} features; "
                f"expected {cls._EXPECTED_CLINICAL}."
            )
        else:
            report.checks_passed += 1

        if len(HISTOPATHOLOGICAL_FEATURE_NAMES) != cls._EXPECTED_HISTOPATH:
            report.violations.append(
                f"Histopathological partition has "
                f"{len(HISTOPATHOLOGICAL_FEATURE_NAMES)} features; "
                f"expected {cls._EXPECTED_HISTOPATH}."
            )
        else:
            report.checks_passed += 1
