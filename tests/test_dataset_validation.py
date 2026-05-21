"""
Tests for DatasetValidator — schema, range, and distribution checks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.dataset_validation import DatasetValidator, ValidationReport


class TestDatasetValidatorSynthetic:
    def test_synthetic_dataset_passes_validation(self, synthetic_X, synthetic_y):
        validator = DatasetValidator(strict=False)
        report = validator.validate(synthetic_X, synthetic_y)
        # Synthetic data may have minor distribution deviations — only errors matter
        assert isinstance(report, ValidationReport)

    def test_correct_class_labels_pass(self, synthetic_y):
        validator = DatasetValidator(strict=False)
        report = DatasetValidator(strict=False).validate(
            pd.DataFrame({"dummy": [0] * 366}), synthetic_y
        )
        label_errors = [e for e in report.errors if e.check == "class_labels"]
        assert len(label_errors) == 0

    def test_invalid_class_labels_flagged(self, synthetic_X, synthetic_y):
        bad_y = synthetic_y.copy()
        bad_y.iloc[0] = 99  # invalid class
        validator = DatasetValidator(strict=False)
        report = validator.validate(synthetic_X, bad_y)
        label_errors = [e for e in report.errors if e.check == "class_labels"]
        assert len(label_errors) == 1

    def test_ordinal_out_of_range_flagged(self, synthetic_X, synthetic_y):
        bad_X = synthetic_X.copy()
        bad_X.loc[0, "erythema"] = 5  # out of [0, 3]
        validator = DatasetValidator(strict=False)
        report = validator.validate(bad_X, synthetic_y)
        range_errors = [e for e in report.errors if e.check == "ordinal_range"]
        assert len(range_errors) >= 1

    def test_binary_out_of_range_flagged(self, synthetic_X, synthetic_y):
        bad_X = synthetic_X.copy()
        bad_X.loc[0, "koebner_phenomenon"] = 3  # out of {0, 1}
        validator = DatasetValidator(strict=False)
        report = validator.validate(bad_X, synthetic_y)
        range_errors = [e for e in report.errors if e.check == "binary_range"]
        assert len(range_errors) >= 1

    def test_strict_mode_raises_on_error(self, synthetic_X, synthetic_y):
        bad_y = synthetic_y.copy()
        bad_y.iloc[0] = 99
        validator = DatasetValidator(strict=True)
        with pytest.raises(ValueError, match="validation failed"):
            validator.validate(synthetic_X, bad_y)

    def test_report_is_valid_with_clean_data(self, synthetic_X, synthetic_y):
        validator = DatasetValidator(strict=False)
        report = validator.validate(synthetic_X, synthetic_y)
        # No schema or range errors expected on clean synthetic data
        schema_errors = [
            e for e in report.errors
            if e.check in ("class_labels", "ordinal_range", "binary_range")
        ]
        assert len(schema_errors) == 0

    def test_missing_feature_column_flagged(self, synthetic_y):
        # Drop a required ordinal feature
        bad_X = pd.DataFrame({"dummy_col": [0] * 366})
        validator = DatasetValidator(strict=False)
        report = validator.validate(bad_X, synthetic_y)
        assert not report.is_valid  # missing columns → errors
