"""
Tests for RuleSchemaValidator and DiagnosticRuleRepository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.symbolic_engine.schema_validator import (
    RuleSchemaValidationError,
    RuleSchemaValidator,
)

DISEASE_FILES = [
    "psoriasis.yaml",
    "seborrheic_dermatitis.yaml",
    "lichen_planus.yaml",
    "pityriasis_rosea.yaml",
    "chronic_dermatitis.yaml",
    "pityriasis_rubra_pilaris.yaml",
]

EXPECTED_RULE_COUNTS = {
    "psoriasis":               6,
    "seborrheic_dermatitis":   4,
    "lichen_planus":           5,
    "pityriasis_rosea":        4,
    "chronic_dermatitis":      4,
    "pityriasis_rubra_pilaris": 4,
}


class TestRuleSchemaValidator:
    def test_all_disease_files_pass_schema(self, schema_validator, rules_dir):
        for filename in DISEASE_FILES:
            path = rules_dir / filename
            rules = schema_validator.load_and_validate(path)
            assert isinstance(rules, list)
            assert len(rules) > 0

    def test_discriminators_file_passes_schema(self, schema_validator, rules_dir):
        path = rules_dir / "discriminators.yaml"
        rules = schema_validator.load_and_validate_discriminators(path)
        assert len(rules) == 6

    def test_invalid_rule_raises(self, schema_validator, tmp_path):
        bad_yaml = tmp_path / "bad_rule.yaml"
        bad_yaml.write_text(
            "rules:\n"
            "  - rule_id: BAD_001\n"
            "    disease_target: psoriasis\n"
            "    # evidence_tier missing — required field\n"
            "    activation_logic: binary\n"
            "    confidence_weight: 0.85\n"
            "    supporting_features: []\n"
            "    min_activation_threshold: 0.50\n"
            "    literature_source: 'test'\n",
            encoding="utf-8",
        )
        with pytest.raises(RuleSchemaValidationError):
            schema_validator.load_and_validate(bad_yaml)

    def test_missing_file_raises(self, schema_validator, tmp_path):
        with pytest.raises(FileNotFoundError):
            schema_validator.load_and_validate(tmp_path / "nonexistent.yaml")


class TestDiagnosticRuleRepository:
    def test_repository_loads_without_error(self, rule_repository):
        assert rule_repository is not None

    def test_total_disease_rule_count(self, rule_repository):
        expected_total = sum(EXPECTED_RULE_COUNTS.values())
        assert rule_repository.rule_count() == expected_total

    def test_per_disease_rule_counts(self, rule_repository):
        for disease, expected in EXPECTED_RULE_COUNTS.items():
            rules = rule_repository.rules_for_disease(disease)
            assert len(rules) == expected, (
                f"Disease '{disease}': expected {expected} rules, got {len(rules)}"
            )

    def test_discriminator_count(self, rule_repository):
        assert len(rule_repository.discriminators()) == 6

    def test_tier_a_rules_present(self, rule_repository):
        tier_a = rule_repository.tier_a_rules()
        assert len(tier_a) > 0
        for rule in tier_a:
            assert rule["evidence_tier"] == "A"

    def test_psoriasis_has_tier_a_rule(self, rule_repository):
        tier_a_pso = rule_repository.tier_a_rules(disease="psoriasis")
        assert len(tier_a_pso) == 1
        assert tier_a_pso[0]["rule_id"] == "PSO_001"

    def test_get_existing_rule(self, rule_repository):
        rule = rule_repository.get("LP_001")
        assert rule["disease_target"] == "lichen_planus"
        assert rule["evidence_tier"] == "A"
        assert rule["confidence_weight"] == 0.90

    def test_get_nonexistent_rule_raises(self, rule_repository):
        from src.symbolic_engine.rule_registry import RuleNotFoundError
        with pytest.raises(RuleNotFoundError):
            rule_repository.get("DOES_NOT_EXIST")

    def test_confusion_zone_pairs_populated(self, rule_repository):
        pairs = rule_repository.confusion_zone_pairs()
        assert len(pairs) > 0
        flat = [item for pair in pairs for item in pair]
        assert "psoriasis" in flat
        assert "seborrheic_dermatitis" in flat

    def test_discriminator_pair_filter(self, rule_repository):
        pso_sd = rule_repository.discriminators(
            disease_pair=("psoriasis", "seborrheic_dermatitis")
        )
        assert len(pso_sd) >= 1
        for rule in pso_sd:
            diseases = {rule.get("disease_target"), rule.get("competing_disease")}
            assert "psoriasis" in diseases
            assert "seborrheic_dermatitis" in diseases
