"""
YAML Rule Schema Validator.

Validates diagnostic rule YAML files against the canonical JSON Schema before
they are loaded into the DiagnosticRuleRepository. This catches authoring
errors at load time rather than propagating malformed rules into inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__, subsystem="SchemaValidator")

# ── JSON Schema for a single supporting feature entry ─────────────────────────
_SUPPORTING_FEATURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["feature", "condition", "threshold", "partial_weight"],
    "additionalProperties": False,
    "properties": {
        "feature":        {"type": "string"},
        "condition":      {"type": "string", "enum": ["eq", "gte", "lte", "gt", "lt"]},
        "threshold":      {"type": "number"},
        "partial_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

# ── JSON Schema for a single contradiction feature entry ──────────────────────
_CONTRADICTION_FEATURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["feature", "condition", "threshold", "penalty", "competing_disease"],
    "additionalProperties": False,
    "properties": {
        "feature":           {"type": "string"},
        "condition":         {"type": "string", "enum": ["eq", "gte", "lte", "gt", "lt"]},
        "threshold":         {"type": "number"},
        "penalty":           {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "competing_disease": {"type": "string"},
    },
}

# ── JSON Schema for a discriminating feature entry ───────────────────────────
_DISCRIMINATING_FEATURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["feature", "condition", "threshold", "favours", "partial_weight"],
    "additionalProperties": False,
    "properties": {
        "feature":        {"type": "string"},
        "condition":      {"type": "string", "enum": ["eq", "gte", "lte", "gt", "lt"]},
        "threshold":      {"type": "number"},
        "favours":        {"type": "string"},
        "partial_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

# ── JSON Schema for disease diagnostic rules (Tiers A, B, C) ─────────────────
RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "rule_id",
        "disease_target",
        "evidence_tier",
        "activation_logic",
        "confidence_weight",
        "supporting_features",
        "min_activation_threshold",
        "literature_source",
    ],
    "additionalProperties": True,
    "properties": {
        "rule_id":          {"type": "string", "pattern": "^[A-Z0-9_]+$"},
        "disease_target":   {"type": "string"},
        "evidence_tier":    {"type": "string", "enum": ["A", "B", "C", "D"]},
        "activation_logic": {"type": "string", "enum": ["binary", "threshold", "fuzzy", "composite"]},
        "confidence_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "supporting_features": {
            "type": "array",
            "minItems": 1,
            "items": _SUPPORTING_FEATURE_SCHEMA,
        },
        "contradiction_features": {
            "type": "array",
            "items": _CONTRADICTION_FEATURE_SCHEMA,
        },
        "min_activation_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "literature_source": {"type": "string"},
        "clinical_rationale": {"type": "string"},
    },
}

# ── JSON Schema for discriminator rules (Tier D) ──────────────────────────────
DISCRIMINATOR_RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "rule_id",
        "disease_target",
        "competing_disease",
        "evidence_tier",
        "activation_logic",
        "confidence_weight",
        "discriminating_features",
        "trigger_condition",
        "literature_source",
    ],
    "additionalProperties": True,
    "properties": {
        "rule_id":              {"type": "string", "pattern": "^[A-Z0-9_]+$"},
        "disease_target":       {"type": "string"},
        "competing_disease":    {"type": "string"},
        "evidence_tier":        {"type": "string", "enum": ["D"]},
        "activation_logic":     {"type": "string", "enum": ["binary", "threshold", "fuzzy", "composite"]},
        "confidence_weight":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "discriminating_features": {
            "type": "array",
            "minItems": 1,
            "items": _DISCRIMINATING_FEATURE_SCHEMA,
        },
        "trigger_condition":    {"type": "object"},
        "literature_source":    {"type": "string"},
        "clinical_rationale":   {"type": "string"},
    },
}

# ── JSON Schema for disease rules YAML file ───────────────────────────────────
RULES_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["rules"],
    "properties": {
        "rules": {
            "type": "array",
            "minItems": 1,
            "items": RULE_SCHEMA,
        }
    },
}

# ── JSON Schema for discriminators YAML file ──────────────────────────────────
DISCRIMINATORS_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["rules"],
    "properties": {
        "rules": {
            "type": "array",
            "minItems": 1,
            "items": DISCRIMINATOR_RULE_SCHEMA,
        }
    },
}


class RuleSchemaValidationError(ValueError):
    """Raised when a rule YAML file fails JSON Schema validation."""


class RuleSchemaValidator:
    """
    Validates diagnostic rule YAML files against the canonical schema.

    Usage
    -----
    validator = RuleSchemaValidator()
    rules = validator.load_and_validate(Path("rules/psoriasis.yaml"))
    discriminators = validator.load_and_validate_discriminators(Path("rules/discriminators.yaml"))
    """

    def __init__(self) -> None:
        self._validator = jsonschema.Draft7Validator(RULES_FILE_SCHEMA)
        self._discriminator_validator = jsonschema.Draft7Validator(DISCRIMINATORS_FILE_SCHEMA)

    def load_and_validate(self, path: Path) -> list[dict[str, Any]]:
        """
        Load a rule YAML file and validate each rule against the schema.

        Returns the list of rule dicts on success.
        Raises RuleSchemaValidationError on the first schema violation.
        """
        raw = self._load_yaml(path)
        errors = list(self._validator.iter_errors(raw))
        if errors:
            messages = [
                f"  [{e.json_path}] {e.message}" for e in errors
            ]
            raise RuleSchemaValidationError(
                f"Schema validation failed for {path}:\n" + "\n".join(messages)
            )
        rules = raw["rules"]
        self._check_rule_id_uniqueness(rules, path)
        log.debug("Rule file validated", path=str(path), n_rules=len(rules))
        return rules

    def load_and_validate_discriminators(self, path: Path) -> list[dict[str, Any]]:
        """Load and validate a discriminators YAML file (Tier D rules)."""
        raw = self._load_yaml(path)
        errors = list(self._discriminator_validator.iter_errors(raw))
        if errors:
            messages = [f"  [{e.json_path}] {e.message}" for e in errors]
            raise RuleSchemaValidationError(
                f"Discriminator schema validation failed for {path}:\n"
                + "\n".join(messages)
            )
        rules = raw["rules"]
        self._check_rule_id_uniqueness(rules, path)
        log.debug("Discriminator file validated", path=str(path), n_rules=len(rules))
        return rules

    def validate_rule(self, rule: dict[str, Any]) -> None:
        """Validate a single rule dict. Raises RuleSchemaValidationError on failure."""
        errors = list(jsonschema.Draft7Validator(RULE_SCHEMA).iter_errors(rule))
        if errors:
            messages = [e.message for e in errors]
            raise RuleSchemaValidationError(
                f"Rule '{rule.get('rule_id', 'UNKNOWN')}' failed schema validation: "
                + "; ".join(messages)
            )

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Rule file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    @staticmethod
    def _check_rule_id_uniqueness(rules: list[dict], path: Path) -> None:
        ids = [r.get("rule_id") for r in rules]
        seen: set[str] = set()
        duplicates = []
        for rule_id in ids:
            if rule_id in seen:
                duplicates.append(rule_id)
            seen.add(rule_id)
        if duplicates:
            raise RuleSchemaValidationError(
                f"Duplicate rule_ids in {path}: {duplicates}"
            )
