"""
Diagnostic Rule Repository.

Loads, indexes, and provides structured access to the full diagnostic rule
base from YAML files under the rules/ directory. All symbolic reasoning
subsystems query rules through this repository — no direct YAML access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.symbolic_engine.schema_validator import RuleSchemaValidator
from src.utils.logger import get_logger

log = get_logger(__name__, subsystem="DiagnosticRuleRepository")

_RULES_DIR = Path("rules")

DISEASE_RULE_FILES: dict[str, str] = {
    "psoriasis":               "psoriasis.yaml",
    "seborrheic_dermatitis":   "seborrheic_dermatitis.yaml",
    "lichen_planus":           "lichen_planus.yaml",
    "pityriasis_rosea":        "pityriasis_rosea.yaml",
    "chronic_dermatitis":      "chronic_dermatitis.yaml",
    "pityriasis_rubra_pilaris": "pityriasis_rubra_pilaris.yaml",
}

DISCRIMINATOR_FILE      = "discriminators.yaml"
CONTRADICTION_MATRIX_FILE = "contradiction_matrix.yaml"


class RuleNotFoundError(KeyError):
    """Raised when a requested rule_id does not exist in the repository."""


class DiagnosticRuleRepository:
    """
    Repository for all diagnostic rules across the six erythemato-squamous
    disease categories, plus cross-disease discriminators.

    On initialisation, all rule files are loaded and validated. Rules are
    indexed by rule_id, disease_target, and evidence_tier for efficient lookup.

    Parameters
    ----------
    rules_dir:
        Directory containing the YAML rule files. Defaults to rules/.
    validate:
        If True, validate each file against the canonical JSON Schema on load.
    """

    def __init__(
        self,
        rules_dir: Path | str = _RULES_DIR,
        validate: bool = True,
    ) -> None:
        self._rules_dir = Path(rules_dir)
        self._validate = validate
        self._validator = RuleSchemaValidator() if validate else None

        # Primary indexes
        self._by_id:      dict[str, dict[str, Any]] = {}
        self._by_disease: dict[str, list[dict[str, Any]]] = {}
        self._by_tier:    dict[str, list[dict[str, Any]]] = {}
        self._discriminators: list[dict[str, Any]] = []
        self._contradiction_matrix: dict[str, Any] = {}

        self._load_all()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, rule_id: str) -> dict[str, Any]:
        """Return a rule dict by its unique rule_id."""
        if rule_id not in self._by_id:
            raise RuleNotFoundError(f"Rule '{rule_id}' not found in repository.")
        return self._by_id[rule_id]

    def rules_for_disease(self, disease: str) -> list[dict[str, Any]]:
        """Return all rules (Tiers A, B, C) for a given disease target."""
        return list(self._by_disease.get(disease, []))

    def rules_for_tier(self, tier: str) -> list[dict[str, Any]]:
        """Return all rules with a given evidence tier (A, B, C, D)."""
        return list(self._by_tier.get(tier, []))

    def tier_a_rules(self, disease: str | None = None) -> list[dict[str, Any]]:
        """Return Tier-A (pathognomonic) rules, optionally filtered by disease."""
        rules = self.rules_for_tier("A")
        if disease:
            rules = [r for r in rules if r["disease_target"] == disease]
        return rules

    def discriminators(
        self,
        disease_pair: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return cross-disease discriminator rules (Tier D).
        If disease_pair is provided, return only rules involving that pair.
        """
        if disease_pair is None:
            return list(self._discriminators)
        a, b = disease_pair
        return [
            r for r in self._discriminators
            if (r.get("disease_target") == a and r.get("competing_disease") == b)
            or (r.get("disease_target") == b and r.get("competing_disease") == a)
        ]

    def contradiction_matrix(self) -> dict[str, Any]:
        """Return the full parsed contradiction matrix."""
        return dict(self._contradiction_matrix)

    def confusion_zone_pairs(self) -> list[list[str]]:
        """Return list of known confusion-zone disease pairs."""
        zones = self._contradiction_matrix.get("confusion_zones", [])
        return [zone["pair"] for zone in zones]

    def all_rules(self) -> list[dict[str, Any]]:
        """Return all disease rules (Tiers A, B, C) across all diseases."""
        return list(self._by_id.values())

    def rule_count(self) -> int:
        return len(self._by_id)

    # ── Loader ────────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        for disease, filename in DISEASE_RULE_FILES.items():
            path = self._rules_dir / filename
            rules = self._load_file(path)
            for rule in rules:
                self._index_rule(rule)
            log.debug(
                "Disease rules loaded",
                disease=disease,
                n_rules=len(rules),
            )

        # Discriminators
        disc_path = self._rules_dir / DISCRIMINATOR_FILE
        discriminators = self._load_discriminator_file(disc_path)
        for rule in discriminators:
            rule.setdefault("evidence_tier", "D")
            self._discriminators.append(rule)
            self._by_tier.setdefault("D", []).append(rule)
        log.debug("Discriminators loaded", n_rules=len(discriminators))

        # Contradiction matrix (not a rules file — loaded separately)
        matrix_path = self._rules_dir / CONTRADICTION_MATRIX_FILE
        if matrix_path.exists():
            import yaml
            with open(matrix_path, encoding="utf-8") as fh:
                self._contradiction_matrix = yaml.safe_load(fh)
            log.debug("Contradiction matrix loaded")

        log.info(
            "Diagnostic rule repository initialised",
            total_disease_rules=len(self._by_id),
            discriminators=len(self._discriminators),
        )

    def _load_file(self, path: Path) -> list[dict[str, Any]]:
        if self._validate and self._validator is not None:
            return self._validator.load_and_validate(path)
        import yaml
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return raw.get("rules", [])

    def _load_discriminator_file(self, path: Path) -> list[dict[str, Any]]:
        if self._validate and self._validator is not None:
            return self._validator.load_and_validate_discriminators(path)
        import yaml
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return raw.get("rules", [])

    def _index_rule(self, rule: dict[str, Any]) -> None:
        rule_id = rule["rule_id"]
        if rule_id in self._by_id:
            raise ValueError(
                f"Duplicate rule_id '{rule_id}' detected across rule files."
            )
        self._by_id[rule_id] = rule
        disease = rule.get("disease_target", "unknown")
        self._by_disease.setdefault(disease, []).append(rule)
        tier = rule.get("evidence_tier", "C")
        self._by_tier.setdefault(tier, []).append(rule)
