"""
reasoning_contract_finalizer.py
=================================
Freezes the symbolic reasoning output schema for the CASDRE clinical inference
pipeline.

All downstream consumers (replay engine, graph serializer, frontend API) must
conform to the canonical reasoning output contract defined here.  Provides
Pydantic-based validation and a contract-diff tool to detect schema drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]


# ──────────────────────────────────────────────────────────────────────────────
# Contract version
# ──────────────────────────────────────────────────────────────────────────────

REASONING_CONTRACT_VERSION = "1.0.0"
REASONING_CONTRACT_FROZEN  = True   # schema may not be altered without version bump


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations (schema)
# ──────────────────────────────────────────────────────────────────────────────

class DiagnosticStateCode(str, Enum):
    INITIAL              = "INITIAL"
    CLINICAL_ASSESSMENT  = "CLINICAL_ASSESSMENT"
    SYMBOLIC_ENRICHMENT  = "SYMBOLIC_ENRICHMENT"
    CONTRADICTION_CHECK  = "CONTRADICTION_CHECK"
    AMBIGUITY_RESOLUTION = "AMBIGUITY_RESOLUTION"
    ESCALATION_REVIEW    = "ESCALATION_REVIEW"
    RECOVERY_ATTEMPT     = "RECOVERY_ATTEMPT"
    FINAL_DECISION       = "FINAL_DECISION"
    BIOPSY_REQUIRED      = "BIOPSY_REQUIRED"


class ContradictionTierCode(str, Enum):
    NONE     = "NONE"
    MINOR    = "MINOR"
    MODERATE = "MODERATE"
    CRITICAL = "CRITICAL"


class RecoveryMechanismCode(str, Enum):
    CONTRADICTION = "CONTRADICTION"
    LEADERSHIP    = "LEADERSHIP"
    AMBIGUITY     = "AMBIGUITY"
    TRAJECTORY    = "TRAJECTORY"
    COMPETITION   = "COMPETITION"
    ESCALATION    = "ESCALATION"
    UNEXPLAINED   = "UNEXPLAINED"


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schema (canonical contract)
# ──────────────────────────────────────────────────────────────────────────────

if _PYDANTIC_AVAILABLE:

    class SymbolicSignalSchema(BaseModel):
        """One symbolic signal value in the reasoning output."""
        name: str
        value: float
        activated: bool
        contribution_weight: float = Field(ge=0.0, le=1.0)

    class DifferentialEntrySchema(BaseModel):
        """One disease entry in the differential diagnosis table."""
        disease: str
        certainty: float = Field(ge=0.0, le=1.0)
        rank: int = Field(ge=1)
        is_leading: bool

    class ContradictionSummarySchema(BaseModel):
        """Contradiction summary in the reasoning output."""
        overall_load: float = Field(ge=0.0, le=0.40)   # ceiling enforced
        tier: ContradictionTierCode
        n_contradicting_signals: int = Field(ge=0)
        escalation_triggered_by_contradiction: bool

        @field_validator("overall_load") if _PYDANTIC_AVAILABLE else lambda x: x
        @classmethod
        def _ceiling(cls, v: float) -> float:
            if v > 0.40:
                raise ValueError(f"Contradiction load {v:.4f} exceeds ceiling 0.40")
            return v

    class TrajectoryStateSchema(BaseModel):
        """Snapshot of reasoning trajectory at a given step."""
        step: int = Field(ge=0)
        certainty: float = Field(ge=0.0, le=1.0)
        ambiguity_bits: float = Field(ge=0.0)
        leading_disease: str
        fsm_state: DiagnosticStateCode

    class ReasoningOutputSchema(BaseModel):
        """
        Canonical frozen schema for a single CASDRE reasoning output.
        Version: 1.0.0
        """
        schema_version: str = Field(default=REASONING_CONTRACT_VERSION)
        case_id: str
        fsm_state: DiagnosticStateCode
        leading_diagnosis: str
        certainty: float = Field(ge=0.0, le=1.0)
        ambiguity_bits: float = Field(ge=0.0)
        requires_biopsy: bool
        is_safe_triage: bool
        contradiction: ContradictionSummarySchema
        differential: List[DifferentialEntrySchema]
        symbolic_signals: List[SymbolicSignalSchema]
        trajectory: List[TrajectoryStateSchema]
        recovery_mechanism: Optional[RecoveryMechanismCode] = None
        recovery_successful: Optional[bool] = None

        @model_validator(mode="after") if _PYDANTIC_AVAILABLE else lambda x: x
        def _consistency_check(self):
            if self.requires_biopsy and self.is_safe_triage:
                raise ValueError(
                    "requires_biopsy and is_safe_triage cannot both be True."
                )
            leading_entries = [d for d in self.differential if d.is_leading]
            if len(leading_entries) > 1:
                raise ValueError("Only one differential entry may be marked is_leading.")
            return self

else:
    # Fallback plain dataclass when Pydantic is not installed
    @dataclass
    class SymbolicSignalSchema:
        name: str; value: float; activated: bool; contribution_weight: float

    @dataclass
    class DifferentialEntrySchema:
        disease: str; certainty: float; rank: int; is_leading: bool

    @dataclass
    class ContradictionSummarySchema:
        overall_load: float; tier: str; n_contradicting_signals: int
        escalation_triggered_by_contradiction: bool

    @dataclass
    class TrajectoryStateSchema:
        step: int; certainty: float; ambiguity_bits: float
        leading_disease: str; fsm_state: str

    @dataclass
    class ReasoningOutputSchema:
        schema_version: str; case_id: str; fsm_state: str
        leading_diagnosis: str; certainty: float; ambiguity_bits: float
        requires_biopsy: bool; is_safe_triage: bool
        contradiction: Any; differential: list; symbolic_signals: list
        trajectory: list; recovery_mechanism: Optional[str] = None
        recovery_successful: Optional[bool] = None


# ──────────────────────────────────────────────────────────────────────────────
# Contract validation utilities
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContractValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    schema_version: str = REASONING_CONTRACT_VERSION


@dataclass
class ContractDriftReport:
    """
    Detects schema drift between a live output dict and the frozen contract.
    """
    n_outputs_checked: int
    n_valid: int
    n_invalid: int
    validation_rate: float
    common_errors: List[str]
    missing_fields: List[str]
    extra_fields: List[str]
    contract_version: str = REASONING_CONTRACT_VERSION
    contract_frozen: bool = REASONING_CONTRACT_FROZEN

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "REASONING CONTRACT VALIDATION REPORT",
            f"  Contract version : {self.contract_version}  (frozen={self.contract_frozen})",
            "=" * 70,
            f"  Outputs checked  : {self.n_outputs_checked}",
            f"  Valid            : {self.n_valid}  ({self.validation_rate:.1%})",
            f"  Invalid          : {self.n_invalid}",
        ]
        if self.missing_fields:
            lines.append(f"  Missing fields   : {self.missing_fields}")
        if self.extra_fields:
            lines.append(f"  Extra fields     : {self.extra_fields}")
        if self.common_errors:
            lines.append("  Common errors:")
            for e in self.common_errors:
                lines.append(f"    • {e}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Finalizer
# ──────────────────────────────────────────────────────────────────────────────

# Canonical set of required top-level fields
_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "case_id", "fsm_state", "leading_diagnosis",
    "certainty", "ambiguity_bits", "requires_biopsy", "is_safe_triage",
    "contradiction", "differential", "symbolic_signals", "trajectory",
})

_CONTRADICTION_REQUIRED_FIELDS: frozenset = frozenset({
    "overall_load", "tier", "n_contradicting_signals",
    "escalation_triggered_by_contradiction",
})


class ReasoningContractFinalizer:
    """
    Validates reasoning outputs against the frozen contract schema.

    Usage
    -----
    finalizer = ReasoningContractFinalizer()
    result    = finalizer.validate_output(output_dict)
    report    = finalizer.audit_batch(list_of_output_dicts)
    """

    def validate_output(self, output: Dict[str, Any]) -> ContractValidationResult:
        """Validate a single reasoning output dict against the frozen contract."""
        errors:   List[str] = []
        warnings: List[str] = []

        # Required fields
        missing = [f for f in _REQUIRED_FIELDS if f not in output]
        if missing:
            errors.append(f"Missing required fields: {missing}")

        # Type / range checks on present fields
        if "certainty" in output:
            c = output["certainty"]
            if not (isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0):
                errors.append(f"certainty={c!r} must be float in [0, 1]")

        if "contradiction" in output:
            contra = output["contradiction"]
            if isinstance(contra, dict):
                missing_c = [f for f in _CONTRADICTION_REQUIRED_FIELDS if f not in contra]
                if missing_c:
                    errors.append(f"contradiction missing fields: {missing_c}")
                load = contra.get("overall_load", 0.0)
                if float(load) > 0.40:
                    errors.append(
                        f"CRITICAL: contradiction.overall_load={load} exceeds ceiling 0.40"
                    )
            else:
                errors.append("contradiction field must be a dict")

        if "requires_biopsy" in output and "is_safe_triage" in output:
            if output["requires_biopsy"] and output["is_safe_triage"]:
                errors.append("requires_biopsy and is_safe_triage cannot both be True")

        if "schema_version" in output:
            if output["schema_version"] != REASONING_CONTRACT_VERSION:
                warnings.append(
                    f"Schema version mismatch: output has "
                    f"'{output['schema_version']}', contract is "
                    f"'{REASONING_CONTRACT_VERSION}'"
                )

        # Extra fields (warning only)
        extra = [f for f in output if f not in _REQUIRED_FIELDS
                 and f not in {"recovery_mechanism", "recovery_successful"}]
        if extra:
            warnings.append(f"Extra fields not in contract: {extra}")

        return ContractValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def audit_batch(
        self,
        outputs: List[Dict[str, Any]],
    ) -> ContractDriftReport:
        """Audit a batch of reasoning outputs and produce a drift report."""
        from collections import Counter
        n   = len(outputs)
        results = [self.validate_output(o) for o in outputs]
        n_valid   = sum(1 for r in results if r.is_valid)
        n_invalid = n - n_valid

        error_counter: Counter = Counter()
        for r in results:
            for e in r.errors:
                # Normalise: strip case-specific values
                key = e.split("=")[0].split(":")[0].strip()
                error_counter[key] += 1

        all_missing: List[str] = []
        all_extra:   List[str] = []
        for r in results:
            for e in r.errors:
                if e.startswith("Missing required fields"):
                    all_missing.extend(e.replace("Missing required fields: ", "").strip("[]'").split("', '"))
            for w in r.warnings:
                if w.startswith("Extra fields"):
                    all_extra.extend(w.replace("Extra fields not in contract: ", "").strip("[]'").split("', '"))

        return ContractDriftReport(
            n_outputs_checked=n,
            n_valid=n_valid,
            n_invalid=n_invalid,
            validation_rate=n_valid / n if n > 0 else 0.0,
            common_errors=[f"{k}: {v}×" for k, v in error_counter.most_common(5)],
            missing_fields=list(dict.fromkeys(all_missing))[:10],
            extra_fields=list(dict.fromkeys(all_extra))[:10],
        )

    @staticmethod
    def canonical_empty_output(case_id: str) -> Dict[str, Any]:
        """Return a canonical empty reasoning output conforming to the contract."""
        return {
            "schema_version": REASONING_CONTRACT_VERSION,
            "case_id": case_id,
            "fsm_state": DiagnosticStateCode.INITIAL.value,
            "leading_diagnosis": "",
            "certainty": 0.0,
            "ambiguity_bits": 0.0,
            "requires_biopsy": False,
            "is_safe_triage": False,
            "contradiction": {
                "overall_load": 0.0,
                "tier": ContradictionTierCode.NONE.value,
                "n_contradicting_signals": 0,
                "escalation_triggered_by_contradiction": False,
            },
            "differential": [],
            "symbolic_signals": [],
            "trajectory": [],
            "recovery_mechanism": None,
            "recovery_successful": None,
        }
