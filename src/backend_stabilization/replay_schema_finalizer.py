"""
replay_schema_finalizer.py
============================
Finalizes the replay JSON schema for the CASDRE clinical inference pipeline.

Defines the canonical replay event format, validates existing pipeline outputs
against the schema, and exports the frozen schema specification.  The replay
engine in src/graph_reasoning/ must produce replay events conforming to this
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]


# ──────────────────────────────────────────────────────────────────────────────
# Contract version
# ──────────────────────────────────────────────────────────────────────────────

REPLAY_SCHEMA_VERSION = "1.0.0"
REPLAY_SCHEMA_FROZEN  = True


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class ReplayEventType(str, Enum):
    CASE_START            = "case_start"
    CLINICAL_EVAL         = "clinical_eval"
    SYMBOLIC_ENRICHMENT   = "symbolic_enrichment"
    CONTRADICTION_CHECK   = "contradiction_check"
    AMBIGUITY_RESOLUTION  = "ambiguity_resolution"
    TRAJECTORY_STEP       = "trajectory_step"
    ESCALATION_DECISION   = "escalation_decision"
    RECOVERY_ATTEMPT      = "recovery_attempt"
    FINAL_DECISION        = "final_decision"
    CASE_END              = "case_end"


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic / plain-dataclass schema
# ──────────────────────────────────────────────────────────────────────────────

if _PYDANTIC_AVAILABLE:

    class ReplayEventSchema(BaseModel):
        """One replay event in the canonical replay stream."""
        schema_version: str    = Field(default=REPLAY_SCHEMA_VERSION)
        event_type: ReplayEventType
        case_id: str
        step: int              = Field(ge=0)
        fsm_state: str
        certainty: float       = Field(ge=0.0, le=1.0)
        ambiguity_bits: float  = Field(ge=0.0)
        leading_diagnosis: str
        contradiction_load: float = Field(ge=0.0, le=0.40)
        payload: Dict[str, Any] = Field(default_factory=dict)
        timestamp_ms: Optional[int] = None

        @field_validator("contradiction_load")
        @classmethod
        def _ceiling(cls, v: float) -> float:
            if v > 0.40:
                raise ValueError(f"contradiction_load {v:.4f} exceeds ceiling 0.40")
            return v

    class ReplayCaseSchema(BaseModel):
        """Full replay record for a single diagnostic case."""
        schema_version: str    = Field(default=REPLAY_SCHEMA_VERSION)
        case_id: str
        true_label: Optional[str]     = None
        final_diagnosis: str
        events: List[ReplayEventSchema]
        total_steps: int       = Field(ge=1)
        converged: bool
        requires_biopsy: bool
        is_safe_triage: bool
        final_certainty: float = Field(ge=0.0, le=1.0)

else:
    @dataclass
    class ReplayEventSchema:
        schema_version: str; event_type: str; case_id: str
        step: int; fsm_state: str; certainty: float; ambiguity_bits: float
        leading_diagnosis: str; contradiction_load: float
        payload: Dict[str, Any] = field(default_factory=dict)
        timestamp_ms: Optional[int] = None

    @dataclass
    class ReplayCaseSchema:
        schema_version: str; case_id: str; final_diagnosis: str
        events: list; total_steps: int; converged: bool
        requires_biopsy: bool; is_safe_triage: bool; final_certainty: float
        true_label: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Required fields
# ──────────────────────────────────────────────────────────────────────────────

_EVENT_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "event_type", "case_id", "step",
    "fsm_state", "certainty", "ambiguity_bits", "leading_diagnosis",
    "contradiction_load",
})

_CASE_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "case_id", "final_diagnosis", "events",
    "total_steps", "converged", "requires_biopsy", "is_safe_triage",
    "final_certainty",
})


# ──────────────────────────────────────────────────────────────────────────────
# Validation results
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ReplayValidationResult:
    case_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    n_events: int


@dataclass
class ReplaySchemaAuditReport:
    """Audit result for a batch of replay records."""
    n_cases_audited: int
    n_valid: int
    n_invalid: int
    validation_rate: float
    n_events_audited: int
    n_invalid_events: int
    common_errors: List[str]
    schema_version: str = REPLAY_SCHEMA_VERSION
    schema_frozen: bool = REPLAY_SCHEMA_FROZEN

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "REPLAY SCHEMA AUDIT REPORT",
            f"  Schema version   : {self.schema_version}  (frozen={self.schema_frozen})",
            "=" * 70,
            f"  Cases audited    : {self.n_cases_audited}",
            f"  Valid cases      : {self.n_valid}  ({self.validation_rate:.1%})",
            f"  Invalid cases    : {self.n_invalid}",
            f"  Events audited   : {self.n_events_audited}",
            f"  Invalid events   : {self.n_invalid_events}",
        ]
        if self.common_errors:
            lines.append("  Common errors:")
            for e in self.common_errors:
                lines.append(f"    • {e}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Finalizer
# ──────────────────────────────────────────────────────────────────────────────

class ReplaySchemaFinalizer:
    """
    Validates replay records against the frozen replay schema.

    Usage
    -----
    finalizer = ReplaySchemaFinalizer()
    result    = finalizer.validate_case(case_dict)
    report    = finalizer.audit_batch(list_of_case_dicts)
    """

    def validate_event(self, event: Dict[str, Any]) -> List[str]:
        """Return list of errors for a single event dict (empty = valid)."""
        errors: List[str] = []
        missing = [f for f in _EVENT_REQUIRED_FIELDS if f not in event]
        if missing:
            errors.append(f"Event missing required fields: {missing}")

        load = event.get("contradiction_load", 0.0)
        if float(load) > 0.40:
            errors.append(f"CRITICAL: event contradiction_load={load} exceeds ceiling 0.40")

        cert = event.get("certainty", 0.0)
        if not (0.0 <= float(cert) <= 1.0):
            errors.append(f"event certainty={cert!r} out of [0,1]")

        step = event.get("step", 0)
        if int(step) < 0:
            errors.append(f"event step={step} must be >= 0")

        return errors

    def validate_case(self, case: Dict[str, Any]) -> ReplayValidationResult:
        """Validate a full replay case dict."""
        case_id = str(case.get("case_id", "<unknown>"))
        errors:   List[str] = []
        warnings: List[str] = []

        missing = [f for f in _CASE_REQUIRED_FIELDS if f not in case]
        if missing:
            errors.append(f"Case missing required fields: {missing}")

        events = case.get("events", [])
        if not isinstance(events, list):
            errors.append("case.events must be a list")
            events = []

        for idx, ev in enumerate(events):
            ev_errors = self.validate_event(ev)
            for e in ev_errors:
                errors.append(f"  event[{idx}]: {e}")

        total_steps = case.get("total_steps", 0)
        if len(events) != int(total_steps):
            warnings.append(
                f"total_steps={total_steps} != len(events)={len(events)}"
            )

        if case.get("requires_biopsy") and case.get("is_safe_triage"):
            errors.append("requires_biopsy and is_safe_triage cannot both be True")

        if "schema_version" in case and case["schema_version"] != REPLAY_SCHEMA_VERSION:
            warnings.append(
                f"schema_version mismatch: {case['schema_version']!r} vs {REPLAY_SCHEMA_VERSION!r}"
            )

        return ReplayValidationResult(
            case_id=case_id,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            n_events=len(events),
        )

    def audit_batch(self, cases: List[Dict[str, Any]]) -> ReplaySchemaAuditReport:
        """Audit a batch of replay case dicts."""
        from collections import Counter
        results   = [self.validate_case(c) for c in cases]
        n_valid   = sum(1 for r in results if r.is_valid)
        n_invalid = len(results) - n_valid
        n_events  = sum(r.n_events for r in results)
        n_inv_ev  = sum(
            sum(1 for e in r.errors if "event[" in e)
            for r in results
        )

        error_counter: Counter = Counter()
        for r in results:
            for e in r.errors:
                key = e.split(":")[0].strip()
                error_counter[key] += 1

        return ReplaySchemaAuditReport(
            n_cases_audited=len(results),
            n_valid=n_valid,
            n_invalid=n_invalid,
            validation_rate=n_valid / len(results) if results else 0.0,
            n_events_audited=n_events,
            n_invalid_events=n_inv_ev,
            common_errors=[f"{k}: {v}×" for k, v in error_counter.most_common(5)],
        )

    @staticmethod
    def canonical_event(
        case_id: str,
        step: int,
        event_type: str,
        fsm_state: str,
        certainty: float,
        ambiguity_bits: float,
        leading_diagnosis: str,
        contradiction_load: float = 0.0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce a canonical replay event dict."""
        return {
            "schema_version":      REPLAY_SCHEMA_VERSION,
            "event_type":          event_type,
            "case_id":             case_id,
            "step":                step,
            "fsm_state":           fsm_state,
            "certainty":           float(max(0.0, min(1.0, certainty))),
            "ambiguity_bits":      float(max(0.0, ambiguity_bits)),
            "leading_diagnosis":   leading_diagnosis,
            "contradiction_load":  float(max(0.0, min(0.40, contradiction_load))),
            "payload":             payload or {},
            "timestamp_ms":        None,
        }

    @staticmethod
    def canonical_case(
        case_id: str,
        final_diagnosis: str,
        events: List[Dict[str, Any]],
        converged: bool,
        requires_biopsy: bool,
        is_safe_triage: bool,
        final_certainty: float,
        true_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Produce a canonical replay case dict."""
        return {
            "schema_version":    REPLAY_SCHEMA_VERSION,
            "case_id":           case_id,
            "true_label":        true_label,
            "final_diagnosis":   final_diagnosis,
            "events":            events,
            "total_steps":       len(events),
            "converged":         converged,
            "requires_biopsy":   requires_biopsy,
            "is_safe_triage":    is_safe_triage,
            "final_certainty":   float(max(0.0, min(1.0, final_certainty))),
        }
