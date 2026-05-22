"""
ClinicalSafetyGate — Stage 5 safety layer (escalation-only).

Evaluates three formal safety invariants and five safety gates against
the current certainty distribution, conflict analysis, and evidence
vector. All gate evaluations are escalation-only: they can only move the
triage recommendation toward greater caution, never toward greater certainty.

Safety invariants (hard overrides)
-----------------------------------
  I1 — Contradiction Safety Ceiling:
       contradiction_load >= 0.40 → BIOPSY_RECOMMENDED (mandatory)
  I2 — Evidence Sufficiency Floor:
       activated_rule_count < 2   → AMBIGUOUS_PRESENTATION (cap)
  I3 — Entropy Escalation Ceiling:
       ambiguity_index > 1.5 bits → BIOPSY_RECOMMENDED (mandatory)

Safety gates (conditional caps)
---------------------------------
  G1 — Single-Source Dominance:
       One rule contributing > 60% of certainty → cap at MODERATE_CERTAINTY
  G2 — Pathognomonic Absence under High Certainty:
       certainty > 0.75 but no Tier-A rule active → cap at MODERATE_CERTAINTY
  G3 — Critical Feature Missingness:
       >= 3 critical features missing → cap at AMBIGUOUS_PRESENTATION
  G4 — Confusion Zone Proximity:
       Leading pair in confusion zone with gap < 0.30 → apply 0.15 penalty
  G5 — Overconfidence Prevention:
       certainty > 0.92 AND contradiction > 0.10 → cap at MODERATE_CERTAINTY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.reasoning.certainty_propagator import CertaintyDistribution
from src.reasoning.conflict_analyzer import ConflictAnalysisResult
from src.reasoning.evidence_evaluator import EvidenceEvaluationResult


# ── Triage recommendation ─────────────────────────────────────────────────────

class TriageRecommendation(str, Enum):
    SAFE_NON_INVASIVE_TRIAGE = "SAFE_NON_INVASIVE_TRIAGE"
    MODERATE_CERTAINTY       = "MODERATE_CERTAINTY"
    AMBIGUOUS_PRESENTATION   = "AMBIGUOUS_PRESENTATION"
    BIOPSY_RECOMMENDED       = "BIOPSY_RECOMMENDED"
    HIGH_RISK_CONTRADICTION  = "HIGH_RISK_CONTRADICTION"

    @property
    def severity_rank(self) -> int:
        _ranks = {
            "SAFE_NON_INVASIVE_TRIAGE": 0,
            "MODERATE_CERTAINTY":       1,
            "AMBIGUOUS_PRESENTATION":   2,
            "BIOPSY_RECOMMENDED":       3,
            "HIGH_RISK_CONTRADICTION":  4,
        }
        return _ranks.get(self.value, 0)


# ── Gate evaluation result ────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of evaluating a single safety invariant or gate."""

    gate_id:        str
    gate_name:      str
    triggered:      bool
    cap:            TriageRecommendation | None
    rationale:      str
    measured_value: float | None = None
    threshold:      float | None = None


# ── Full safety report ────────────────────────────────────────────────────────

@dataclass
class SafetyGateReport:
    """
    Aggregate result of all invariant and gate evaluations.
    Produced by ClinicalSafetyGate.evaluate().
    """

    invariant_results: list[GateResult] = field(default_factory=list)
    gate_results:      list[GateResult] = field(default_factory=list)
    effective_cap:     TriageRecommendation | None = None
    any_triggered:     bool = False
    certainty_penalty: float = 0.0   # from confusion zone proximity gate

    @property
    def all_results(self) -> list[GateResult]:
        return self.invariant_results + self.gate_results

    @property
    def triggered_gates(self) -> list[GateResult]:
        return [r for r in self.all_results if r.triggered]

    def apply_cap(self, base: TriageRecommendation) -> TriageRecommendation:
        """
        Apply the effective cap to a base recommendation.
        Returns the more severe of base and effective_cap (escalation-only).
        """
        if self.effective_cap is None:
            return base
        if self.effective_cap.severity_rank > base.severity_rank:
            return self.effective_cap
        return base


# ── Safety gate ───────────────────────────────────────────────────────────────

class ClinicalSafetyGate:
    """
    Evaluates all safety invariants and gates, producing an aggregate
    SafetyGateReport that can cap the final triage recommendation.

    Parameters
    ----------
    contradiction_ceiling:
        Contradiction load above which I1 mandatory escalation fires.
    min_activated_rules:
        Minimum rules for I2 evidence sufficiency gate.
    entropy_ceiling:
        Ambiguity index above which I3 mandatory escalation fires.
    single_rule_dominance:
        Fraction of leading certainty from one rule triggering G1.
    pathognomonic_certainty_threshold:
        Certainty above which Tier-A absence triggers G2.
    max_critical_missing:
        Number of missing critical features triggering G3.
    confusion_zone_max_gap:
        Certainty gap below which confusion zone penalty (G4) applies.
    confusion_zone_penalty:
        Certainty value to subtract when G4 applies.
    overconfidence_certainty:
        Certainty above which G5 may trigger.
    overconfidence_min_contradiction:
        Minimum contradiction load for G5 to trigger.
    """

    _CRITICAL_FEATURES: frozenset[str] = frozenset({
        "koebner_phenomenon", "polygonal_papules",
        "follicular_papules", "oral_mucosal_involvement",
    })

    def __init__(
        self,
        contradiction_ceiling: float = 0.40,
        min_activated_rules: int = 2,
        entropy_ceiling: float = 1.50,
        single_rule_dominance: float = 0.60,
        pathognomonic_certainty_threshold: float = 0.75,
        max_critical_missing: int = 2,
        confusion_zone_max_gap: float = 0.30,
        confusion_zone_penalty: float = 0.15,
        overconfidence_certainty: float = 0.92,
        overconfidence_min_contradiction: float = 0.10,
    ) -> None:
        self._contra_ceiling     = contradiction_ceiling
        self._min_rules          = min_activated_rules
        self._entropy_ceiling    = entropy_ceiling
        self._dominance          = single_rule_dominance
        self._patho_threshold    = pathognomonic_certainty_threshold
        self._max_missing        = max_critical_missing
        self._confusion_max_gap  = confusion_zone_max_gap
        self._confusion_penalty  = confusion_zone_penalty
        self._overconf_cert      = overconfidence_certainty
        self._overconf_contra    = overconfidence_min_contradiction

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        certainty: CertaintyDistribution,
        conflict: ConflictAnalysisResult,
        evidence: EvidenceEvaluationResult,
        missing_features: list[str] | None = None,
    ) -> SafetyGateReport:
        """
        Evaluate all invariants and gates. Returns a SafetyGateReport
        with the aggregate effective cap.

        Parameters
        ----------
        missing_features:
            List of feature names with None values in the input vector.
        """
        missing = set(missing_features or [])

        invariants = [
            self._i1_contradiction_ceiling(conflict),
            self._i2_evidence_sufficiency(evidence),
            self._i3_entropy_ceiling(certainty),
        ]
        gates = [
            self._g1_single_source_dominance(certainty, evidence),
            self._g2_pathognomonic_absence(certainty, evidence),
            self._g3_critical_missingness(missing),
            self._g4_confusion_zone_proximity(certainty, conflict),
            self._g5_overconfidence_prevention(certainty, conflict),
        ]

        report = SafetyGateReport(
            invariant_results=invariants,
            gate_results=gates,
        )

        # Confusion zone certainty penalty (G4 — applied to certainty score)
        g4 = gates[3]
        if g4.triggered:
            report.certainty_penalty = self._confusion_penalty

        # Determine effective cap (most severe triggered cap wins)
        all_triggered = [r for r in invariants + gates if r.triggered and r.cap is not None]
        if all_triggered:
            report.any_triggered = True
            report.effective_cap = max(
                all_triggered, key=lambda r: r.cap.severity_rank
            ).cap

        return report

    # ── Invariants ────────────────────────────────────────────────────────────

    def _i1_contradiction_ceiling(self, conflict: ConflictAnalysisResult) -> GateResult:
        triggered = conflict.contradiction_load >= self._contra_ceiling
        return GateResult(
            gate_id="I1",
            gate_name="Contradiction Safety Ceiling",
            triggered=triggered,
            cap=TriageRecommendation.BIOPSY_RECOMMENDED if triggered else None,
            rationale=(
                f"Contradiction load {conflict.contradiction_load:.3f} "
                f">= ceiling {self._contra_ceiling}" if triggered
                else f"Contradiction load {conflict.contradiction_load:.3f} within safe range."
            ),
            measured_value=conflict.contradiction_load,
            threshold=self._contra_ceiling,
        )

    def _i2_evidence_sufficiency(self, evidence: EvidenceEvaluationResult) -> GateResult:
        total_active = evidence.total_rules_active
        triggered = total_active < self._min_rules
        return GateResult(
            gate_id="I2",
            gate_name="Evidence Sufficiency Floor",
            triggered=triggered,
            cap=TriageRecommendation.AMBIGUOUS_PRESENTATION if triggered else None,
            rationale=(
                f"Only {total_active} rule(s) active; minimum required: {self._min_rules}." if triggered
                else f"{total_active} active rules satisfy the evidence floor."
            ),
            measured_value=float(total_active),
            threshold=float(self._min_rules),
        )

    def _i3_entropy_ceiling(self, certainty: CertaintyDistribution) -> GateResult:
        triggered = certainty.ambiguity_index > self._entropy_ceiling
        return GateResult(
            gate_id="I3",
            gate_name="Entropy Escalation Ceiling",
            triggered=triggered,
            cap=TriageRecommendation.BIOPSY_RECOMMENDED if triggered else None,
            rationale=(
                f"Ambiguity index {certainty.ambiguity_index:.3f} bits "
                f"> ceiling {self._entropy_ceiling} bits." if triggered
                else f"Ambiguity index {certainty.ambiguity_index:.3f} bits within safe range."
            ),
            measured_value=certainty.ambiguity_index,
            threshold=self._entropy_ceiling,
        )

    # ── Gates ─────────────────────────────────────────────────────────────────

    def _g1_single_source_dominance(
        self,
        certainty: CertaintyDistribution,
        evidence: EvidenceEvaluationResult,
    ) -> GateResult:
        """Check if a single rule contributes > 60% of leading certainty."""
        leading_vec = evidence.get(certainty.leading_disease)
        triggered = False
        if leading_vec and leading_vec.active_rule_count == 1:
            # Single rule drives all evidence
            triggered = certainty.max_certainty > 0.0
        return GateResult(
            gate_id="G1",
            gate_name="Single-Source Dominance",
            triggered=triggered,
            cap=TriageRecommendation.MODERATE_CERTAINTY if triggered else None,
            rationale=(
                "Entire certainty for leading hypothesis derived from a single rule." if triggered
                else "Multiple rules contribute to leading certainty."
            ),
        )

    def _g2_pathognomonic_absence(
        self,
        certainty: CertaintyDistribution,
        evidence: EvidenceEvaluationResult,
    ) -> GateResult:
        """High certainty without any Tier-A (pathognomonic) rule active."""
        leading_vec = evidence.get(certainty.leading_disease)
        has_patho = leading_vec.has_pathognomonic if leading_vec else False
        triggered = (certainty.max_certainty > self._patho_threshold and not has_patho)
        return GateResult(
            gate_id="G2",
            gate_name="Pathognomonic Absence under High Certainty",
            triggered=triggered,
            cap=TriageRecommendation.MODERATE_CERTAINTY if triggered else None,
            rationale=(
                f"Certainty {certainty.max_certainty:.3f} > {self._patho_threshold} "
                f"without pathognomonic (Tier-A) evidence." if triggered
                else "Pathognomonic evidence present or certainty within safe range."
            ),
            measured_value=certainty.max_certainty,
            threshold=self._patho_threshold,
        )

    def _g3_critical_missingness(self, missing: set[str]) -> GateResult:
        """More than max_critical_missing critical features absent."""
        missing_critical = missing & self._CRITICAL_FEATURES
        count = len(missing_critical)
        triggered = count > self._max_missing
        return GateResult(
            gate_id="G3",
            gate_name="Critical Feature Missingness",
            triggered=triggered,
            cap=TriageRecommendation.AMBIGUOUS_PRESENTATION if triggered else None,
            rationale=(
                f"{count} critical features missing: {sorted(missing_critical)}. "
                f"Maximum tolerated: {self._max_missing}." if triggered
                else f"{count} critical feature(s) missing — within tolerated threshold."
            ),
            measured_value=float(count),
            threshold=float(self._max_missing),
        )

    def _g4_confusion_zone_proximity(
        self,
        certainty: CertaintyDistribution,
        conflict: ConflictAnalysisResult,
    ) -> GateResult:
        """Leading pair in known confusion zone with small certainty gap."""
        in_zone = bool(conflict.confusion_zone_active)
        triggered = in_zone and certainty.certainty_gap < self._confusion_max_gap
        return GateResult(
            gate_id="G4",
            gate_name="Confusion Zone Proximity",
            triggered=triggered,
            cap=None,  # G4 applies a penalty rather than a cap
            rationale=(
                f"Leading pair in confusion zone with gap={certainty.certainty_gap:.3f} "
                f"< {self._confusion_max_gap}. Certainty penalty applied." if triggered
                else "No active confusion zone proximity concern."
            ),
            measured_value=certainty.certainty_gap,
            threshold=self._confusion_max_gap,
        )

    def _g5_overconfidence_prevention(
        self,
        certainty: CertaintyDistribution,
        conflict: ConflictAnalysisResult,
    ) -> GateResult:
        """Very high certainty combined with non-trivial contradiction load."""
        triggered = (
            certainty.max_certainty > self._overconf_cert
            and conflict.contradiction_load >= self._overconf_contra
        )
        return GateResult(
            gate_id="G5",
            gate_name="Overconfidence Prevention",
            triggered=triggered,
            cap=TriageRecommendation.MODERATE_CERTAINTY if triggered else None,
            rationale=(
                f"Certainty {certainty.max_certainty:.3f} > {self._overconf_cert} "
                f"with contradiction_load={conflict.contradiction_load:.3f} "
                f">= {self._overconf_contra}. Overconfidence risk." if triggered
                else "Certainty and contradiction load within acceptable bounds."
            ),
            measured_value=certainty.max_certainty,
            threshold=self._overconf_cert,
        )
