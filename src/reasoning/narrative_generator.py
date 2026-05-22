"""
DiagnosticNarrativeGenerator — Stage 6 clinical reasoning narrative.

Produces a structured, clinically-grounded reasoning narrative from the
completed inference trajectory. The narrative is NOT a statistical
justification — it reads as a differential diagnosis rationale that a
clinician can trace, verify, and challenge.

Narrative structure
-------------------
1. Clinical presentation summary (activated features)
2. Evidence interpretation (Tier A/B findings, by disease)
3. Contradiction analysis (active cross-disease conflicts)
4. Certainty evolution summary (how confidence developed)
5. Safety assessment (triggered gates with clinical implications)
6. Triage rationale (final recommendation with justification)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.reasoning.certainty_propagator import CertaintyDistribution
from src.reasoning.conflict_analyzer import ConflictAnalysisResult
from src.reasoning.escalation_engine import TriageDecision
from src.reasoning.evidence_evaluator import EvidenceEvaluationResult
from src.reasoning.safety_gate import SafetyGateReport
from src.reasoning.state_tracker import DiagnosticState


# ── Narrative output ──────────────────────────────────────────────────────────

@dataclass
class ClinicalNarrative:
    """Structured clinical reasoning narrative for a single case."""

    presentation_summary:   str
    evidence_interpretation: str
    contradiction_summary:  str
    certainty_evolution:    str
    safety_assessment:      str
    triage_rationale:       str

    def full_text(self, separator: str = "\n\n") -> str:
        """Concatenate all sections into a single narrative document."""
        sections = [
            ("Clinical Presentation", self.presentation_summary),
            ("Evidence Interpretation", self.evidence_interpretation),
            ("Contradiction Analysis", self.contradiction_summary),
            ("Certainty Assessment", self.certainty_evolution),
            ("Safety Assessment", self.safety_assessment),
            ("Triage Rationale", self.triage_rationale),
        ]
        parts = []
        for heading, content in sections:
            parts.append(f"[{heading}]\n{content}")
        return separator.join(parts)


# ── Narrative generator ───────────────────────────────────────────────────────

class DiagnosticNarrativeGenerator:
    """
    Generates structured clinical reasoning narratives from completed
    inference outputs.

    The generator translates quantitative reasoning signals (certainty
    scores, contradiction loads, safety gate triggers) into clinically
    meaningful prose that remains interpretable without reference to
    the underlying computational framework.
    """

    _DISEASE_LABELS: dict[str, str] = {
        "psoriasis":               "psoriasis",
        "seborrheic_dermatitis":   "seborrheic dermatitis",
        "lichen_planus":           "lichen planus",
        "pityriasis_rosea":        "pityriasis rosea",
        "chronic_dermatitis":      "chronic dermatitis",
        "pityriasis_rubra_pilaris": "pityriasis rubra pilaris",
    }

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        evidence:      EvidenceEvaluationResult,
        conflict:      ConflictAnalysisResult,
        certainty:     CertaintyDistribution,
        safety_report: SafetyGateReport,
        decision:      TriageDecision,
        final_state:   DiagnosticState,
    ) -> ClinicalNarrative:
        """Generate the complete clinical narrative."""
        return ClinicalNarrative(
            presentation_summary=self._presentation(evidence),
            evidence_interpretation=self._evidence(evidence, certainty),
            contradiction_summary=self._contradictions(conflict),
            certainty_evolution=self._certainty(certainty),
            safety_assessment=self._safety(safety_report),
            triage_rationale=self._triage(decision, final_state),
        )

    # ── Section generators ────────────────────────────────────────────────────

    def _presentation(self, evidence: EvidenceEvaluationResult) -> str:
        """Summarise the clinical features that contributed to the differential."""
        leading_vec = evidence.get(evidence.leading_disease)
        if not leading_vec:
            return "Insufficient clinical feature data to characterise the presentation."

        active_rules = [r for r in leading_vec.activated_rules if r.status != "dormant"]
        contributing = sorted(
            {feat for r in active_rules for feat in r.contributing_features}
        )

        if not contributing:
            return "No clinical features met activation thresholds."

        feature_list = ", ".join(
            f.replace("_", " ") for f in contributing
        )
        return (
            f"The clinical evaluation identified {len(contributing)} active feature(s) "
            f"contributing to the differential: {feature_list}. "
            f"A total of {evidence.total_rules_active} diagnostic rule(s) reached "
            f"activation threshold across all disease hypotheses."
        )

    def _evidence(
        self, evidence: EvidenceEvaluationResult, certainty: CertaintyDistribution
    ) -> str:
        """Interpret activated evidence by disease and tier."""
        lines = []
        ranked = certainty.top_n(3)
        for hyp in ranked:
            vec = evidence.get(hyp.disease)
            if not vec:
                continue
            label = self._label(hyp.disease)
            tier_a = [r for r in vec.activated_rules if r.evidence_tier == "A" and r.status != "dormant"]
            tier_b = [r for r in vec.activated_rules if r.evidence_tier == "B" and r.status != "dormant"]
            cert_pct = f"{hyp.certainty * 100:.1f}%"

            tier_a_txt = (
                f"pathognomonic evidence present ({', '.join(r.rule_id for r in tier_a)})"
                if tier_a else "no pathognomonic evidence identified"
            )
            tier_b_txt = (
                f"{len(tier_b)} supportive rule(s) activated"
                if tier_b else "no supportive rules activated"
            )
            lines.append(
                f"  {label.capitalize()} [certainty {cert_pct}, rank {hyp.rank}]: "
                f"{tier_a_txt}; {tier_b_txt}."
            )

        if not lines:
            return "No differential hypotheses reached evidence threshold."

        return (
            f"The leading differential hypotheses, ranked by certainty:\n"
            + "\n".join(lines)
        )

    def _contradictions(self, conflict: ConflictAnalysisResult) -> str:
        """Summarise active cross-disease contradictions."""
        if conflict.is_contradiction_free:
            return (
                "No cross-disease contradictions detected. "
                "The clinical feature profile is internally consistent."
            )

        load = conflict.contradiction_load
        n    = len(conflict.active_contradictions)
        lines = [
            f"Contradiction load: {load:.3f} ({n} active contradiction(s))."
        ]
        for c in conflict.active_contradictions[:5]:  # show up to 5
            feature = c.trigger_feature.replace("_", " ")
            lines.append(
                f"  · {feature} (value={int(c.trigger_value)}) supports "
                f"{self._label(c.source_disease)} but penalises "
                f"{self._label(c.target_disease)} (penalty={c.penalty_weight:.2f})."
            )
        if n > 5:
            lines.append(f"  · ... and {n - 5} additional contradiction(s).")

        if conflict.mandatory_escalation:
            lines.append(
                "Contradiction load exceeds the mandatory escalation ceiling. "
                "Histological confirmation is required."
            )
        return "\n".join(lines)

    def _certainty(self, certainty: CertaintyDistribution) -> str:
        """Summarise the certainty distribution and its stability."""
        lead  = self._label(certainty.leading_disease)
        cert  = certainty.max_certainty
        gap   = certainty.certainty_gap
        entropy = certainty.ambiguity_index
        damped  = certainty.contradiction_dampened

        stability = (
            "highly certain" if certainty.is_highly_certain else
            "stable" if certainty.is_stable else
            "ambiguous" if certainty.is_ambiguous else
            "partially aligned"
        )

        dampening_note = (
            " Contradiction dampening was applied to reduce overconfidence "
            "in the presence of active cross-disease conflicts."
            if damped else ""
        )

        return (
            f"The leading hypothesis ({lead}) reached certainty={cert:.3f} "
            f"with a gap of {gap:.3f} over the second-ranking hypothesis. "
            f"The diagnostic distribution is classified as {stability} "
            f"(Shannon entropy={entropy:.3f} bits).{dampening_note}"
        )

    def _safety(self, safety_report: SafetyGateReport) -> str:
        """Describe triggered safety gates and their clinical implications."""
        if not safety_report.any_triggered:
            return (
                "All safety invariants and gates were satisfied. "
                "No safety-driven escalation was applied."
            )

        triggered = safety_report.triggered_gates
        lines = [f"{len(triggered)} safety condition(s) triggered:"]
        for gate in triggered:
            lines.append(f"  · [{gate.gate_id}] {gate.gate_name}: {gate.rationale}")

        if safety_report.effective_cap:
            lines.append(
                f"Effective safety cap applied: "
                f"{safety_report.effective_cap.value.replace('_', ' ').lower()}."
            )
        return "\n".join(lines)

    def _triage(self, decision: TriageDecision, final_state: DiagnosticState) -> str:
        """State the final triage recommendation with its rationale."""
        rec   = decision.recommendation.value.replace("_", " ").lower()
        lead  = self._label(decision.leading_disease)
        state = final_state.value.replace("_", " ").lower()

        return (
            f"Triage recommendation: {rec.upper()}.\n"
            f"Leading differential: {lead} "
            f"(certainty={decision.max_certainty:.3f}, gap={decision.certainty_gap:.3f}).\n"
            f"Final reasoning state: {state}.\n"
            f"Rationale: {decision.decision_rationale}"
        )

    @staticmethod
    def _label(disease: str) -> str:
        labels = {
            "psoriasis":               "psoriasis",
            "seborrheic_dermatitis":   "seborrheic dermatitis",
            "lichen_planus":           "lichen planus",
            "pityriasis_rosea":        "pityriasis rosea",
            "chronic_dermatitis":      "chronic dermatitis",
            "pityriasis_rubra_pilaris": "pityriasis rubra pilaris",
        }
        return labels.get(disease, disease.replace("_", " "))
