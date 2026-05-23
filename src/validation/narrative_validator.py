"""
NarrativeValidator — clinical narrative plausibility assessment.

Evaluates whether the reasoning output is clinically coherent as a narrative:
the decision rationale, leading disease, and recommendation must together
form a consistent, interpretable, and clinically plausible explanation.

Narrative plausibility is distinct from technical correctness — it asks
whether a clinician reading the output would find it:

  · Internally consistent  — rationale aligns with recommendation
  · Disease-specific       — leading disease is named and unambiguous
  · Clinically grounded    — reasoning references observable evidence
  · Appropriately hedged   — high-uncertainty outputs express uncertainty
  · Non-contradictory      — conflicting narrative signals absent

The NarrativeValidator does not evaluate writing quality — it evaluates
clinical reasoning coherence as expressed in the structured output.
"""

from __future__ import annotations

import re

from src.pipeline.pipeline_runner import PipelineResult
from src.validation.behavioral_validator import Severity, ValidationSignal


# Keywords expected in rationale strings for each recommendation type
_BIOPSY_KEYWORDS   = {"biopsy", "histolog", "contradiction", "conflict",
                       "ambig", "uncertain", "insufficient", "entropy"}
_SAFE_KEYWORDS     = {"safe", "consistent", "stable", "confident", "clear",
                       "sufficient", "adequate", "certainty"}
_MODERATE_KEYWORDS = {"moderate", "reasonable", "partial", "incomplete",
                       "consider", "monitor", "follow"}
_AMBIGUOUS_KEYWORDS = {"ambig", "compet", "multiple", "unclear",
                        "insufficient", "overlap", "differential"}

# Known disease names in the 6-class panel
_KNOWN_DISEASES = {
    "psoriasis", "seborrheic_dermatitis", "lichen_planus",
    "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris",
}

# Human-readable alias fragments that may appear in narratives
_DISEASE_FRAGMENTS = {
    "psoriasis", "seborrheic", "lichen", "pityriasis", "chronic", "pilaris",
    "dermatitis", "rosea",
}


class NarrativeValidator:
    """
    Validates clinical narrative plausibility and reasoning coherence.

    Parameters
    ----------
    min_rationale_length:
        Minimum character length for a meaningful decision rationale.
    min_rationale_word_count:
        Minimum word count for a decision rationale to be considered
        clinically informative.
    require_disease_in_rationale:
        When True, the leading disease name (or a recognisable fragment)
        must appear in the decision rationale.
    require_recommendation_keywords:
        When True, the rationale should contain at least one keyword
        consistent with the recommendation type.
    """

    def __init__(
        self,
        min_rationale_length:          int  = 30,
        min_rationale_word_count:      int  = 5,
        require_disease_in_rationale:  bool = True,
        require_recommendation_keywords: bool = True,
    ) -> None:
        self._min_length     = min_rationale_length
        self._min_words      = min_rationale_word_count
        self._require_disease = require_disease_in_rationale
        self._require_keywords = require_recommendation_keywords

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        result: PipelineResult,
    ) -> tuple[list[ValidationSignal], float]:
        """
        Validate clinical narrative plausibility.

        Returns (signals, score) where score in [0, 1].
        """
        signals: list[ValidationSignal] = []

        signals.extend(self._check_rationale_substance(result))
        signals.extend(self._check_leading_disease_named(result))
        signals.extend(self._check_rationale_recommendation_alignment(result))
        signals.extend(self._check_uncertainty_expression(result))
        signals.extend(self._check_completed_stages_coherence(result))

        score = sum(1 for s in signals if s.passed) / max(len(signals), 1)
        return signals, score

    # ── Sub-checks ────────────────────────────────────────────────────────────

    def _check_rationale_substance(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        Decision rationale must be non-trivial: long enough and word-rich
        enough to constitute a clinical explanation rather than a label.
        """
        rationale = result.decision_rationale or ""
        signals: list[ValidationSignal] = []

        char_count = len(rationale.strip())
        long_enough = char_count >= self._min_length
        signals.append(ValidationSignal(
            validator="narrative",
            signal_name="rationale_length_sufficient",
            passed=long_enough,
            severity="warning",
            description=(
                f"Decision rationale has {char_count} characters. "
                + ("Sufficient." if long_enough else
                   f"Below minimum {self._min_length} — rationale is too brief.")
            ),
            measured_value=float(char_count),
            expected_range=(float(self._min_length), 2000.0),
        ))

        words = len(rationale.split())
        word_rich = words >= self._min_words
        signals.append(ValidationSignal(
            validator="narrative",
            signal_name="rationale_word_count_sufficient",
            passed=word_rich,
            severity="warning",
            description=(
                f"Decision rationale has {words} words. "
                + ("Sufficient." if word_rich else
                   f"Below minimum {self._min_words} — not clinically informative.")
            ),
            measured_value=float(words),
            expected_range=(float(self._min_words), 500.0),
        ))

        return signals

    def _check_leading_disease_named(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        The leading disease must be set and, when require_disease_in_rationale
        is True, at least a fragment of its name must appear in the rationale.
        """
        signals: list[ValidationSignal] = []
        disease  = result.leading_disease or ""
        rationale = (result.decision_rationale or "").lower()

        disease_set = bool(disease and disease in _KNOWN_DISEASES)
        signals.append(ValidationSignal(
            validator="narrative",
            signal_name="leading_disease_is_known",
            passed=disease_set,
            severity="critical",
            description=(
                f"Leading disease '{disease}' "
                + ("is a recognised" if disease_set else "is NOT a recognised")
                + " diagnostic category."
            ),
        ))

        if self._require_disease and disease:
            # Check for any recognisable fragment of the disease name
            disease_fragments = set(disease.lower().split("_")) & _DISEASE_FRAGMENTS
            mentioned = any(frag in rationale for frag in disease_fragments) if disease_fragments else False
            signals.append(ValidationSignal(
                validator="narrative",
                signal_name="leading_disease_mentioned_in_rationale",
                passed=mentioned,
                severity="info",
                description=(
                    f"Leading disease '{disease}' "
                    + ("is mentioned in" if mentioned else "is NOT mentioned in")
                    + " the decision rationale."
                ),
            ))

        return signals

    def _check_rationale_recommendation_alignment(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        The decision rationale must contain lexical signals consistent with
        the issued recommendation. A biopsy rationale should mention clinical
        uncertainty; a safe-triage rationale should mention certainty/stability.
        """
        if not self._require_keywords:
            return []

        signals: list[ValidationSignal] = []
        rec       = result.recommendation or ""
        rationale = (result.decision_rationale or "").lower()

        if rec in ("BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"):
            keyword_present = any(kw in rationale for kw in _BIOPSY_KEYWORDS)
            signals.append(ValidationSignal(
                validator="narrative",
                signal_name="biopsy_rationale_contains_clinical_concern",
                passed=keyword_present,
                severity="warning",
                description=(
                    f"Biopsy recommendation rationale "
                    + ("references clinical concern terms (coherent)."
                       if keyword_present else
                       "lacks clinical concern keywords — rationale may be generic.")
                ),
            ))

        elif rec == "SAFE_NON_INVASIVE_TRIAGE":
            keyword_present = any(kw in rationale for kw in _SAFE_KEYWORDS)
            signals.append(ValidationSignal(
                validator="narrative",
                signal_name="safe_triage_rationale_expresses_confidence",
                passed=keyword_present,
                severity="warning",
                description=(
                    f"Safe triage rationale "
                    + ("expresses certainty/stability (coherent)."
                       if keyword_present else
                       "lacks confidence-related keywords — rationale may be generic.")
                ),
            ))

        elif rec == "MODERATE_CERTAINTY":
            keyword_present = any(kw in rationale for kw in _MODERATE_KEYWORDS)
            signals.append(ValidationSignal(
                validator="narrative",
                signal_name="moderate_rationale_appropriate_language",
                passed=keyword_present,
                severity="info",
                description=(
                    f"Moderate-certainty rationale "
                    + ("uses appropriately measured language."
                       if keyword_present else
                       "lacks hedging/partial-confidence language.")
                ),
            ))

        elif rec == "AMBIGUOUS_PRESENTATION":
            keyword_present = any(kw in rationale for kw in _AMBIGUOUS_KEYWORDS)
            signals.append(ValidationSignal(
                validator="narrative",
                signal_name="ambiguous_rationale_reflects_uncertainty",
                passed=keyword_present,
                severity="info",
                description=(
                    f"Ambiguous-presentation rationale "
                    + ("reflects multi-hypothesis uncertainty."
                       if keyword_present else
                       "lacks uncertainty language for an ambiguous case.")
                ),
            ))

        return signals

    def _check_uncertainty_expression(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        High-entropy cases should express uncertainty in their rationale.
        High-certainty safe-triage cases should not express doubt.
        """
        signals: list[ValidationSignal] = []
        entropy   = result.ambiguity_index
        cert      = result.max_certainty
        rationale = (result.decision_rationale or "").lower()
        rec       = result.recommendation or ""

        uncertainty_words = {"uncertain", "ambig", "unclear", "conflict", "contradict"}
        confidence_words  = {"confident", "clear", "stable", "definit", "consistent"}

        # High-entropy case should express uncertainty
        if entropy > 1.50:
            uncertainty_expressed = any(w in rationale for w in uncertainty_words)
            signals.append(ValidationSignal(
                validator="narrative",
                signal_name="high_entropy_case_expresses_uncertainty",
                passed=uncertainty_expressed,
                severity="info",
                description=(
                    f"High entropy case (ambiguity_index={entropy:.3f} bits). "
                    + ("Rationale appropriately expresses uncertainty."
                       if uncertainty_expressed else
                       "Rationale does not acknowledge high ambiguity.")
                ),
                measured_value=entropy,
            ))

        # High-certainty safe case should not be hedging heavily
        if rec == "SAFE_NON_INVASIVE_TRIAGE" and cert >= 0.70:
            over_hedged = sum(1 for w in uncertainty_words if w in rationale) >= 2
            signals.append(ValidationSignal(
                validator="narrative",
                signal_name="high_certainty_safe_not_over_hedged",
                passed=not over_hedged,
                severity="info",
                description=(
                    f"SAFE triage with high certainty ({cert:.3f}). "
                    + ("Rationale appropriately confident."
                       if not over_hedged else
                       "Rationale is excessively hedged for a high-certainty safe case.")
                ),
                measured_value=cert,
            ))

        return signals

    def _check_completed_stages_coherence(
        self,
        result: PipelineResult,
    ) -> list[ValidationSignal]:
        """
        The reasoning pipeline must have completed the key stages to support
        a meaningful clinical narrative. A rationale produced from incomplete
        reasoning is clinically untrustworthy.
        """
        signals: list[ValidationSignal] = []
        completed = set(result.completed_stages)

        # Minimum required stages for clinical interpretation
        required_stages = {
            "clinical_grading",
            "evidence_activation",
            "contradiction_analysis",
            "certainty_propagation",
        }
        missing = required_stages - completed
        all_present = len(missing) == 0

        signals.append(ValidationSignal(
            validator="narrative",
            signal_name="required_stages_completed_for_narrative",
            passed=all_present,
            severity="warning",
            description=(
                "All minimum reasoning stages completed — narrative is fully grounded."
                if all_present else
                f"Missing stages: {sorted(missing)}. "
                "Narrative may be based on incomplete reasoning."
            ),
        ))

        # Escalation stage must have completed for a valid recommendation
        escalation_completed = "escalation" in completed or "fsm_escalation" in completed
        signals.append(ValidationSignal(
            validator="narrative",
            signal_name="escalation_stage_completed",
            passed=escalation_completed,
            severity="critical",
            description=(
                "Escalation stage completed — recommendation is formally produced."
                if escalation_completed else
                "Escalation stage NOT completed — recommendation may be default/fallback."
            ),
        ))

        return signals
