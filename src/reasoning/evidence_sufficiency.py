"""
EvidenceSufficiencyAnalyzer — evidence quality and coverage assessment.

Distinguishes between two clinically distinct forms of high certainty:

  (A) High certainty from WEAK evidence:
      A single pathognomonic feature dominates; supporting evidence is absent.
      This produces fragile certainty — one missing feature collapses the case.

  (B) High certainty from STABLE sufficient evidence:
      Multiple independent evidence threads converge. The diagnosis is robust
      to individual feature removal.

This distinction is crucial for biopsy triage: SAFE_NON_INVASIVE_TRIAGE
requires not just high certainty but sufficient evidence diversity.

Sufficiency dimensions
----------------------
1. Anatomical coverage     — evidence spans multiple anatomical domains
2. Evidence tier diversity — both Tier A and Tier B evidence present
3. Rule count adequacy     — sufficient rules activated for the leading disease
4. Consistency score       — low variance in rule activation across features
5. Biopsy-free sufficiency — all four criteria above satisfied
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.reasoning.certainty_propagator import CertaintyDistribution
from src.reasoning.evidence_evaluator import DiseaseEvidenceVector, EvidenceEvaluationResult


# ── Anatomical domain classification ─────────────────────────────────────────

_ANATOMICAL_DOMAINS: dict[str, str] = {
    "koebner_phenomenon":       "skin_response",
    "polygonal_papules":        "skin_morphology",
    "follicular_papules":       "follicular",
    "oral_mucosal_involvement": "mucosal",
    "knee_and_elbow_involvement": "topographic",
    "scalp_involvement":        "topographic",
    "family_history":           "hereditary",
    "erythema":                 "inflammatory",
    "scaling":                  "inflammatory",
    "definite_borders":         "skin_morphology",
    "itching":                  "symptomatic",
}


# ── Sufficiency report ────────────────────────────────────────────────────────

@dataclass
class SufficiencyReport:
    """
    Evidence quality and sufficiency analysis for the leading disease hypothesis.
    Produced by EvidenceSufficiencyAnalyzer.analyze().
    """

    disease:                str
    anatomical_domains_covered: list[str]
    domain_coverage_fraction: float        # domains covered / total possible
    has_tier_a_evidence:    bool
    has_tier_b_evidence:    bool
    tier_diversity_score:   float          # 0.0 = single tier, 1.0 = both tiers
    active_rule_count:      int
    rule_adequacy_score:    float          # normalised rule count metric
    consistency_score:      float          # rule activation uniformity
    aggregate_sufficiency:  float          # composite [0.0, 1.0]
    is_biopsy_free_sufficient: bool        # all dimensions meet threshold
    fragility_risk:         str            # "low" | "moderate" | "high"
    insufficiency_reasons:  list[str]      # human-readable gaps

    @property
    def summary(self) -> str:
        status = "sufficient" if self.is_biopsy_free_sufficient else "insufficient"
        return (
            f"Evidence {status} for {self.disease.replace('_', ' ')}: "
            f"sufficiency={self.aggregate_sufficiency:.2f}, "
            f"domains={len(self.anatomical_domains_covered)}, "
            f"rules={self.active_rule_count}, "
            f"fragility={self.fragility_risk}."
        )


# ── Analyzer ──────────────────────────────────────────────────────────────────

class EvidenceSufficiencyAnalyzer:
    """
    Assesses the quality and coverage of evidence supporting the leading
    disease hypothesis, independent of its raw certainty score.

    Parameters
    ----------
    min_anatomical_domains:
        Minimum anatomical domains required for coverage adequacy. Default: 2.
    min_active_rules:
        Minimum activated rules for leading disease. Default: 3.
    biopsy_free_sufficiency_threshold:
        Aggregate sufficiency score required for SAFE recommendation. Default: 0.60.
    """

    _MIN_DOMAIN_SCORE    = 0.25   # ≥ 1 domain (minimum)
    _GOOD_DOMAIN_SCORE   = 0.50   # ≥ 2 domains (coverage)
    _STRONG_DOMAIN_SCORE = 0.75   # ≥ 3 domains (strong coverage)

    def __init__(
        self,
        min_anatomical_domains: int = 2,
        min_active_rules: int = 3,
        biopsy_free_sufficiency_threshold: float = 0.60,
    ) -> None:
        self._min_domains    = min_anatomical_domains
        self._min_rules      = min_active_rules
        self._biopsy_thresh  = biopsy_free_sufficiency_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        evidence: EvidenceEvaluationResult,
        certainty: CertaintyDistribution,
    ) -> SufficiencyReport:
        """
        Analyze evidence sufficiency for the leading disease hypothesis.
        """
        disease = certainty.leading_disease
        vec     = evidence.get(disease)

        if not vec or vec.active_rule_count == 0:
            return self._empty_report(disease)

        # ── Anatomical domain coverage ────────────────────────────────────────
        contributing = {
            feat
            for r in vec.activated_rules if r.status != "dormant"
            for feat in r.contributing_features
        }
        covered_domains = list({
            _ANATOMICAL_DOMAINS.get(f, "other")
            for f in contributing
        })
        total_domains = len(set(_ANATOMICAL_DOMAINS.values()))
        domain_fraction = len(covered_domains) / max(total_domains, 1)

        # ── Tier diversity ────────────────────────────────────────────────────
        has_a = vec.has_pathognomonic
        has_b = vec.tier_b_count > 0
        tier_diversity = (
            1.0 if (has_a and has_b) else
            0.5 if (has_a or has_b) else
            0.0
        )

        # ── Rule adequacy ─────────────────────────────────────────────────────
        rule_count = vec.active_rule_count
        rule_adequacy = min(rule_count / max(self._min_rules * 1.5, 1), 1.0)

        # ── Consistency score (activation uniformity across active rules) ─────
        active_rules = [r for r in vec.activated_rules if r.status != "dormant"]
        consistency = self._activation_consistency(active_rules)

        # ── Aggregate sufficiency ─────────────────────────────────────────────
        weights = {
            "domain": 0.30,
            "tier":   0.30,
            "rules":  0.25,
            "consistency": 0.15,
        }
        aggregate = (
            weights["domain"]      * domain_fraction
            + weights["tier"]      * tier_diversity
            + weights["rules"]     * rule_adequacy
            + weights["consistency"] * consistency
        )

        # ── Biopsy-free sufficiency decision ──────────────────────────────────
        reasons: list[str] = []
        if len(covered_domains) < self._min_domains:
            reasons.append(
                f"Anatomical coverage insufficient "
                f"({len(covered_domains)} domain(s) < {self._min_domains} required)."
            )
        if not has_a:
            reasons.append("No pathognomonic (Tier-A) evidence identified.")
        if rule_count < self._min_rules:
            reasons.append(
                f"Insufficient activated rules ({rule_count} < {self._min_rules})."
            )
        is_sufficient = (
            aggregate >= self._biopsy_thresh
            and len(covered_domains) >= self._min_domains
        )

        # ── Fragility risk ────────────────────────────────────────────────────
        fragility = (
            "low"      if aggregate >= 0.75 else
            "moderate" if aggregate >= 0.50 else
            "high"
        )

        return SufficiencyReport(
            disease=disease,
            anatomical_domains_covered=covered_domains,
            domain_coverage_fraction=domain_fraction,
            has_tier_a_evidence=has_a,
            has_tier_b_evidence=has_b,
            tier_diversity_score=tier_diversity,
            active_rule_count=rule_count,
            rule_adequacy_score=rule_adequacy,
            consistency_score=consistency,
            aggregate_sufficiency=aggregate,
            is_biopsy_free_sufficient=is_sufficient,
            fragility_risk=fragility,
            insufficiency_reasons=reasons,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _activation_consistency(
        active_rules: list,
    ) -> float:
        """
        Measures how uniformly evidence is distributed across active rules.
        High consistency → evidence is spread; low → dominated by one rule.
        Returns a score in [0.0, 1.0].
        """
        if not active_rules:
            return 0.0
        scores = [r.activation_score for r in active_rules]
        if len(scores) == 1:
            return 0.5  # single rule — partial consistency
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance)
        # Normalise: low std → high consistency
        cv = std / max(mean, 1e-6)   # coefficient of variation
        return max(0.0, 1.0 - min(cv, 1.0))

    @staticmethod
    def _empty_report(disease: str) -> "SufficiencyReport":
        return SufficiencyReport(
            disease=disease,
            anatomical_domains_covered=[],
            domain_coverage_fraction=0.0,
            has_tier_a_evidence=False,
            has_tier_b_evidence=False,
            tier_diversity_score=0.0,
            active_rule_count=0,
            rule_adequacy_score=0.0,
            consistency_score=0.0,
            aggregate_sufficiency=0.0,
            is_biopsy_free_sufficient=False,
            fragility_risk="high",
            insufficiency_reasons=["No evidence rules activated."],
        )
