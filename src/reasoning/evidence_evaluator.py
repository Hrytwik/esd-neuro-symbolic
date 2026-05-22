"""
DiagnosticEvidenceEvaluator — Stages 1, 2, and 4 of the reasoning pipeline.

Evaluates diagnostic rules against a graded feature vector, computing
per-disease evidence scores across three evidence tiers:

  Tier A — pathognomonic evidence (high discriminating power, singular)
  Tier B — supportive evidence (reinforcing, compound contribution)
  Tier D — discriminating evidence (cross-disease differentiation)

Activation semantics by rule logic type
----------------------------------------
  binary    — ALL supporting features must satisfy their conditions.
              Score = confidence_weight if all pass, else 0.
  threshold — ANY supporting feature satisfying its condition contributes.
              Score = confidence_weight × (n_passing / n_features).
  composite — Weighted partial sum: Σ (partial_weight × fuzzy_grade).
              Score = min(weighted_sum, confidence_weight).
  fuzzy     — Direct fuzzy aggregation: Σ (partial_weight × fuzzy_grade).
              Equivalent to composite but semantically distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.reasoning.clinical_grading import ClinicalGradingModule, GradingResult

_BINARY_FEATURES: set[str] = {
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history",
}

_DORMANT_THRESHOLD  = 0.05
_MIN_PARTIAL        = 0.10


# ── Per-rule activation result ────────────────────────────────────────────────

@dataclass
class RuleEvaluationResult:
    """Activation outcome for a single diagnostic rule."""

    rule_id:              str
    disease_target:       str
    evidence_tier:        str          # "A" | "B" | "D"
    activation_logic:     str
    activation_score:     float        # 0.0 – 1.0, penalised
    raw_score:            float        # pre-penalty
    confidence_weight:    float
    status:               str          # "active" | "partial" | "dormant"
    contributing_features: list[str]   # features that passed their conditions
    failed_features:      list[str]    # features that failed their conditions
    is_tier_a:            bool


# ── Per-disease evidence vector ───────────────────────────────────────────────

@dataclass
class DiseaseEvidenceVector:
    """Aggregated evidence for a single disease hypothesis."""

    disease:            str
    raw_evidence_score: float           # sum of raw rule scores
    tier_a_score:       float
    tier_b_score:       float
    tier_d_score:       float
    activated_rules:    list[RuleEvaluationResult] = field(default_factory=list)
    active_rule_count:  int = 0
    tier_a_count:       int = 0
    tier_b_count:       int = 0
    tier_d_count:       int = 0
    coverage_fraction:  float = 0.0    # fraction of available rules activated
    has_pathognomonic:  bool = False


# ── Full evaluation result ────────────────────────────────────────────────────

@dataclass
class EvidenceEvaluationResult:
    """
    Complete evidence evaluation across all six disease hypotheses.
    Produced by DiagnosticEvidenceEvaluator.evaluate().
    """

    disease_vectors:      dict[str, DiseaseEvidenceVector]
    evaluated_tiers:      list[str]
    total_rules_checked:  int
    total_rules_active:   int
    leading_disease:      str
    second_disease:       str

    def get(self, disease: str) -> DiseaseEvidenceVector | None:
        return self.disease_vectors.get(disease)

    def ranked(self) -> list[DiseaseEvidenceVector]:
        """Disease vectors sorted by raw_evidence_score descending."""
        return sorted(
            self.disease_vectors.values(),
            key=lambda v: v.raw_evidence_score,
            reverse=True,
        )

    def score(self, disease: str) -> float:
        vec = self.disease_vectors.get(disease)
        return vec.raw_evidence_score if vec else 0.0


# ── Evidence evaluator ────────────────────────────────────────────────────────

class DiagnosticEvidenceEvaluator:
    """
    Evaluates the diagnostic rule base against a graded clinical feature
    vector to produce disease-specific evidence scores.

    The evaluator respects evidence tier ordering — Tier A rules (pathognomonic)
    are evaluated first and carry the greatest discriminating weight. Tier B
    rules (supportive) reinforce existing evidence without independently
    establishing a diagnosis. Tier D rules (discriminating) are evaluated only
    when two leading hypotheses are in close competition.

    Parameters
    ----------
    grading_module:
        Instance of ClinicalGradingModule for condition evaluation.
    min_activation_threshold:
        Rules with activation_score below this value are classified dormant
        and excluded from evidence accumulation.
    """

    _ALL_DISEASES: tuple[str, ...] = (
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris",
    )

    def __init__(
        self,
        grading_module: ClinicalGradingModule | None = None,
        min_activation_threshold: float = 0.10,
    ) -> None:
        self._grading = grading_module or ClinicalGradingModule()
        self._min_activation = min_activation_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        grading_result: GradingResult,
        rules: list[dict[str, Any]],
        tiers: list[str] | None = None,
    ) -> EvidenceEvaluationResult:
        """
        Evaluate all rules in the supplied list against the graded feature
        vector and return per-disease evidence vectors.

        Parameters
        ----------
        grading_result:
            Graded feature vector from ClinicalGradingModule.
        rules:
            List of rule dicts (from DiagnosticRuleRepository).
        tiers:
            Tier filter; if None, all tiers are evaluated.
        """
        active_tiers = set(tiers) if tiers else {"A", "B", "D"}
        filtered = [
            r for r in rules
            if r.get("evidence_tier") in active_tiers
        ]

        # Initialise per-disease accumulators
        vectors: dict[str, DiseaseEvidenceVector] = {
            disease: DiseaseEvidenceVector(
                disease=disease,
                raw_evidence_score=0.0,
                tier_a_score=0.0,
                tier_b_score=0.0,
                tier_d_score=0.0,
            )
            for disease in self._ALL_DISEASES
        }

        total_active = 0

        for rule in filtered:
            result = self._evaluate_rule(rule, grading_result)
            disease = result.disease_target

            if disease not in vectors:
                vectors[disease] = DiseaseEvidenceVector(
                    disease=disease,
                    raw_evidence_score=0.0,
                    tier_a_score=0.0,
                    tier_b_score=0.0,
                    tier_d_score=0.0,
                )

            vec = vectors[disease]
            vec.activated_rules.append(result)

            if result.status != "dormant":
                vec.raw_evidence_score += result.activation_score
                if result.evidence_tier == "A":
                    vec.tier_a_score += result.activation_score
                    vec.tier_a_count += 1
                    vec.has_pathognomonic = True
                elif result.evidence_tier == "B":
                    vec.tier_b_score += result.activation_score
                    vec.tier_b_count += 1
                elif result.evidence_tier == "D":
                    vec.tier_d_score += result.activation_score
                    vec.tier_d_count += 1
                vec.active_rule_count += 1
                total_active += 1

        # Compute coverage fractions
        rules_per_disease: dict[str, int] = {}
        for r in filtered:
            d = r.get("disease_target", "")
            rules_per_disease[d] = rules_per_disease.get(d, 0) + 1

        for disease, vec in vectors.items():
            total = rules_per_disease.get(disease, 1)
            vec.coverage_fraction = vec.active_rule_count / max(total, 1)

        # Identify leading and second disease
        ranked = sorted(
            vectors.values(),
            key=lambda v: v.raw_evidence_score,
            reverse=True,
        )
        leading = ranked[0].disease if ranked else "unknown"
        second  = ranked[1].disease if len(ranked) > 1 else "unknown"

        return EvidenceEvaluationResult(
            disease_vectors=vectors,
            evaluated_tiers=sorted(active_tiers),
            total_rules_checked=len(filtered),
            total_rules_active=total_active,
            leading_disease=leading,
            second_disease=second,
        )

    # ── Rule activation logic ─────────────────────────────────────────────────

    def _evaluate_rule(
        self,
        rule: dict[str, Any],
        grading: GradingResult,
    ) -> RuleEvaluationResult:
        """Evaluate a single rule and return its activation result."""

        rule_id       = rule.get("rule_id", "UNKNOWN")
        disease       = rule.get("disease_target", "unknown")
        tier          = rule.get("evidence_tier", "B")
        logic         = rule.get("activation_logic", "binary")
        conf_weight   = float(rule.get("confidence_weight", 0.5))
        min_threshold = float(rule.get("min_activation_threshold", 0.10))
        supporting    = rule.get("supporting_features", [])

        # Compute raw activation score
        raw_score, contributing, failed = self._compute_activation(
            logic, supporting, grading, conf_weight
        )

        # Apply min_activation_threshold
        if raw_score < min_threshold:
            raw_score = 0.0

        status = (
            "active"  if raw_score >= conf_weight * 0.80 else
            "partial" if raw_score >= self._min_activation else
            "dormant"
        )

        return RuleEvaluationResult(
            rule_id=rule_id,
            disease_target=disease,
            evidence_tier=tier,
            activation_logic=logic,
            activation_score=raw_score,
            raw_score=raw_score,
            confidence_weight=conf_weight,
            status=status,
            contributing_features=contributing,
            failed_features=failed,
            is_tier_a=(tier == "A"),
        )

    def _compute_activation(
        self,
        logic: str,
        supporting: list[dict],
        grading: GradingResult,
        conf_weight: float,
    ) -> tuple[float, list[str], list[str]]:
        """
        Compute raw activation score for a rule given its logic type.
        Returns (score, contributing_feature_names, failed_feature_names).
        """
        if not supporting:
            return 0.0, [], []

        contributing: list[str] = []
        failed: list[str] = []

        if logic == "binary":
            # ALL features must pass
            for sf in supporting:
                feature  = sf["feature"]
                raw_val  = grading.raw_value(feature)
                condition = sf["condition"]
                threshold = float(sf["threshold"])
                if self._condition_met(raw_val, condition, threshold):
                    contributing.append(feature)
                else:
                    failed.append(feature)
            if failed:
                return 0.0, contributing, failed
            return conf_weight, contributing, failed

        elif logic in ("threshold", "fuzzy", "composite"):
            # Weighted partial sum
            total_weight = sum(float(sf.get("partial_weight", 1.0)) for sf in supporting)
            score = 0.0
            for sf in supporting:
                feature   = sf["feature"]
                raw_val   = grading.raw_value(feature)
                condition = sf["condition"]
                threshold = float(sf["threshold"])
                p_weight  = float(sf.get("partial_weight", 1.0 / len(supporting)))
                is_bin    = feature in _BINARY_FEATURES

                if self._condition_met(raw_val, condition, threshold):
                    # For composite/fuzzy: weight by fuzzy grade
                    if logic in ("composite", "fuzzy"):
                        fuzzy = grading.fuzzy_value(feature)
                        score += p_weight * fuzzy
                    else:
                        score += p_weight
                    contributing.append(feature)
                else:
                    failed.append(feature)

            # Normalise by total weight and scale by confidence_weight
            if total_weight > 0:
                normalised = score / total_weight
            else:
                normalised = 0.0
            return min(normalised * conf_weight, conf_weight), contributing, failed

        # Unknown logic — return zero
        return 0.0, [], []

    @staticmethod
    def _condition_met(
        raw_value: int | float | None,
        condition: str,
        threshold: float,
    ) -> bool:
        """Evaluate a single feature condition."""
        if raw_value is None:
            return False
        v = float(raw_value)
        t = float(threshold)
        if condition == "eq":  return abs(v - t) < 0.5   # integer equality with tolerance
        if condition == "gte": return v >= t
        if condition == "lte": return v <= t
        if condition == "gt":  return v > t
        if condition == "lt":  return v < t
        return False
