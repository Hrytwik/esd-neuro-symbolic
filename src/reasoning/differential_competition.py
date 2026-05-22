"""
DifferentialCompetitionEngine — inter-hypothesis competition dynamics.

Models the competitive relationship between disease hypotheses, where
evidence accumulation for one disease suppresses competing diseases
and vice versa. This captures the inherently exclusive nature of
erythemato-squamous differential diagnosis.

Competition mechanics
---------------------
When a disease hypothesis gains tier-A (pathognomonic) evidence, it
exerts suppression pressure on competing diseases proportional to the
diagnostic specificity of that evidence. Similarly, when a contradiction
is active between two diseases, the penalised disease's competitive
standing is further weakened.

Suppression propagation
-----------------------
  suppression[target] += tier_a_score[source] × specificity_weight
  suppression[target] += contradiction_penalty[source→target]

Competition score
-----------------
  competition_score[disease] = certainty[disease] − suppression_received[disease]

This score drives the DifferentialCompetitionEngine's rankings, which
may differ from raw certainty rankings in high-contradiction cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.reasoning.certainty_propagator import CertaintyDistribution
from src.reasoning.conflict_analyzer import ConflictAnalysisResult
from src.reasoning.evidence_evaluator import EvidenceEvaluationResult


# ── Competition state ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HypothesisCompetitionState:
    """Competition-adjusted state for a single disease hypothesis."""

    disease:             str
    certainty:           float        # raw certainty from propagator
    suppression_received: float       # total suppression from competitors
    competition_score:   float        # certainty − suppression_received
    dominant_suppressor: str | None   # disease exerting strongest suppression
    rank:                int


@dataclass
class CompetitionResult:
    """
    Full inter-hypothesis competition analysis for a single reasoning step.
    Produced by DifferentialCompetitionEngine.evaluate().
    """

    competition_states:      list[HypothesisCompetitionState]
    leading_by_competition:  str
    competition_gap:         float    # top1 − top2 on competition score
    suppression_map:         dict[str, float]   # total suppression per disease
    highest_tension_pair:    tuple[str, str] | None
    divergence_amplified:    bool    # True if competition gap > certainty gap

    def get(self, disease: str) -> HypothesisCompetitionState | None:
        for s in self.competition_states:
            if s.disease == disease:
                return s
        return None

    def competition_score(self, disease: str) -> float:
        state = self.get(disease)
        return state.competition_score if state else 0.0


# ── Differential competition engine ──────────────────────────────────────────

class DifferentialCompetitionEngine:
    """
    Models inter-hypothesis suppression and competition dynamics.

    Pathognomonic evidence (Tier A) for one disease suppresses competing
    diseases proportional to a specificity weight. Contradiction penalties
    amplify suppression toward penalised diseases.

    Parameters
    ----------
    tier_a_specificity_weight:
        Suppression pressure exerted per unit of Tier-A evidence on
        all competing diseases. Default: 0.35.
    contradiction_amplification:
        Additional suppression multiplier applied to contradiction penalties.
        Default: 1.20.
    """

    _ALL_DISEASES: tuple[str, ...] = (
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris",
    )

    def __init__(
        self,
        tier_a_specificity_weight: float = 0.35,
        contradiction_amplification: float = 1.20,
    ) -> None:
        self._tier_a_weight     = tier_a_specificity_weight
        self._contra_amplify    = contradiction_amplification

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        certainty: CertaintyDistribution,
        evidence:  EvidenceEvaluationResult,
        conflict:  ConflictAnalysisResult,
    ) -> CompetitionResult:
        """
        Compute competition-adjusted scores for all disease hypotheses.
        """
        diseases = [h.disease for h in certainty.hypotheses]
        certainty_map = {h.disease: h.certainty for h in certainty.hypotheses}

        # ── Build suppression map ─────────────────────────────────────────────
        suppression: dict[str, float] = {d: 0.0 for d in diseases}

        # Tier-A suppression: pathognomonic evidence for disease A → suppresses all others
        for disease in diseases:
            vec = evidence.get(disease)
            if not vec or vec.tier_a_score <= 0.0:
                continue
            for competitor in diseases:
                if competitor == disease:
                    continue
                suppression[competitor] += vec.tier_a_score * self._tier_a_weight

        # Contradiction suppression: amplified penalty propagation
        for disease, penalty in conflict.penalty_by_disease.items():
            if disease in suppression:
                suppression[disease] += penalty * self._contra_amplify

        # ── Compute competition scores ─────────────────────────────────────────
        competition_scores: dict[str, float] = {}
        for disease in diseases:
            raw_cert  = certainty_map.get(disease, 0.0)
            sup       = suppression.get(disease, 0.0)
            competition_scores[disease] = max(raw_cert - sup, 0.0)

        # ── Rank by competition score ──────────────────────────────────────────
        ranked = sorted(diseases, key=lambda d: competition_scores[d], reverse=True)

        # Identify dominant suppressor for each disease
        states: list[HypothesisCompetitionState] = []
        for rank_idx, disease in enumerate(ranked, start=1):
            # Find which disease suppresses this one the most
            max_suppressor: str | None = None
            max_sup = 0.0
            for source in diseases:
                if source == disease:
                    continue
                vec = evidence.get(source)
                tier_a_contribution = (
                    (vec.tier_a_score * self._tier_a_weight) if vec else 0.0
                )
                contra_contribution = (
                    conflict.penalty_for(disease) * self._contra_amplify
                    if source == certainty.leading_disease else 0.0
                )
                total_from_source = tier_a_contribution + contra_contribution
                if total_from_source > max_sup:
                    max_sup = total_from_source
                    max_suppressor = source

            states.append(HypothesisCompetitionState(
                disease=disease,
                certainty=certainty_map.get(disease, 0.0),
                suppression_received=suppression.get(disease, 0.0),
                competition_score=competition_scores[disease],
                dominant_suppressor=max_suppressor,
                rank=rank_idx,
            ))

        leading   = ranked[0] if ranked else "unknown"
        second    = ranked[1] if len(ranked) > 1 else "unknown"
        comp_gap  = competition_scores.get(leading, 0.0) - competition_scores.get(second, 0.0)
        cert_gap  = certainty.certainty_gap

        # Identify highest tension pair from conflict data
        highest_pair: tuple[str, str] | None = None
        if conflict.highest_tension_pair:
            t = conflict.highest_tension_pair
            highest_pair = (t.source_disease, t.target_disease)

        return CompetitionResult(
            competition_states=states,
            leading_by_competition=leading,
            competition_gap=comp_gap,
            suppression_map=suppression,
            highest_tension_pair=highest_pair,
            divergence_amplified=(comp_gap > cert_gap),
        )
