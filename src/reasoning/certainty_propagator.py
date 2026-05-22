"""
HypothesisCertaintyPropagator — Stage 5 of the progressive reasoning pipeline.

Converts raw per-disease evidence scores (net of contradiction penalties)
into a normalised certainty distribution. Certainty is not a static snapshot
— it evolves with each reasoning stage as evidence accumulates and
contradictions propagate.

Certainty computation
---------------------
1. Penalised score = raw_evidence_score − penalty_for_disease
2. Softmax normalisation over all six diseases (temperature = 1.0):
   certainty[d] = exp(penalised[d]) / Σ exp(penalised[d'])
3. Certainty gap = certainty[rank_1] − certainty[rank_2]
4. Ambiguity index = Shannon entropy H = −Σ p·log₂(p)
   (higher entropy → more ambiguous differential)

Certainty evolution
-------------------
Contradiction load decays certainty in a contradiction-sensitive manner:
if contradiction_load ≥ 0.20 and the leading certainty is not well-separated
from competitors, the effective certainty is dampened. This prevents false
confidence in the presence of active contradictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.reasoning.conflict_analyzer import ConflictAnalysisResult
from src.reasoning.evidence_evaluator import EvidenceEvaluationResult


# ── Per-hypothesis certainty entry ────────────────────────────────────────────

@dataclass(frozen=True)
class HypothesisCertainty:
    """Certainty state for a single disease hypothesis."""

    disease:              str
    raw_evidence:         float   # pre-penalty raw score
    penalised_score:      float   # after contradiction penalty
    certainty:            float   # softmax-normalised [0, 1]
    rank:                 int     # 1 = leading
    active_rule_count:    int
    tier_a_count:         int
    has_pathognomonic:    bool


# ── Full certainty distribution ───────────────────────────────────────────────

@dataclass
class CertaintyDistribution:
    """
    Normalised certainty distribution across all six disease hypotheses,
    produced by HypothesisCertaintyPropagator.propagate().
    """

    hypotheses:            list[HypothesisCertainty]
    leading_disease:       str
    second_disease:        str
    max_certainty:         float       # certainty of leading hypothesis
    certainty_gap:         float       # top1 − top2
    ambiguity_index:       float       # Shannon entropy in bits
    contradiction_load:    float       # from ConflictAnalysisResult
    contradiction_dampened: bool       # True if certainty was dampened

    # Stability metrics
    is_stable:             bool        # gap > 0.20 and certainty > 0.55
    is_highly_certain:     bool        # gap > 0.35 and certainty > 0.65
    is_ambiguous:          bool        # entropy > 1.0 bit

    def certainty_for(self, disease: str) -> float:
        for h in self.hypotheses:
            if h.disease == disease:
                return h.certainty
        return 0.0

    def rank_of(self, disease: str) -> int | None:
        for h in self.hypotheses:
            if h.disease == disease:
                return h.rank
        return None

    def top_n(self, n: int) -> list[HypothesisCertainty]:
        return self.hypotheses[:n]


# ── Certainty propagator ──────────────────────────────────────────────────────

class HypothesisCertaintyPropagator:
    """
    Propagates evidence scores through contradiction penalties and softmax
    normalisation to produce an evolving certainty distribution.

    Parameters
    ----------
    softmax_temperature:
        Temperature parameter for softmax normalisation. Lower values
        sharpen the distribution; higher values flatten it. Default: 1.0.
    contradiction_damping_threshold:
        Contradiction load at which certainty damping activates. Default: 0.20.
    certainty_decay_rate:
        Fraction of leading certainty to subtract per unit of contradiction
        load above the damping threshold. Default: 0.15.
    stability_gap_threshold:
        Minimum certainty_gap for a distribution to be considered stable.
    stability_certainty_threshold:
        Minimum max_certainty for a distribution to be considered stable.
    """

    def __init__(
        self,
        softmax_temperature: float = 1.0,
        contradiction_damping_threshold: float = 0.20,
        certainty_decay_rate: float = 0.15,
        stability_gap_threshold: float = 0.20,
        stability_certainty_threshold: float = 0.55,
        high_certainty_gap_threshold: float = 0.35,
        high_certainty_threshold: float = 0.65,
    ) -> None:
        self._temperature           = max(softmax_temperature, 1e-6)
        self._damping_threshold     = contradiction_damping_threshold
        self._decay_rate            = certainty_decay_rate
        self._stability_gap         = stability_gap_threshold
        self._stability_cert        = stability_certainty_threshold
        self._high_cert_gap         = high_certainty_gap_threshold
        self._high_cert             = high_certainty_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def propagate(
        self,
        evidence: EvidenceEvaluationResult,
        conflict: ConflictAnalysisResult,
    ) -> CertaintyDistribution:
        """
        Compute the certainty distribution from evidence and conflict analysis.

        Steps:
          1. Apply contradiction penalties to raw evidence scores.
          2. Softmax-normalise the penalised scores.
          3. Apply contradiction damping if load ≥ threshold.
          4. Compute certainty_gap and Shannon entropy.
          5. Classify stability.
        """
        diseases = list(evidence.disease_vectors.keys())

        # Step 1 — penalised scores
        penalised: dict[str, float] = {}
        for disease in diseases:
            vec     = evidence.disease_vectors[disease]
            penalty = conflict.penalty_for(disease)
            penalised[disease] = max(vec.raw_evidence_score - penalty, 0.0)

        # Step 2 — softmax normalisation
        certainties = self._softmax(penalised)

        # Step 3 — contradiction damping
        dampened = False
        load = conflict.contradiction_load
        if load >= self._damping_threshold:
            excess = load - self._damping_threshold
            decay  = excess * self._decay_rate
            # Identify the leading disease before damping
            leading_pre = max(certainties, key=certainties.get)
            if certainties[leading_pre] > 0.0:
                delta = min(decay, certainties[leading_pre] * 0.30)
                certainties[leading_pre] -= delta
                # Redistribute removed mass uniformly to non-leading diseases
                n_others = len(diseases) - 1
                if n_others > 0:
                    per_other = delta / n_others
                    for d in diseases:
                        if d != leading_pre:
                            certainties[d] += per_other
                dampened = True
            # Re-normalise to ensure sum = 1.0
            certainties = self._renormalise(certainties)

        # Step 4 — sort and rank
        ranked = sorted(diseases, key=lambda d: certainties[d], reverse=True)
        leading = ranked[0] if ranked else "unknown"
        second  = ranked[1] if len(ranked) > 1 else "unknown"

        max_cert = certainties.get(leading, 0.0)
        second_cert = certainties.get(second, 0.0)
        gap = max_cert - second_cert

        entropy = self._shannon_entropy(certainties)

        # Step 5 — build hypothesis objects
        hypotheses: list[HypothesisCertainty] = []
        for rank_idx, disease in enumerate(ranked, start=1):
            vec = evidence.disease_vectors.get(disease)
            raw = vec.raw_evidence_score if vec else 0.0
            pen = penalised.get(disease, 0.0)
            hypotheses.append(HypothesisCertainty(
                disease=disease,
                raw_evidence=raw,
                penalised_score=pen,
                certainty=certainties[disease],
                rank=rank_idx,
                active_rule_count=vec.active_rule_count if vec else 0,
                tier_a_count=vec.tier_a_count if vec else 0,
                has_pathognomonic=vec.has_pathognomonic if vec else False,
            ))

        return CertaintyDistribution(
            hypotheses=hypotheses,
            leading_disease=leading,
            second_disease=second,
            max_certainty=max_cert,
            certainty_gap=gap,
            ambiguity_index=entropy,
            contradiction_load=load,
            contradiction_dampened=dampened,
            is_stable=(gap >= self._stability_gap and max_cert >= self._stability_cert),
            is_highly_certain=(gap >= self._high_cert_gap and max_cert >= self._high_cert),
            is_ambiguous=(entropy >= 1.0),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _softmax(self, scores: dict[str, float]) -> dict[str, float]:
        """Compute softmax-normalised certainties over a score dictionary."""
        if not scores:
            return {}
        values = {d: s / self._temperature for d, s in scores.items()}
        max_v  = max(values.values())
        exps   = {d: math.exp(v - max_v) for d, v in values.items()}
        total  = sum(exps.values())
        if total < 1e-12:
            n = len(exps)
            return {d: 1.0 / n for d in exps}
        return {d: e / total for d, e in exps.items()}

    @staticmethod
    def _renormalise(certainties: dict[str, float]) -> dict[str, float]:
        """Re-normalise certainties to sum to 1.0 after damping adjustment."""
        total = sum(certainties.values())
        if total < 1e-12:
            n = len(certainties)
            return {d: 1.0 / n for d in certainties}
        return {d: v / total for d, v in certainties.items()}

    @staticmethod
    def _shannon_entropy(certainties: dict[str, float]) -> float:
        """Shannon entropy in bits: H = −Σ p·log₂(p)."""
        entropy = 0.0
        for p in certainties.values():
            if p > 1e-12:
                entropy -= p * math.log2(p)
        return entropy
