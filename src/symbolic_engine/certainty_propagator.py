"""
HypothesisCertaintyPropagator — Stage 5 of the reasoning pipeline.

Applies softmax normalisation over penalised evidence scores to produce
a certainty distribution. Computes certainty_gap and ambiguity_index.
Stub for Phase 2.
"""

from __future__ import annotations


class HypothesisCertaintyPropagator:
    def propagate(self, *args, **kwargs):
        raise NotImplementedError("HypothesisCertaintyPropagator is implemented in Phase 2.")
