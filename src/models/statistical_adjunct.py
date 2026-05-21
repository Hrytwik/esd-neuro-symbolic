"""
StatisticalRefinementAdjunct — hybrid extension for Model C.

Provides optional statistical post-processing layer that refines
the symbolic certainty scores using XGBoost-derived probability
estimates. Used only in the hybrid ablation configuration.
Stub for Phase 3.
"""

from __future__ import annotations


class StatisticalRefinementAdjunct:
    def refine(self, *args, **kwargs):
        raise NotImplementedError("StatisticalRefinementAdjunct is implemented in Phase 3.")
