"""
ClinicalGradingModule — Stage 0 of the progressive reasoning pipeline.

Converts ordinal clinical grades (0–3) to fuzzy membership values using
the mapping: 0→0.00, 1→0.33, 2→0.67, 3→1.00. Stub for Phase 2.
"""

from __future__ import annotations

# Phase 2 implementation placeholder.
# Raises NotImplementedError until Phase 2 symbolic engine is implemented.


class ClinicalGradingModule:
    def grade(self, feature_name: str, ordinal_value: int) -> float:
        raise NotImplementedError("ClinicalGradingModule is implemented in Phase 2.")
