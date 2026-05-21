"""
ClinicalSafetyGate — Stage 5 of the reasoning pipeline.

Evaluates 3 invariants and 5 safety gates. Applies escalation-only caps
to the triage recommendation when safety conditions are triggered.
Stub for Phase 2.
"""

from __future__ import annotations


class ClinicalSafetyGate:
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("ClinicalSafetyGate is implemented in Phase 2.")
