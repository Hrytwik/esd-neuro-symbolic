"""
Certainty-Aware Symbolic Reasoning Engine — Phase 2 reasoning subsystems.

This package implements the core clinical inference pipeline:
Stage 0 → ClinicalGradingModule         (ordinal→fuzzy conversion)
Stage 1 → DiagnosticEvidenceEvaluator   (Tier A/B rule activation)
Stage 2 → DiagnosticEvidenceEvaluator   (continued reinforcement)
Stage 3 → DiagnosticConflictAnalyzer    (contradiction detection)
Stage 4 → DiagnosticEvidenceEvaluator   (Tier D discriminators)
Stage 5 → HypothesisCertaintyPropagator + ClinicalSafetyGate
Stage 6 → ClinicalEscalationEngine + DiagnosticNarrativeGenerator

Supporting subsystems (Novelty Layer):
  DiagnosticTrajectoryMemory          — replayable reasoning trace
  DifferentialCompetitionEngine       — inter-hypothesis suppression
  EvidenceSufficiencyAnalyzer         — coverage and diversity analysis
  DiagnosticInstabilityMonitor        — certainty volatility tracking
  LightweightCounterfactualReasoner   — feature-sensitivity analysis
"""
