"""
src/validation — Behavioral Validation + Clinical Reasoning Calibration layer.

Provides a structured framework for evaluating whether the symbolic diagnostic
reasoning engine behaves in a clinically believable and diagnostically coherent
manner.

Validators are stateless and accept PipelineResult / DiagnosticTrajectory
instances as inputs — they contain no reasoning logic of their own.

Available validators
--------------------
  BehavioralValidator     — overall reasoning coherence coordinator
  TrajectoryValidator     — certainty evolution smoothness and stability
  EscalationValidator     — biopsy escalation appropriateness
  ContradictionValidator  — contradiction propagation realism
  CertaintyValidator      — certainty stabilisation and entropy behavior
  NarrativeValidator      — clinical plausibility of reasoning narratives
  StabilityValidator      — deterministic output and replay consistency
  SyntheticCaseExpander   — expanded clinical case library (30–50 scenarios)
"""
