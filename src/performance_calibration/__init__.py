"""
src/performance_calibration — Performance Calibration + Reasoning Optimisation.

This package diagnoses and addresses the gap between current performance
(Model B ≈ 80%, Model C ≈ 82%) and target performance (Model B ≥ 86%,
Model C 88–91%) while preserving:
  · Escalation safety — never suppress clinically warranted biopsy requests
  · Contradiction sensitivity — contradiction signals must remain informative
  · Interpretability — classification decisions must remain traceable
  · Deterministic reasoning — symbolic pipeline output is reproducible

Module inventory
----------------
performance_diagnostics        Per-disease failure audit; certainty-collapse
                               analysis; escalation over-activation diagnosis.
failure_mode_analyzer          Root-cause analysis of Model B / C underperformance.
escalation_sensitivity_analysis  Threshold sensitivity sweep (ambiguity, certainty,
                               contradiction); safe rebalancing candidates.
rule_discrimination_analysis   Rule firing frequency, inter-disease discriminative
                               power, and rule conflict heat-map.
hypothesis_separation_analysis Inter-hypothesis certainty gap analysis; confusion
                               zone separation scoring.
baseline_calibration           Multi-algorithm stratified repeated CV for Model B/C;
                               optimal hyperparameter discovery.
"""

from src.performance_calibration.performance_diagnostics import (
    PerformanceDiagnostics,
    DiagnosticReport,
    DiseaseFailureProfile,
    CertaintyCollapseProfile,
)
from src.performance_calibration.failure_mode_analyzer import (
    FailureModeAnalyzer,
    FailureModeReport,
    FailurePattern,
)
from src.performance_calibration.escalation_sensitivity_analysis import (
    EscalationSensitivityAnalyzer,
    SensitivityReport,
    ThresholdSweepResult,
)
from src.performance_calibration.rule_discrimination_analysis import (
    RuleDiscriminationAnalyzer,
    RuleDiscriminationReport,
    RuleProfile,
)
from src.performance_calibration.hypothesis_separation_analysis import (
    HypothesisSeparationAnalyzer,
    SeparationReport,
    ConfusionZoneProfile,
)
from src.performance_calibration.baseline_calibration import (
    BaselineCalibrator,
    CalibrationResult,
    AlgorithmTrialResult,
)

__all__ = [
    "PerformanceDiagnostics",
    "DiagnosticReport",
    "DiseaseFailureProfile",
    "CertaintyCollapseProfile",
    "FailureModeAnalyzer",
    "FailureModeReport",
    "FailurePattern",
    "EscalationSensitivityAnalyzer",
    "SensitivityReport",
    "ThresholdSweepResult",
    "RuleDiscriminationAnalyzer",
    "RuleDiscriminationReport",
    "RuleProfile",
    "HypothesisSeparationAnalyzer",
    "SeparationReport",
    "ConfusionZoneProfile",
    "BaselineCalibrator",
    "CalibrationResult",
    "AlgorithmTrialResult",
]
