"""
src/performance_calibration — Performance Calibration + Reasoning Optimisation.

This package diagnoses and addresses the gap between current performance
(Model B ~= 80%, Model C ~= 82%) and target performance (Model B >= 86%,
Model C 88-91%) while preserving:
  * Escalation safety -- never suppress clinically warranted biopsy requests
  * Contradiction sensitivity -- contradiction signals must remain informative
  * Interpretability -- classification decisions must remain traceable
  * Deterministic reasoning -- symbolic pipeline output is reproducible

Module inventory (Phase 5 Step 1 — Diagnostics)
------------------------------------------------
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

Module inventory (Phase 5 Step 2 — Recalibration + Optimisation)
-----------------------------------------------------------------
threshold_recalibration        Post-processing recalibration of escalation thresholds;
                               ambiguity/certainty sweep with zero-violation constraint.
disease_signature_refinement   Discriminative rule-signature analysis; per-disease
                               feature-weight recommendations.
competition_sharpening         Inter-hypothesis competition signal enrichment;
                               gap amplification and entropy contrast.
contradiction_rebalancing      Severity-tiered contradiction signal calibration;
                               NONE/MINOR/MODERATE/CRITICAL taxonomy.
certainty_rebalancing          Monotone certainty normalisation; convergence-weighted
                               composite; context-aware sufficiency threshold.
symbolic_signal_enrichment_v2  Expanded symbolic signal set (22 -> 40 signals);
                               trajectory dynamics, competition topology, clinical context.
advanced_baseline_calibration  Extended cross-validation with per-disease analysis;
                               XGBoost/RF/LightGBM grid search with class balancing.
symbolic_recovery_analysis     Model B failure -> Model C recovery attribution;
                               7-mechanism taxonomy for symbolic contribution evidence.
biopsy_reduction_analysis      Disease-wise biopsy necessity analysis; safe-triage
                               conditions; reduction potential under recalibrated thresholds.
final_calibration_report       Publication-grade calibration summary; pre/post comparisons;
                               clinical safety audit; JSON export.
"""

# ── Phase 5 Step 1 — Diagnostics ─────────────────────────────────────────────

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

# ── Phase 5 Step 2 — Recalibration + Optimisation ────────────────────────────

from src.performance_calibration.threshold_recalibration import (
    ThresholdRecalibrator,
    ThresholdConfig,
    CalibrationReport as ThresholdCalibrationReport,
    ThresholdEvaluationResult,
    CONTRADICTION_CEILING,
)
from src.performance_calibration.disease_signature_refinement import (
    DiseaseSignatureRefiner,
    DiseaseSignatureReport,
    RuleActivationProfile,
)
from src.performance_calibration.competition_sharpening import (
    CompetitionSharpener,
    CompetitionEnrichedSignals,
)
from src.performance_calibration.contradiction_rebalancing import (
    ContradictionRebalancer,
    ContradictionRebalancingReport,
    ContradictionTier,
    ContradictionEnrichedSignals,
)
from src.performance_calibration.certainty_rebalancing import (
    CertaintyRebalancer,
    CertaintyRebalancingReport,
    CertaintyEnrichedSignals,
    CertaintyDistributionProfile,
)
from src.performance_calibration.symbolic_signal_enrichment_v2 import (
    SymbolicSignalEnricherV2,
    EnrichedSignalSet,
    EnrichmentReport,
)
from src.performance_calibration.advanced_baseline_calibration import (
    AdvancedBaselineCalibrator,
    AdvancedCalibrationResult,
    AdvancedTrialResult,
)
from src.performance_calibration.symbolic_recovery_analysis import (
    SymbolicRecoveryAnalyzer,
    RecoveryReport,
    RecoveryRecord,
    RecoveryMechanism,
    MechanismStats,
)
from src.performance_calibration.biopsy_reduction_analysis import (
    BiopsyReductionAnalyzer,
    BiopsyReductionReport,
    DiseaseBiopsyProfile,
    SafeTriageCondition,
)
from src.performance_calibration.final_calibration_report import (
    FinalCalibrationReporter,
    FinalCalibrationReport,
    PerformanceComparison,
    EscalationSummary,
    CertaintyImprovementSummary,
    SymbolicLiftSummary,
    ContradictionAuditSummary,
    DiseaseImprovementRow,
    TrajectoryStabilizationSummary,
)


__all__ = [
    # Phase 5 Step 1
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
    # Phase 5 Step 2
    "ThresholdRecalibrator",
    "ThresholdConfig",
    "ThresholdCalibrationReport",
    "ThresholdEvaluationResult",
    "CONTRADICTION_CEILING",
    "DiseaseSignatureRefiner",
    "DiseaseSignatureReport",
    "RuleActivationProfile",
    "CompetitionSharpener",
    "CompetitionEnrichedSignals",
    "ContradictionRebalancer",
    "ContradictionRebalancingReport",
    "ContradictionTier",
    "ContradictionEnrichedSignals",
    "CertaintyRebalancer",
    "CertaintyRebalancingReport",
    "CertaintyEnrichedSignals",
    "CertaintyDistributionProfile",
    "SymbolicSignalEnricherV2",
    "EnrichedSignalSet",
    "EnrichmentReport",
    "AdvancedBaselineCalibrator",
    "AdvancedCalibrationResult",
    "AdvancedTrialResult",
    "SymbolicRecoveryAnalyzer",
    "RecoveryReport",
    "RecoveryRecord",
    "RecoveryMechanism",
    "MechanismStats",
    "BiopsyReductionAnalyzer",
    "BiopsyReductionReport",
    "DiseaseBiopsyProfile",
    "SafeTriageCondition",
    "FinalCalibrationReporter",
    "FinalCalibrationReport",
    "PerformanceComparison",
    "EscalationSummary",
    "CertaintyImprovementSummary",
    "SymbolicLiftSummary",
    "ContradictionAuditSummary",
    "DiseaseImprovementRow",
    "TrajectoryStabilizationSummary",
]
