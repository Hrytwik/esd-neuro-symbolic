"""
src/backend_refinement/__init__.py
===================================
CASDRE — Backend Refinement Package
Certainty-Aware Symbolic Dermatological Reasoning Engine

Exports all public symbols from the eleven backend-refinement subsystems:
  1. model_c_optimizer               — Model-C configuration search and optimisation
  2. symbolic_recovery_refinement    — Seven-mechanism symbolic recovery taxonomy
  3. disease_discrimination_refinement — Pairwise separation and confusion analysis
  4. escalation_behavior_refinement  — Selective biopsy-escalation behaviour audits
  5. contradiction_competition_refinement — Localised contradiction propagation
  6. trajectory_realism_refinement   — Certainty-trajectory smoothness and realism
  7. rare_disease_refinement         — Imbalanced-class stabilisation (PRP focus)
  8. symbolic_rule_refinement_v2     — Sharpened symbolic rule library (16 rules)
  9. certainty_behavior_refinement   — Calibration, ECE/MCE, entropy separation
 10. publication_evaluation_suite    — Publication-grade evaluation tables
 11. backend_refinement_report       — Holistic refinement phase report
"""

# ---------------------------------------------------------------------------
# 1. Model-C Optimisation
# ---------------------------------------------------------------------------
from .model_c_optimizer import (
    DiseasePerformance,
    ModelCConfiguration,
    ModelBBaseline,
    ModelCOptimizationReport,
    build_model_c_features,
    ModelCOptimizer,
)

# ---------------------------------------------------------------------------
# 2. Symbolic Recovery Refinement
# ---------------------------------------------------------------------------
from .symbolic_recovery_refinement import (
    RecoveryDifficultyTier,
    RecoveryMechanism,
    RecoveryOutcome,
    RecoveryCase,
    RecoveryTaxonomyEntry,
    DiseaseRecoveryProfile,
    RecoveryStrengthReport,
    SymbolicRecoveryRefiner,
)

# ---------------------------------------------------------------------------
# 3. Disease Discrimination Refinement
# ---------------------------------------------------------------------------
from .disease_discrimination_refinement import (
    SeparabilityTier,
    PairwiseSeparationScore,
    DiseaseConfusionProfile,
    SymbolicSeparabilityMatrix,
    DiscriminationStrengthProfile,
    DiseaseDiscriminationReport,
    DiseaseDiscriminationRefiner,
)

# ---------------------------------------------------------------------------
# 4. Escalation Behaviour Refinement
# ---------------------------------------------------------------------------
from .escalation_behavior_refinement import (
    EscalationDecision,
    StabilisationSafety,
    EscalationTrigger,
    EscalationDecisionRecord,
    SelectivityCurvePoint,
    StabilisationPrevalenceReport,
    DiseaseEscalationBehavior,
    EscalationBehaviorReport,
    EscalationBehaviorRefiner,
)

# ---------------------------------------------------------------------------
# 5. Contradiction & Competition Refinement
# ---------------------------------------------------------------------------
from .contradiction_competition_refinement import (
    ContradictionScope,
    CompetitionOutcome,
    ContradictionPropagationProfile,
    CompetitionAnalysis,
    DiseasePropagationTendency,
    ContradictionCompetitionReport,
    ContradictionCompetitionRefiner,
)

# ---------------------------------------------------------------------------
# 6. Trajectory Realism Refinement
# ---------------------------------------------------------------------------
from .trajectory_realism_refinement import (
    TrajectoryQuality,
    SmoothnessGrade,
    TrajectoryQualityRecord,
    SmoothnessProfile,
    ConvergenceRealismProfile,
    DiseaseTrajectoryRealismProfile,
    TrajectoryRealismReport,
    TrajectoryRealismRefiner,
)

# ---------------------------------------------------------------------------
# 7. Rare-Disease Refinement
# ---------------------------------------------------------------------------
from .rare_disease_refinement import (
    ImbalanceSeverity,
    RareDiseasePerformance,
    SymbolicStabilisationAssistance,
    TrajectoryAssistedDiscrimination,
    RareDiseaseRefinementReport,
    RareDiseaseRefiner,
)

# ---------------------------------------------------------------------------
# 8. Symbolic Rule Refinement v2
# ---------------------------------------------------------------------------
from .symbolic_rule_refinement_v2 import (
    RuleStrength,
    RuleDirection,
    SymbolicRule,
    RulePerformanceMetrics,
    DiseaseRuleSet,
    RuleRefinementReport,
    SymbolicRuleRefinerV2,
)

# ---------------------------------------------------------------------------
# 9. Certainty Behaviour Refinement
# ---------------------------------------------------------------------------
from .certainty_behavior_refinement import (
    CalibrationStatus,
    AmbiguityRealism,
    CertaintyCalibrationBin,
    CalibrationCurve,
    EntropyCalibrationProfile,
    StabilisationThresholdProfile,
    CertaintyBehaviorReport,
    CertaintyBehaviorRefiner,
)

# ---------------------------------------------------------------------------
# 10. Publication Evaluation Suite
# ---------------------------------------------------------------------------
from .publication_evaluation_suite import (
    DiseaseMetricRow,
    ModelComparisonTable,
    SymbolicRecoveryTable,
    EscalationAnalysisTable,
    ContradictionAnalysisTable,
    StabilisationAnalysisTable,
    BiopsyReductionTable,
    TrajectoryQualityTable,
    PublicationEvaluationReport,
    PublicationEvaluationSuite,
)

# ---------------------------------------------------------------------------
# 11. Backend Refinement Report
# ---------------------------------------------------------------------------
from .backend_refinement_report import (
    RefinementPhase,
    FrontendTransitionDecision,
    ModelProgressEntry,
    SubsystemRefinementEntry,
    ProgressionTimeline,
    BackendRefinementReport,
    BackendRefinementReporter,
)

__all__ = [
    # model_c_optimizer
    "DiseasePerformance",
    "ModelCConfiguration",
    "ModelBBaseline",
    "ModelCOptimizationReport",
    "build_model_c_features",
    "ModelCOptimizer",
    # symbolic_recovery_refinement
    "RecoveryDifficultyTier",
    "RecoveryMechanism",
    "RecoveryOutcome",
    "RecoveryCase",
    "RecoveryTaxonomyEntry",
    "DiseaseRecoveryProfile",
    "RecoveryStrengthReport",
    "SymbolicRecoveryRefiner",
    # disease_discrimination_refinement
    "SeparabilityTier",
    "PairwiseSeparationScore",
    "DiseaseConfusionProfile",
    "SymbolicSeparabilityMatrix",
    "DiscriminationStrengthProfile",
    "DiseaseDiscriminationReport",
    "DiseaseDiscriminationRefiner",
    # escalation_behavior_refinement
    "EscalationDecision",
    "StabilisationSafety",
    "EscalationTrigger",
    "EscalationDecisionRecord",
    "SelectivityCurvePoint",
    "StabilisationPrevalenceReport",
    "DiseaseEscalationBehavior",
    "EscalationBehaviorReport",
    "EscalationBehaviorRefiner",
    # contradiction_competition_refinement
    "ContradictionScope",
    "CompetitionOutcome",
    "ContradictionPropagationProfile",
    "CompetitionAnalysis",
    "DiseasePropagationTendency",
    "ContradictionCompetitionReport",
    "ContradictionCompetitionRefiner",
    # trajectory_realism_refinement
    "TrajectoryQuality",
    "SmoothnessGrade",
    "TrajectoryQualityRecord",
    "SmoothnessProfile",
    "ConvergenceRealismProfile",
    "DiseaseTrajectoryRealismProfile",
    "TrajectoryRealismReport",
    "TrajectoryRealismRefiner",
    # rare_disease_refinement
    "ImbalanceSeverity",
    "RareDiseasePerformance",
    "SymbolicStabilisationAssistance",
    "TrajectoryAssistedDiscrimination",
    "RareDiseaseRefinementReport",
    "RareDiseaseRefiner",
    # symbolic_rule_refinement_v2
    "RuleStrength",
    "RuleDirection",
    "SymbolicRule",
    "RulePerformanceMetrics",
    "DiseaseRuleSet",
    "RuleRefinementReport",
    "SymbolicRuleRefinerV2",
    # certainty_behavior_refinement
    "CalibrationStatus",
    "AmbiguityRealism",
    "CertaintyCalibrationBin",
    "CalibrationCurve",
    "EntropyCalibrationProfile",
    "StabilisationThresholdProfile",
    "CertaintyBehaviorReport",
    "CertaintyBehaviorRefiner",
    # publication_evaluation_suite
    "DiseaseMetricRow",
    "ModelComparisonTable",
    "SymbolicRecoveryTable",
    "EscalationAnalysisTable",
    "ContradictionAnalysisTable",
    "StabilisationAnalysisTable",
    "BiopsyReductionTable",
    "TrajectoryQualityTable",
    "PublicationEvaluationReport",
    "PublicationEvaluationSuite",
    # backend_refinement_report
    "RefinementPhase",
    "FrontendTransitionDecision",
    "ModelProgressEntry",
    "SubsystemRefinementEntry",
    "ProgressionTimeline",
    "BackendRefinementReport",
    "BackendRefinementReporter",
]
