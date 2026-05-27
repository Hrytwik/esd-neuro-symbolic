"""
src/backend_stabilization/__init__.py
=======================================
Backend maturation and reasoning stabilisation package for the CASDRE
clinical inference pipeline.

Modules
-------
1.  discriminative_optimization        — Model B/C discrimination analysis
2.  disease_separation_refinement      — Deep per-disease separation analysis
3.  symbolic_recovery_optimizer        — Contradiction/trajectory/competition recovery
4.  escalation_selectivity_optimizer   — Selective escalation threshold optimisation
5.  trajectory_stabilization           — Convergence-realism characterisation
6.  contradiction_localization         — Localized contradiction propagation
7.  reasoning_contract_finalizer       — Frozen reasoning output schema
8.  replay_schema_finalizer            — Frozen replay JSON schema
9.  graph_contract_finalizer           — Frozen graph serialization schema
10. backend_maturity_report            — Comprehensive maturity audit
"""

# ── Module 1: Discriminative Optimisation ────────────────────────────────────
from .discriminative_optimization import (
    DiscriminationTier,
    FeaturePairSeparation,
    DiseaseDiscriminativeProfile,
    OptimizationRecommendation,
    DiscriminativeOptimizationReport,
    DiscriminativeOptimizer,
)

# ── Module 2: Disease Separation Refinement ───────────────────────────────────
from .disease_separation_refinement import (
    BiopsyNecessityTier,
    SymbolicRecoverabilityTier,
    AmbiguityTier,
    DifferentialCompetitor,
    DiseaseBiopsyProfile,
    DiseaseStabilisationProfile,
    SymbolicRecoveryHeatmapCell,
    DiseaseSeparationReport,
    DiseaseSeparationRefiner,
)

# ── Module 3: Symbolic Recovery Optimiser ─────────────────────────────────────
from .symbolic_recovery_optimizer import (
    RecoveryMechanism,
    RecoveryOpportunityTier,
    RecoveryCandidate,
    MechanismBreakdown,
    ContradictionRecoveryProfile,
    TrajectoryRecoveryProfile,
    CompetitionRecoveryProfile,
    SymbolicRecoveryOptimizationReport,
    SymbolicRecoveryOptimizer,
)

# ── Module 4: Escalation Selectivity Optimiser ───────────────────────────────
from .escalation_selectivity_optimizer import (
    SelectivityTier,
    EscalationJustification,
    ThresholdCandidate,
    SelectivityCurvePoint,
    EscalationAuditRow,
    DiseaseEscalationProfile,
    StabilisationPrevalenceReport,
    EscalationSelectivityReport,
    EscalationSelectivityOptimizer,
)

# ── Module 5: Trajectory Stabilisation ───────────────────────────────────────
from .trajectory_stabilization import (
    ConvergenceTier,
    OscillationSeverity,
    CertaintyEvolutionPattern,
    TrajectorySnapshot,
    CaseTrajectory,
    DiseaseTrajectoryProfile,
    ConvergenceRealism,
    TrajectoryStabilisationReport,
    TrajectoryStabilizer,
)

# ── Module 6: Contradiction Localisation ─────────────────────────────────────
from .contradiction_localization import (
    ContradictionSeverity,
    PropagationDepth,
    SignalContradictionProfile,
    DiseaseContradictionAffinity,
    ContradictionCluster,
    CeilingAudit,
    ContradictionLocalizationReport,
    ContradictionLocalizer,
)

# ── Module 7: Reasoning Contract Finalizer ───────────────────────────────────
from .reasoning_contract_finalizer import (
    REASONING_CONTRACT_VERSION,
    REASONING_CONTRACT_FROZEN,
    DiagnosticStateCode,
    ContradictionTierCode,
    RecoveryMechanismCode,
    ContractValidationResult,
    ContractDriftReport,
    ReasoningContractFinalizer,
)

# ── Module 8: Replay Schema Finalizer ────────────────────────────────────────
from .replay_schema_finalizer import (
    REPLAY_SCHEMA_VERSION,
    REPLAY_SCHEMA_FROZEN,
    ReplayEventType,
    ReplayValidationResult,
    ReplaySchemaAuditReport,
    ReplaySchemaFinalizer,
)

# ── Module 9: Graph Contract Finalizer ───────────────────────────────────────
from .graph_contract_finalizer import (
    GRAPH_CONTRACT_VERSION,
    GRAPH_CONTRACT_FROZEN,
    GraphNodeType,
    GraphEdgeType,
    GraphValidationResult,
    GraphContractAuditReport,
    GraphContractFinalizer,
)

# ── Module 10: Backend Maturity Report ───────────────────────────────────────
from .backend_maturity_report import (
    MaturityLevel,
    FrontendReadiness,
    SubsystemStatus,
    SubsystemAuditEntry,
    ModelPerformanceSummary,
    SafetyConstraintAudit,
    SchemaReadinessAudit,
    BackendMaturityReport,
    BackendMaturityReporter,
)

__all__ = [
    # Module 1
    "DiscriminationTier",
    "FeaturePairSeparation",
    "DiseaseDiscriminativeProfile",
    "OptimizationRecommendation",
    "DiscriminativeOptimizationReport",
    "DiscriminativeOptimizer",
    # Module 2
    "BiopsyNecessityTier",
    "SymbolicRecoverabilityTier",
    "AmbiguityTier",
    "DifferentialCompetitor",
    "DiseaseBiopsyProfile",
    "DiseaseStabilisationProfile",
    "SymbolicRecoveryHeatmapCell",
    "DiseaseSeparationReport",
    "DiseaseSeparationRefiner",
    # Module 3
    "RecoveryMechanism",
    "RecoveryOpportunityTier",
    "RecoveryCandidate",
    "MechanismBreakdown",
    "ContradictionRecoveryProfile",
    "TrajectoryRecoveryProfile",
    "CompetitionRecoveryProfile",
    "SymbolicRecoveryOptimizationReport",
    "SymbolicRecoveryOptimizer",
    # Module 4
    "SelectivityTier",
    "EscalationJustification",
    "ThresholdCandidate",
    "SelectivityCurvePoint",
    "EscalationAuditRow",
    "DiseaseEscalationProfile",
    "StabilisationPrevalenceReport",
    "EscalationSelectivityReport",
    "EscalationSelectivityOptimizer",
    # Module 5
    "ConvergenceTier",
    "OscillationSeverity",
    "CertaintyEvolutionPattern",
    "TrajectorySnapshot",
    "CaseTrajectory",
    "DiseaseTrajectoryProfile",
    "ConvergenceRealism",
    "TrajectoryStabilisationReport",
    "TrajectoryStabilizer",
    # Module 6
    "ContradictionSeverity",
    "PropagationDepth",
    "SignalContradictionProfile",
    "DiseaseContradictionAffinity",
    "ContradictionCluster",
    "CeilingAudit",
    "ContradictionLocalizationReport",
    "ContradictionLocalizer",
    # Module 7
    "REASONING_CONTRACT_VERSION",
    "REASONING_CONTRACT_FROZEN",
    "DiagnosticStateCode",
    "ContradictionTierCode",
    "RecoveryMechanismCode",
    "ContractValidationResult",
    "ContractDriftReport",
    "ReasoningContractFinalizer",
    # Module 8
    "REPLAY_SCHEMA_VERSION",
    "REPLAY_SCHEMA_FROZEN",
    "ReplayEventType",
    "ReplayValidationResult",
    "ReplaySchemaAuditReport",
    "ReplaySchemaFinalizer",
    # Module 9
    "GRAPH_CONTRACT_VERSION",
    "GRAPH_CONTRACT_FROZEN",
    "GraphNodeType",
    "GraphEdgeType",
    "GraphValidationResult",
    "GraphContractAuditReport",
    "GraphContractFinalizer",
    # Module 10
    "MaturityLevel",
    "FrontendReadiness",
    "SubsystemStatus",
    "SubsystemAuditEntry",
    "ModelPerformanceSummary",
    "SafetyConstraintAudit",
    "SchemaReadinessAudit",
    "BackendMaturityReport",
    "BackendMaturityReporter",
]
