"""
graph_nodes — Typed node system for the diagnostic reasoning graph.

Every entity in the reasoning process is represented as a typed node.
Nodes are immutable once created (frozen dataclass) and identified by a
deterministic string ID that is stable across graph rebuilds for the same
clinical case.

Node taxonomy
-------------
  HYPOTHESIS        — Disease hypothesis (one per disease, six total)
  STAGE             — Pipeline reasoning stage (one per trajectory snapshot)
  CONTRADICTION     — Active cross-disease contradiction event or cluster
  ESCALATION        — Terminal triage decision node
  INSTABILITY       — Instability detection event (oscillatory trajectory)
  CERTAINTY_ANCHOR  — Certainty distribution state at a given stage

Node lifecycle
--------------
Nodes begin as INACTIVE (no supporting evidence) and transition to ACTIVE
when the relevant reasoning stage fires. Activation strength reflects the
evidential weight or certainty at that node.

Node IDs
--------
Deterministic string identifiers for stable cross-snapshot reference:
  Hypothesis:       "hyp:{disease}"
  Stage:            "stage:{n}:{stage_name}"
  Contradiction:    "contra:{source}:{target}" or "contra:aggregate"
  Escalation:       "esc:{recommendation}"
  Instability:      "instab:{stage}"
  Certainty anchor: "cert:{stage}"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Six-disease panel (canonical order) ───────────────────────────────────────

KNOWN_DISEASES: tuple[str, ...] = (
    "psoriasis",
    "seborrheic_dermatitis",
    "lichen_planus",
    "pityriasis_rosea",
    "chronic_dermatitis",
    "pityriasis_rubra_pilaris",
)

DISEASE_INDEX: dict[str, int] = {d: i for i, d in enumerate(KNOWN_DISEASES)}


# ── Node types ────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    HYPOTHESIS       = "hypothesis"
    STAGE            = "stage"
    CONTRADICTION    = "contradiction"
    ESCALATION       = "escalation"
    INSTABILITY      = "instability"
    CERTAINTY_ANCHOR = "certainty_anchor"


# ── Activation states ─────────────────────────────────────────────────────────

class ActivationState(str, Enum):
    INACTIVE  = "inactive"   # node exists but has no active evidence
    PARTIAL   = "partial"    # some evidence, below threshold
    ACTIVE    = "active"     # evidence above threshold
    DOMINANT  = "dominant"   # node is the leading hypothesis
    TRIGGERED = "triggered"  # escalation or contradiction node fired
    SUPPRESSED = "suppressed" # certainty or hypothesis suppressed by contradiction


# ── Base node ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphNode:
    """
    Base typed graph node.

    All domain-specific node types inherit from or are aliases of this class.
    The 'meta' field holds node-type-specific structured attributes.
    """

    node_id:          str
    node_type:        NodeType
    label:            str
    activation:       ActivationState
    activation_strength: float           # [0, 1] — evidential weight or certainty
    created_at_stage: int                # pipeline stage that created this node (-1 = static)
    meta:             dict[str, Any]     # node-type-specific structured attributes

    @property
    def is_active(self) -> bool:
        return self.activation not in (ActivationState.INACTIVE, ActivationState.SUPPRESSED)

    @property
    def is_hypothesis(self) -> bool:
        return self.node_type == NodeType.HYPOTHESIS

    @property
    def is_dominant(self) -> bool:
        return self.activation == ActivationState.DOMINANT

    def __str__(self) -> str:
        return (
            f"[{self.node_type.value}] {self.label} "
            f"({self.activation.value}, strength={self.activation_strength:.3f})"
        )


# ── Node factory helpers ───────────────────────────────────────────────────────

def make_hypothesis_node(
    disease: str,
    certainty: float,
    is_leading: bool,
    is_suppressed: bool = False,
    stage: int = -1,
) -> GraphNode:
    """
    Create a hypothesis node for one of the six disease classes.

    Parameters
    ----------
    disease:
        Disease name — must be one of KNOWN_DISEASES.
    certainty:
        Normalised certainty [0, 1].
    is_leading:
        True if this disease is the current leading hypothesis.
    is_suppressed:
        True if contradiction has dampened this hypothesis's certainty.
    stage:
        Pipeline stage at which this node state was captured.
    """
    if is_suppressed:
        activation = ActivationState.SUPPRESSED
    elif is_leading:
        activation = ActivationState.DOMINANT
    elif certainty >= 0.10:
        activation = ActivationState.ACTIVE
    elif certainty >= 0.02:
        activation = ActivationState.PARTIAL
    else:
        activation = ActivationState.INACTIVE

    return GraphNode(
        node_id=f"hyp:{disease}",
        node_type=NodeType.HYPOTHESIS,
        label=disease.replace("_", " ").title(),
        activation=activation,
        activation_strength=certainty,
        created_at_stage=stage,
        meta={
            "disease": disease,
            "disease_index": DISEASE_INDEX.get(disease, -1),
            "is_leading": is_leading,
            "is_suppressed": is_suppressed,
        },
    )


def make_stage_node(
    stage: int,
    stage_name: str,
    fsm_state: str,
    certainty: float,
    contradiction_load: float,
    ambiguity_index: float,
    active_rule_count: int,
    safety_triggered: bool = False,
) -> GraphNode:
    """Create a pipeline stage node representing one reasoning step."""
    if safety_triggered:
        activation = ActivationState.TRIGGERED
    elif certainty >= 0.55:
        activation = ActivationState.DOMINANT
    elif certainty >= 0.10:
        activation = ActivationState.ACTIVE
    else:
        activation = ActivationState.PARTIAL

    return GraphNode(
        node_id=f"stage:{stage}:{stage_name}",
        node_type=NodeType.STAGE,
        label=f"Stage {stage}: {stage_name.replace('_', ' ').title()}",
        activation=activation,
        activation_strength=certainty,
        created_at_stage=stage,
        meta={
            "stage": stage,
            "stage_name": stage_name,
            "fsm_state": fsm_state,
            "certainty": certainty,
            "contradiction_load": contradiction_load,
            "ambiguity_index": ambiguity_index,
            "active_rule_count": active_rule_count,
            "safety_triggered": safety_triggered,
        },
    )


def make_contradiction_node(
    source_disease: str | None = None,
    target_disease: str | None = None,
    load: float = 0.0,
    trigger_feature: str | None = None,
    penalty_weight: float = 0.0,
    stage: int = 2,
) -> GraphNode:
    """
    Create a contradiction node.

    If source_disease and target_disease are both provided, this represents
    a specific disease-pair contradiction. Otherwise, it represents the
    aggregate contradiction state.
    """
    if source_disease and target_disease:
        node_id = f"contra:{source_disease}:{target_disease}"
        label   = f"Conflict: {source_disease.replace('_',' ')} vs {target_disease.replace('_',' ')}"
        strength = penalty_weight
    else:
        node_id = "contra:aggregate"
        label   = f"Contradiction Cluster (load={load:.3f})"
        strength = min(load / 0.40, 1.0)  # normalise against escalation ceiling

    activation = ActivationState.TRIGGERED if load > 0.0 else ActivationState.INACTIVE

    return GraphNode(
        node_id=node_id,
        node_type=NodeType.CONTRADICTION,
        label=label,
        activation=activation,
        activation_strength=strength,
        created_at_stage=stage,
        meta={
            "source_disease": source_disease,
            "target_disease": target_disease,
            "contradiction_load": load,
            "trigger_feature": trigger_feature,
            "penalty_weight": penalty_weight,
        },
    )


def make_escalation_node(
    recommendation: str,
    certainty: float,
    stage: int = 7,
) -> GraphNode:
    """Create the terminal escalation decision node."""
    biopsy_recs = {"BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"}
    activation  = (
        ActivationState.TRIGGERED if recommendation in biopsy_recs
        else ActivationState.ACTIVE
    )
    return GraphNode(
        node_id=f"esc:{recommendation}",
        node_type=NodeType.ESCALATION,
        label=recommendation.replace("_", " ").title(),
        activation=activation,
        activation_strength=certainty,
        created_at_stage=stage,
        meta={
            "recommendation": recommendation,
            "requires_biopsy": recommendation in biopsy_recs,
        },
    )


def make_instability_node(
    stage: int,
    instability_index: float,
    oscillation_count: int = 0,
) -> GraphNode:
    """Create an instability event node."""
    return GraphNode(
        node_id=f"instab:{stage}",
        node_type=NodeType.INSTABILITY,
        label=f"Instability (stage={stage}, index={instability_index:.3f})",
        activation=ActivationState.TRIGGERED if instability_index > 0 else ActivationState.INACTIVE,
        activation_strength=instability_index,
        created_at_stage=stage,
        meta={
            "stage": stage,
            "instability_index": instability_index,
            "oscillation_count": oscillation_count,
        },
    )
