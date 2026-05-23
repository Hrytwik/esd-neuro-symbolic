"""
graph_edges — Typed edge system for the diagnostic reasoning graph.

Edges encode the directional relationships between reasoning nodes —
how evidence flows, where contradictions propagate, which hypothesis leads,
and how the FSM transitions connect stages.

Edge taxonomy
-------------
  REINFORCEMENT   — Evidence or rule strengthens a hypothesis
  CONTRADICTION   — Active cross-disease conflict (penalty flow)
  SUPPRESSION     — Contradiction dampens a target hypothesis's certainty
  ESCALATION      — Safety gate / threshold triggers a triage decision
  PROPAGATION     — Certainty flows from one hypothesis to downstream nodes
  TRAJECTORY      — Stage-to-stage temporal progression link
  LEADERSHIP      — Marks the current leading hypothesis at a stage
  COMPETITION     — Differential competition tension between hypotheses

Edge directionality
-------------------
  REINFORCEMENT:  evidence_node → hypothesis_node
  CONTRADICTION:  hypothesis_A → contradiction_node → hypothesis_B
  SUPPRESSION:    contradiction_node → hypothesis_node (dampening)
  ESCALATION:     stage_node → escalation_node
  PROPAGATION:    stage_node → hypothesis_node (certainty assignment)
  TRAJECTORY:     stage_node(n) → stage_node(n+1)
  LEADERSHIP:     stage_node → hypothesis_node (dominant hypothesis)
  COMPETITION:    hypothesis_A ↔ hypothesis_B (mutual tension)

Edge weights
------------
  REINFORCEMENT:  rule evidence strength [0, 1]
  CONTRADICTION:  penalty_weight (cumulative if multiple contradictions)
  SUPPRESSION:    dampening factor applied to target certainty
  ESCALATION:     trigger severity [0, 1]
  PROPAGATION:    certainty value [0, 1]
  TRAJECTORY:     certainty delta (signed — positive = improving)
  LEADERSHIP:     certainty of the leading hypothesis
  COMPETITION:    certainty_gap between the two hypotheses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Edge types ────────────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    REINFORCEMENT = "reinforcement"
    CONTRADICTION = "contradiction"
    SUPPRESSION   = "suppression"
    ESCALATION    = "escalation"
    PROPAGATION   = "propagation"
    TRAJECTORY    = "trajectory"
    LEADERSHIP    = "leadership"
    COMPETITION   = "competition"


# ── Edge direction semantics ──────────────────────────────────────────────────

class EdgeDirection(str, Enum):
    FORWARD  = "forward"    # source → target
    REVERSE  = "reverse"    # target → source (for display)
    BIDIRECT = "bidirect"   # mutual (competition, confusion zones)


# ── Graph edge ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphEdge:
    """
    Typed directed edge between two graph nodes.

    The edge carries a weight, a direction, and a structured metadata dict
    that records edge-type-specific clinical attributes.
    """

    edge_id:   str
    edge_type: EdgeType
    source_id: str
    target_id: str
    weight:    float             # type-specific weight (see module docstring)
    direction: EdgeDirection
    active:    bool              # False for dormant/unfired relationships
    stage:     int               # pipeline stage at which this edge is active
    meta:      dict[str, Any]    # edge-type-specific structured attributes

    @property
    def is_conflict(self) -> bool:
        return self.edge_type in (EdgeType.CONTRADICTION, EdgeType.SUPPRESSION)

    @property
    def is_temporal(self) -> bool:
        return self.edge_type == EdgeType.TRAJECTORY

    def __str__(self) -> str:
        arrow = "<->" if self.direction == EdgeDirection.BIDIRECT else "->"
        status = "ON" if self.active else "OFF"
        return (
            f"[{self.edge_type.value}] {self.source_id} {arrow} {self.target_id} "
            f"w={self.weight:.3f} [{status}]"
        )


# ── Edge factory helpers ───────────────────────────────────────────────────────

def _edge_id(source: str, target: str, etype: EdgeType, suffix: str = "") -> str:
    """Generate a deterministic edge ID."""
    base = f"{source}=>{target}::{etype.value}"
    return f"{base}:{suffix}" if suffix else base


def make_trajectory_edge(
    from_stage_id: str,
    to_stage_id: str,
    certainty_delta: float,
    contradiction_delta: float,
    entropy_delta: float,
    state_changed: bool,
    stage: int,
) -> GraphEdge:
    """Create a temporal trajectory edge linking consecutive stage nodes."""
    return GraphEdge(
        edge_id=_edge_id(from_stage_id, to_stage_id, EdgeType.TRAJECTORY),
        edge_type=EdgeType.TRAJECTORY,
        source_id=from_stage_id,
        target_id=to_stage_id,
        weight=certainty_delta,   # positive = certainty improved
        direction=EdgeDirection.FORWARD,
        active=True,
        stage=stage,
        meta={
            "certainty_delta":     certainty_delta,
            "contradiction_delta": contradiction_delta,
            "entropy_delta":       entropy_delta,
            "state_changed":       state_changed,
        },
    )


def make_propagation_edge(
    stage_id: str,
    hypothesis_id: str,
    certainty: float,
    stage: int,
) -> GraphEdge:
    """Create a certainty propagation edge from a stage node to a hypothesis."""
    return GraphEdge(
        edge_id=_edge_id(stage_id, hypothesis_id, EdgeType.PROPAGATION, str(stage)),
        edge_type=EdgeType.PROPAGATION,
        source_id=stage_id,
        target_id=hypothesis_id,
        weight=certainty,
        direction=EdgeDirection.FORWARD,
        active=certainty > 0.0,
        stage=stage,
        meta={"certainty": certainty, "stage": stage},
    )


def make_leadership_edge(
    stage_id: str,
    hypothesis_id: str,
    certainty: float,
    certainty_gap: float,
    stage: int,
) -> GraphEdge:
    """Create a leadership edge designating the dominant hypothesis at a stage."""
    return GraphEdge(
        edge_id=_edge_id(stage_id, hypothesis_id, EdgeType.LEADERSHIP, str(stage)),
        edge_type=EdgeType.LEADERSHIP,
        source_id=stage_id,
        target_id=hypothesis_id,
        weight=certainty,
        direction=EdgeDirection.FORWARD,
        active=True,
        stage=stage,
        meta={
            "certainty": certainty,
            "certainty_gap": certainty_gap,
            "stage": stage,
        },
    )


def make_contradiction_edge(
    source_hyp_id: str,
    target_hyp_id: str,
    load: float,
    trigger_feature: str | None = None,
    penalty_weight: float = 0.0,
    stage: int = 2,
) -> GraphEdge:
    """Create a contradiction edge between two competing hypothesis nodes."""
    return GraphEdge(
        edge_id=_edge_id(source_hyp_id, target_hyp_id, EdgeType.CONTRADICTION),
        edge_type=EdgeType.CONTRADICTION,
        source_id=source_hyp_id,
        target_id=target_hyp_id,
        weight=penalty_weight if penalty_weight > 0 else load,
        direction=EdgeDirection.FORWARD,
        active=load > 0.0,
        stage=stage,
        meta={
            "contradiction_load": load,
            "trigger_feature": trigger_feature,
            "penalty_weight": penalty_weight,
        },
    )


def make_suppression_edge(
    contradiction_id: str,
    hypothesis_id: str,
    dampening_factor: float,
    stage: int = 3,
) -> GraphEdge:
    """Create a suppression edge from a contradiction node to a dampened hypothesis."""
    return GraphEdge(
        edge_id=_edge_id(contradiction_id, hypothesis_id, EdgeType.SUPPRESSION),
        edge_type=EdgeType.SUPPRESSION,
        source_id=contradiction_id,
        target_id=hypothesis_id,
        weight=dampening_factor,
        direction=EdgeDirection.FORWARD,
        active=dampening_factor > 0.0,
        stage=stage,
        meta={"dampening_factor": dampening_factor},
    )


def make_escalation_edge(
    stage_id: str,
    escalation_id: str,
    trigger_reason: str,
    severity: float,
    stage: int = 7,
) -> GraphEdge:
    """Create an escalation edge from the terminal stage to the triage decision."""
    return GraphEdge(
        edge_id=_edge_id(stage_id, escalation_id, EdgeType.ESCALATION),
        edge_type=EdgeType.ESCALATION,
        source_id=stage_id,
        target_id=escalation_id,
        weight=severity,
        direction=EdgeDirection.FORWARD,
        active=True,
        stage=stage,
        meta={
            "trigger_reason": trigger_reason,
            "severity": severity,
        },
    )


def make_competition_edge(
    hyp_a_id: str,
    hyp_b_id: str,
    certainty_gap: float,
    stage: int,
) -> GraphEdge:
    """Create a bidirectional competition edge between two competing hypotheses."""
    return GraphEdge(
        edge_id=_edge_id(hyp_a_id, hyp_b_id, EdgeType.COMPETITION, str(stage)),
        edge_type=EdgeType.COMPETITION,
        source_id=hyp_a_id,
        target_id=hyp_b_id,
        weight=certainty_gap,
        direction=EdgeDirection.BIDIRECT,
        active=certainty_gap < 0.30,  # active competition = small gap
        stage=stage,
        meta={
            "certainty_gap": certainty_gap,
            "close_competition": certainty_gap < 0.15,
        },
    )
