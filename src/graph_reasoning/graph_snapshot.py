"""
graph_snapshot — Immutable point-in-time graph state capture.

A GraphSnapshot is the complete graph state frozen at the moment a specific
pipeline stage completes. Snapshots are the atomic unit of graph replay:
the replay engine steps from snapshot to snapshot, reconstructing the
reasoning state at each stage.

Snapshots are immutable once created — they represent historical reasoning
states and should not be modified. The full trajectory is a sequence of
snapshots ordered by stage index.

Snapshot metrics
----------------
Each snapshot carries quantitative clinical reasoning signals extracted
from the corresponding trajectory point:
  · certainty         — leading hypothesis certainty at this stage
  · certainty_gap     — gap between top-1 and top-2 hypotheses
  · contradiction_load — aggregate cross-disease penalty weight
  · ambiguity_index   — Shannon entropy in bits
  · leading_disease   — dominant hypothesis at this stage
  · fsm_state         — FSM diagnostic state value (string)
  · active_rule_count — number of fired diagnostic rules
  · safety_triggered  — whether any safety gate was active

Graph state
-----------
The snapshot stores the full node and edge state at this stage by holding
copies of active node IDs and edge IDs with their weights. Full node/edge
objects are not embedded (to avoid circular references and memory overhead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeState:
    """
    Lightweight record of one node's state at a specific snapshot.
    Avoids embedding full GraphNode to keep snapshots compact.
    """

    node_id:          str
    node_type:        str
    label:            str
    activation:       str       # ActivationState.value
    activation_strength: float
    meta:             dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id":            self.node_id,
            "node_type":          self.node_type,
            "label":              self.label,
            "activation":         self.activation,
            "activation_strength": self.activation_strength,
            "meta":               dict(self.meta),
        }


@dataclass(frozen=True)
class EdgeState:
    """
    Lightweight record of one edge's state at a specific snapshot.
    """

    edge_id:   str
    edge_type: str
    source_id: str
    target_id: str
    weight:    float
    active:    bool
    meta:      dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id":   self.edge_id,
            "edge_type": self.edge_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight":    self.weight,
            "active":    self.active,
            "meta":      dict(self.meta),
        }


@dataclass(frozen=True)
class SnapshotMetrics:
    """
    Quantitative clinical reasoning signals at a specific stage.
    Mirrors the ReasoningSnapshot fields from trajectory_memory.
    """

    stage:              int
    stage_name:         str
    certainty:          float
    certainty_gap:      float
    contradiction_load: float
    ambiguity_index:    float
    leading_disease:    str
    fsm_state:          str
    active_rule_count:  int
    tier_a_count:       int
    safety_triggered:   bool
    triage_so_far:      str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage":              self.stage,
            "stage_name":         self.stage_name,
            "certainty":          self.certainty,
            "certainty_gap":      self.certainty_gap,
            "contradiction_load": self.contradiction_load,
            "ambiguity_index":    self.ambiguity_index,
            "leading_disease":    self.leading_disease,
            "fsm_state":          self.fsm_state,
            "active_rule_count":  self.active_rule_count,
            "tier_a_count":       self.tier_a_count,
            "safety_triggered":   self.safety_triggered,
            "triage_so_far":      self.triage_so_far,
        }


@dataclass(frozen=True)
class GraphSnapshot:
    """
    Complete graph state frozen at one pipeline stage.

    The snapshot is the atomic unit of graph replay — stepping through
    an ordered sequence of snapshots reconstructs the full reasoning
    trajectory as a graph evolution.

    Parameters
    ----------
    case_id:
        The clinical case this snapshot belongs to.
    snapshot_index:
        Ordinal position in the trajectory (0-indexed).
    metrics:
        Quantitative reasoning signals at this stage.
    node_states:
        State of every node in the graph at this stage.
    edge_states:
        State of every edge in the graph at this stage.
    active_node_ids:
        IDs of nodes that are active at this stage (fast lookup).
    active_edge_ids:
        IDs of edges that are active at this stage.
    delta_description:
        Human-readable description of what changed at this stage.
    """

    case_id:          str
    snapshot_index:   int
    metrics:          SnapshotMetrics
    node_states:      tuple[NodeState, ...]
    edge_states:      tuple[EdgeState, ...]
    active_node_ids:  frozenset[str]
    active_edge_ids:  frozenset[str]
    delta_description: str = ""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def stage(self) -> int:
        return self.metrics.stage

    @property
    def stage_name(self) -> str:
        return self.metrics.stage_name

    @property
    def leading_disease(self) -> str:
        return self.metrics.leading_disease

    @property
    def certainty(self) -> float:
        return self.metrics.certainty

    @property
    def contradiction_load(self) -> float:
        return self.metrics.contradiction_load

    @property
    def is_contradiction_active(self) -> bool:
        return self.metrics.contradiction_load > 0.0

    @property
    def is_safety_triggered(self) -> bool:
        return self.metrics.safety_triggered

    @property
    def hypothesis_nodes(self) -> list[NodeState]:
        return [n for n in self.node_states if n.node_type == "hypothesis"]

    @property
    def stage_nodes(self) -> list[NodeState]:
        return [n for n in self.node_states if n.node_type == "stage"]

    @property
    def contradiction_nodes(self) -> list[NodeState]:
        return [n for n in self.node_states if n.node_type == "contradiction"]

    @property
    def active_hypotheses(self) -> list[NodeState]:
        return [
            n for n in self.hypothesis_nodes
            if n.node_id in self.active_node_ids
        ]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id":          self.case_id,
            "snapshot_index":   self.snapshot_index,
            "metrics":          self.metrics.to_dict(),
            "node_states":      [n.to_dict() for n in self.node_states],
            "edge_states":      [e.to_dict() for e in self.edge_states],
            "active_node_ids":  sorted(self.active_node_ids),
            "active_edge_ids":  sorted(self.active_edge_ids),
            "delta_description": self.delta_description,
        }

    def __str__(self) -> str:
        return (
            f"GraphSnapshot[{self.snapshot_index}] "
            f"stage={self.stage}:{self.stage_name} "
            f"leading={self.leading_disease} "
            f"cert={self.certainty:.3f} "
            f"load={self.contradiction_load:.3f}"
        )
