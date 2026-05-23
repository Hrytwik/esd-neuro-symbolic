"""
GraphSerializer — low-level dict serialization of graph components.

Provides atomic serialization functions for every graph component type.
These are the building blocks used by GraphExporter to produce structured
output in multiple target formats.

All serialization functions produce plain Python dicts suitable for
JSON encoding via the standard library json module. No external
serialization dependencies are required.

Serialization design
--------------------
  · All float values are rounded to 6 decimal places (avoids float noise)
  · All enum values are serialized as their .value strings
  · All frozensets are converted to sorted lists (deterministic JSON output)
  · None values are preserved (not omitted) for schema completeness
  · Timestamps are omitted (graph layer is stateless w.r.t. time)

Usage
-----
  from src.graph_reasoning.graph_serializer import GraphSerializer

  node_dict  = GraphSerializer.node(my_node)
  edge_dict  = GraphSerializer.edge(my_edge)
  snap_dict  = GraphSerializer.snapshot(my_snapshot)
  graph_dict = GraphSerializer.graph(my_reasoning_graph)
"""

from __future__ import annotations

import math
from typing import Any

from src.graph_reasoning.graph_nodes import GraphNode
from src.graph_reasoning.graph_edges import GraphEdge
from src.graph_reasoning.graph_snapshot import GraphSnapshot, SnapshotMetrics
from src.graph_reasoning.reasoning_graph import ReasoningGraph
from src.graph_reasoning.trajectory_graph import TrajectoryGraph, TrajectoryDelta
from src.graph_reasoning.certainty_graph import CertaintyGraph, CertaintyPoint
from src.graph_reasoning.contradiction_graph import (
    ContradictionGraph, PairTension, ContradictionCluster,
)
from src.graph_reasoning.replay_engine import ReplayEvent, ReplayResult


def _r(v: float | None, places: int = 6) -> float | None:
    """Round float to `places` decimal places. Preserves None."""
    if v is None:
        return None
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return round(v, places)


class GraphSerializer:
    """
    Collection of static serialization methods for all graph component types.

    All methods accept the typed graph object and return a plain dict.
    The dict is guaranteed to be JSON-serializable via the standard library.
    """

    # ── Node serialization ────────────────────────────────────────────────────

    @staticmethod
    def node(node: GraphNode) -> dict[str, Any]:
        """Serialize a GraphNode to a plain dict."""
        return {
            "node_id":            node.node_id,
            "node_type":          node.node_type.value,
            "label":              node.label,
            "activation":         node.activation.value,
            "activation_strength": _r(node.activation_strength),
            "created_at_stage":   node.created_at_stage,
            "is_active":          node.is_active,
            "is_dominant":        node.is_dominant,
            "meta":               _serialize_meta(node.meta),
        }

    # ── Edge serialization ────────────────────────────────────────────────────

    @staticmethod
    def edge(edge: GraphEdge) -> dict[str, Any]:
        """Serialize a GraphEdge to a plain dict."""
        return {
            "edge_id":   edge.edge_id,
            "edge_type": edge.edge_type.value,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "weight":    _r(edge.weight),
            "direction": edge.direction.value,
            "active":    edge.active,
            "stage":     edge.stage,
            "meta":      _serialize_meta(edge.meta),
        }

    # ── Snapshot serialization ────────────────────────────────────────────────

    @staticmethod
    def snapshot_metrics(metrics: SnapshotMetrics) -> dict[str, Any]:
        """Serialize SnapshotMetrics."""
        return {
            "stage":              metrics.stage,
            "stage_name":         metrics.stage_name,
            "certainty":          _r(metrics.certainty),
            "certainty_gap":      _r(metrics.certainty_gap),
            "contradiction_load": _r(metrics.contradiction_load),
            "ambiguity_index":    _r(metrics.ambiguity_index),
            "leading_disease":    metrics.leading_disease,
            "fsm_state":          metrics.fsm_state,
            "active_rule_count":  metrics.active_rule_count,
            "tier_a_count":       metrics.tier_a_count,
            "safety_triggered":   metrics.safety_triggered,
            "triage_so_far":      metrics.triage_so_far,
        }

    @staticmethod
    def snapshot(snap: GraphSnapshot) -> dict[str, Any]:
        """Serialize a GraphSnapshot to a plain dict."""
        return {
            "case_id":          snap.case_id,
            "snapshot_index":   snap.snapshot_index,
            "delta_description": snap.delta_description,
            "metrics":          GraphSerializer.snapshot_metrics(snap.metrics),
            "node_states": [
                {
                    "node_id":            ns.node_id,
                    "node_type":          ns.node_type,
                    "label":              ns.label,
                    "activation":         ns.activation,
                    "activation_strength": _r(ns.activation_strength),
                    "meta":               _serialize_meta(ns.meta),
                }
                for ns in snap.node_states
            ],
            "edge_states": [
                {
                    "edge_id":   es.edge_id,
                    "edge_type": es.edge_type,
                    "source_id": es.source_id,
                    "target_id": es.target_id,
                    "weight":    _r(es.weight),
                    "active":    es.active,
                    "meta":      _serialize_meta(es.meta),
                }
                for es in snap.edge_states
            ],
            "active_node_ids": sorted(snap.active_node_ids),
            "active_edge_ids": sorted(snap.active_edge_ids),
        }

    # ── ReasoningGraph serialization ──────────────────────────────────────────

    @staticmethod
    def graph(g: ReasoningGraph) -> dict[str, Any]:
        """Serialize a ReasoningGraph to a plain dict."""
        leading = g.leading_hypothesis()
        esc     = g.escalation_node()
        return {
            "case_id":         g.case_id,
            "run_id":          g.run_id,
            "node_count":      g.node_count,
            "edge_count":      g.edge_count,
            "leading_disease": leading.meta.get("disease") if leading else None,
            "recommendation":  esc.meta.get("recommendation") if esc else None,
            "nodes":           [GraphSerializer.node(n) for n in g.nodes()],
            "edges":           [GraphSerializer.edge(e) for e in g.edges()],
            "hypothesis_nodes": [
                GraphSerializer.node(n) for n in g.hypothesis_nodes()
            ],
            "stage_nodes":     [GraphSerializer.node(n) for n in g.stage_nodes()],
            "contradiction_nodes": [
                GraphSerializer.node(n) for n in g.contradiction_nodes()
            ],
        }

    # ── TrajectoryGraph serialization ─────────────────────────────────────────

    @staticmethod
    def trajectory(tg: TrajectoryGraph) -> dict[str, Any]:
        """Serialize a TrajectoryGraph to a plain dict."""
        deltas = tg.deltas()
        return {
            "case_id":              tg.case_id,
            "run_id":               tg.run_id,
            "stage_count":          tg.length,
            "certainty_series":     [(s, _r(c)) for s, c in tg.certainty_series()],
            "gap_series":           [(s, _r(g)) for s, g in tg.gap_series()],
            "contradiction_series": [(s, _r(l)) for s, l in tg.contradiction_series()],
            "entropy_series":       [(s, _r(e)) for s, e in tg.entropy_series()],
            "fsm_state_series":     tg.fsm_state_series(),
            "leading_disease_series": tg.leading_disease_series(),
            "oscillation_count":    tg.oscillation_count(),
            "convergence_index":    _r(tg.convergence_index()),
            "snapshots":            [GraphSerializer.snapshot(s) for s in tg.snapshots],
            "deltas": [
                {
                    "from_index":      d.from_index,
                    "to_index":        d.to_index,
                    "from_stage":      d.from_stage,
                    "to_stage":        d.to_stage,
                    "certainty_delta": _r(d.certainty_delta),
                    "gap_delta":       _r(d.gap_delta),
                    "contra_delta":    _r(d.contra_delta),
                    "entropy_delta":   _r(d.entropy_delta),
                    "state_changed":   d.state_changed,
                    "from_state":      d.from_state,
                    "to_state":        d.to_state,
                    "from_leader":     d.from_leader,
                    "to_leader":       d.to_leader,
                    "leader_changed":  d.leader_changed,
                }
                for d in deltas
            ],
        }

    # ── CertaintyGraph serialization ──────────────────────────────────────────

    @staticmethod
    def certainty_graph(cg: CertaintyGraph) -> dict[str, Any]:
        """Serialize a CertaintyGraph to a plain dict."""
        stab = cg.stabilisation_stage()
        peak = cg.peak_certainty()
        return {
            "case_id":              cg.case_id,
            "recommendation":       cg.recommendation,
            "certainty_series":     [_r(v) for v in cg.certainty_series()],
            "gap_series":           [_r(v) for v in cg.gap_series()],
            "entropy_series":       [_r(v) for v in cg.entropy_series()],
            "normalised_entropy":   [_r(v) for v in cg.normalised_entropy_series()],
            "stage_labels":         cg.stage_labels(),
            "leading_disease_series": cg.leading_disease_series(),
            "convergence_index":    _r(cg.convergence_index()),
            "entropy_reduction":    _r(cg.entropy_reduction()),
            "oscillation_count":    cg.oscillation_count(),
            "leadership_changes":   cg.leadership_changes(),
            "was_dampened":         cg.was_dampened(),
            "stabilisation_stage":  stab.stage if stab else None,
            "peak_certainty_stage": peak.stage if peak else None,
            "peak_certainty_value": _r(peak.certainty) if peak else None,
            "points":               [p.to_dict() for p in cg.points()],
        }

    # ── ContradictionGraph serialization ──────────────────────────────────────

    @staticmethod
    def contradiction_graph(cg: ContradictionGraph) -> dict[str, Any]:
        """Serialize a ContradictionGraph to a plain dict."""
        return cg.to_dict()

    # ── ReplayEvent serialization ─────────────────────────────────────────────

    @staticmethod
    def replay_event(event: ReplayEvent) -> dict[str, Any]:
        """Serialize a ReplayEvent to a plain dict."""
        return event.to_dict()

    # ── ReplayResult serialization ────────────────────────────────────────────

    @staticmethod
    def replay_result(result: ReplayResult) -> dict[str, Any]:
        """Serialize a complete ReplayResult to a plain dict."""
        return result.to_dict()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _serialize_meta(meta: dict) -> dict:
    """
    Serialize a meta dict, converting non-JSON-serializable types to
    JSON-safe equivalents.
    """
    out = {}
    for k, v in meta.items():
        if isinstance(v, float):
            out[k] = _r(v)
        elif isinstance(v, frozenset):
            out[k] = sorted(v)
        elif isinstance(v, set):
            out[k] = sorted(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif hasattr(v, "value"):   # Enum
            out[k] = v.value
        else:
            out[k] = v
    return out
