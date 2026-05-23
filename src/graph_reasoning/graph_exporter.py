"""
GraphExporter — high-level graph export to JSON, React Flow, and Cytoscape.

The GraphExporter translates graph structures into formats suitable for:
  · React Flow    — interactive reasoning graph visualisation (frontend)
  · Cytoscape.js  — network analysis and publication-quality rendering
  · Plain JSON    — storage, API responses, and offline replay

All exports are deterministic — same input always produces identical output.
File exports use atomic writes (temp file → rename) to avoid partial writes.

React Flow format
-----------------
  {
    "nodes": [{"id", "type", "position": {"x", "y"}, "data": {...}}, ...],
    "edges": [{"id", "type", "source", "target", "data": {...}}, ...],
    "meta":  {"case_id", "recommendation", "certainty", ...}
  }

React Flow node types map to:
  hypothesis    → "hypothesisNode"
  stage         → "stageNode"
  contradiction → "contradictionNode"
  escalation    → "escalationNode"
  instability   → "instabilityNode"

Cytoscape format
----------------
  {
    "elements": {
      "nodes": [{"data": {"id", "label", ...}}, ...],
      "edges": [{"data": {"id", "source", "target", "weight", ...}}, ...]
    },
    "meta": {...}
  }

Layout
------
  Hypothesis nodes:    x=-200, y=disease_index * 120
  Stage nodes:         x=stage * 250, y=-180
  Contradiction nodes: x=stage_midpoint * 250, y=400
  Escalation node:     x=(max_stage+1) * 250, y=300
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.graph_reasoning.graph_nodes import DISEASE_INDEX, NodeType
from src.graph_reasoning.graph_edges import EdgeType
from src.graph_reasoning.graph_snapshot import GraphSnapshot
from src.graph_reasoning.reasoning_graph import ReasoningGraph
from src.graph_reasoning.trajectory_graph import TrajectoryGraph
from src.graph_reasoning.certainty_graph import CertaintyGraph
from src.graph_reasoning.contradiction_graph import ContradictionGraph
from src.graph_reasoning.replay_engine import ReplayResult
from src.graph_reasoning.graph_serializer import GraphSerializer


# ── Layout constants ──────────────────────────────────────────────────────────

_STAGE_X_STEP: int   = 250      # horizontal spacing between stages
_HYPO_Y_STEP: int    = 120      # vertical spacing between hypothesis nodes
_HYPO_X:      int    = -220     # fixed x position for hypothesis column
_STAGE_Y:     int    = -200     # y position for stage row (above hypotheses)
_CONTRA_Y:    int    = 440      # y position for contradiction row
_ESC_X_OFFSET: int  = 250      # x offset for escalation node after last stage


def _stage_x(stage: int) -> int:
    return stage * _STAGE_X_STEP


def _hypo_y(disease: str) -> int:
    idx = DISEASE_INDEX.get(disease, 0)
    return idx * _HYPO_Y_STEP


def _esc_x(max_stage: int) -> int:
    return (max_stage + 1) * _STAGE_X_STEP


# ── React Flow exporter ───────────────────────────────────────────────────────

class ReactFlowExporter:
    """
    Converts a ReasoningGraph to React Flow compatible JSON.

    React Flow is the primary frontend visualisation target.
    """

    @staticmethod
    def export(graph: ReasoningGraph) -> dict[str, Any]:
        """
        Convert a ReasoningGraph to React Flow format.

        Returns a dict with keys: 'nodes', 'edges', 'meta'.
        """
        rf_nodes: list[dict] = []
        rf_edges: list[dict] = []

        stage_nodes = graph.stage_nodes()
        max_stage = max((n.meta.get("stage", 0) for n in stage_nodes), default=0)

        # Hypothesis nodes (fixed left column)
        for node in graph.hypothesis_nodes():
            disease = node.meta.get("disease", "")
            rf_nodes.append({
                "id":   node.node_id,
                "type": "hypothesisNode",
                "position": {
                    "x": _HYPO_X,
                    "y": _hypo_y(disease),
                },
                "data": {
                    "label":          node.label,
                    "activation":     node.activation.value,
                    "certainty":      node.activation_strength,
                    "is_leading":     node.meta.get("is_leading", False),
                    "is_suppressed":  node.meta.get("is_suppressed", False),
                    "disease":        disease,
                    "disease_index":  node.meta.get("disease_index", -1),
                },
            })

        # Stage nodes (horizontal timeline)
        for node in stage_nodes:
            stage = node.meta.get("stage", 0)
            rf_nodes.append({
                "id":   node.node_id,
                "type": "stageNode",
                "position": {
                    "x": _stage_x(stage),
                    "y": _STAGE_Y,
                },
                "data": {
                    "label":            node.label,
                    "stage":            stage,
                    "stage_name":       node.meta.get("stage_name", ""),
                    "certainty":        node.activation_strength,
                    "fsm_state":        node.meta.get("fsm_state", ""),
                    "contradiction_load": node.meta.get("contradiction_load", 0.0),
                    "ambiguity_index":  node.meta.get("ambiguity_index", 0.0),
                    "active_rule_count": node.meta.get("active_rule_count", 0),
                    "safety_triggered": node.meta.get("safety_triggered", False),
                },
            })

        # Contradiction nodes (below hypothesis column)
        contra_nodes = graph.contradiction_nodes()
        for i, node in enumerate(contra_nodes):
            stage = node.meta.get("stage", 2)
            rf_nodes.append({
                "id":   node.node_id,
                "type": "contradictionNode",
                "position": {
                    "x": _stage_x(stage),
                    "y": _CONTRA_Y,
                },
                "data": {
                    "label":             node.label,
                    "activation":        node.activation.value,
                    "contradiction_load": node.meta.get("contradiction_load", 0.0),
                    "source_disease":    node.meta.get("source_disease"),
                    "target_disease":    node.meta.get("target_disease"),
                    "penalty_weight":    node.meta.get("penalty_weight", 0.0),
                },
            })

        # Escalation node (far right)
        esc = graph.escalation_node()
        if esc:
            rf_nodes.append({
                "id":   esc.node_id,
                "type": "escalationNode",
                "position": {
                    "x": _esc_x(max_stage),
                    "y": (_hypo_y("lichen_planus") + _hypo_y("pityriasis_rosea")) // 2,
                },
                "data": {
                    "label":          esc.label,
                    "recommendation": esc.meta.get("recommendation", ""),
                    "requires_biopsy": esc.meta.get("requires_biopsy", False),
                    "certainty":      esc.activation_strength,
                    "activation":     esc.activation.value,
                },
            })

        # All edges
        for edge in graph.edges():
            edge_data: dict[str, Any] = {
                "edge_type": edge.edge_type.value,
                "weight":    edge.weight,
                "active":    edge.active,
                **{k: v for k, v in edge.meta.items()},
            }
            rf_edges.append({
                "id":     edge.edge_id,
                "type":   _react_flow_edge_type(edge.edge_type),
                "source": edge.source_id,
                "target": edge.target_id,
                "data":   edge_data,
                "style":  _edge_style(edge.edge_type, edge.weight, edge.active),
            })

        leading = graph.leading_hypothesis()
        esc_node = graph.escalation_node()
        return {
            "nodes": rf_nodes,
            "edges": rf_edges,
            "meta":  {
                "case_id":          graph.case_id,
                "run_id":           graph.run_id,
                "node_count":       len(rf_nodes),
                "edge_count":       len(rf_edges),
                "leading_disease":  leading.meta.get("disease") if leading else None,
                "recommendation":   esc_node.meta.get("recommendation") if esc_node else None,
                "stage_count":      len(stage_nodes),
                "max_stage":        max_stage,
            },
        }


# ── Cytoscape exporter ────────────────────────────────────────────────────────

class CytoscapeExporter:
    """
    Converts a ReasoningGraph to Cytoscape.js compatible JSON.

    Cytoscape is suitable for publication-quality network rendering and
    offline graph analysis.
    """

    @staticmethod
    def export(graph: ReasoningGraph) -> dict[str, Any]:
        """
        Convert a ReasoningGraph to Cytoscape.js elements format.

        Returns a dict with keys: 'elements', 'meta'.
        """
        cy_nodes: list[dict] = []
        cy_edges: list[dict] = []

        stage_nodes = graph.stage_nodes()
        max_stage   = max((n.meta.get("stage", 0) for n in stage_nodes), default=0)

        for node in graph.nodes():
            stage = node.meta.get("stage", 0)
            disease = node.meta.get("disease", "")
            if node.node_type == NodeType.HYPOTHESIS:
                x, y = _HYPO_X, _hypo_y(disease)
            elif node.node_type == NodeType.STAGE:
                x, y = _stage_x(stage), _STAGE_Y
            elif node.node_type == NodeType.CONTRADICTION:
                x, y = _stage_x(node.meta.get("stage", 2)), _CONTRA_Y
            elif node.node_type == NodeType.ESCALATION:
                x, y = _esc_x(max_stage), 300
            else:
                x, y = 0, 0

            cy_nodes.append({
                "data": {
                    "id":                node.node_id,
                    "label":             node.label,
                    "node_type":         node.node_type.value,
                    "activation":        node.activation.value,
                    "activation_strength": round(node.activation_strength, 4),
                    **_serialize_meta_cy(node.meta),
                },
                "position": {"x": x, "y": y},
                "classes":  _cytoscape_classes(node),
            })

        for edge in graph.edges():
            cy_edges.append({
                "data": {
                    "id":        edge.edge_id,
                    "source":    edge.source_id,
                    "target":    edge.target_id,
                    "edge_type": edge.edge_type.value,
                    "weight":    round(edge.weight, 4),
                    "active":    edge.active,
                    **_serialize_meta_cy(edge.meta),
                },
                "classes": edge.edge_type.value + ("" if edge.active else " inactive"),
            })

        leading = graph.leading_hypothesis()
        esc_node = graph.escalation_node()
        return {
            "elements": {
                "nodes": cy_nodes,
                "edges": cy_edges,
            },
            "meta": {
                "case_id":         graph.case_id,
                "run_id":          graph.run_id,
                "leading_disease": leading.meta.get("disease") if leading else None,
                "recommendation":  esc_node.meta.get("recommendation") if esc_node else None,
                "stage_count":     len(stage_nodes),
            },
        }


# ── Full graph exporter ───────────────────────────────────────────────────────

class GraphExporter:
    """
    High-level export interface for all graph types and formats.

    Provides convenience methods that write JSON files and return dicts
    in React Flow and Cytoscape formats.

    All file-writing methods use pathlib.Path and create parent directories
    automatically.
    """

    # ── Format exports ────────────────────────────────────────────────────────

    @staticmethod
    def to_react_flow(graph: ReasoningGraph) -> dict[str, Any]:
        """Export a ReasoningGraph to React Flow format."""
        return ReactFlowExporter.export(graph)

    @staticmethod
    def to_cytoscape(graph: ReasoningGraph) -> dict[str, Any]:
        """Export a ReasoningGraph to Cytoscape.js format."""
        return CytoscapeExporter.export(graph)

    @staticmethod
    def to_json(graph: ReasoningGraph) -> dict[str, Any]:
        """Export a ReasoningGraph to plain JSON dict."""
        return GraphSerializer.graph(graph)

    @staticmethod
    def trajectory_to_json(tg: TrajectoryGraph) -> dict[str, Any]:
        """Export a TrajectoryGraph to plain JSON dict."""
        return GraphSerializer.trajectory(tg)

    @staticmethod
    def certainty_to_json(cg: CertaintyGraph) -> dict[str, Any]:
        """Export a CertaintyGraph to plain JSON dict."""
        return GraphSerializer.certainty_graph(cg)

    @staticmethod
    def contradiction_to_json(cg: ContradictionGraph) -> dict[str, Any]:
        """Export a ContradictionGraph to plain JSON dict."""
        return GraphSerializer.contradiction_graph(cg)

    @staticmethod
    def replay_to_json(result: ReplayResult) -> dict[str, Any]:
        """Export a ReplayResult to plain JSON dict."""
        return GraphSerializer.replay_result(result)

    # ── Trajectory snapshot exports ───────────────────────────────────────────

    @staticmethod
    def snapshot_to_react_flow(snap: GraphSnapshot) -> dict[str, Any]:
        """
        Convert a single GraphSnapshot to a React Flow compatible dict.

        Reconstructs a minimal React Flow graph from the lightweight node/edge
        state records embedded in the snapshot.
        """
        rf_nodes: list[dict] = []
        rf_edges: list[dict] = []

        stage_nodes = [n for n in snap.node_states if n.node_type == "stage"]
        max_stage   = max((n.meta.get("stage", 0) for n in stage_nodes), default=0)

        for ns in snap.node_states:
            disease = ns.meta.get("disease", "")
            stage   = ns.meta.get("stage", 0)
            node_type = ns.node_type

            if node_type == "hypothesis":
                pos = {"x": _HYPO_X, "y": _hypo_y(disease)}
                rf_type = "hypothesisNode"
            elif node_type == "stage":
                pos = {"x": _stage_x(stage), "y": _STAGE_Y}
                rf_type = "stageNode"
            elif node_type == "contradiction":
                pos = {"x": _stage_x(stage), "y": _CONTRA_Y}
                rf_type = "contradictionNode"
            elif node_type == "escalation":
                pos = {"x": _esc_x(max_stage), "y": 300}
                rf_type = "escalationNode"
            else:
                pos = {"x": 0, "y": 0}
                rf_type = node_type

            rf_nodes.append({
                "id":       ns.node_id,
                "type":     rf_type,
                "position": pos,
                "data": {
                    "label":             ns.label,
                    "activation":        ns.activation,
                    "activation_strength": ns.activation_strength,
                    "is_active":         ns.node_id in snap.active_node_ids,
                    **ns.meta,
                },
            })

        for es in snap.edge_states:
            rf_edges.append({
                "id":     es.edge_id,
                "source": es.source_id,
                "target": es.target_id,
                "data": {
                    "edge_type": es.edge_type,
                    "weight":    es.weight,
                    "active":    es.active,
                    "is_active": es.edge_id in snap.active_edge_ids,
                    **es.meta,
                },
            })

        return {
            "nodes":  rf_nodes,
            "edges":  rf_edges,
            "meta": {
                "case_id":          snap.case_id,
                "snapshot_index":   snap.snapshot_index,
                "stage":            snap.metrics.stage,
                "stage_name":       snap.metrics.stage_name,
                "leading_disease":  snap.metrics.leading_disease,
                "certainty":        snap.metrics.certainty,
                "contradiction_load": snap.metrics.contradiction_load,
                "ambiguity_index":  snap.metrics.ambiguity_index,
                "fsm_state":        snap.metrics.fsm_state,
                "delta_description": snap.delta_description,
            },
        }

    # ── File writing ──────────────────────────────────────────────────────────

    @staticmethod
    def write_json(data: dict[str, Any], path: Path | str, indent: int = 2) -> Path:
        """
        Write a dict to a JSON file. Creates parent directories if needed.

        Parameters
        ----------
        data:
            JSON-serializable dict.
        path:
            Output file path.
        indent:
            JSON indentation (default 2).

        Returns
        -------
        Path:
            The resolved output path.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=_json_fallback)
        tmp_path.replace(out_path)
        return out_path

    @staticmethod
    def export_graph(
        graph: ReasoningGraph,
        output_dir: Path | str,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """
        Export a ReasoningGraph to one or more formats in the given directory.

        Parameters
        ----------
        graph:
            The ReasoningGraph to export.
        output_dir:
            Directory to write output files into.
        formats:
            List of format names: 'json', 'react_flow', 'cytoscape'.
            Defaults to all three.

        Returns
        -------
        dict[str, Path]:
            Mapping from format name to written file path.
        """
        if formats is None:
            formats = ["json", "react_flow", "cytoscape"]

        out_dir  = Path(output_dir)
        case_id  = graph.case_id.replace("/", "_")
        written: dict[str, Path] = {}

        if "json" in formats:
            p = GraphExporter.write_json(
                GraphExporter.to_json(graph),
                out_dir / f"{case_id}_reasoning_graph.json",
            )
            written["json"] = p

        if "react_flow" in formats:
            p = GraphExporter.write_json(
                GraphExporter.to_react_flow(graph),
                out_dir / f"{case_id}_react_flow.json",
            )
            written["react_flow"] = p

        if "cytoscape" in formats:
            p = GraphExporter.write_json(
                GraphExporter.to_cytoscape(graph),
                out_dir / f"{case_id}_cytoscape.json",
            )
            written["cytoscape"] = p

        return written

    @staticmethod
    def export_trajectory(
        tg: TrajectoryGraph,
        cg: CertaintyGraph,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """
        Export trajectory and certainty graphs to JSON files.

        Parameters
        ----------
        tg:
            TrajectoryGraph to export.
        cg:
            CertaintyGraph to export.
        output_dir:
            Output directory.

        Returns
        -------
        dict[str, Path]:
            Written file paths by type.
        """
        out_dir  = Path(output_dir)
        case_id  = tg.case_id.replace("/", "_")
        written: dict[str, Path] = {}

        written["trajectory"] = GraphExporter.write_json(
            GraphExporter.trajectory_to_json(tg),
            out_dir / f"{case_id}_trajectory.json",
        )
        written["certainty"] = GraphExporter.write_json(
            GraphExporter.certainty_to_json(cg),
            out_dir / f"{case_id}_certainty.json",
        )
        return written

    @staticmethod
    def export_replay(
        result: ReplayResult,
        output_dir: Path | str,
    ) -> Path:
        """Export a ReplayResult to a JSON file."""
        out_dir  = Path(output_dir)
        case_id  = result.case_id.replace("/", "_")
        return GraphExporter.write_json(
            GraphExporter.replay_to_json(result),
            out_dir / f"{case_id}_replay.json",
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _react_flow_edge_type(edge_type: EdgeType) -> str:
    """Map EdgeType to React Flow edge type string."""
    mapping = {
        EdgeType.REINFORCEMENT: "reinforcementEdge",
        EdgeType.CONTRADICTION: "contradictionEdge",
        EdgeType.SUPPRESSION:   "suppressionEdge",
        EdgeType.ESCALATION:    "escalationEdge",
        EdgeType.PROPAGATION:   "propagationEdge",
        EdgeType.TRAJECTORY:    "trajectoryEdge",
        EdgeType.LEADERSHIP:    "leadershipEdge",
        EdgeType.COMPETITION:   "competitionEdge",
    }
    return mapping.get(edge_type, "default")


def _edge_style(edge_type: EdgeType, weight: float, active: bool) -> dict:
    """Generate React Flow edge inline style based on type and weight."""
    if not active:
        return {"opacity": 0.2, "strokeDasharray": "4 4"}

    colour_map = {
        EdgeType.REINFORCEMENT: "#22c55e",   # green
        EdgeType.CONTRADICTION: "#ef4444",   # red
        EdgeType.SUPPRESSION:   "#f97316",   # orange
        EdgeType.ESCALATION:    "#8b5cf6",   # violet
        EdgeType.PROPAGATION:   "#3b82f6",   # blue
        EdgeType.TRAJECTORY:    "#6b7280",   # grey
        EdgeType.LEADERSHIP:    "#eab308",   # yellow
        EdgeType.COMPETITION:   "#ec4899",   # pink
    }
    colour = colour_map.get(edge_type, "#6b7280")
    stroke_width = max(1, min(4, int(weight * 5)))  # scale by weight [1, 4px]
    return {"stroke": colour, "strokeWidth": stroke_width}


def _cytoscape_classes(node) -> str:
    """Generate Cytoscape CSS classes for a node."""
    classes = [node.node_type.value, node.activation.value]
    if node.is_dominant:
        classes.append("dominant")
    if not node.is_active:
        classes.append("inactive")
    return " ".join(classes)


def _serialize_meta_cy(meta: dict) -> dict:
    """Serialize a meta dict for Cytoscape (all values must be primitives)."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, bool)) or v is None:
            out[k] = v
        elif isinstance(v, float):
            out[k] = round(v, 4)
        elif isinstance(v, (frozenset, set)):
            out[k] = sorted(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif hasattr(v, "value"):
            out[k] = v.value
        else:
            out[k] = str(v)
    return out


def _json_fallback(obj: Any) -> Any:
    """JSON serialization fallback for non-standard types."""
    if hasattr(obj, "value"):       # Enum
        return obj.value
    if hasattr(obj, "__iter__"):    # sets, frozensets, etc.
        return list(obj)
    return str(obj)
