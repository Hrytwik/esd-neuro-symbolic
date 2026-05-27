"""
graph_contract_finalizer.py
=============================
Finalizes the graph serialization schema and frontend export contracts for
the CASDRE clinical inference pipeline.

Coordinates with src/graph_reasoning/ infrastructure to define the canonical
node/edge/snapshot format that the frontend visualization layer must consume.
Validates graph outputs for structural integrity and schema conformance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from pydantic import BaseModel, Field, model_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]


# ──────────────────────────────────────────────────────────────────────────────
# Contract version
# ──────────────────────────────────────────────────────────────────────────────

GRAPH_CONTRACT_VERSION = "1.0.0"
GRAPH_CONTRACT_FROZEN  = True


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class GraphNodeType(str, Enum):
    DISEASE      = "disease"
    FEATURE      = "feature"
    SIGNAL       = "signal"
    FSM_STATE    = "fsm_state"
    CASE         = "case"
    TRAJECTORY   = "trajectory"


class GraphEdgeType(str, Enum):
    SUPPORTS        = "supports"        # feature → disease support
    CONTRADICTS     = "contradicts"     # feature → disease contradiction
    TRANSITIONS_TO  = "transitions_to"  # FSM state transitions
    ACTIVATES       = "activates"       # signal → derived signal
    COMPETES_WITH   = "competes_with"   # disease ↔ disease competition
    RECOVERS_VIA    = "recovers_via"    # case → recovery mechanism


# ──────────────────────────────────────────────────────────────────────────────
# Schema definitions
# ──────────────────────────────────────────────────────────────────────────────

if _PYDANTIC_AVAILABLE:

    class GraphNodeSchema(BaseModel):
        schema_version: str       = Field(default=GRAPH_CONTRACT_VERSION)
        node_id: str
        node_type: GraphNodeType
        label: str
        properties: Dict[str, Any] = Field(default_factory=dict)

    class GraphEdgeSchema(BaseModel):
        schema_version: str       = Field(default=GRAPH_CONTRACT_VERSION)
        edge_id: str
        source_id: str
        target_id: str
        edge_type: GraphEdgeType
        weight: float             = Field(ge=0.0, le=1.0)
        properties: Dict[str, Any] = Field(default_factory=dict)

    class GraphSnapshotSchema(BaseModel):
        """Complete graph snapshot for a case or the full population."""
        schema_version: str       = Field(default=GRAPH_CONTRACT_VERSION)
        snapshot_id: str
        case_id: Optional[str]    = None
        step: Optional[int]       = None
        nodes: List[GraphNodeSchema]
        edges: List[GraphEdgeSchema]
        metadata: Dict[str, Any]  = Field(default_factory=dict)

        @model_validator(mode="after")
        def _check_edge_refs(self):
            node_ids = {n.node_id for n in self.nodes}
            for e in self.edges:
                if e.source_id not in node_ids:
                    raise ValueError(
                        f"Edge {e.edge_id!r} source_id={e.source_id!r} not in nodes"
                    )
                if e.target_id not in node_ids:
                    raise ValueError(
                        f"Edge {e.edge_id!r} target_id={e.target_id!r} not in nodes"
                    )
            return self

else:
    @dataclass
    class GraphNodeSchema:
        schema_version: str; node_id: str; node_type: str; label: str
        properties: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class GraphEdgeSchema:
        schema_version: str; edge_id: str; source_id: str; target_id: str
        edge_type: str; weight: float
        properties: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class GraphSnapshotSchema:
        schema_version: str; snapshot_id: str; nodes: list; edges: list
        case_id: Optional[str] = None; step: Optional[int] = None
        metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Required fields
# ──────────────────────────────────────────────────────────────────────────────

_NODE_REQUIRED   = frozenset({"schema_version", "node_id", "node_type", "label"})
_EDGE_REQUIRED   = frozenset({"schema_version", "edge_id", "source_id", "target_id",
                               "edge_type", "weight"})
_SNAP_REQUIRED   = frozenset({"schema_version", "snapshot_id", "nodes", "edges"})

_VALID_NODE_TYPES = {t.value for t in GraphNodeType}
_VALID_EDGE_TYPES = {t.value for t in GraphEdgeType}


# ──────────────────────────────────────────────────────────────────────────────
# Validation results
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphValidationResult:
    snapshot_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    n_nodes: int
    n_edges: int


@dataclass
class GraphContractAuditReport:
    n_snapshots_audited: int
    n_valid: int
    n_invalid: int
    validation_rate: float
    n_nodes_audited: int
    n_edges_audited: int
    common_errors: List[str]
    contract_version: str = GRAPH_CONTRACT_VERSION
    contract_frozen:  bool = GRAPH_CONTRACT_FROZEN

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "GRAPH CONTRACT AUDIT REPORT",
            f"  Contract version   : {self.contract_version}  (frozen={self.contract_frozen})",
            "=" * 70,
            f"  Snapshots audited  : {self.n_snapshots_audited}",
            f"  Valid              : {self.n_valid}  ({self.validation_rate:.1%})",
            f"  Invalid            : {self.n_invalid}",
            f"  Nodes audited      : {self.n_nodes_audited}",
            f"  Edges audited      : {self.n_edges_audited}",
        ]
        if self.common_errors:
            lines.append("  Common errors:")
            for e in self.common_errors:
                lines.append(f"    • {e}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Finalizer
# ──────────────────────────────────────────────────────────────────────────────

class GraphContractFinalizer:
    """
    Validates graph snapshots against the frozen graph export contract.

    Usage
    -----
    finalizer = GraphContractFinalizer()
    result    = finalizer.validate_snapshot(snapshot_dict)
    report    = finalizer.audit_batch(list_of_snapshot_dicts)
    """

    def _validate_node(self, node: Dict[str, Any], idx: int) -> List[str]:
        errors: List[str] = []
        missing = [f for f in _NODE_REQUIRED if f not in node]
        if missing:
            errors.append(f"node[{idx}] missing fields: {missing}")
        nt = node.get("node_type", "")
        if nt not in _VALID_NODE_TYPES:
            errors.append(f"node[{idx}] invalid node_type={nt!r}")
        return errors

    def _validate_edge(
        self,
        edge: Dict[str, Any],
        idx: int,
        node_ids: Set[str],
    ) -> List[str]:
        errors: List[str] = []
        missing = [f for f in _EDGE_REQUIRED if f not in edge]
        if missing:
            errors.append(f"edge[{idx}] missing fields: {missing}")
        et = edge.get("edge_type", "")
        if et not in _VALID_EDGE_TYPES:
            errors.append(f"edge[{idx}] invalid edge_type={et!r}")
        w = edge.get("weight", 0.0)
        if not (0.0 <= float(w) <= 1.0):
            errors.append(f"edge[{idx}] weight={w!r} out of [0,1]")
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        if src and src not in node_ids:
            errors.append(f"edge[{idx}] source_id={src!r} not in nodes")
        if tgt and tgt not in node_ids:
            errors.append(f"edge[{idx}] target_id={tgt!r} not in nodes")
        return errors

    def validate_snapshot(self, snapshot: Dict[str, Any]) -> GraphValidationResult:
        snap_id = str(snapshot.get("snapshot_id", "<unknown>"))
        errors:   List[str] = []
        warnings: List[str] = []

        missing = [f for f in _SNAP_REQUIRED if f not in snapshot]
        if missing:
            errors.append(f"Snapshot missing required fields: {missing}")

        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])
        if not isinstance(nodes, list):
            errors.append("snapshot.nodes must be a list")
            nodes = []
        if not isinstance(edges, list):
            errors.append("snapshot.edges must be a list")
            edges = []

        node_ids: Set[str] = set()
        for idx, n in enumerate(nodes):
            node_errs = self._validate_node(n, idx)
            errors.extend(node_errs)
            if "node_id" in n:
                node_ids.add(str(n["node_id"]))

        for idx, e in enumerate(edges):
            edge_errs = self._validate_edge(e, idx, node_ids)
            errors.extend(edge_errs)

        if "schema_version" in snapshot:
            if snapshot["schema_version"] != GRAPH_CONTRACT_VERSION:
                warnings.append(
                    f"schema_version mismatch: {snapshot['schema_version']!r} "
                    f"vs {GRAPH_CONTRACT_VERSION!r}"
                )

        return GraphValidationResult(
            snapshot_id=snap_id,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            n_nodes=len(nodes),
            n_edges=len(edges),
        )

    def audit_batch(
        self,
        snapshots: List[Dict[str, Any]],
    ) -> GraphContractAuditReport:
        from collections import Counter
        results   = [self.validate_snapshot(s) for s in snapshots]
        n_valid   = sum(1 for r in results if r.is_valid)
        n_invalid = len(results) - n_valid
        n_nodes   = sum(r.n_nodes for r in results)
        n_edges   = sum(r.n_edges for r in results)

        error_counter: Counter = Counter()
        for r in results:
            for e in r.errors:
                key = e.split(":")[0].strip()
                error_counter[key] += 1

        return GraphContractAuditReport(
            n_snapshots_audited=len(results),
            n_valid=n_valid,
            n_invalid=n_invalid,
            validation_rate=n_valid / len(results) if results else 0.0,
            n_nodes_audited=n_nodes,
            n_edges_audited=n_edges,
            common_errors=[f"{k}: {v}×" for k, v in error_counter.most_common(5)],
        )

    @staticmethod
    def canonical_node(
        node_id: str,
        node_type: str,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "schema_version": GRAPH_CONTRACT_VERSION,
            "node_id":        node_id,
            "node_type":      node_type,
            "label":          label,
            "properties":     properties or {},
        }

    @staticmethod
    def canonical_edge(
        edge_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "schema_version": GRAPH_CONTRACT_VERSION,
            "edge_id":        edge_id,
            "source_id":      source_id,
            "target_id":      target_id,
            "edge_type":      edge_type,
            "weight":         float(max(0.0, min(1.0, weight))),
            "properties":     properties or {},
        }

    @staticmethod
    def canonical_snapshot(
        snapshot_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        case_id: Optional[str] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "schema_version": GRAPH_CONTRACT_VERSION,
            "snapshot_id":    snapshot_id,
            "case_id":        case_id,
            "step":           step,
            "nodes":          nodes,
            "edges":          edges,
            "metadata":       metadata or {},
        }
