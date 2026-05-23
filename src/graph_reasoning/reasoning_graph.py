"""
ReasoningGraph — core directed graph for diagnostic reasoning representation.

The ReasoningGraph is the central data structure of the graph_reasoning layer.
It holds all nodes and edges representing one clinical case's inference process
and provides query, navigation, and snapshot APIs.

Graph structure
---------------
The graph is constructed from a completed PipelineResult and its embedded
DiagnosticTrajectory. It does NOT re-execute reasoning — it translates
the structured outputs into graph topology.

Node composition
----------------
  Six HypothesisNodes  — one per disease, certainty from terminal state
  N StageNodes         — one per trajectory snapshot (pipeline stage)
  0-M ContradictionNodes — aggregate + individual pairs (when data available)
  One EscalationNode   — terminal triage decision

Edge composition
----------------
  N-1 TrajectoryEdges  — stage(n) → stage(n+1) temporal links
  N*6 PropagationEdges — stage(n) → hypothesis(d) certainty assignments
  N LeadershipEdges    — stage(n) → leading_hypothesis(n)
  0-M ContradictionEdges — hypothesis pairs with active contradiction
  0-M SuppressionEdges   — contradiction → suppressed hypothesis
  1   EscalationEdge   — terminal stage → escalation decision

Builder pattern
---------------
  ReasoningGraph.from_result(result)          ← primary constructor
  ReasoningGraph.from_result(result, conflict) ← with contradiction detail

Direct construction is also supported for testing:
  g = ReasoningGraph(case_id="...", run_id="...")
  g.add_node(node)
  g.add_edge(edge)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from src.graph_reasoning.graph_nodes import (
    GraphNode, NodeType, ActivationState, KNOWN_DISEASES,
    make_hypothesis_node, make_stage_node, make_contradiction_node,
    make_escalation_node, make_instability_node,
)
from src.graph_reasoning.graph_edges import (
    GraphEdge, EdgeType,
    make_trajectory_edge, make_propagation_edge, make_leadership_edge,
    make_contradiction_edge, make_suppression_edge, make_escalation_edge,
    make_competition_edge,
)
from src.graph_reasoning.graph_snapshot import (
    GraphSnapshot, SnapshotMetrics, NodeState, EdgeState,
)


# ── Certainty estimation helpers ──────────────────────────────────────────────

def _estimate_second_certainty(max_certainty: float, certainty_gap: float) -> float:
    """
    Estimate the second-ranked hypothesis certainty from top-1 and gap.
    second_cert = max_certainty - certainty_gap (clamped to [0, 1]).
    """
    return max(0.0, min(1.0, max_certainty - certainty_gap))


def _estimate_field_certainty(max_certainty: float, second_certainty: float) -> float:
    """
    Estimate the certainty shared by the remaining 4 hypotheses (equal split).
    field = (1 - max - second) / 4.
    """
    remaining = max(0.0, 1.0 - max_certainty - second_certainty)
    return remaining / 4.0


# ── Core graph ────────────────────────────────────────────────────────────────

class ReasoningGraph:
    """
    Directed typed graph representing one clinical case's reasoning trajectory.

    Nodes and edges are stored in insertion order. All mutation methods
    return self for chaining.

    Parameters
    ----------
    case_id:
        The clinical case identifier.
    run_id:
        The pipeline run identifier.
    """

    def __init__(self, case_id: str, run_id: str) -> None:
        self._case_id    = case_id
        self._run_id     = run_id
        self._nodes:     dict[str, GraphNode] = {}
        self._edges:     dict[str, GraphEdge] = {}
        self._adjacency: dict[str, list[str]] = {}  # source_id → [edge_id, ...]
        self._reverse:   dict[str, list[str]] = {}  # target_id → [edge_id, ...]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # ── Mutation API ──────────────────────────────────────────────────────────

    def add_node(self, node: GraphNode) -> "ReasoningGraph":
        """Add a node. Silently replaces an existing node with the same ID."""
        self._nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, [])
        self._reverse.setdefault(node.node_id, [])
        return self

    def add_edge(self, edge: GraphEdge) -> "ReasoningGraph":
        """
        Add an edge. Both source and target nodes must exist.
        Silently replaces an edge with the same ID.
        """
        if edge.source_id not in self._nodes:
            raise ValueError(
                f"Edge source '{edge.source_id}' not found in graph — "
                "add source node before adding the edge."
            )
        if edge.target_id not in self._nodes:
            raise ValueError(
                f"Edge target '{edge.target_id}' not found in graph — "
                "add target node before adding the edge."
            )
        self._edges[edge.edge_id] = edge
        self._adjacency.setdefault(edge.source_id, [])
        self._reverse.setdefault(edge.target_id, [])
        if edge.edge_id not in self._adjacency[edge.source_id]:
            self._adjacency[edge.source_id].append(edge.edge_id)
        if edge.edge_id not in self._reverse[edge.target_id]:
            self._reverse[edge.target_id].append(edge.edge_id)
        return self

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        return self._edges.get(edge_id)

    def nodes(self) -> Iterator[GraphNode]:
        return iter(self._nodes.values())

    def edges(self) -> Iterator[GraphEdge]:
        return iter(self._edges.values())

    def nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def edges_by_type(self, edge_type: EdgeType) -> list[GraphEdge]:
        return [e for e in self._edges.values() if e.edge_type == edge_type]

    def active_nodes(self) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.is_active]

    def active_edges(self) -> list[GraphEdge]:
        return [e for e in self._edges.values() if e.active]

    def outgoing_edges(self, node_id: str) -> list[GraphEdge]:
        """Return all edges where source_id == node_id."""
        return [
            self._edges[eid]
            for eid in self._adjacency.get(node_id, [])
            if eid in self._edges
        ]

    def incoming_edges(self, node_id: str) -> list[GraphEdge]:
        """Return all edges where target_id == node_id."""
        return [
            self._edges[eid]
            for eid in self._reverse.get(node_id, [])
            if eid in self._edges
        ]

    def neighbors(self, node_id: str) -> list[GraphNode]:
        """Return all nodes reachable by one outgoing edge from node_id."""
        targets = {e.target_id for e in self.outgoing_edges(node_id)}
        return [self._nodes[t] for t in targets if t in self._nodes]

    def hypothesis_nodes(self) -> list[GraphNode]:
        return self.nodes_by_type(NodeType.HYPOTHESIS)

    def stage_nodes(self) -> list[GraphNode]:
        return sorted(
            self.nodes_by_type(NodeType.STAGE),
            key=lambda n: n.meta.get("stage", 0),
        )

    def contradiction_nodes(self) -> list[GraphNode]:
        return self.nodes_by_type(NodeType.CONTRADICTION)

    def leading_hypothesis(self) -> GraphNode | None:
        """Return the hypothesis node marked as DOMINANT."""
        for node in self.hypothesis_nodes():
            if node.activation == ActivationState.DOMINANT:
                return node
        return None

    def escalation_node(self) -> GraphNode | None:
        """Return the terminal escalation node (should be exactly one)."""
        nodes = self.nodes_by_type(NodeType.ESCALATION)
        return nodes[0] if nodes else None

    def contradiction_load(self) -> float:
        """
        Aggregate contradiction load from contradiction edges.
        """
        return sum(
            e.weight for e in self.edges_by_type(EdgeType.CONTRADICTION)
            if e.active
        )

    # ── Snapshot API ──────────────────────────────────────────────────────────

    def snapshot(
        self,
        snapshot_index: int,
        metrics: SnapshotMetrics,
        delta_description: str = "",
    ) -> GraphSnapshot:
        """
        Capture the current graph state as an immutable snapshot.

        Parameters
        ----------
        snapshot_index:
            Ordinal position in the trajectory sequence.
        metrics:
            Quantitative signals from the corresponding trajectory point.
        delta_description:
            Human-readable description of what changed at this stage.
        """
        node_states = tuple(
            NodeState(
                node_id=n.node_id,
                node_type=n.node_type.value,
                label=n.label,
                activation=n.activation.value,
                activation_strength=n.activation_strength,
                meta=dict(n.meta),
            )
            for n in self._nodes.values()
        )
        edge_states = tuple(
            EdgeState(
                edge_id=e.edge_id,
                edge_type=e.edge_type.value,
                source_id=e.source_id,
                target_id=e.target_id,
                weight=e.weight,
                active=e.active,
                meta=dict(e.meta),
            )
            for e in self._edges.values()
        )
        active_node_ids = frozenset(
            n.node_id for n in self._nodes.values() if n.is_active
        )
        active_edge_ids = frozenset(
            e.edge_id for e in self._edges.values() if e.active
        )
        return GraphSnapshot(
            case_id=self._case_id,
            snapshot_index=snapshot_index,
            metrics=metrics,
            node_states=node_states,
            edge_states=edge_states,
            active_node_ids=active_node_ids,
            active_edge_ids=active_edge_ids,
            delta_description=delta_description,
        )

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        hyp     = len(self.hypothesis_nodes())
        stages  = len(self.stage_nodes())
        contra  = len(self.contradiction_nodes())
        esc_n   = self.escalation_node()
        leading = self.leading_hypothesis()
        return (
            f"ReasoningGraph[case={self._case_id}] "
            f"nodes={self.node_count}(hyp={hyp}, stages={stages}, "
            f"contra={contra}) "
            f"edges={self.edge_count} "
            f"leading={leading.meta.get('disease', '?') if leading else '?'} "
            f"recommendation={esc_n.meta.get('recommendation', '?') if esc_n else '?'}"
        )

    # ── Primary factory ───────────────────────────────────────────────────────

    @classmethod
    def from_result(
        cls,
        result: "PipelineResult",  # type: ignore[name-defined]
        conflict: "ConflictAnalysisResult | None" = None,  # type: ignore[name-defined]
    ) -> "ReasoningGraph":
        """
        Build a complete ReasoningGraph from a finished PipelineResult.

        The graph includes:
        - Six HypothesisNodes (one per disease)
        - One StageNode per trajectory snapshot
        - Trajectory edges linking consecutive stages
        - Leadership edges from each stage to its leading hypothesis
        - Propagation edges from each stage to all hypotheses (weighted by
          estimated certainty)
        - Contradiction edges (from ConflictAnalysisResult if provided,
          otherwise aggregate)
        - One EscalationNode with an escalation edge from the terminal stage

        Parameters
        ----------
        result:
            Completed PipelineResult from PipelineRunner.run().
        conflict:
            Optional ConflictAnalysisResult for detailed contradiction edges.
            When None, an aggregate contradiction node is used.
        """
        from src.pipeline.pipeline_runner import PipelineResult  # local import
        g = cls(case_id=result.case_id, run_id=result.run_id)

        traj = result.trajectory

        # ── 1. Hypothesis nodes (terminal state certainty) ─────────────────
        leading = result.leading_disease or ""
        second_cert = _estimate_second_certainty(
            result.max_certainty, result.certainty_gap
        )
        field_cert = _estimate_field_certainty(result.max_certainty, second_cert)
        is_dampened = result.contradiction_load > 0.20

        for disease in KNOWN_DISEASES:
            if disease == leading:
                cert = result.max_certainty
                is_lead = True
            else:
                cert = field_cert  # equal-split estimate for non-leaders
                is_lead = False
            suppressed = is_dampened and not is_lead
            g.add_node(make_hypothesis_node(
                disease=disease,
                certainty=cert,
                is_leading=is_lead,
                is_suppressed=suppressed,
                stage=-1,
            ))

        # ── 2. Stage nodes + trajectory + propagation + leadership edges ───
        stage_node_ids: list[str] = []

        if traj is not None and traj.snapshots:
            snapshots = traj.snapshots
            for i, snap in enumerate(snapshots):
                stage_id = f"stage:{snap.stage}:{snap.stage_name}"
                stage_node = make_stage_node(
                    stage=snap.stage,
                    stage_name=snap.stage_name,
                    fsm_state=snap.state.value,
                    certainty=snap.max_certainty,
                    contradiction_load=snap.contradiction_load,
                    ambiguity_index=snap.ambiguity_index,
                    active_rule_count=snap.active_rule_count,
                    safety_triggered=snap.safety_triggered,
                )
                g.add_node(stage_node)
                stage_node_ids.append(stage_id)

                # Propagation edges: stage → all hypotheses
                snap_leading = snap.leading_disease
                snap_second  = _estimate_second_certainty(
                    snap.max_certainty, snap.certainty_gap
                )
                snap_field   = _estimate_field_certainty(snap.max_certainty, snap_second)

                for disease in KNOWN_DISEASES:
                    hyp_id = f"hyp:{disease}"
                    cert   = snap.max_certainty if disease == snap_leading else snap_field
                    g.add_edge(make_propagation_edge(
                        stage_id=stage_id,
                        hypothesis_id=hyp_id,
                        certainty=cert,
                        stage=snap.stage,
                    ))

                # Leadership edge: stage → leading hypothesis
                g.add_edge(make_leadership_edge(
                    stage_id=stage_id,
                    hypothesis_id=f"hyp:{snap_leading}",
                    certainty=snap.max_certainty,
                    certainty_gap=snap.certainty_gap,
                    stage=snap.stage,
                ))

            # Trajectory edges: stage(n) → stage(n+1)
            deltas = traj.deltas()
            for i, delta in enumerate(deltas):
                from_id = stage_node_ids[i]
                to_id   = stage_node_ids[i + 1]
                g.add_edge(make_trajectory_edge(
                    from_stage_id=from_id,
                    to_stage_id=to_id,
                    certainty_delta=delta.certainty_delta,
                    contradiction_delta=delta.contradiction_delta,
                    entropy_delta=delta.entropy_delta,
                    state_changed=delta.state_changed,
                    stage=delta.to_stage,
                ))
        else:
            # No trajectory — create a single terminal stage node
            stage_id = "stage:0:terminal"
            g.add_node(make_stage_node(
                stage=0,
                stage_name="terminal",
                fsm_state=result.final_state or "UNKNOWN",
                certainty=result.max_certainty,
                contradiction_load=result.contradiction_load,
                ambiguity_index=result.ambiguity_index,
                active_rule_count=0,
            ))
            stage_node_ids = [stage_id]

        # ── 3. Contradiction nodes + edges ─────────────────────────────────
        load = result.contradiction_load
        if load > 0.0:
            if conflict is not None:
                # Detailed per-pair contradiction nodes from ConflictAnalysisResult
                for tension in conflict.pair_tensions:
                    if tension.cumulative_penalty > 0.0:
                        contra_node = make_contradiction_node(
                            source_disease=tension.source_disease,
                            target_disease=tension.target_disease,
                            load=load,
                            penalty_weight=tension.cumulative_penalty,
                            stage=2,
                        )
                        g.add_node(contra_node)
                        # Contradiction edge: source_hyp → contra_node → target_hyp
                        g.add_edge(make_contradiction_edge(
                            source_hyp_id=f"hyp:{tension.source_disease}",
                            target_hyp_id=f"hyp:{tension.target_disease}",
                            load=load,
                            penalty_weight=tension.cumulative_penalty,
                            stage=2,
                        ))
                        # Suppression edge: contra_node → dampened target
                        dampening = tension.cumulative_penalty / max(load, 0.001)
                        g.add_edge(make_suppression_edge(
                            contradiction_id=contra_node.node_id,
                            hypothesis_id=f"hyp:{tension.target_disease}",
                            dampening_factor=dampening,
                            stage=3,
                        ))
            else:
                # Aggregate contradiction node
                agg_node = make_contradiction_node(load=load, stage=2)
                g.add_node(agg_node)
                # Suppression edges: aggregate → all non-leading hypotheses
                for disease in KNOWN_DISEASES:
                    if disease != leading:
                        hyp_node = g.get_node(f"hyp:{disease}")
                        if hyp_node and hyp_node.activation == ActivationState.SUPPRESSED:
                            g.add_edge(make_suppression_edge(
                                contradiction_id="contra:aggregate",
                                hypothesis_id=f"hyp:{disease}",
                                dampening_factor=load,
                                stage=3,
                            ))

            # Competition edge between leading and estimated second hypothesis
            # (second disease is unknown without full distribution, so skip unless
            #  conflict provides it)
            if conflict is not None and conflict.highest_tension_pair:
                htp = conflict.highest_tension_pair
                g.add_edge(make_competition_edge(
                    hyp_a_id=f"hyp:{htp.source_disease}",
                    hyp_b_id=f"hyp:{htp.target_disease}",
                    certainty_gap=result.certainty_gap,
                    stage=4,
                ))

        # ── 4. Escalation node + escalation edge ──────────────────────────
        rec = result.recommendation or "UNKNOWN"
        esc_node = make_escalation_node(
            recommendation=rec,
            certainty=result.max_certainty,
            stage=7,
        )
        g.add_node(esc_node)

        if stage_node_ids:
            last_stage_id = stage_node_ids[-1]
            biopsy_recs = {"BIOPSY_RECOMMENDED", "HIGH_RISK_CONTRADICTION"}
            severity = (
                min(load / 0.40, 1.0) if rec in biopsy_recs
                else result.max_certainty
            )
            g.add_edge(make_escalation_edge(
                stage_id=last_stage_id,
                escalation_id=esc_node.node_id,
                trigger_reason=result.decision_rationale[:120] if result.decision_rationale else rec,
                severity=severity,
                stage=7,
            ))

        return g
