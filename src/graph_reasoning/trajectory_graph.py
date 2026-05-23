"""
TrajectoryGraph — ordered stage-by-stage graph trajectory.

A TrajectoryGraph is a sequence of GraphSnapshots, one per pipeline stage,
representing the temporal evolution of the reasoning graph as evidence
accumulates and hypotheses compete.

Each snapshot captures:
  · The complete graph state (nodes + edges) at that stage
  · Quantitative reasoning metrics (certainty, contradiction, entropy)
  · The leading hypothesis and FSM state at that moment
  · The delta (change) from the previous stage

The TrajectoryGraph provides the foundation for:
  · ReplayEngine (step through reasoning stage by stage)
  · Temporal analysis (which stage drove the most certainty change?)
  · Convergence detection (did certainty stabilise or oscillate?)
  · Contradiction emergence detection (when did the load first exceed threshold?)

Building from PipelineResult
-----------------------------
  tg = TrajectoryGraph.from_result(result)

This is the standard construction path. The TrajectoryGraph is built by
iterating over the DiagnosticTrajectory snapshots embedded in the result
and constructing a graph state for each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from src.graph_reasoning.graph_nodes import (
    KNOWN_DISEASES, make_hypothesis_node, make_stage_node,
    make_contradiction_node, make_escalation_node,
)
from src.graph_reasoning.graph_edges import (
    make_trajectory_edge, make_propagation_edge, make_leadership_edge,
    make_suppression_edge, make_escalation_edge,
)
from src.graph_reasoning.graph_snapshot import GraphSnapshot, SnapshotMetrics
from src.graph_reasoning.reasoning_graph import (
    ReasoningGraph, _estimate_second_certainty, _estimate_field_certainty,
)


# ── Stage delta ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrajectoryDelta:
    """
    Quantitative change between two consecutive trajectory snapshots.
    """

    from_index:         int
    to_index:           int
    from_stage:         int
    to_stage:           int
    certainty_delta:    float   # positive = leading certainty increased
    gap_delta:          float   # positive = gap widened (more decisive)
    contra_delta:       float   # positive = more contradiction load
    entropy_delta:      float   # positive = more ambiguous
    state_changed:      bool
    from_state:         str
    to_state:           str
    from_leader:        str
    to_leader:          str
    leader_changed:     bool

    @property
    def is_positive_step(self) -> bool:
        """True if this stage improved diagnostic clarity."""
        return self.certainty_delta > 0 and self.contra_delta <= 0

    @property
    def is_contradiction_emergence(self) -> bool:
        """True if contradiction load first appeared at this step."""
        return self.contra_delta > 0.0

    @property
    def is_convergence_step(self) -> bool:
        """True if certainty and gap both increased."""
        return self.certainty_delta > 0 and self.gap_delta > 0


# ── Trajectory graph ──────────────────────────────────────────────────────────

class TrajectoryGraph:
    """
    Ordered sequence of GraphSnapshots representing the stage-by-stage
    evolution of the reasoning graph.

    Parameters
    ----------
    case_id:
        Clinical case identifier.
    run_id:
        Pipeline run identifier.
    """

    def __init__(self, case_id: str, run_id: str) -> None:
        self._case_id   = case_id
        self._run_id    = run_id
        self._snapshots: list[GraphSnapshot] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def snapshots(self) -> list[GraphSnapshot]:
        return list(self._snapshots)

    @property
    def length(self) -> int:
        return len(self._snapshots)

    @property
    def is_empty(self) -> bool:
        return len(self._snapshots) == 0

    def __iter__(self) -> Iterator[GraphSnapshot]:
        return iter(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)

    def get(self, index: int) -> GraphSnapshot:
        """Return snapshot at ordinal index. Raises IndexError if out of range."""
        return self._snapshots[index]

    def first(self) -> GraphSnapshot | None:
        return self._snapshots[0] if self._snapshots else None

    def last(self) -> GraphSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    # ── Time series ───────────────────────────────────────────────────────────

    def certainty_series(self) -> list[tuple[int, float]]:
        """Return (stage, max_certainty) pairs across the trajectory."""
        return [(s.metrics.stage, s.metrics.certainty) for s in self._snapshots]

    def gap_series(self) -> list[tuple[int, float]]:
        """Return (stage, certainty_gap) pairs."""
        return [(s.metrics.stage, s.metrics.certainty_gap) for s in self._snapshots]

    def contradiction_series(self) -> list[tuple[int, float]]:
        """Return (stage, contradiction_load) pairs."""
        return [(s.metrics.stage, s.metrics.contradiction_load) for s in self._snapshots]

    def entropy_series(self) -> list[tuple[int, float]]:
        """Return (stage, ambiguity_index) pairs (bits)."""
        return [(s.metrics.stage, s.metrics.ambiguity_index) for s in self._snapshots]

    def fsm_state_series(self) -> list[tuple[int, str]]:
        """Return (stage, fsm_state) pairs."""
        return [(s.metrics.stage, s.metrics.fsm_state) for s in self._snapshots]

    def leading_disease_series(self) -> list[tuple[int, str]]:
        """Return (stage, leading_disease) pairs — tracks leadership changes."""
        return [(s.metrics.stage, s.metrics.leading_disease) for s in self._snapshots]

    # ── Delta analysis ────────────────────────────────────────────────────────

    def deltas(self) -> list[TrajectoryDelta]:
        """Compute stage-to-stage quantitative deltas."""
        if len(self._snapshots) < 2:
            return []
        result = []
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            result.append(TrajectoryDelta(
                from_index=i - 1,
                to_index=i,
                from_stage=prev.metrics.stage,
                to_stage=curr.metrics.stage,
                certainty_delta=curr.metrics.certainty - prev.metrics.certainty,
                gap_delta=curr.metrics.certainty_gap - prev.metrics.certainty_gap,
                contra_delta=curr.metrics.contradiction_load - prev.metrics.contradiction_load,
                entropy_delta=curr.metrics.ambiguity_index - prev.metrics.ambiguity_index,
                state_changed=curr.metrics.fsm_state != prev.metrics.fsm_state,
                from_state=prev.metrics.fsm_state,
                to_state=curr.metrics.fsm_state,
                from_leader=prev.metrics.leading_disease,
                to_leader=curr.metrics.leading_disease,
                leader_changed=curr.metrics.leading_disease != prev.metrics.leading_disease,
            ))
        return result

    def max_certainty_stage(self) -> GraphSnapshot | None:
        """Return the snapshot with the highest certainty (peak of the series)."""
        if not self._snapshots:
            return None
        return max(self._snapshots, key=lambda s: s.metrics.certainty)

    def max_contradiction_stage(self) -> GraphSnapshot | None:
        """Return the snapshot with the highest contradiction load."""
        if not self._snapshots:
            return None
        return max(self._snapshots, key=lambda s: s.metrics.contradiction_load)

    def contradiction_emergence_stage(self) -> GraphSnapshot | None:
        """
        Return the first snapshot where contradiction load became non-zero.
        Returns None if no contradictions were ever active.
        """
        for s in self._snapshots:
            if s.metrics.contradiction_load > 0.0:
                return s
        return None

    def convergence_index(self) -> float:
        """
        Measure of certainty trajectory convergence: ratio of the final
        certainty to the peak certainty. 1.0 = perfect convergence (no
        decay from peak). < 0.70 = significant convergence failure.
        """
        if not self._snapshots:
            return 0.0
        peak = max(s.metrics.certainty for s in self._snapshots)
        final = self._snapshots[-1].metrics.certainty
        return final / peak if peak > 0 else 0.0

    def oscillation_count(self) -> int:
        """Count direction reversals in the certainty series."""
        series = [s.metrics.certainty for s in self._snapshots]
        if len(series) < 3:
            return 0
        directions = [
            1 if series[i] > series[i - 1] else (-1 if series[i] < series[i - 1] else 0)
            for i in range(1, len(series))
        ]
        return sum(
            1 for i in range(1, len(directions))
            if directions[i] != 0 and directions[i - 1] != 0
            and directions[i] != directions[i - 1]
        )

    # ── Append (used by builder) ──────────────────────────────────────────────

    def _append(self, snapshot: GraphSnapshot) -> None:
        self._snapshots.append(snapshot)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        if self.is_empty:
            return f"TrajectoryGraph[case={self._case_id}] empty"
        first = self._snapshots[0].metrics
        last  = self._snapshots[-1].metrics
        return (
            f"TrajectoryGraph[case={self._case_id}] "
            f"stages={self.length} "
            f"certainty={first.certainty:.3f}→{last.certainty:.3f} "
            f"contradiction={last.contradiction_load:.3f} "
            f"entropy={last.ambiguity_index:.3f}bits "
            f"oscillations={self.oscillation_count()} "
            f"convergence={self.convergence_index():.3f}"
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_result(
        cls,
        result: "PipelineResult",  # type: ignore[name-defined]
    ) -> "TrajectoryGraph":
        """
        Build a TrajectoryGraph from a completed PipelineResult.

        One GraphSnapshot is created per trajectory stage. Each snapshot
        captures a complete graph state at that pipeline moment using a
        temporary ReasoningGraph built from the snapshot's metrics.

        Parameters
        ----------
        result:
            Completed PipelineResult with an embedded DiagnosticTrajectory.
        """
        tg = cls(case_id=result.case_id, run_id=result.run_id)

        traj = result.trajectory
        if traj is None or not traj.snapshots:
            # Build a single-snapshot trajectory from terminal result
            metrics = SnapshotMetrics(
                stage=0,
                stage_name="terminal",
                certainty=result.max_certainty,
                certainty_gap=result.certainty_gap,
                contradiction_load=result.contradiction_load,
                ambiguity_index=result.ambiguity_index,
                leading_disease=result.leading_disease or "unknown",
                fsm_state=result.final_state or "UNKNOWN",
                active_rule_count=0,
                tier_a_count=0,
                safety_triggered=False,
                triage_so_far=result.recommendation,
            )
            # Build a minimal graph for this terminal state
            g = ReasoningGraph(case_id=result.case_id, run_id=result.run_id)
            _populate_graph_from_snapshot_metrics(g, result, metrics)
            tg._append(g.snapshot(0, metrics, "Terminal reasoning state"))
            return tg

        # Build one graph + snapshot per trajectory stage
        for i, snap in enumerate(traj.snapshots):
            metrics = SnapshotMetrics(
                stage=snap.stage,
                stage_name=snap.stage_name,
                certainty=snap.max_certainty,
                certainty_gap=snap.certainty_gap,
                contradiction_load=snap.contradiction_load,
                ambiguity_index=snap.ambiguity_index,
                leading_disease=snap.leading_disease,
                fsm_state=snap.state.value,
                active_rule_count=snap.active_rule_count,
                tier_a_count=snap.tier_a_count,
                safety_triggered=snap.safety_triggered,
                triage_so_far=snap.triage_so_far,
            )

            # Build a graph representing the state at this snapshot
            g = ReasoningGraph(case_id=result.case_id, run_id=result.run_id)
            _populate_graph_from_snapshot_metrics(g, result, metrics)

            tg._append(g.snapshot(
                snapshot_index=i,
                metrics=metrics,
                delta_description=snap.delta_description,
            ))

        return tg


# ── Internal builder helper ───────────────────────────────────────────────────

def _populate_graph_from_snapshot_metrics(
    g: ReasoningGraph,
    result: "PipelineResult",  # type: ignore[name-defined]
    metrics: SnapshotMetrics,
) -> None:
    """
    Populate a ReasoningGraph with nodes and edges reflecting a single
    snapshot's metrics. Used internally by TrajectoryGraph.from_result().
    """
    leading    = metrics.leading_disease
    load       = metrics.contradiction_load
    is_dampened = load > 0.20

    second_cert = _estimate_second_certainty(metrics.certainty, metrics.certainty_gap)
    field_cert  = _estimate_field_certainty(metrics.certainty, second_cert)

    # Hypothesis nodes
    for disease in KNOWN_DISEASES:
        is_lead = (disease == leading)
        cert    = metrics.certainty if is_lead else field_cert
        suppressed = is_dampened and not is_lead
        g.add_node(make_hypothesis_node(
            disease=disease,
            certainty=cert,
            is_leading=is_lead,
            is_suppressed=suppressed,
            stage=metrics.stage,
        ))

    # Stage node
    stage_id = f"stage:{metrics.stage}:{metrics.stage_name}"
    g.add_node(make_stage_node(
        stage=metrics.stage,
        stage_name=metrics.stage_name,
        fsm_state=metrics.fsm_state,
        certainty=metrics.certainty,
        contradiction_load=load,
        ambiguity_index=metrics.ambiguity_index,
        active_rule_count=metrics.active_rule_count,
        safety_triggered=metrics.safety_triggered,
    ))

    # Leadership edge
    g.add_edge(make_leadership_edge(
        stage_id=stage_id,
        hypothesis_id=f"hyp:{leading}",
        certainty=metrics.certainty,
        certainty_gap=metrics.certainty_gap,
        stage=metrics.stage,
    ))

    # Propagation edges to all hypotheses
    for disease in KNOWN_DISEASES:
        hyp_id = f"hyp:{disease}"
        cert   = metrics.certainty if disease == leading else field_cert
        g.add_edge(make_propagation_edge(
            stage_id=stage_id,
            hypothesis_id=hyp_id,
            certainty=cert,
            stage=metrics.stage,
        ))

    # Contradiction node (aggregate if load > 0)
    if load > 0.0:
        contra_node = make_contradiction_node(load=load, stage=metrics.stage)
        g.add_node(contra_node)

    # Escalation node (terminal)
    rec = result.recommendation or "UNKNOWN"
    g.add_node(make_escalation_node(
        recommendation=rec,
        certainty=result.max_certainty,
        stage=7,
    ))
