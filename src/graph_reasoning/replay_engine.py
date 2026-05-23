"""
ReplayEngine — stepwise reasoning trajectory playback controller.

The ReplayEngine drives stage-by-stage reconstruction of the diagnostic
reasoning process by stepping through a TrajectoryGraph's ordered snapshots.

This is the primary infrastructure that will later power:
  · Frontend trajectory explorer (step through reasoning interactively)
  · Publication animations (reasoning graph evolution visualisation)
  · Demonstration workflows (synthetic case walkthroughs)
  · Audit replay (verify reasoning decisions at any intermediate stage)
  · Counterfactual analysis (freeze at stage N, modify, continue)

Replay semantics
----------------
The engine maintains a cursor pointing at the current snapshot. Each step()
advances the cursor by one stage and returns the snapshot at that position.
The engine does not re-execute reasoning — it replays pre-computed graph states.

Replay modes
------------
  STANDARD   — step through all snapshots in order
  FAST       — jump directly to a specific stage index
  ANNOTATED  — include delta annotations between each step

Replay events
-------------
Each step emits a ReplayEvent that describes the graph change at that step:
  · SNAPSHOT_ADVANCED     — cursor moved to next snapshot
  · CONTRADICTION_EMERGED — contradiction load became non-zero
  · LEADERSHIP_CHANGED    — leading disease changed at this step
  · STATE_TRANSITIONED    — FSM state changed at this step
  · CERTAINTY_PEAKED      — certainty reached its maximum this step
  · ESCALATION_TRIGGERED  — safety gate or escalation threshold fired
  · TRAJECTORY_COMPLETE   — all stages have been replayed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from src.graph_reasoning.graph_snapshot import GraphSnapshot
from src.graph_reasoning.trajectory_graph import TrajectoryGraph, TrajectoryDelta


# ── Replay events ─────────────────────────────────────────────────────────────

class ReplayEventType(str, Enum):
    SNAPSHOT_ADVANCED      = "snapshot_advanced"
    CONTRADICTION_EMERGED  = "contradiction_emerged"
    LEADERSHIP_CHANGED     = "leadership_changed"
    STATE_TRANSITIONED     = "state_transitioned"
    CERTAINTY_PEAKED       = "certainty_peaked"
    ESCALATION_TRIGGERED   = "escalation_triggered"
    TRAJECTORY_COMPLETE    = "trajectory_complete"


@dataclass(frozen=True)
class ReplayEvent:
    """
    An annotated event emitted during trajectory replay.

    Captures what changed at the current step and why it is clinically
    significant.
    """

    event_type:       ReplayEventType
    step_index:       int             # cursor position (0-indexed)
    snapshot:         GraphSnapshot
    delta:            TrajectoryDelta | None   # None for step 0 and final step
    description:      str
    clinical_note:    str            # what this means clinically
    meta:             dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type":    self.event_type.value,
            "step_index":    self.step_index,
            "stage":         self.snapshot.metrics.stage,
            "stage_name":    self.snapshot.metrics.stage_name,
            "description":   self.description,
            "clinical_note": self.clinical_note,
            "metrics":       self.snapshot.metrics.to_dict(),
            "meta":          self.meta,
        }


# ── Replay result ─────────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    """
    Complete replay outcome — all events from start to terminal stage.
    """

    case_id:         str
    total_steps:     int
    events:          list[ReplayEvent]
    final_snapshot:  GraphSnapshot | None
    significant_events: list[ReplayEvent]   # non-SNAPSHOT_ADVANCED events

    @property
    def contradiction_emergence_step(self) -> ReplayEvent | None:
        for e in self.events:
            if e.event_type == ReplayEventType.CONTRADICTION_EMERGED:
                return e
        return None

    @property
    def escalation_event(self) -> ReplayEvent | None:
        for e in self.events:
            if e.event_type == ReplayEventType.ESCALATION_TRIGGERED:
                return e
        return None

    @property
    def leadership_changes(self) -> list[ReplayEvent]:
        return [e for e in self.events if e.event_type == ReplayEventType.LEADERSHIP_CHANGED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id":     self.case_id,
            "total_steps": self.total_steps,
            "events":      [e.to_dict() for e in self.events],
            "significant_events": [e.to_dict() for e in self.significant_events],
            "final_stage": (
                self.final_snapshot.metrics.to_dict()
                if self.final_snapshot else None
            ),
        }


# ── Replay engine ─────────────────────────────────────────────────────────────

class ReplayEngine:
    """
    Stepwise reasoning trajectory replay controller.

    The engine wraps a TrajectoryGraph and provides a cursor-based playback
    interface. It detects clinically significant events at each step and
    emits annotated ReplayEvent objects.

    Parameters
    ----------
    trajectory:
        A TrajectoryGraph built from a PipelineResult.

    Usage
    -----
    engine = ReplayEngine(trajectory)
    while not engine.is_complete:
        event = engine.step()
        print(event.description)

    Or replay all at once:
    result = engine.replay_all()
    """

    def __init__(self, trajectory: TrajectoryGraph) -> None:
        self._traj    = trajectory
        self._cursor  = -1          # -1 = before start
        self._deltas  = trajectory.deltas()
        self._peak_certainty_stage: int | None = self._find_peak_certainty_stage()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def trajectory(self) -> TrajectoryGraph:
        return self._traj

    @property
    def cursor(self) -> int:
        """Current position (0-indexed). -1 means not started."""
        return self._cursor

    @property
    def is_started(self) -> bool:
        return self._cursor >= 0

    @property
    def is_complete(self) -> bool:
        return self._cursor >= self._traj.length - 1

    @property
    def current_snapshot(self) -> GraphSnapshot | None:
        """The snapshot at the current cursor position."""
        if self._cursor < 0 or self._cursor >= self._traj.length:
            return None
        return self._traj.get(self._cursor)

    @property
    def steps_remaining(self) -> int:
        return max(0, self._traj.length - 1 - self._cursor)

    # ── Playback controls ─────────────────────────────────────────────────────

    def reset(self) -> "ReplayEngine":
        """Reset cursor to before-start position."""
        self._cursor = -1
        return self

    def step(self) -> ReplayEvent:
        """
        Advance the cursor by one step and return the annotated ReplayEvent.

        Raises
        ------
        StopIteration:
            When the trajectory is already complete.
        RuntimeError:
            When the trajectory is empty.
        """
        if self._traj.is_empty:
            raise RuntimeError("Cannot step through an empty trajectory.")
        if self.is_complete:
            raise StopIteration("Trajectory replay is complete.")

        self._cursor += 1
        snap  = self._traj.get(self._cursor)
        delta = self._deltas[self._cursor - 1] if self._cursor > 0 else None

        return self._classify_event(self._cursor, snap, delta)

    def seek(self, index: int) -> GraphSnapshot:
        """
        Jump directly to a specific snapshot index.

        Parameters
        ----------
        index:
            Target snapshot index (0 to length-1).
        """
        if index < 0 or index >= self._traj.length:
            raise IndexError(
                f"Snapshot index {index} out of range [0, {self._traj.length - 1}]."
            )
        self._cursor = index
        return self._traj.get(self._cursor)

    def __iter__(self) -> Iterator[ReplayEvent]:
        """Iterate over all steps from the beginning."""
        self.reset()
        while not self.is_complete:
            yield self.step()

    # ── Full replay ───────────────────────────────────────────────────────────

    def replay_all(self) -> ReplayResult:
        """
        Execute a complete replay from start to finish.

        Returns a ReplayResult containing all events and a list of
        significant events (contradictions, escalations, leadership changes).
        """
        self.reset()
        events: list[ReplayEvent] = []
        for event in self:
            events.append(event)

        # Final TRAJECTORY_COMPLETE event
        final_snap = self._traj.last()
        if final_snap:
            events.append(ReplayEvent(
                event_type=ReplayEventType.TRAJECTORY_COMPLETE,
                step_index=self._cursor,
                snapshot=final_snap,
                delta=None,
                description=(
                    f"Reasoning trajectory complete at stage {final_snap.metrics.stage}. "
                    f"Recommendation: {self._traj.run_id or '?'}"
                ),
                clinical_note=(
                    f"Terminal state: {final_snap.metrics.fsm_state}. "
                    f"Leading disease: {final_snap.metrics.leading_disease} "
                    f"({final_snap.metrics.certainty:.3f} certainty)."
                ),
            ))

        significant = [
            e for e in events
            if e.event_type != ReplayEventType.SNAPSHOT_ADVANCED
        ]
        return ReplayResult(
            case_id=self._traj.case_id,
            total_steps=len(events),
            events=events,
            final_snapshot=final_snap,
            significant_events=significant,
        )

    # ── Event classification ──────────────────────────────────────────────────

    def _classify_event(
        self,
        index: int,
        snap: GraphSnapshot,
        delta: TrajectoryDelta | None,
    ) -> ReplayEvent:
        """
        Classify a snapshot step into a typed ReplayEvent with clinical annotation.
        """
        metrics = snap.metrics

        # Check for clinically significant transitions
        if (
            delta is not None
            and delta.contra_delta > 0.0
            and (index == 0 or self._delta_at(index - 1) is None
                 or self._delta_at(index - 1).contra_delta == 0.0)
        ):
            return ReplayEvent(
                event_type=ReplayEventType.CONTRADICTION_EMERGED,
                step_index=index,
                snapshot=snap,
                delta=delta,
                description=(
                    f"Stage {metrics.stage}: Contradiction emerged — "
                    f"load={metrics.contradiction_load:.3f}."
                ),
                clinical_note=(
                    "Cross-disease evidential conflict detected. "
                    f"Contradiction load of {metrics.contradiction_load:.3f} "
                    + ("exceeds escalation ceiling (≥0.40) — biopsy indicated."
                       if metrics.contradiction_load >= 0.40
                       else "does not yet require mandatory escalation.")
                ),
                meta={"contradiction_load": metrics.contradiction_load},
            )

        if delta is not None and delta.leader_changed:
            return ReplayEvent(
                event_type=ReplayEventType.LEADERSHIP_CHANGED,
                step_index=index,
                snapshot=snap,
                delta=delta,
                description=(
                    f"Stage {metrics.stage}: Leadership changed — "
                    f"{delta.from_leader} → {delta.to_leader}."
                ),
                clinical_note=(
                    f"The leading hypothesis shifted from {delta.from_leader.replace('_', ' ')} "
                    f"to {delta.to_leader.replace('_', ' ')} at this reasoning stage. "
                    "Evidence re-weighting caused the differential to shift."
                ),
                meta={
                    "from_leader": delta.from_leader,
                    "to_leader":   delta.to_leader,
                },
            )

        if delta is not None and delta.state_changed:
            return ReplayEvent(
                event_type=ReplayEventType.STATE_TRANSITIONED,
                step_index=index,
                snapshot=snap,
                delta=delta,
                description=(
                    f"Stage {metrics.stage}: FSM transitioned — "
                    f"{delta.from_state} → {delta.to_state}."
                ),
                clinical_note=(
                    f"Diagnostic state advanced from {delta.from_state} "
                    f"to {delta.to_state}. "
                    + self._state_transition_note(delta.from_state, delta.to_state)
                ),
                meta={
                    "from_state": delta.from_state,
                    "to_state":   delta.to_state,
                },
            )

        if index == self._peak_certainty_stage and index > 0:
            return ReplayEvent(
                event_type=ReplayEventType.CERTAINTY_PEAKED,
                step_index=index,
                snapshot=snap,
                delta=delta,
                description=(
                    f"Stage {metrics.stage}: Certainty peaked at {metrics.certainty:.3f}."
                ),
                clinical_note=(
                    f"Maximum diagnostic certainty of {metrics.certainty:.3f} "
                    f"reached for {metrics.leading_disease.replace('_', ' ')}. "
                    + ("Certainty is sufficient for non-invasive diagnosis."
                       if metrics.certainty >= 0.72
                       else "Certainty remains below the safe-triage threshold.")
                ),
                meta={"peak_certainty": metrics.certainty},
            )

        if metrics.safety_triggered:
            return ReplayEvent(
                event_type=ReplayEventType.ESCALATION_TRIGGERED,
                step_index=index,
                snapshot=snap,
                delta=delta,
                description=(
                    f"Stage {metrics.stage}: Safety gate triggered — "
                    f"escalation active (load={metrics.contradiction_load:.3f})."
                ),
                clinical_note=(
                    "A clinical safety gate has been triggered at this stage. "
                    "The safety gate enforces mandatory biopsy escalation when "
                    "contradiction load, ambiguity, or evidence insufficiency "
                    "exceeds calibrated thresholds."
                ),
                meta={"safety_triggered": True},
            )

        # Default: standard snapshot advance
        direction = ""
        if delta is not None:
            if delta.certainty_delta > 0.02:
                direction = f"certainty +{delta.certainty_delta:.3f}"
            elif delta.certainty_delta < -0.02:
                direction = f"certainty {delta.certainty_delta:.3f}"
            else:
                direction = "certainty stable"

        return ReplayEvent(
            event_type=ReplayEventType.SNAPSHOT_ADVANCED,
            step_index=index,
            snapshot=snap,
            delta=delta,
            description=(
                f"Stage {metrics.stage} [{metrics.stage_name}]: "
                f"leading={metrics.leading_disease} "
                f"cert={metrics.certainty:.3f} "
                + (f"({direction})" if direction else "")
            ),
            clinical_note=(
                f"Reasoning in state {metrics.fsm_state}. "
                f"{metrics.active_rule_count} rules active. "
                f"Entropy={metrics.ambiguity_index:.3f} bits."
            ),
        )

    def _delta_at(self, index: int) -> TrajectoryDelta | None:
        """Return delta at a given step index, or None if out of range."""
        if index < 0 or index >= len(self._deltas):
            return None
        return self._deltas[index]

    def _find_peak_certainty_stage(self) -> int | None:
        """Pre-compute the index of the peak certainty snapshot."""
        if self._traj.is_empty:
            return None
        snap = self._traj.max_certainty_stage()
        if snap is None:
            return None
        for i, s in enumerate(self._traj.snapshots):
            if s.snapshot_index == snap.snapshot_index:
                return i
        return None

    @staticmethod
    def _state_transition_note(from_state: str, to_state: str) -> str:
        """Generate a clinical note for a specific FSM state transition."""
        transitions = {
            ("INITIAL_EVIDENCE", "PARTIAL_ALIGNMENT"):
                "Initial evidence has begun accumulating toward one hypothesis.",
            ("PARTIAL_ALIGNMENT", "REINFORCING_ALIGNMENT"):
                "Multiple rules now converge on the leading hypothesis.",
            ("REINFORCING_ALIGNMENT", "CERTAINTY_STABILIZATION"):
                "Certainty is stabilising — hypothesis is separating from competitors.",
            ("CERTAINTY_STABILIZATION", "SAFE_TRIAGE"):
                "Certainty sufficient for safe non-invasive triage.",
            ("PARTIAL_ALIGNMENT", "CONTRADICTION_DETECTED"):
                "Cross-disease contradictions have been detected.",
            ("CONTRADICTION_DETECTED", "BIOPSY_ESCALATION"):
                "Contradiction load exceeded the safety ceiling — biopsy required.",
            ("REINFORCING_ALIGNMENT", "AMBIGUITY_ESCALATION"):
                "Shannon entropy exceeds 1.5 bits — differential is ambiguous.",
            ("AMBIGUITY_ESCALATION", "BIOPSY_ESCALATION"):
                "Ambiguity unresolved — biopsy escalation required.",
        }
        key = (from_state, to_state)
        return transitions.get(key, f"Transition from {from_state} to {to_state}.")
