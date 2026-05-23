"""
CertaintyGraph — certainty evolution and entropy tracking across pipeline stages.

The CertaintyGraph captures the temporal dynamics of the certainty distribution
as it evolves from the first evidence-activation stage through to the terminal
escalation decision.

Key metrics tracked
-------------------
  leading_certainty   — max_certainty of the dominant hypothesis at each stage
  certainty_gap       — top1 − top2 gap at each stage
  ambiguity_index     — Shannon entropy (bits) at each stage
  leading_disease     — which disease is leading at each stage
  dampening_active    — whether contradiction dampening suppressed certainty

Analysis capabilities
---------------------
  · Convergence index: final certainty / peak certainty
  · Stabilisation stage: first stage where gap > 0.20 (certainty stabilised)
  · Entropy reduction: total entropy drop from maximum to final
  · Peak certainty stage: the stage with maximum certainty
  · Leadership stability: how often the leading disease changes
  · Oscillation detection: direction reversals in certainty series

The CertaintyGraph is derived from a DiagnosticTrajectory and does not
re-execute any reasoning logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Per-stage certainty record ────────────────────────────────────────────────

@dataclass(frozen=True)
class CertaintyPoint:
    """
    Certainty snapshot at a single pipeline stage.

    Attributes
    ----------
    stage:
        Pipeline stage index.
    stage_name:
        Human-readable stage name.
    leading_disease:
        The hypothesis with highest certainty at this stage.
    certainty:
        Leading hypothesis certainty [0, 1].
    certainty_gap:
        Gap between leading and second hypothesis [0, 1].
    ambiguity_index:
        Shannon entropy in bits.
    dampening_active:
        Whether contradiction load exceeded the dampening threshold (0.20).
    contradiction_load:
        Contradiction load at this stage.
    active_rule_count:
        Number of diagnostic rules active at this stage.
    """

    stage:              int
    stage_name:         str
    leading_disease:    str
    certainty:          float
    certainty_gap:      float
    ambiguity_index:    float
    dampening_active:   bool
    contradiction_load: float
    active_rule_count:  int

    @property
    def is_stable(self) -> bool:
        """True if this point meets the certainty stabilisation criteria."""
        return self.certainty >= 0.55 and self.certainty_gap >= 0.20

    @property
    def is_ambiguous(self) -> bool:
        """True if entropy exceeds 1.5 bits (ambiguity escalation threshold)."""
        return self.ambiguity_index > 1.5

    @property
    def is_convergent(self) -> bool:
        """True if gap > 0.20 (hypothesis is pulling away from competitors)."""
        return self.certainty_gap >= 0.20

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage":              self.stage,
            "stage_name":         self.stage_name,
            "leading_disease":    self.leading_disease,
            "certainty":          self.certainty,
            "certainty_gap":      self.certainty_gap,
            "ambiguity_index":    self.ambiguity_index,
            "dampening_active":   self.dampening_active,
            "contradiction_load": self.contradiction_load,
            "active_rule_count":  self.active_rule_count,
            "is_stable":          self.is_stable,
            "is_ambiguous":       self.is_ambiguous,
            "is_convergent":      self.is_convergent,
        }


# ── CertaintyGraph ────────────────────────────────────────────────────────────

class CertaintyGraph:
    """
    Time-series representation of certainty evolution across pipeline stages.

    Built from a DiagnosticTrajectory, the CertaintyGraph provides a
    structured view of how diagnostic confidence developed, where it peaked,
    and whether it converged or oscillated.

    Parameters
    ----------
    case_id:
        Clinical case identifier.
    recommendation:
        Terminal triage recommendation from PipelineResult.
    """

    # Constant thresholds mirroring the pipeline configuration
    DAMPENING_THRESHOLD: float = 0.20
    STABILITY_GAP_THRESHOLD: float = 0.20
    STABILITY_CERT_THRESHOLD: float = 0.55
    AMBIGUITY_THRESHOLD: float = 1.50  # bits
    MAX_ENTROPY_6CLASS: float = math.log2(6)  # ≈ 2.585

    def __init__(self, case_id: str, recommendation: str) -> None:
        self._case_id        = case_id
        self._recommendation = recommendation
        self._points: list[CertaintyPoint] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def recommendation(self) -> str:
        return self._recommendation

    @property
    def length(self) -> int:
        return len(self._points)

    @property
    def is_empty(self) -> bool:
        return len(self._points) == 0

    def points(self) -> list[CertaintyPoint]:
        """Return all certainty points in stage order."""
        return list(self._points)

    def first(self) -> CertaintyPoint | None:
        return self._points[0] if self._points else None

    def last(self) -> CertaintyPoint | None:
        return self._points[-1] if self._points else None

    # ── Time series extraction ────────────────────────────────────────────────

    def certainty_series(self) -> list[float]:
        """Raw certainty values in stage order."""
        return [p.certainty for p in self._points]

    def gap_series(self) -> list[float]:
        """Certainty gap values in stage order."""
        return [p.certainty_gap for p in self._points]

    def entropy_series(self) -> list[float]:
        """Ambiguity index (bits) values in stage order."""
        return [p.ambiguity_index for p in self._points]

    def stage_labels(self) -> list[str]:
        """Stage name labels for axis annotation."""
        return [f"{p.stage}:{p.stage_name}" for p in self._points]

    def leading_disease_series(self) -> list[str]:
        """Leading disease at each stage — useful for tracking leadership changes."""
        return [p.leading_disease for p in self._points]

    # ── Analysis ──────────────────────────────────────────────────────────────

    def peak_certainty(self) -> CertaintyPoint | None:
        """The stage with maximum leading certainty."""
        if not self._points:
            return None
        return max(self._points, key=lambda p: p.certainty)

    def peak_gap(self) -> CertaintyPoint | None:
        """The stage with maximum certainty gap."""
        if not self._points:
            return None
        return max(self._points, key=lambda p: p.certainty_gap)

    def stabilisation_stage(self) -> CertaintyPoint | None:
        """
        First stage where certainty >= 0.55 AND gap >= 0.20 simultaneously.
        Returns None if certainty never stabilised.
        """
        for p in self._points:
            if p.is_stable:
                return p
        return None

    def ambiguity_peak(self) -> CertaintyPoint | None:
        """The stage with highest entropy (maximum ambiguity)."""
        if not self._points:
            return None
        return max(self._points, key=lambda p: p.ambiguity_index)

    def convergence_index(self) -> float:
        """
        Ratio of final certainty to peak certainty.
        1.0 = perfect convergence. < 0.70 = significant decay from peak.
        """
        if not self._points:
            return 0.0
        peak  = max(p.certainty for p in self._points)
        final = self._points[-1].certainty
        return final / peak if peak > 0 else 0.0

    def entropy_reduction(self) -> float:
        """
        Total entropy drop from the maximum entropy stage to the final stage.
        Positive = entropy decreased (system became less ambiguous).
        """
        if not self._points:
            return 0.0
        peak_entropy  = max(p.ambiguity_index for p in self._points)
        final_entropy = self._points[-1].ambiguity_index
        return peak_entropy - final_entropy

    def oscillation_count(self) -> int:
        """Direction reversals in certainty series (mirrors TrajectoryValidator)."""
        series = self.certainty_series()
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

    def leadership_changes(self) -> int:
        """Count how many times the leading disease changed across stages."""
        diseases = self.leading_disease_series()
        if len(diseases) < 2:
            return 0
        return sum(1 for i in range(1, len(diseases)) if diseases[i] != diseases[i - 1])

    def dampening_stages(self) -> list[CertaintyPoint]:
        """Return stages where contradiction dampening was active."""
        return [p for p in self._points if p.dampening_active]

    def was_dampened(self) -> bool:
        """True if contradiction dampening occurred at any stage."""
        return any(p.dampening_active for p in self._points)

    # ── Normalised entropy for display ────────────────────────────────────────

    def normalised_entropy_series(self) -> list[float]:
        """
        Entropy series normalised to [0, 1] by dividing by log2(6).
        Useful for overlaying with certainty in visualisations.
        """
        return [p.ambiguity_index / self.MAX_ENTROPY_6CLASS for p in self._points]

    # ── Data management ───────────────────────────────────────────────────────

    def _append(self, point: CertaintyPoint) -> None:
        self._points.append(point)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        if self.is_empty:
            return f"CertaintyGraph[case={self._case_id}] empty"
        first = self._points[0]
        last  = self._points[-1]
        stab  = self.stabilisation_stage()
        return (
            f"CertaintyGraph[case={self._case_id}] "
            f"stages={self.length} "
            f"certainty={first.certainty:.3f}→{last.certainty:.3f} "
            f"peak={max(p.certainty for p in self._points):.3f} "
            f"convergence={self.convergence_index():.3f} "
            f"entropy_reduction={self.entropy_reduction():.3f}bits "
            f"oscillations={self.oscillation_count()} "
            f"stabilised_at_stage={stab.stage if stab else 'never'} "
            f"dampened={self.was_dampened()} "
            f"recommendation={self._recommendation}"
        )

    def to_dict(self) -> dict[str, Any]:
        stab = self.stabilisation_stage()
        return {
            "case_id":              self._case_id,
            "recommendation":       self._recommendation,
            "points":               [p.to_dict() for p in self._points],
            "certainty_series":     self.certainty_series(),
            "gap_series":           self.gap_series(),
            "entropy_series":       self.entropy_series(),
            "normalised_entropy":   self.normalised_entropy_series(),
            "convergence_index":    self.convergence_index(),
            "entropy_reduction":    self.entropy_reduction(),
            "oscillation_count":    self.oscillation_count(),
            "leadership_changes":   self.leadership_changes(),
            "was_dampened":         self.was_dampened(),
            "stabilisation_stage":  stab.stage if stab else None,
            "peak_certainty_stage": (self.peak_certainty().stage if self.peak_certainty() else None),
        }

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_result(
        cls,
        result: "PipelineResult",  # type: ignore[name-defined]
    ) -> "CertaintyGraph":
        """
        Build a CertaintyGraph from a completed PipelineResult.

        One CertaintyPoint is created per trajectory snapshot.
        When no trajectory is available, a single-point graph is built
        from the terminal result metrics.

        Parameters
        ----------
        result:
            Completed PipelineResult with embedded DiagnosticTrajectory.
        """
        cg = cls(
            case_id=result.case_id,
            recommendation=result.recommendation or "UNKNOWN",
        )

        traj = result.trajectory
        if traj is None or not traj.snapshots:
            cg._append(CertaintyPoint(
                stage=0,
                stage_name="terminal",
                leading_disease=result.leading_disease or "unknown",
                certainty=result.max_certainty,
                certainty_gap=result.certainty_gap,
                ambiguity_index=result.ambiguity_index,
                dampening_active=result.contradiction_load > cls.DAMPENING_THRESHOLD,
                contradiction_load=result.contradiction_load,
                active_rule_count=0,
            ))
            return cg

        for snap in traj.snapshots:
            cg._append(CertaintyPoint(
                stage=snap.stage,
                stage_name=snap.stage_name,
                leading_disease=snap.leading_disease,
                certainty=snap.max_certainty,
                certainty_gap=snap.certainty_gap,
                ambiguity_index=snap.ambiguity_index,
                dampening_active=snap.contradiction_load > cls.DAMPENING_THRESHOLD,
                contradiction_load=snap.contradiction_load,
                active_rule_count=snap.active_rule_count,
            ))

        return cg
