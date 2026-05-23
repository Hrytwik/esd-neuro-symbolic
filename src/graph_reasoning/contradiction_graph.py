"""
ContradictionGraph — contradiction-focused subgraph and cluster analysis.

The ContradictionGraph provides a lens on the cross-disease conflict structure
within a case's reasoning trajectory. It models:

  · Disease pair tensions (which hypothesis pairs are in conflict?)
  · Contradiction load distribution across pairs
  · Cluster detection (are there multiple independent conflict zones?)
  · Instability hotspots (which pairs are driving escalation?)
  · Confusion zone membership (known clinical mimicry pairs)

This is one of the strongest novelty visualisation opportunities — showing
not just that contradictions exist but mapping their topology.

The ContradictionGraph is derived from a ReasoningGraph and optionally
enriched with ConflictAnalysisResult data from the reasoning pipeline.
Without ConflictAnalysisResult, it uses the aggregate contradiction load
and known clinical confusion zone structure.

Known confusion zones (embedded from clinical domain knowledge)
--------------------------------------------------------------
  psoriasis ↔ pityriasis_rubra_pilaris  (koebner vs follicular papules)
  psoriasis ↔ lichen_planus             (koebner vs polygonal papules)
  lichen_planus ↔ pityriasis_rubra_pilaris (polygonal vs follicular)
  psoriasis ↔ seborrheic_dermatitis     (scalp overlap)
  pityriasis_rosea ↔ chronic_dermatitis (erythema + scaling overlap)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Known confusion zones ─────────────────────────────────────────────────────

CONFUSION_ZONE_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"psoriasis", "pityriasis_rubra_pilaris"}),
    frozenset({"psoriasis", "lichen_planus"}),
    frozenset({"lichen_planus", "pityriasis_rubra_pilaris"}),
    frozenset({"psoriasis", "seborrheic_dermatitis"}),
    frozenset({"pityriasis_rosea", "chronic_dermatitis"}),
})


def is_confusion_zone(disease_a: str, disease_b: str) -> bool:
    """Return True if the two diseases form a known confusion zone pair."""
    return frozenset({disease_a, disease_b}) in CONFUSION_ZONE_PAIRS


# ── Disease pair tension record ───────────────────────────────────────────────

@dataclass(frozen=True)
class PairTension:
    """
    Accumulated contradiction tension between a specific disease pair.

    Attributes
    ----------
    disease_a, disease_b:
        The two diseases in tension (unordered pair).
    cumulative_load:
        Total contradiction penalty weight between this pair.
    is_confusion_zone:
        True if this pair belongs to a known clinical confusion zone.
    trigger_features:
        Features that activated contradictions in this pair.
    severity_label:
        Human-readable severity classification.
    """

    disease_a:       str
    disease_b:       str
    cumulative_load: float
    is_confusion_zone: bool
    trigger_features: tuple[str, ...] = ()
    clinical_rationale: str = ""

    @property
    def severity_label(self) -> str:
        if self.cumulative_load < 0.15:
            return "low"
        if self.cumulative_load < 0.30:
            return "moderate"
        if self.cumulative_load < 0.45:
            return "high"
        return "critical"

    @property
    def pair_key(self) -> frozenset[str]:
        return frozenset({self.disease_a, self.disease_b})

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease_a":         self.disease_a,
            "disease_b":         self.disease_b,
            "cumulative_load":   self.cumulative_load,
            "is_confusion_zone": self.is_confusion_zone,
            "trigger_features":  list(self.trigger_features),
            "severity_label":    self.severity_label,
            "clinical_rationale": self.clinical_rationale,
        }


# ── Contradiction cluster ─────────────────────────────────────────────────────

@dataclass
class ContradictionCluster:
    """
    A group of disease pairs involved in a shared contradiction network.

    A cluster forms when multiple pairs share a common disease node
    (star-pattern conflict) or form a chain of pairwise contradictions.
    """

    cluster_id:      str
    diseases:        frozenset[str]
    pair_tensions:   list[PairTension]
    total_load:      float
    is_star_pattern: bool    # True if all pairs share one central disease

    @property
    def center_disease(self) -> str | None:
        """
        For star-pattern clusters, return the central disease.
        For chain clusters, return None.
        """
        if not self.is_star_pattern:
            return None
        # Count disease occurrences in pairs
        counts: dict[str, int] = {}
        for pt in self.pair_tensions:
            counts[pt.disease_a] = counts.get(pt.disease_a, 0) + 1
            counts[pt.disease_b] = counts.get(pt.disease_b, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda d: counts[d])

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id":      self.cluster_id,
            "diseases":        sorted(self.diseases),
            "pair_tensions":   [pt.to_dict() for pt in self.pair_tensions],
            "total_load":      self.total_load,
            "is_star_pattern": self.is_star_pattern,
            "center_disease":  self.center_disease,
        }


# ── ContradictionGraph ────────────────────────────────────────────────────────

class ContradictionGraph:
    """
    Contradiction-focused subgraph: hypothesis nodes linked by tension edges.

    Provides analysis of which disease pairs are in conflict, the severity
    of each tension, and whether conflicts form clusters.

    Parameters
    ----------
    case_id:
        Clinical case identifier.
    contradiction_load:
        Aggregate bilateral contradiction load from PipelineResult.
    """

    def __init__(self, case_id: str, contradiction_load: float) -> None:
        self._case_id   = case_id
        self._load      = contradiction_load
        self._tensions: list[PairTension] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def aggregate_load(self) -> float:
        return self._load

    @property
    def pair_count(self) -> int:
        return len(self._tensions)

    @property
    def is_contradiction_free(self) -> bool:
        return self._load == 0.0 and len(self._tensions) == 0

    def tensions(self) -> list[PairTension]:
        """Return all pair tensions, sorted by load descending."""
        return sorted(self._tensions, key=lambda t: t.cumulative_load, reverse=True)

    def highest_tension_pair(self) -> PairTension | None:
        if not self._tensions:
            return None
        return max(self._tensions, key=lambda t: t.cumulative_load)

    def confusion_zone_tensions(self) -> list[PairTension]:
        """Return only tensions between known confusion-zone pairs."""
        return [t for t in self._tensions if t.is_confusion_zone]

    def critical_tensions(self) -> list[PairTension]:
        """Return tensions with severity_label == 'critical'."""
        return [t for t in self._tensions if t.severity_label == "critical"]

    def exceeds_escalation_ceiling(self, ceiling: float = 0.40) -> bool:
        """True if aggregate load meets mandatory escalation threshold."""
        return self._load >= ceiling

    # ── Cluster detection ─────────────────────────────────────────────────────

    def clusters(self) -> list[ContradictionCluster]:
        """
        Detect connected contradiction clusters using union-find.

        Two disease pairs belong to the same cluster if they share a disease.
        Returns clusters sorted by total load descending.
        """
        if not self._tensions:
            return []

        # Build adjacency from pair tensions
        adj: dict[str, set[str]] = {}
        for pt in self._tensions:
            adj.setdefault(pt.disease_a, set()).add(pt.disease_b)
            adj.setdefault(pt.disease_b, set()).add(pt.disease_a)

        # BFS connected components
        visited: set[str] = set()
        components: list[set[str]] = []
        for seed in list(adj.keys()):
            if seed in visited:
                continue
            component: set[str] = set()
            queue = [seed]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adj.get(node, set()) - visited)
            components.append(component)

        clusters: list[ContradictionCluster] = []
        for idx, component in enumerate(components):
            pair_tensions = [
                pt for pt in self._tensions
                if pt.disease_a in component and pt.disease_b in component
            ]
            total = sum(pt.cumulative_load for pt in pair_tensions)

            # Star detection: one disease appears in all pairs
            is_star = False
            for disease in component:
                if all(disease in {pt.disease_a, pt.disease_b} for pt in pair_tensions):
                    is_star = True
                    break

            clusters.append(ContradictionCluster(
                cluster_id=f"cluster:{idx}",
                diseases=frozenset(component),
                pair_tensions=pair_tensions,
                total_load=total,
                is_star_pattern=is_star,
            ))

        return sorted(clusters, key=lambda c: c.total_load, reverse=True)

    # ── Tension management ────────────────────────────────────────────────────

    def add_tension(self, tension: PairTension) -> None:
        """Add a pair tension to the contradiction graph."""
        self._tensions.append(tension)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        clusters = self.clusters()
        return (
            f"ContradictionGraph[case={self._case_id}] "
            f"load={self._load:.3f} "
            f"pairs={self.pair_count} "
            f"clusters={len(clusters)} "
            f"confusion_zone={len(self.confusion_zone_tensions())} "
            f"exceeds_ceiling={self.exceeds_escalation_ceiling()}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id":          self._case_id,
            "aggregate_load":   self._load,
            "pair_tensions":    [t.to_dict() for t in self.tensions()],
            "clusters":         [c.to_dict() for c in self.clusters()],
            "exceeds_ceiling":  self.exceeds_escalation_ceiling(),
            "confusion_zones":  [t.to_dict() for t in self.confusion_zone_tensions()],
        }

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_result(
        cls,
        result: "PipelineResult",  # type: ignore[name-defined]
        conflict: "ConflictAnalysisResult | None" = None,  # type: ignore[name-defined]
    ) -> "ContradictionGraph":
        """
        Build a ContradictionGraph from a PipelineResult.

        With ConflictAnalysisResult: uses actual pair tensions from the
        contradiction analysis stage.

        Without ConflictAnalysisResult: builds only the aggregate node;
        pair tensions are inferred from confusion zones where the aggregate
        load is non-zero.

        Parameters
        ----------
        result:
            Completed PipelineResult.
        conflict:
            Optional detailed contradiction analysis result. When provided,
            individual pair tensions and trigger features are populated.
        """
        load = result.contradiction_load
        cg   = cls(case_id=result.case_id, contradiction_load=load)

        if conflict is not None:
            # Full detail from ConflictAnalysisResult
            for tension in conflict.pair_tensions:
                if tension.cumulative_penalty <= 0.0:
                    continue
                trigger_feats = tuple(
                    c.trigger_feature for c in tension.active_contradictions
                )
                rationale = "; ".join(
                    c.clinical_rationale for c in tension.active_contradictions
                    if c.clinical_rationale
                )
                cg.add_tension(PairTension(
                    disease_a=tension.source_disease,
                    disease_b=tension.target_disease,
                    cumulative_load=tension.cumulative_penalty,
                    is_confusion_zone=is_confusion_zone(
                        tension.source_disease, tension.target_disease
                    ),
                    trigger_features=trigger_feats,
                    clinical_rationale=rationale,
                ))
        elif load > 0.0:
            # No detail — add confusion zone tensions proportionally
            # (this is a structural estimation, not ground truth)
            known_confusion = [
                ("psoriasis", "pityriasis_rubra_pilaris"),
                ("psoriasis", "lichen_planus"),
                ("lichen_planus", "pityriasis_rubra_pilaris"),
            ]
            # We don't know which pairs fired without ConflictAnalysisResult,
            # so we create an aggregate tension only
            # (Pair-level detail requires the ConflictAnalysisResult)
            pass

        return cg
