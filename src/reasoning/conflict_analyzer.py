"""
DiagnosticConflictAnalyzer — Stage 3 of the progressive reasoning pipeline.

Detects cross-disease contradictions and propagates penalty weights through
the hypothesis evidence space. This is a PRIMARY novelty component of the
reasoning engine — contradiction awareness differentiates the symbolic
inference approach from simple evidence accumulation.

Contradiction semantics
-----------------------
A contradiction occurs when a feature that supports Disease A simultaneously
and mechanistically argues against Disease B. The penalty is applied to the
Disease B evidence score, not Disease A — preserving the asymmetric nature
of clinical contradictions.

Contradiction load
------------------
The aggregate contradiction load quantifies the total evidential conflict
present in a case. High contradiction load is the primary trigger for
BIOPSY_RECOMMENDED escalation, independent of the leading hypothesis certainty.

  contradiction_load = Σ active_penalty_weights

  Threshold for mandatory biopsy escalation: ≥ 0.40 (configurable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Contradiction event ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActiveContradiction:
    """
    A single resolved contradiction — one feature activating a penalty
    between two specific diseases.
    """

    contradiction_id:  str
    trigger_feature:   str
    trigger_value:     int | float
    source_disease:    str   # disease the feature supports
    target_disease:    str   # disease receiving the penalty
    penalty_weight:    float
    clinical_rationale: str = ""


# ── Disease-pair tension ──────────────────────────────────────────────────────

@dataclass
class DiseasePairTension:
    """
    Accumulated contradiction tension between a specific disease pair.
    Captures the asymmetric conflict between two competing hypotheses.
    """

    source_disease:       str
    target_disease:       str
    cumulative_penalty:   float   # sum of all active penalties from source → target
    active_contradictions: list[ActiveContradiction] = field(default_factory=list)

    @property
    def severity_label(self) -> str:
        if self.cumulative_penalty < 0.15:
            return "low"
        if self.cumulative_penalty < 0.30:
            return "moderate"
        if self.cumulative_penalty < 0.45:
            return "high"
        return "critical"


# ── Full conflict analysis result ─────────────────────────────────────────────

@dataclass
class ConflictAnalysisResult:
    """
    Complete contradiction analysis for a single case, produced by
    DiagnosticConflictAnalyzer.analyze().
    """

    active_contradictions:  list[ActiveContradiction]
    pair_tensions:          list[DiseasePairTension]
    penalty_by_disease:     dict[str, float]    # total penalty weight per target disease
    contradiction_load:     float               # aggregate penalty (all pairs)
    confusion_zone_active:  list[tuple[str, str]]  # pairs in known confusion zones
    instability_contribution: float             # load × max_pair_tension_ratio
    mandatory_escalation:   bool                # load >= escalation_ceiling

    @property
    def is_contradiction_free(self) -> bool:
        return len(self.active_contradictions) == 0

    @property
    def highest_tension_pair(self) -> DiseasePairTension | None:
        if not self.pair_tensions:
            return None
        return max(self.pair_tensions, key=lambda t: t.cumulative_penalty)

    def penalty_for(self, disease: str) -> float:
        return self.penalty_by_disease.get(disease, 0.0)


# ── Analyzer ──────────────────────────────────────────────────────────────────

class DiagnosticConflictAnalyzer:
    """
    Analyses cross-disease contradictions by scanning the active feature
    values against the structured contradiction matrix.

    Parameters
    ----------
    contradiction_entries:
        List of contradiction dicts from contradiction_matrix.yaml
        (the 'contradictions' key). Each entry specifies trigger_feature,
        trigger_value, supports_disease, contradicts_disease, penalty_weight.
    confusion_zones:
        List of disease-pair dicts from contradiction_matrix.yaml
        (the 'confusion_zones' key).
    escalation_ceiling:
        Contradiction load at which mandatory BIOPSY escalation is triggered.
    """

    def __init__(
        self,
        contradiction_entries: list[dict[str, Any]],
        confusion_zones: list[dict[str, Any]] | None = None,
        escalation_ceiling: float = 0.40,
    ) -> None:
        self._entries       = contradiction_entries
        self._confusion     = confusion_zones or []
        self._escalation    = escalation_ceiling
        self._confusion_pairs: set[frozenset[str]] = {
            frozenset(z["pair"]) for z in self._confusion
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        feature_values: dict[str, int | float | None],
    ) -> ConflictAnalysisResult:
        """
        Evaluate all contradiction entries against the current feature values.

        Parameters
        ----------
        feature_values:
            Raw feature values {name: value}. Missing features should be None.
        """
        active: list[ActiveContradiction] = []
        penalty_by_disease: dict[str, float] = {}

        for entry in self._entries:
            trigger_feat  = entry["trigger_feature"]
            trigger_val   = float(entry["trigger_value"])
            source        = entry["supports_disease"]
            target        = entry["contradicts_disease"]
            penalty       = float(entry["penalty_weight"])
            c_id          = entry.get("contradiction_id", f"{source}→{target}")
            rationale     = entry.get("clinical_rationale", "")

            observed = feature_values.get(trigger_feat)
            if observed is None:
                continue

            # Contradiction fires when feature value equals trigger value
            if abs(float(observed) - trigger_val) < 0.5:
                contradiction = ActiveContradiction(
                    contradiction_id=c_id,
                    trigger_feature=trigger_feat,
                    trigger_value=float(observed),
                    source_disease=source,
                    target_disease=target,
                    penalty_weight=penalty,
                    clinical_rationale=rationale,
                )
                active.append(contradiction)
                penalty_by_disease[target] = (
                    penalty_by_disease.get(target, 0.0) + penalty
                )

        # Aggregate tension by disease pair
        pair_map: dict[tuple[str, str], DiseasePairTension] = {}
        for c in active:
            key = (c.source_disease, c.target_disease)
            if key not in pair_map:
                pair_map[key] = DiseasePairTension(
                    source_disease=c.source_disease,
                    target_disease=c.target_disease,
                    cumulative_penalty=0.0,
                )
            pair_map[key].cumulative_penalty += c.penalty_weight
            pair_map[key].active_contradictions.append(c)

        pair_tensions = list(pair_map.values())

        # Total contradiction load
        contradiction_load = sum(c.penalty_weight for c in active)

        # Confusion zone activity
        confusion_active: list[tuple[str, str]] = []
        for pair in pair_tensions:
            fs = frozenset([pair.source_disease, pair.target_disease])
            if fs in self._confusion_pairs:
                confusion_active.append((pair.source_disease, pair.target_disease))

        # Instability contribution
        max_pair_penalty = (
            max(t.cumulative_penalty for t in pair_tensions)
            if pair_tensions else 0.0
        )
        instability_contribution = (
            contradiction_load * (max_pair_penalty / max(contradiction_load, 1e-6))
            if contradiction_load > 0 else 0.0
        )

        return ConflictAnalysisResult(
            active_contradictions=active,
            pair_tensions=pair_tensions,
            penalty_by_disease=penalty_by_disease,
            contradiction_load=contradiction_load,
            confusion_zone_active=confusion_active,
            instability_contribution=instability_contribution,
            mandatory_escalation=(contradiction_load >= self._escalation),
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: dict[str, Any],
        escalation_ceiling: float = 0.40,
    ) -> "DiagnosticConflictAnalyzer":
        """
        Construct from the parsed contradiction_matrix.yaml dict.

        Parameters
        ----------
        matrix:
            Full parsed YAML dict with 'contradictions' and 'confusion_zones'.
        """
        entries = matrix.get("contradictions", [])
        zones   = matrix.get("confusion_zones", [])
        return cls(
            contradiction_entries=entries,
            confusion_zones=zones,
            escalation_ceiling=escalation_ceiling,
        )
