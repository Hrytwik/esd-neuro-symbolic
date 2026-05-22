"""
PipelineOutputs — structured export of reasoning traces and escalation reports.

Exports five output types into the configured output directories:

  1. ReasoningTraceExporter   — JSON per-stage snapshot sequence
  2. NarrativeExporter        — plain-text clinical reasoning narrative
  3. EscalationReportExporter — JSON triage decision + rationale
  4. ReplaySnapshotExporter   — compact JSON replay bundle (all stages)
  5. ValidationReportExporter — batch validation summary across multiple cases

All exporters are stateless and accept their inputs explicitly, making
them independently testable and replay-safe.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.reasoning.escalation_engine import TriageDecision
from src.reasoning.narrative_generator import ClinicalNarrative
from src.reasoning.trajectory_memory import DiagnosticTrajectory


# ── Utility helpers ───────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_value(value: Any) -> Any:
    """Recursively convert non-JSON-serializable types to strings."""
    if isinstance(value, dict):
        return {k: _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    if hasattr(value, "value"):        # Enum
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: _safe_value(getattr(value, k))
                for k in value.__dataclass_fields__}
    return value


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ── 1. Reasoning trace exporter ───────────────────────────────────────────────

class ReasoningTraceExporter:
    """
    Exports the full per-stage reasoning trace as a JSON file.

    Output format
    -------------
    {
      "case_id":    str,
      "run_id":     str,
      "exported_at": ISO8601,
      "stage_count": int,
      "snapshots":  [{stage, stage_name, state, leading_disease, ...}, ...],
      "deltas":     [{from_stage, to_stage, certainty_delta, ...}, ...],
      "certainty_series": [float, ...],
      "state_sequence":   [str, ...]
    }
    """

    @staticmethod
    def export(trajectory: DiagnosticTrajectory, output_dir: Path) -> Path:
        snapshots_data = []
        for snap in trajectory.snapshots:
            snapshots_data.append({
                "stage":             snap.stage,
                "stage_name":        snap.stage_name,
                "state":             snap.state.value,
                "leading_disease":   snap.leading_disease,
                "max_certainty":     round(snap.max_certainty, 5),
                "certainty_gap":     round(snap.certainty_gap, 5),
                "contradiction_load": round(snap.contradiction_load, 5),
                "ambiguity_index":   round(snap.ambiguity_index, 5),
                "active_rule_count": snap.active_rule_count,
                "tier_a_count":      snap.tier_a_count,
                "safety_triggered":  snap.safety_triggered,
                "triage_so_far":     snap.triage_so_far,
                "delta_description": snap.delta_description,
            })

        deltas_data = []
        for d in trajectory.deltas():
            deltas_data.append({
                "from_stage":        d.from_stage,
                "to_stage":          d.to_stage,
                "certainty_delta":   round(d.certainty_delta, 5),
                "gap_delta":         round(d.gap_delta, 5),
                "contradiction_delta": round(d.contradiction_delta, 5),
                "entropy_delta":     round(d.entropy_delta, 5),
                "state_changed":     d.state_changed,
                "from_state":        d.from_state.value,
                "to_state":          d.to_state.value,
            })

        final_decision = None
        if trajectory.final_decision:
            fd = trajectory.final_decision
            final_decision = {
                "recommendation":     fd.recommendation.value,
                "leading_disease":    fd.leading_disease,
                "max_certainty":      round(fd.max_certainty, 5),
                "certainty_gap":      round(fd.certainty_gap, 5),
                "contradiction_load": round(fd.contradiction_load, 5),
                "ambiguity_index":    round(fd.ambiguity_index, 5),
                "final_state":        fd.final_state.value,
                "safety_gate_applied": fd.safety_gate_applied,
                "applied_gate_ids":   fd.applied_gate_ids,
                "decision_rationale": fd.decision_rationale,
            }

        data = {
            "case_id":         trajectory.case_id,
            "run_id":          trajectory.run_id,
            "exported_at":     _timestamp(),
            "stage_count":     trajectory.stage_count,
            "snapshots":       snapshots_data,
            "deltas":          deltas_data,
            "certainty_series": [round(v, 5) for v in trajectory.certainty_series()],
            "state_sequence":  trajectory.state_sequence(),
            "final_decision":  final_decision,
        }

        path = output_dir / f"{trajectory.case_id}_{trajectory.run_id}_trace.json"
        return _write_json(path, data)


# ── 2. Narrative exporter ─────────────────────────────────────────────────────

class NarrativeExporter:
    """
    Exports the clinical reasoning narrative as a structured plain-text file.
    """

    @staticmethod
    def export(
        narrative: ClinicalNarrative,
        case_id: str,
        run_id: str,
        output_dir: Path,
    ) -> Path:
        header = (
            f"CLINICAL REASONING NARRATIVE\n"
            f"{'=' * 60}\n"
            f"Case ID : {case_id}\n"
            f"Run ID  : {run_id}\n"
            f"Generated: {_timestamp()}\n"
            f"{'=' * 60}\n\n"
        )
        body = narrative.full_text(separator="\n\n")
        content = header + body + "\n"
        path = output_dir / f"{case_id}_{run_id}_narrative.txt"
        return _write_text(path, content)


# ── 3. Escalation report exporter ────────────────────────────────────────────

class EscalationReportExporter:
    """
    Exports the terminal triage decision as a structured JSON report.
    """

    @staticmethod
    def export(
        decision: TriageDecision,
        case_id: str,
        run_id: str,
        output_dir: Path,
    ) -> Path:
        data = {
            "report_type":   "escalation_report",
            "case_id":       case_id,
            "run_id":        run_id,
            "exported_at":   _timestamp(),
            "recommendation": decision.recommendation.value,
            "requires_biopsy": decision.requires_biopsy,
            "is_safe_triage":  decision.is_safe_triage,
            "leading_disease": decision.leading_disease,
            "second_disease":  decision.second_disease,
            "max_certainty":   round(decision.max_certainty, 5),
            "certainty_gap":   round(decision.certainty_gap, 5),
            "contradiction_load": round(decision.contradiction_load, 5),
            "ambiguity_index": round(decision.ambiguity_index, 5),
            "final_state":     decision.final_state.value,
            "safety_gate_applied": decision.safety_gate_applied,
            "applied_gate_ids": decision.applied_gate_ids,
            "decision_rationale": decision.decision_rationale,
        }
        path = output_dir / f"{case_id}_{run_id}_escalation.json"
        return _write_json(path, data)


# ── 4. Replay snapshot exporter ───────────────────────────────────────────────

class ReplaySnapshotExporter:
    """
    Exports a compact replay bundle: feature inputs + full trace + decision.
    Sufficient to re-run or visualise any completed pipeline execution.
    """

    @staticmethod
    def export(
        trajectory: DiagnosticTrajectory,
        feature_values: dict,
        output_dir: Path,
    ) -> Path:
        snapshot_stages = []
        for snap in trajectory.snapshots:
            snapshot_stages.append({
                "stage":           snap.stage,
                "stage_name":      snap.stage_name,
                "state":           snap.state.value,
                "leading_disease": snap.leading_disease,
                "max_certainty":   round(snap.max_certainty, 5),
                "contradiction_load": round(snap.contradiction_load, 5),
                "ambiguity_index": round(snap.ambiguity_index, 5),
            })

        final_decision = None
        if trajectory.final_decision:
            fd = trajectory.final_decision
            final_decision = {
                "recommendation": fd.recommendation.value,
                "leading_disease": fd.leading_disease,
                "max_certainty":  round(fd.max_certainty, 5),
            }

        data = {
            "replay_format":    "symbolic_reasoning_v1",
            "case_id":          trajectory.case_id,
            "run_id":           trajectory.run_id,
            "exported_at":      _timestamp(),
            "feature_inputs":   {k: (v if v is not None else None)
                                  for k, v in feature_values.items()},
            "stages":           snapshot_stages,
            "certainty_series": [round(v, 5) for v in trajectory.certainty_series()],
            "state_sequence":   trajectory.state_sequence(),
            "final_decision":   final_decision,
        }

        path = output_dir / f"{trajectory.case_id}_{trajectory.run_id}_replay.json"
        return _write_json(path, data)


# ── 5. Validation report exporter ────────────────────────────────────────────

class ValidationReportExporter:
    """
    Exports a batch validation summary across multiple pipeline runs.
    """

    @staticmethod
    def export(
        validation_results: list[dict],
        run_label: str,
        output_dir: Path,
    ) -> Path:
        passed = sum(1 for r in validation_results if r.get("passed"))
        failed = len(validation_results) - passed

        data = {
            "report_type":       "validation_run",
            "run_label":         run_label,
            "exported_at":       _timestamp(),
            "total_cases":       len(validation_results),
            "passed":            passed,
            "failed":            failed,
            "pass_rate":         round(passed / max(len(validation_results), 1), 4),
            "results":           validation_results,
        }

        path = output_dir / f"validation_{run_label}_{_timestamp()}.json"
        return _write_json(path, data)
