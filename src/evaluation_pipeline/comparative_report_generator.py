"""
ComparativeReportGenerator — publication-grade A/B/C comparison tables.

Generates structured clinical evaluation reports from the outputs of all
evaluation modules. Reports are designed for two audiences:

  1. Scientific publication (tables, numerical summaries, statistical comparisons)
  2. Clinical demonstration (plain-text interpretation, decision analysis)

Report sections
---------------
  1. Model comparison table (A/B/C accuracy, F1, per-disease recall)
  2. Biopsy-free safety analysis (escalation rates, avoidance statistics)
  3. Contradiction analysis (prevalence, certainty decay, confusion zones)
  4. Trajectory stability summary (convergence, oscillation, stabilisation)
  5. Symbolic contribution analysis (feature importance breakdown)
  6. Per-disease performance comparison (A vs B vs C per disease)
  7. Reasoning quality indicators (certainty, entropy, sufficiency)

Output formats
--------------
  · Plain text tables (console-ready)
  · Python dict (JSON-serialisable)
  · Structured file (outputs/ directory)

Usage
-----
  from src.evaluation_pipeline.comparative_report_generator import (
      ComparativeReportGenerator, ComparativeReport,
  )

  report = ComparativeReportGenerator.generate(
      evaluation_result=runner_result,
      escalation_result=escalation_result,
      contradiction_result=contradiction_result,
      trajectory_result=trajectory_result,
      reasoning_metrics=reasoning_metrics,
  )

  print(ComparativeReportGenerator.to_text_table(report))
  ComparativeReportGenerator.write_report(report, output_dir=Path("outputs/"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation_pipeline.evaluation_runner import TripartiteEvaluationResult
from src.evaluation_pipeline.escalation_evaluator import EscalationEvaluationResult
from src.evaluation_pipeline.contradiction_evaluator import ContradictionEvaluationResult
from src.evaluation_pipeline.trajectory_evaluator import TrajectoryEvaluationResult
from src.evaluation_pipeline.reasoning_metrics import AggregatedReasoningMetrics
from src.dataset_integration.dataset_loader import CANONICAL_DISEASES


# ── Report data model ─────────────────────────────────────────────────────────

@dataclass
class ComparativeReport:
    """
    Complete comparative evaluation report.

    Attributes
    ----------
    evaluation_result:
        Three-model classification comparison.
    escalation_result:
        Escalation behaviour analysis from symbolic reasoning.
    contradiction_result:
        Contradiction prevalence and certainty decay analysis.
    trajectory_result:
        Trajectory stability and convergence analysis.
    reasoning_metrics:
        Aggregated symbolic reasoning quality indicators.
    generation_timestamp:
        ISO 8601 timestamp when the report was generated.
    """

    evaluation_result:      TripartiteEvaluationResult
    escalation_result:      EscalationEvaluationResult
    contradiction_result:   ContradictionEvaluationResult
    trajectory_result:      TrajectoryEvaluationResult
    reasoning_metrics:      AggregatedReasoningMetrics
    generation_timestamp:   str = ""

    @property
    def symbolic_recovery_rate(self) -> float:
        return self.evaluation_result.symbolic_recovery_rate

    @property
    def biopsy_avoidance_rate(self) -> float:
        return self.escalation_result.safe_rate

    @property
    def contradiction_prevalence(self) -> float:
        return self.contradiction_result.contradiction_prevalence


# ── Report generator ──────────────────────────────────────────────────────────

class ComparativeReportGenerator:
    """
    Generates and formats comparative evaluation reports.

    All methods are static — no instance state required.
    """

    @staticmethod
    def generate(
        evaluation_result:    TripartiteEvaluationResult,
        escalation_result:    EscalationEvaluationResult,
        contradiction_result: ContradictionEvaluationResult,
        trajectory_result:    TrajectoryEvaluationResult,
        reasoning_metrics:    AggregatedReasoningMetrics,
    ) -> ComparativeReport:
        """
        Assemble a ComparativeReport from all evaluator outputs.

        Parameters
        ----------
        evaluation_result:
            From EvaluationRunner.run().
        escalation_result:
            From EscalationEvaluator.evaluate_vectors().
        contradiction_result:
            From ContradictionEvaluator.evaluate_vectors().
        trajectory_result:
            From TrajectoryEvaluator.evaluate_vectors().
        reasoning_metrics:
            From ReasoningMetricsAggregator.aggregate().
        """
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return ComparativeReport(
            evaluation_result=evaluation_result,
            escalation_result=escalation_result,
            contradiction_result=contradiction_result,
            trajectory_result=trajectory_result,
            reasoning_metrics=reasoning_metrics,
            generation_timestamp=ts,
        )

    @staticmethod
    def to_text_table(report: ComparativeReport) -> str:
        """
        Format the full report as a human-readable plain-text table.

        Suitable for console output, publication appendices, and
        demonstration workflows.
        """
        ev  = report.evaluation_result
        esc = report.escalation_result
        con = report.contradiction_result
        trj = report.trajectory_result
        rmq = report.reasoning_metrics
        a   = ev.model_a
        b   = ev.model_b
        c   = ev.model_c

        W = 72

        def _line(label: str, a_val: str, b_val: str, c_val: str) -> str:
            return f"  {label:<30} {a_val:>10} {b_val:>10} {c_val:>10}"

        def _sep() -> str:
            return "-" * W

        def _head(title: str) -> str:
            return f"\n{'=' * W}\n  {title}\n{'=' * W}"

        # ── Section 1: Model comparison ───────────────────────────────────────
        lines = [
            "=" * W,
            "  COMPARATIVE CLINICAL EVALUATION REPORT",
            f"  Generated: {report.generation_timestamp}",
            "=" * W,
            "",
            _head("1. THREE-MODEL CLASSIFICATION COMPARISON"),
            "",
            "  " + " " * 30 + f"{'Model A':>10}  {'Model B':>10}  {'Model C':>10}",
            "  " + " " * 30 + f"{'(34 feat)':>10}  {'(12 feat)':>10}  {'(+symbolic)':>10}",
            _sep(),
            _line("Accuracy",
                  f"{a.accuracy:.4f}",
                  f"{b.accuracy:.4f}",
                  f"{c.accuracy:.4f}"),
            _line("Macro F1",
                  f"{a.macro_f1:.4f}",
                  f"{b.macro_f1:.4f}",
                  f"{c.macro_f1:.4f}"),
            _line("Feature count",
                  str(a.feature_count),
                  str(b.feature_count),
                  str(c.total_feature_count)),
            _line("Training samples",
                  str(a.n_train),
                  str(b.n_train),
                  str(c.n_train)),
            _line("Test samples",
                  str(a.n_test),
                  str(b.n_test),
                  str(c.n_test)),
            _sep(),
            f"  Biopsy-free accuracy gap  (A - B) : "
            f"{ev.biopsy_free_accuracy_gap:+.4f}",
            f"  Symbolic accuracy lift    (C - B) : "
            f"{ev.symbolic_lift:+.4f}",
            f"  Symbolic recovery rate    (C-B)/(A-B) : "
            f"{ev.symbolic_recovery_rate:.1%}",
            f"  Biopsy-free F1 gap        (A - B) : "
            f"{ev.biopsy_free_f1_gap:+.4f}",
            f"  Symbolic F1 lift          (C - B) : "
            f"{ev.symbolic_f1_lift:+.4f}",
        ]

        # ── Section 2: Per-disease performance ────────────────────────────────
        lines += [
            _head("2. PER-DISEASE RECALL COMPARISON"),
            "",
            "  " + f"{'Disease':<30} {'Model A':>10}  {'Model B':>10}  {'Model C':>10}",
            _sep(),
        ]
        for disease in CANONICAL_DISEASES:
            ra = a.per_class_recall.get(disease, 0.0)
            rb = b.per_class_recall.get(disease, 0.0)
            rc = c.per_class_recall.get(disease, 0.0)
            lines.append(_line(
                disease.replace("_", " ").title()[:28],
                f"{ra:.3f}", f"{rb:.3f}", f"{rc:.3f}",
            ))

        # ── Section 3: Escalation analysis ───────────────────────────────────
        n_esc = max(esc.total_cases, 1)
        lines += [
            _head("3. BIOPSY ESCALATION ANALYSIS (SYMBOLIC REASONING)"),
            "",
            f"  Total evaluated cases      : {esc.total_cases}",
            f"  Biopsy recommended         : "
            f"{esc.biopsy_recommended_count} ({esc.biopsy_rate:.1%})",
            f"  Safe non-invasive triage   : "
            f"{esc.safe_triage_count} ({esc.safe_rate:.1%})",
            f"  High-risk contradiction    : "
            f"{esc.high_risk_count} ({esc.high_risk_count/n_esc:.1%})",
            f"  Contradiction-driven esc.  : "
            f"{esc.contradiction_driven_escalation_count}",
            f"  Ambiguity-driven esc.      : "
            f"{esc.ambiguity_driven_escalation_count}",
            f"  Safety gate activations    : "
            f"{esc.safety_gate_activation_count}",
            f"  Justified biopsy recs.     : "
            f"{esc.justified_biopsy_count}",
            f"  Justified safe triage      : "
            f"{esc.justified_safe_triage_count}",
            "",
            "  Per-disease biopsy rate:",
            _sep(),
        ]
        for dis in CANONICAL_DISEASES:
            bx_rate = esc.per_disease_biopsy_rate.get(dis, 0.0)
            sf_rate = esc.per_disease_safe_rate.get(dis, 0.0)
            cert    = esc.per_disease_mean_certainty.get(dis, 0.0)
            contra  = esc.per_disease_mean_contradiction.get(dis, 0.0)
            lines.append(
                f"  {dis.replace('_',' ').title()[:28]:<30} "
                f"biopsy={bx_rate:.1%}  safe={sf_rate:.1%}  "
                f"cert={cert:.3f}  contra={contra:.3f}"
            )

        # ── Section 4: Contradiction analysis ─────────────────────────────────
        lines += [
            _head("4. CONTRADICTION ANALYSIS"),
            "",
            f"  Contradicted cases         : "
            f"{con.contradiction_cases} ({con.contradiction_prevalence:.1%})",
            f"  Critical cases (>= 0.40)   : "
            f"{con.critical_cases} ({con.critical_prevalence:.1%})",
            f"  Dampened cases (>= 0.20)   : "
            f"{con.dampened_cases}",
            f"  Mean load (all cases)      : "
            f"{con.mean_contradiction_load:.4f}  (std={con.std_contradiction_load:.4f})",
            f"  Certainty (with contra)    : "
            f"{con.mean_certainty_with_contradiction:.4f}",
            f"  Certainty (no contra)      : "
            f"{con.mean_certainty_without_contradiction:.4f}",
            f"  Certainty decay            : "
            f"{con.certainty_decay_under_contradiction:.4f}",
            f"  Escalation under contra    : "
            f"{con.escalation_under_contradiction_rate:.1%}",
            f"  Severity: " + "  ".join(
                f"{k}={v}" for k, v in con.severity_distribution.items()
            ),
        ]

        # ── Section 5: Trajectory analysis ────────────────────────────────────
        lines += [
            _head("5. TRAJECTORY STABILITY ANALYSIS"),
            "",
            f"  Mean convergence index     : "
            f"{trj.mean_convergence_index:.4f}  (std={trj.std_convergence_index:.4f})",
            f"  Convergent (index >= 0.80) : "
            f"{trj.convergent_case_count} ({trj.convergent_case_count/max(trj.total_cases,1):.1%})",
            f"  Oscillating (>= 2 rev.)    : "
            f"{trj.oscillating_case_count} ({trj.oscillating_case_count/max(trj.total_cases,1):.1%})",
            f"  Stable trajectories        : "
            f"{trj.stable_case_count} ({trj.stable_case_count/max(trj.total_cases,1):.1%})",
            f"  Leadership changes         : "
            f"{trj.leadership_changed_count}",
            f"  Mean peak certainty        : {trj.mean_peak_certainty:.4f}",
            f"  Mean final certainty       : {trj.mean_final_certainty:.4f}",
            f"  Mean certainty delta       : {trj.mean_certainty_delta:+.4f}",
            f"  Stabilised cases           : "
            f"{trj.stabilised_count} (mean stage {trj.mean_stabilisation_stage:.1f})",
        ]

        # ── Section 6: Symbolic contribution ─────────────────────────────────
        lines += [
            _head("6. SYMBOLIC REASONING CONTRIBUTION (MODEL C)"),
            "",
            f"  Clinical feature count     : {c.clinical_feature_count}",
            f"  Symbolic signal count      : {c.symbolic_feature_count}",
            f"  Total feature count        : {c.total_feature_count}",
            f"  Symbolic importance weight : "
            f"{c.symbolic_importance_fraction:.1%}",
            f"  Reasoning vectors used     : {c.reasoning_vectors_used}",
            "",
            "  Top symbolic signals by importance:",
            _sep(),
        ]
        top_sym = sorted(
            c.symbolic_feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:8]
        for sig, imp in top_sym:
            lines.append(f"  {sig:<38} {imp:.5f}")

        # ── Section 7: Reasoning quality ─────────────────────────────────────
        lines += [
            _head("7. SYMBOLIC REASONING QUALITY INDICATORS"),
            "",
            f"  Safe triage rate           : {rmq.safe_triage_rate:.1%}",
            f"  Escalation rate            : {rmq.escalation_rate:.1%}",
            f"  Sufficiency rate           : {rmq.sufficiency_rate:.1%}",
            f"  Certain reasoning rate     : {rmq.certain_reasoning_rate:.1%}",
            f"  Ambiguous reasoning rate   : {rmq.ambiguous_reasoning_rate:.1%}",
            f"  Contradiction prevalence   : {rmq.contradiction_prevalence:.1%}",
            f"  Dampened case rate         : {rmq.dampened_case_rate:.1%}",
            f"  Mean convergence index     : {rmq.mean_convergence_index:.4f}",
            f"  Mean oscillations          : {rmq.mean_oscillation_count:.2f}",
            f"  Mean certainty             : "
            f"{rmq.certainty_metrics.get('mean', 0):.4f}  "
            f"(std={rmq.certainty_metrics.get('std', 0):.4f})",
            f"  Mean entropy               : "
            f"{rmq.entropy_metrics.get('mean', 0):.4f} bits",
        ]

        lines.append("=" * W)
        return "\n".join(lines)

    @staticmethod
    def to_dict(report: ComparativeReport) -> dict[str, Any]:
        """
        Return the full report as a JSON-serialisable dict.

        Suitable for programmatic consumption and automated pipelines.
        """
        ev  = report.evaluation_result
        esc = report.escalation_result
        con = report.contradiction_result
        trj = report.trajectory_result
        rmq = report.reasoning_metrics
        a   = ev.model_a
        b   = ev.model_b
        c   = ev.model_c

        return {
            "generated_at":            report.generation_timestamp,
            "model_comparison": {
                "model_a": {
                    "accuracy":  a.accuracy,
                    "macro_f1":  a.macro_f1,
                    "features":  a.feature_count,
                    "partition": a.partition,
                    "n_train":   a.n_train,
                    "n_test":    a.n_test,
                    "per_class_f1": a.per_class_f1,
                },
                "model_b": {
                    "accuracy":  b.accuracy,
                    "macro_f1":  b.macro_f1,
                    "features":  b.feature_count,
                    "partition": b.partition,
                    "n_train":   b.n_train,
                    "n_test":    b.n_test,
                    "per_class_f1": b.per_class_f1,
                },
                "model_c": {
                    "accuracy":                    c.accuracy,
                    "macro_f1":                    c.macro_f1,
                    "features":                    c.total_feature_count,
                    "clinical_features":           c.clinical_feature_count,
                    "symbolic_features":           c.symbolic_feature_count,
                    "partition":                   c.partition,
                    "n_train":                     c.n_train,
                    "n_test":                      c.n_test,
                    "per_class_f1":                c.per_class_f1,
                    "symbolic_importance_fraction": c.symbolic_importance_fraction,
                    "reasoning_vectors_used":      c.reasoning_vectors_used,
                    "top_symbolic_signals":        sorted(
                        c.symbolic_feature_importances.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10],
                },
                "comparative_metrics": {
                    "biopsy_free_accuracy_gap": ev.biopsy_free_accuracy_gap,
                    "symbolic_accuracy_lift":   ev.symbolic_lift,
                    "symbolic_recovery_rate":   ev.symbolic_recovery_rate,
                    "biopsy_free_f1_gap":       ev.biopsy_free_f1_gap,
                    "symbolic_f1_lift":         ev.symbolic_f1_lift,
                },
            },
            "escalation_analysis": {
                "total_cases":           esc.total_cases,
                "biopsy_rate":           esc.biopsy_rate,
                "safe_rate":             esc.safe_rate,
                "contradiction_driven":  esc.contradiction_driven_escalation_count,
                "ambiguity_driven":      esc.ambiguity_driven_escalation_count,
                "safety_gate_count":     esc.safety_gate_activation_count,
                "justified_biopsies":    esc.justified_biopsy_count,
                "justified_safe_triage": esc.justified_safe_triage_count,
                "per_disease_biopsy_rate": esc.per_disease_biopsy_rate,
            },
            "contradiction_analysis": {
                "contradiction_cases":    con.contradiction_cases,
                "contradiction_prevalence": con.contradiction_prevalence,
                "critical_cases":         con.critical_cases,
                "critical_prevalence":    con.critical_prevalence,
                "mean_load":              con.mean_contradiction_load,
                "std_load":               con.std_contradiction_load,
                "certainty_with_contra":  con.mean_certainty_with_contradiction,
                "certainty_without_contra": con.mean_certainty_without_contradiction,
                "certainty_decay":        con.certainty_decay_under_contradiction,
                "escalation_under_contra": con.escalation_under_contradiction_rate,
                "severity_distribution":  con.severity_distribution,
                "per_disease_prevalence": con.per_disease_contradiction_prevalence,
                "per_disease_mean_load":  con.per_disease_mean_load,
            },
            "trajectory_analysis": {
                "mean_convergence_index": trj.mean_convergence_index,
                "std_convergence_index":  trj.std_convergence_index,
                "convergent_cases":       trj.convergent_case_count,
                "oscillating_cases":      trj.oscillating_case_count,
                "stable_cases":           trj.stable_case_count,
                "dampened_cases":         trj.dampened_case_count,
                "leadership_changes":     trj.leadership_changed_count,
                "mean_peak_certainty":    trj.mean_peak_certainty,
                "mean_final_certainty":   trj.mean_final_certainty,
                "stabilised_cases":       trj.stabilised_count,
                "mean_stabilisation_stage": trj.mean_stabilisation_stage,
                "per_disease_convergence": trj.per_disease_convergence_index,
            },
            "reasoning_quality": rmq.to_dict(),
            "execution_metadata": {
                "execution_time_seconds": ev.execution_time_seconds,
                "pipeline_success_count": ev.pipeline_success_count,
                "pipeline_failure_count": ev.pipeline_failure_count,
                "seed":                   ev.config.seed,
                "model_type":             ev.config.model_type,
                "train_ratio":            ev.config.train_ratio,
                "test_ratio":             ev.config.test_ratio,
            },
        }

    @staticmethod
    def write_report(
        report: ComparativeReport,
        output_dir: Path,
        prefix: str = "comparative_evaluation",
    ) -> dict[str, Path]:
        """
        Write the report to disk in both text and JSON formats.

        Parameters
        ----------
        report:
            Assembled ComparativeReport.
        output_dir:
            Directory for output files.
        prefix:
            Filename prefix.

        Returns
        -------
        dict with 'text_path' and 'json_path'.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = report.generation_timestamp.replace(":", "").replace("-", "")

        # Plain text
        text_path = output_dir / f"{prefix}_{ts}.txt"
        text_path.write_text(
            ComparativeReportGenerator.to_text_table(report),
            encoding="utf-8",
        )

        # JSON
        json_path = output_dir / f"{prefix}_{ts}.json"
        data = ComparativeReportGenerator.to_dict(report)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

        return {"text_path": text_path, "json_path": json_path}
