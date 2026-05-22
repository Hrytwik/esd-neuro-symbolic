"""
Standalone Symbolic Pipeline — executable entrypoint for the diagnostic
reasoning engine.

Runs all synthetic clinical cases through the full symbolic reasoning
pipeline and prints structured diagnostic summaries to the console.
Optionally exports reasoning traces, escalation reports, clinical
narratives, and replay snapshots to the configured output directory.

Usage
-----
    python -m src.pipeline.standalone_symbolic_pipeline
    python -m src.pipeline.standalone_symbolic_pipeline --mode debug
    python -m src.pipeline.standalone_symbolic_pipeline --mode validation
    python -m src.pipeline.standalone_symbolic_pipeline --case SYN_001
    python -m src.pipeline.standalone_symbolic_pipeline --no-export

Options
-------
    --mode      Execution mode: standard (default), debug, validation
    --case      Run a single case by ID (e.g. SYN_001); default: all 8 cases
    --no-export Suppress file export (console output only)
    --rules-dir Override the rules directory path (default: rules/)
    --output-dir Override the output directory path (default: outputs/)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline.pipeline_config import PipelineConfig
from src.pipeline.pipeline_runner import PipelineResult, PipelineRunner
from src.pipeline.synthetic_case_library import SyntheticCase, SyntheticCaseLibrary
from src.symbolic_engine.rule_registry import DiagnosticRuleRepository
from src.utils.logger import get_logger

log = get_logger(__name__, subsystem="StandaloneSymbolicPipeline")


# ── Console presentation helpers ──────────────────────────────────────────────

_SEPARATOR   = "-" * 72
_THICK_SEP   = "=" * 72
_RECOMMENDATION_LABELS = {
    "SAFE_NON_INVASIVE_TRIAGE":  "[OK]  SAFE NON-INVASIVE TRIAGE",
    "MODERATE_CERTAINTY":        "[~~]  MODERATE CERTAINTY",
    "AMBIGUOUS_PRESENTATION":    "[??]  AMBIGUOUS PRESENTATION",
    "BIOPSY_RECOMMENDED":        "[!!]  BIOPSY RECOMMENDED",
    "HIGH_RISK_CONTRADICTION":   "[XX]  HIGH-RISK CONTRADICTION",
}


def _recommendation_label(rec: str | None) -> str:
    if rec is None:
        return "?  INCOMPLETE"
    return _RECOMMENDATION_LABELS.get(rec, rec)


def _print_case_header(case: SyntheticCase) -> None:
    print()
    print(_THICK_SEP)
    print(f"  CASE  {case.case_id}  --  {case.description[:64]}")
    print(f"  Tags: {', '.join(case.tags)}")
    print(_THICK_SEP)


def _print_result_summary(result: PipelineResult, case: SyntheticCase) -> None:
    rec_label = _recommendation_label(result.recommendation)
    match_leader  = (result.leading_disease == case.expected_leader)
    match_outcome = (result.recommendation  == case.expected_outcome)

    print(f"\n  Recommendation   : {rec_label}")
    print(f"  Leading disease  : {result.leading_disease or '?'}", end="")
    print("  [match]" if match_leader else f"  (expected: {case.expected_leader})")
    print(f"  Max certainty    : {result.max_certainty:.3f}")
    print(f"  Certainty gap    : {result.certainty_gap:.3f}")
    print(f"  Contradiction    : {result.contradiction_load:.3f}")
    print(f"  Ambiguity (bits) : {result.ambiguity_index:.3f}")
    print(f"  Final FSM state  : {result.final_state or '?'}")

    outcome_ok = "[OK]" if match_outcome else f"[MISS]  (expected: {case.expected_outcome})"
    print(f"\n  Outcome match    : {outcome_ok}")


def _print_stage_log(result: PipelineResult) -> None:
    print(f"\n  Stage execution log ({len(result.stage_results)} stages):")
    for sr in result.stage_results:
        status = "OK" if sr.success else "!!"
        print(f"    [{status}] {sr.stage_name:<28}  {sr.summary[:60]}")
        if sr.error:
            print(f"         error: {sr.error}")


def _print_export_paths(result: PipelineResult) -> None:
    paths = {
        "Reasoning trace"  : result.trace_path,
        "Narrative"        : result.narrative_path,
        "Escalation report": result.escalation_path,
        "Replay snapshot"  : result.replay_path,
    }
    exported = {k: v for k, v in paths.items() if v is not None}
    if exported:
        print("\n  Exported files:")
        for label, path in exported.items():
            print(f"    {label:<20}: {path}")


def _print_narrative_excerpt(result: PipelineResult) -> None:
    if result.narrative_path and result.narrative_path.exists():
        lines = result.narrative_path.read_text(encoding="utf-8").splitlines()
        excerpt = [l for l in lines if l.strip() and not l.startswith("=")][:6]
        if excerpt:
            print("\n  Narrative excerpt:")
            for line in excerpt:
                print(f"    {line}")


def _print_batch_summary(
    cases: list[SyntheticCase],
    results: list[PipelineResult],
) -> None:
    print()
    print(_THICK_SEP)
    print("  BATCH SUMMARY")
    print(_THICK_SEP)

    total   = len(results)
    success = sum(1 for r in results if r.success)
    leader_match  = sum(
        1 for r, c in zip(results, cases)
        if r.leading_disease == c.expected_leader
    )
    outcome_match = sum(
        1 for r, c in zip(results, cases)
        if r.recommendation == c.expected_outcome
    )

    print(f"\n  Cases run        : {total}")
    print(f"  Pipeline success : {success}/{total}")
    print(f"  Leader match     : {leader_match}/{total}")
    print(f"  Outcome match    : {outcome_match}/{total}")

    if success < total:
        failed = [
            r.case_id for r, c in zip(results, cases) if not r.success
        ]
        print(f"\n  Failed cases     : {', '.join(failed)}")

    print()
    print(f"  {'Case':<10}  {'Recommendation':<32}  {'Certainty':>9}  {'Match':>5}")
    print(f"  {_SEPARATOR}")
    for r, c in zip(results, cases):
        rec   = r.recommendation or "INCOMPLETE"
        cert  = f"{r.max_certainty:.3f}"
        match = "OK" if r.recommendation == c.expected_outcome else "!!"
        print(f"  {r.case_id:<10}  {rec:<32}  {cert:>9}  {match:>5}")

    print()
    print(_THICK_SEP)


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="standalone_symbolic_pipeline",
        description=(
            "Certainty-Aware Symbolic Dermatological Reasoning Engine -- "
            "standalone diagnostic pipeline runner."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "debug", "validation"],
        default="standard",
        help="Execution mode controlling verbosity and output exports.",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Run a single case by ID (e.g. SYN_001). Default: all cases.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Suppress all file exports; print console output only.",
    )
    parser.add_argument(
        "--rules-dir",
        default="rules",
        help="Path to the YAML rule base directory (default: rules/).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for all pipeline outputs (default: outputs/).",
    )
    parser.add_argument(
        "--verbose-stages",
        action="store_true",
        help="Print per-stage execution log for each case.",
    )
    return parser.parse_args()


# ── Config factory ────────────────────────────────────────────────────────────

def _build_config(args: argparse.Namespace) -> PipelineConfig:
    if args.mode == "debug":
        cfg = PipelineConfig.debug_profile()
    elif args.mode == "validation":
        cfg = PipelineConfig.validation_profile()
    else:
        cfg = PipelineConfig()

    cfg.rules_dir  = Path(args.rules_dir)
    cfg.output_dir = Path(args.output_dir)

    if args.no_export:
        cfg.enable_replay_export = False

    return cfg


# ── Single case runner ────────────────────────────────────────────────────────

def run_case(
    case: SyntheticCase,
    runner: PipelineRunner,
    verbose_stages: bool = False,
    show_narrative: bool = False,
) -> PipelineResult:
    """
    Run a single synthetic case through the pipeline and print its output.

    Parameters
    ----------
    case:
        The SyntheticCase to execute.
    runner:
        Pre-initialised PipelineRunner.
    verbose_stages:
        If True, print the per-stage execution log.
    show_narrative:
        If True, print a short narrative excerpt when available.

    Returns
    -------
    PipelineResult:
        The complete reasoning output for this case.
    """
    _print_case_header(case)
    result = runner.run(
        case_id=case.case_id,
        feature_values=case.feature_values,
    )
    _print_result_summary(result, case)

    if verbose_stages:
        _print_stage_log(result)

    _print_export_paths(result)

    if show_narrative:
        _print_narrative_excerpt(result)

    print(f"\n  {_SEPARATOR}")
    return result


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_all_cases(
    runner: PipelineRunner,
    verbose_stages: bool = False,
    show_narrative: bool = False,
) -> list[PipelineResult]:
    """
    Run the full synthetic case library and print a batch summary.

    Returns
    -------
    list[PipelineResult]:
        One result per case, in library order.
    """
    cases   = list(SyntheticCaseLibrary.all())
    results = []
    for case in cases:
        result = run_case(
            case, runner,
            verbose_stages=verbose_stages,
            show_narrative=show_narrative,
        )
        results.append(result)

    _print_batch_summary(cases, results)
    return results


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Main entrypoint.

    Returns
    -------
    int
        Exit code: 0 = all cases resolved, 1 = one or more pipeline failures.
    """
    args = _parse_args()
    cfg  = _build_config(args)

    print()
    print(_THICK_SEP)
    print("  CASDRE -- Certainty-Aware Symbolic Dermatological Reasoning Engine")
    print(f"  Mode: {cfg.execution_mode}  |  Export: {'disabled' if args.no_export else 'enabled'}")
    print(f"  Rules: {cfg.rules_dir}  |  Output: {cfg.output_dir}")
    print(_THICK_SEP)

    # Initialise rule repository
    try:
        rule_repo = DiagnosticRuleRepository(rules_dir=cfg.rules_dir)
    except Exception as exc:
        print(f"\n  [ERROR] Failed to load rule repository: {exc}", file=sys.stderr)
        return 1

    # Initialise pipeline runner
    runner = PipelineRunner(config=cfg, rule_repository=rule_repo)

    verbose = args.verbose_stages or (args.mode == "debug")
    show_narrative = args.mode == "debug"

    # Single-case or full-batch
    if args.case:
        try:
            case = SyntheticCaseLibrary.get(args.case)
        except KeyError:
            print(
                f"\n  [ERROR] Case '{args.case}' not found. "
                f"Valid IDs: {[c.case_id for c in SyntheticCaseLibrary.all()]}",
                file=sys.stderr,
            )
            return 1
        result = run_case(
            case, runner,
            verbose_stages=verbose,
            show_narrative=show_narrative,
        )
        return 0 if result.success else 1

    else:
        results = run_all_cases(
            runner,
            verbose_stages=verbose,
            show_narrative=show_narrative,
        )
        failed = [r for r in results if not r.success]
        return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
