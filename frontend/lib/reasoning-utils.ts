/**
 * lib/reasoning-utils.ts
 * ========================
 * Pure utility functions for reasoning state derivation and display.
 */

import type { ReasoningOutput, ReplayCaseRecord, GraphSnapshot } from "@/types";

// ─── Certainty colour scale ───────────────────────────────────────────────────

/** Tailwind-compatible class for a certainty value. */
export function certaintyCssClass(cert: number): string {
  if (cert >= 0.85) return "text-emerald-700";
  if (cert >= 0.70) return "text-emerald-600";
  if (cert >= 0.55) return "text-amber-600";
  if (cert >= 0.40) return "text-orange-600";
  return "text-red-600";
}

/** Background class for certainty bar fill. */
export function certaintyBarClass(cert: number): string {
  if (cert >= 0.85) return "bg-emerald-500";
  if (cert >= 0.70) return "bg-emerald-400";
  if (cert >= 0.55) return "bg-amber-400";
  if (cert >= 0.40) return "bg-orange-400";
  return "bg-red-400";
}

/** Hex colour string for chart lines. */
export function certaintyHexColor(cert: number): string {
  if (cert >= 0.85) return "#059669";
  if (cert >= 0.70) return "#10b981";
  if (cert >= 0.55) return "#d97706";
  if (cert >= 0.40) return "#f97316";
  return "#dc2626";
}

// ─── Contradiction colour scale ───────────────────────────────────────────────

export function contradictionCssClass(load: number): string {
  if (load <= 0.05)  return "text-slate-400";
  if (load <= 0.10)  return "text-amber-500";
  if (load <= 0.20)  return "text-orange-600";
  return "text-red-600";
}

export function contradictionBarClass(load: number): string {
  if (load <= 0.05)  return "bg-slate-300";
  if (load <= 0.10)  return "bg-amber-400";
  if (load <= 0.20)  return "bg-orange-500";
  return "bg-red-500";
}

// ─── Formatting helpers ───────────────────────────────────────────────────────

export function formatPercent(value: number, decimals = 0): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatCertainty(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatBits(bits: number): string {
  return `${bits.toFixed(2)} bits`;
}

export function formatLoad(load: number): string {
  return `${(load * 100).toFixed(1)}`;
}

// ─── Derived reasoning state ──────────────────────────────────────────────────

/** Returns the top-N differential entries sorted by certainty. */
export function topDifferential(
  reasoning: ReasoningOutput,
  n = 3
): ReasoningOutput["differential"] {
  return [...reasoning.differential]
    .sort((a, b) => b.certainty - a.certainty)
    .slice(0, n);
}

/** Returns activated symbolic signals sorted by contribution weight. */
export function activatedSignals(
  reasoning: ReasoningOutput
): ReasoningOutput["symbolic_signals"] {
  return [...reasoning.symbolic_signals]
    .filter((s) => s.activated)
    .sort((a, b) => b.contribution_weight - a.contribution_weight);
}

/** Returns contradicting (non-activated) signals. */
export function suppressedSignals(
  reasoning: ReasoningOutput
): ReasoningOutput["symbolic_signals"] {
  return [...reasoning.symbolic_signals].filter((s) => !s.activated);
}

/** Computes the competition margin between rank-1 and rank-2 hypotheses. */
export function competitionMargin(reasoning: ReasoningOutput): number {
  const sorted = [...reasoning.differential].sort((a, b) => b.certainty - a.certainty);
  if (sorted.length < 2) return 1.0;
  return sorted[0].certainty - sorted[1].certainty;
}

/** True if the competition is close (< 0.20 margin). */
export function isCompetitionClose(reasoning: ReasoningOutput): boolean {
  return competitionMargin(reasoning) < 0.20;
}

// ─── Case summary helpers ─────────────────────────────────────────────────────

export function caseStatusLabel(record: ReplayCaseRecord): string {
  if (record.requires_biopsy) return "Biopsy Required";
  if (record.converged && record.is_safe_triage) return "Stable Determination";
  return "Inconclusive";
}

export function caseStatusColor(record: ReplayCaseRecord): string {
  if (record.requires_biopsy) return "text-red-600 bg-red-50 border-red-200";
  if (record.converged) return "text-emerald-700 bg-emerald-50 border-emerald-200";
  return "text-amber-700 bg-amber-50 border-amber-200";
}
