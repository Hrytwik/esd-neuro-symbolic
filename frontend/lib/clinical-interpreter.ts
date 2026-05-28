/**
 * lib/clinical-interpreter.ts
 * =============================
 * Translates internal reasoning outputs into plain clinical language.
 * All signal names, contradiction tiers, and certainty values are mapped
 * to statements appropriate for clinician-facing display.
 */

import type { ReasoningOutput, ContradictionSummary } from "@/types";
import { diseaseLabel } from "./clinical-language";

// ─── Confidence levels ────────────────────────────────────────────────────────

export type ConfidenceTier = "high" | "moderate" | "low" | "inconclusive";

export interface ConfidenceLevel {
  label: string;
  sublabel: string;
  tier: ConfidenceTier;
  dots: number; // filled dots out of 5
}

export function confidenceLevel(certainty: number): ConfidenceLevel {
  if (certainty >= 0.85) return { label: "High Confidence",          sublabel: "Strong diagnostic alignment",             tier: "high",         dots: 5 };
  if (certainty >= 0.70) return { label: "Moderate-High Confidence", sublabel: "Clinical pattern well-supported",         tier: "moderate",     dots: 4 };
  if (certainty >= 0.55) return { label: "Moderate Confidence",      sublabel: "Further assessment may strengthen this",  tier: "moderate",     dots: 3 };
  if (certainty >= 0.40) return { label: "Low-Moderate Confidence",  sublabel: "Competing patterns remain unresolved",    tier: "low",          dots: 2 };
  return                          { label: "Low Confidence",          sublabel: "Biopsy required for confirmation",         tier: "inconclusive", dots: 1 };
}

// ─── Signal name → clinical finding text ─────────────────────────────────────

const SIGNAL_MAP: [RegExp, string][] = [
  [/scaling.*erythema|erythema.*scaling/i,  "Erythematous scaling pattern"],
  [/scalp/i,                                "Scalp involvement with characteristic distribution"],
  [/knee|elbow|extensor/i,                  "Extensor surface distribution (knee and elbow)"],
  [/koebner/i,                              "Koebner phenomenon present"],
  [/family.?hist/i,                         "Positive family history of similar inflammatory skin disease"],
  [/polygonal/i,                            "Polygonal papule morphology (pathognomonic)"],
  [/oral/i,                                 "Oral mucosal involvement"],
  [/follicular/i,                           "Follicular papule distribution"],
  [/melanin/i,                              "Histological melanin incontinence"],
  [/palmoplantar|keratoderm/i,              "Palmoplantar keratoderma pattern"],
  [/sebaceous|greasy/i,                     "Sebaceous area scaling"],
  [/pityriasis.*triad|herald/i,             "Classic pityriasis clinical triad"],
  [/annular/i,                              "Annular lesion distribution"],
  [/border|margin/i,                        "Well-defined lesion borders"],
  [/itch|pruriti/i,                         "Pruritic presentation"],
  [/erythema/i,                             "Erythema"],
  [/scaling/i,                              "Epidermal scaling"],
];

export function signalToFinding(signalName: string): string {
  for (const [pattern, label] of SIGNAL_MAP) {
    if (pattern.test(signalName)) return label;
  }
  // Fallback: humanise the raw identifier
  return signalName
    .replace(/_/g, " ")
    .replace(/×/g, " combined with ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

// ─── Contradiction tier → clinical sentence ───────────────────────────────────

export function contradictionSummaryText(tier: ContradictionSummary["tier"]): {
  text: string;
  severity: "none" | "low" | "moderate" | "high";
} {
  switch (tier) {
    case "NONE":
      return { text: "Clinical findings are internally consistent — no conflicting patterns detected.", severity: "none" };
    case "MINOR":
      return { text: "Minimal conflicting findings identified — does not materially affect the primary assessment.", severity: "low" };
    case "MODERATE":
      return { text: "Moderate conflicting patterns identified — tissue biopsy may provide additional diagnostic clarity.", severity: "moderate" };
    case "CRITICAL":
      return { text: "Significant diagnostic conflict between competing patterns — histological examination is required.", severity: "high" };
  }
}

// ─── Clinical interpretation paragraph ────────────────────────────────────────

export function clinicalInterpretation(reasoning: ReasoningOutput): string {
  const disease = diseaseLabel(reasoning.leading_diagnosis);
  const c       = reasoning.certainty;
  const biopsy  = reasoning.requires_biopsy;
  const tier    = reasoning.contradiction.tier;
  const active  = reasoning.symbolic_signals.filter(s => s.activated);

  const strength =
    c >= 0.85 ? "strongly aligns with"   :
    c >= 0.70 ? "is consistent with"     :
    c >= 0.55 ? "shows features of"      :
    c >= 0.40 ? "partially overlaps with":
    "remains inconclusive relative to";

  const signalPhrase = active.length > 0
    ? ` Key discriminating findings include ${active
        .sort((a, b) => b.contribution_weight - a.contribution_weight)
        .slice(0, 2)
        .map(s => signalToFinding(s.name).toLowerCase())
        .join(" and ")}.`
    : "";

  const contradictionPhrase =
    (tier === "NONE")
      ? " No significant conflicting features were identified."
      : (tier === "MINOR")
        ? " Minor overlapping features are present but do not alter the primary assessment."
        : " Conflicting inflammatory patterns contribute uncertainty and are reflected in the confidence rating.";

  const triagePhrase = biopsy
    ? " Given the level of diagnostic complexity, histological confirmation is recommended before initiating treatment."
    : " Clinical assessment is sufficient to guide conservative management.";

  return `The overall clinical presentation ${strength} ${disease}.${signalPhrase}${contradictionPhrase}${triagePhrase}`;
}

// ─── Biopsy rationale ─────────────────────────────────────────────────────────

export function biopsyRationale(reasoning: ReasoningOutput): string {
  if (!reasoning.requires_biopsy) {
    return `Clinical evidence provides sufficient certainty (${Math.round(reasoning.certainty * 100)}%) for a working diagnosis without tissue confirmation.`;
  }
  const tier = reasoning.contradiction.tier;
  if (tier === "CRITICAL" || tier === "MODERATE") {
    return `Competing inflammatory patterns could not be resolved through clinical assessment alone. Histological examination is required to distinguish between leading diagnostic candidates.`;
  }
  return `Diagnostic certainty (${Math.round(reasoning.certainty * 100)}%) is insufficient for confident clinical assessment. Biopsy will provide definitive tissue-level characterisation.`;
}

// ─── Key findings extraction ──────────────────────────────────────────────────

export function extractKeyFindings(reasoning: ReasoningOutput): string[] {
  return reasoning.symbolic_signals
    .filter(s => s.activated)
    .sort((a, b) => b.contribution_weight - a.contribution_weight)
    .slice(0, 5)
    .map(s => signalToFinding(s.name));
}
