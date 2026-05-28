/**
 * lib/clinical-language.ts
 * ==========================
 * Maps internal schema codes and values to clinician-facing language.
 * All labels, descriptions, and rationale text must use clinical terminology.
 *
 * IMPORTANT: No technical jargon exposed to the clinician interface.
 */

import type {
  DiagnosticStateCode,
  ContradictionTierCode,
  RecoveryMechanismCode,
} from "@/types";

// ─── Disease display names ────────────────────────────────────────────────────

export const DISEASE_DISPLAY_NAMES: Record<string, string> = {
  psoriasis:               "Psoriasis",
  seborrheic_dermatitis:   "Seborrheic Dermatitis",
  lichen_planus:           "Lichen Planus",
  pityriasis_rosea:        "Pityriasis Rosea",
  chronic_dermatitis:      "Chronic Dermatitis",
  pityriasis_rubra_pilaris: "Pityriasis Rubra Pilaris",
};

export function diseaseLabel(key: string): string {
  return DISEASE_DISPLAY_NAMES[key] ?? key;
}

// ─── FSM state clinical descriptions ─────────────────────────────────────────

export const FSM_STATE_DESCRIPTIONS: Record<DiagnosticStateCode, string> = {
  INITIAL:              "Patient case received. Awaiting clinical feature input.",
  CLINICAL_ASSESSMENT:  "Evaluating clinical findings and symptom severity scores.",
  SYMBOLIC_ENRICHMENT:  "Applying disease-specific diagnostic criteria and pattern rules.",
  CONTRADICTION_CHECK:  "Checking for conflicting diagnostic signals across competing hypotheses.",
  AMBIGUITY_RESOLUTION: "Resolving residual uncertainty through differential weighing.",
  ESCALATION_REVIEW:    "Assessing whether available evidence supports safe determination.",
  RECOVERY_ATTEMPT:     "Attempting to resolve diagnostic ambiguity through secondary criteria.",
  FINAL_DECISION:       "Sufficient certainty achieved. Stable triage determination reached.",
  BIOPSY_REQUIRED:      "Competing findings remain unresolved. Histopathological confirmation required.",
};

// ─── Contradiction clinical descriptions ──────────────────────────────────────

export const CONTRADICTION_DESCRIPTIONS: Record<ContradictionTierCode, string> = {
  NONE:     "Clinical findings are internally consistent. No conflicting signals detected.",
  MINOR:    "A minor inconsistency is present. It does not affect the leading determination.",
  MODERATE: "Competing diagnostic patterns detected. Interpretation requires care.",
  CRITICAL: "Significant conflicting evidence. Clinical determination is unreliable without histopathological confirmation.",
};

export const CONTRADICTION_ESCALATION_RATIONALE: Record<ContradictionTierCode, string> = {
  NONE:     "Contradiction load is within safe tolerance. Biopsy not warranted.",
  MINOR:    "Minor conflicts are present but do not compromise diagnostic confidence.",
  MODERATE: "Competing inflammatory patterns remain incompletely resolved. Biopsy escalation is warranted if certainty remains below threshold.",
  CRITICAL: "Critical conflict detected. Biopsy is required before clinical management can proceed.",
};

// ─── Escalation clinical language ─────────────────────────────────────────────

export function escalationRationale(
  requiresBiopsy: boolean,
  contradictionTier: ContradictionTierCode,
  certainty: number,
  ambiguityBits: number
): string {
  if (!requiresBiopsy) {
    if (certainty >= 0.85) {
      return "Diagnostic evidence is strongly convergent. Safe triage is warranted.";
    }
    if (certainty >= 0.70) {
      return "Sufficient diagnostic certainty achieved. Clinical management can proceed.";
    }
    return "Adequate certainty achieved for clinical management. Routine follow-up recommended.";
  }

  // Biopsy required
  if (contradictionTier === "CRITICAL") {
    return "Critical conflicting signals prevent confident determination. Histopathological confirmation is required.";
  }
  if (contradictionTier === "MODERATE") {
    return "Competing inflammatory patterns remain unresolved. Diagnostic certainty is insufficient for clinical management without biopsy.";
  }
  if (certainty < 0.60) {
    return "Insufficient diagnostic certainty. Additional histopathological evidence is required before treatment can be initiated.";
  }
  if (ambiguityBits > 2.0) {
    return "Residual diagnostic ambiguity is too high for safe clinical determination. Biopsy is indicated.";
  }
  return "The clinical evidence does not resolve to a single diagnostic hypothesis with adequate confidence.";
}

// ─── Certainty clinical descriptions ──────────────────────────────────────────

export function certaintyClinicalLabel(certainty: number): string {
  if (certainty >= 0.90) return "Very high confidence";
  if (certainty >= 0.80) return "High confidence";
  if (certainty >= 0.70) return "Moderate–high confidence";
  if (certainty >= 0.60) return "Moderate confidence";
  if (certainty >= 0.50) return "Borderline confidence";
  if (certainty >= 0.35) return "Low confidence";
  return "Very low confidence";
}

export function ambiguityClinicalLabel(bits: number): string {
  if (bits < 0.50) return "Negligible ambiguity";
  if (bits < 1.00) return "Low ambiguity";
  if (bits < 1.80) return "Moderate ambiguity";
  if (bits < 2.50) return "High ambiguity";
  return "Very high ambiguity";
}

// ─── Recovery mechanism descriptions ──────────────────────────────────────────

export const RECOVERY_DESCRIPTIONS: Record<RecoveryMechanismCode, string> = {
  CONTRADICTION: "Contradiction resolved — the conflicting signal was suppressed, allowing the leading hypothesis to stabilise.",
  LEADERSHIP:    "Hypothesis leadership confirmed — the leading diagnosis separated clearly from competing alternatives.",
  AMBIGUITY:     "Ambiguity resolved — secondary criteria removed the diagnostic tie.",
  TRAJECTORY:    "Trajectory convergence — the certainty trajectory stabilised toward the correct diagnosis.",
  COMPETITION:   "Competition tiebreak — pathognomonic criteria distinguished the leading hypothesis.",
  ESCALATION:      "Escalation rerouted — reassessment of clinical criteria corrected the initial direction.",
  SIGNATURE_MATCH: "Pathognomonic signature identified — a disease-specific pattern was detected, resolving diagnostic ambiguity.",
  UNEXPLAINED:     "Recovery pathway unclassified — the leading hypothesis strengthened without identifiable mechanism.",
};

// ─── Symbolic rule clinical labels ────────────────────────────────────────────

export const RULE_STRENGTH_LABELS: Record<string, string> = {
  PATHOGNOMONIC: "Pathognomonic — highly specific for this diagnosis",
  STRONG:        "Strong supporting finding",
  MODERATE:      "Moderate supporting finding",
  WEAK:          "Weak supporting finding",
  EXCLUSIONARY:  "Exclusionary — argues against this diagnosis",
};
