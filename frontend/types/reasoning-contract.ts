/**
 * types/reasoning-contract.ts
 * ==============================
 * TypeScript mirror of the frozen CASDRE reasoning output contract (v1.0.0).
 * Source: src/backend_stabilization/reasoning_contract_finalizer.py
 *
 * DO NOT modify these types without a version bump in the backend contract.
 */

export const REASONING_CONTRACT_VERSION = "1.0.0" as const;

// ─── FSM state codes ────────────────────────────────────────────────────────

export type DiagnosticStateCode =
  | "INITIAL"
  | "CLINICAL_ASSESSMENT"
  | "SYMBOLIC_ENRICHMENT"
  | "CONTRADICTION_CHECK"
  | "AMBIGUITY_RESOLUTION"
  | "ESCALATION_REVIEW"
  | "RECOVERY_ATTEMPT"
  | "FINAL_DECISION"
  | "BIOPSY_REQUIRED";

export const FSM_STATE_LABELS: Record<DiagnosticStateCode, string> = {
  INITIAL:              "Initial Assessment",
  CLINICAL_ASSESSMENT:  "Clinical Feature Evaluation",
  SYMBOLIC_ENRICHMENT:  "Symbolic Signal Enrichment",
  CONTRADICTION_CHECK:  "Contradiction Check",
  AMBIGUITY_RESOLUTION: "Ambiguity Resolution",
  ESCALATION_REVIEW:    "Escalation Review",
  RECOVERY_ATTEMPT:     "Recovery Attempt",
  FINAL_DECISION:       "Final Determination",
  BIOPSY_REQUIRED:      "Biopsy Recommended",
};

// ─── Contradiction tier ──────────────────────────────────────────────────────

export type ContradictionTierCode = "NONE" | "MINOR" | "MODERATE" | "CRITICAL";

export const CONTRADICTION_TIER_LABELS: Record<ContradictionTierCode, string> = {
  NONE:     "No significant contradiction",
  MINOR:    "Minor conflicting signals",
  MODERATE: "Moderate diagnostic tension",
  CRITICAL: "Critical conflict — escalation warranted",
};

// ─── Recovery mechanism ──────────────────────────────────────────────────────

export type RecoveryMechanismCode =
  | "CONTRADICTION"
  | "LEADERSHIP"
  | "AMBIGUITY"
  | "TRAJECTORY"
  | "COMPETITION"
  | "ESCALATION"
  | "SIGNATURE_MATCH"
  | "UNEXPLAINED";

export const RECOVERY_MECHANISM_LABELS: Record<RecoveryMechanismCode, string> = {
  CONTRADICTION: "Contradiction Resolution",
  LEADERSHIP:    "Hypothesis Leadership",
  AMBIGUITY:     "Ambiguity Reduction",
  TRAJECTORY:    "Trajectory Convergence",
  COMPETITION:   "Competition Tiebreak",
  ESCALATION:      "Escalation Reroute",
  SIGNATURE_MATCH: "Pathognomonic Signature",
  UNEXPLAINED:     "Unexplained Recovery",
};

// ─── Sub-schemas ─────────────────────────────────────────────────────────────

export interface SymbolicSignal {
  name: string;
  value: number;          // [0, 1]
  activated: boolean;
  contribution_weight: number;  // [0, 1]
}

export interface DifferentialEntry {
  disease: string;
  certainty: number;   // [0, 1]
  rank: number;        // 1 = leading
  is_leading: boolean;
}

export interface ContradictionSummary {
  overall_load: number;    // [0, 0.40] — ceiling enforced
  tier: ContradictionTierCode;
  n_contradicting_signals: number;
  escalation_triggered_by_contradiction: boolean;
}

export interface TrajectoryState {
  step: number;
  certainty: number;       // [0, 1]
  ambiguity_bits: number;
  leading_disease: string;
  fsm_state: DiagnosticStateCode;
}

// ─── Root reasoning output ───────────────────────────────────────────────────

export interface ReasoningOutput {
  schema_version: typeof REASONING_CONTRACT_VERSION;
  case_id: string;
  fsm_state: DiagnosticStateCode;
  leading_diagnosis: string;
  certainty: number;        // [0, 1]
  ambiguity_bits: number;
  requires_biopsy: boolean;
  is_safe_triage: boolean;
  contradiction: ContradictionSummary;
  differential: DifferentialEntry[];
  symbolic_signals: SymbolicSignal[];
  trajectory: TrajectoryState[];
  recovery_mechanism?: RecoveryMechanismCode;
  recovery_successful?: boolean;
}
