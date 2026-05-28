/**
 * types/replay-schema.ts
 * ========================
 * TypeScript mirror of the frozen CASDRE replay schema (v1.0.0).
 * Source: src/backend_stabilization/replay_schema_finalizer.py
 */

export const REPLAY_SCHEMA_VERSION = "1.0.0" as const;

// ─── Event types ─────────────────────────────────────────────────────────────

export type ReplayEventType =
  | "case_start"
  | "clinical_eval"
  | "symbolic_enrichment"
  | "contradiction_check"
  | "ambiguity_resolution"
  | "trajectory_step"
  | "escalation_decision"
  | "recovery_attempt"
  | "final_decision"
  | "case_end";

export const REPLAY_EVENT_LABELS: Record<ReplayEventType, string> = {
  case_start:            "Case Initiated",
  clinical_eval:         "Clinical Feature Evaluation",
  symbolic_enrichment:   "Symbolic Signal Enrichment",
  contradiction_check:   "Contradiction Analysis",
  ambiguity_resolution:  "Ambiguity Resolution",
  trajectory_step:       "Trajectory Step",
  escalation_decision:   "Escalation Assessment",
  recovery_attempt:      "Recovery Attempt",
  final_decision:        "Final Determination",
  case_end:              "Case Closed",
};

export const REPLAY_EVENT_COLORS: Record<ReplayEventType, string> = {
  case_start:            "#94a3b8",
  clinical_eval:         "#2563eb",
  symbolic_enrichment:   "#7c3aed",
  contradiction_check:   "#dc2626",
  ambiguity_resolution:  "#d97706",
  trajectory_step:       "#0891b2",
  escalation_decision:   "#dc2626",
  recovery_attempt:      "#059669",
  final_decision:        "#1d3461",
  case_end:              "#94a3b8",
};

// ─── Event schema ─────────────────────────────────────────────────────────────

export interface ReplayEvent {
  schema_version: typeof REPLAY_SCHEMA_VERSION;
  event_type: ReplayEventType;
  case_id: string;
  step: number;
  fsm_state: string;
  certainty: number;           // [0, 1]
  ambiguity_bits: number;
  leading_diagnosis: string;
  contradiction_load: number;  // [0, 0.40] — ceiling enforced
  payload: Record<string, unknown>;
  timestamp_ms?: number;
}

// ─── Full case replay record ──────────────────────────────────────────────────

export interface ReplayCaseRecord {
  schema_version: typeof REPLAY_SCHEMA_VERSION;
  case_id: string;
  true_label?: string;
  final_diagnosis: string;
  events: ReplayEvent[];
  total_steps: number;
  converged: boolean;
  requires_biopsy: boolean;
  is_safe_triage: boolean;
  final_certainty: number;    // [0, 1]
}
