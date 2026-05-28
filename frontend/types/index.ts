/**
 * types/index.ts
 * ================
 * Central re-export for all CASDRE workstation types.
 */

export * from "./reasoning-contract";
export * from "./replay-schema";
export * from "./graph-contract";
export * from "./clinical-input";

// ─── Shared UI state types ────────────────────────────────────────────────────

export interface CaseListItem {
  case_id: string;
  final_diagnosis: string;
  requires_biopsy: boolean;
  final_certainty: number;
  total_steps: number;
  converged: boolean;
  true_label?: string;
}

export type PanelId = "features" | "graph" | "certainty" | "replay";
