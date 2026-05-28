/**
 * types/graph-contract.ts
 * =========================
 * TypeScript mirror of the frozen CASDRE graph contract (v1.0.0).
 * Source: src/backend_stabilization/graph_contract_finalizer.py
 */

export const GRAPH_CONTRACT_VERSION = "1.0.0" as const;

// ─── Node and edge types ─────────────────────────────────────────────────────

export type GraphNodeType =
  | "disease"
  | "feature"
  | "signal"
  | "fsm_state"
  | "case"
  | "trajectory";

export type GraphEdgeType =
  | "supports"
  | "contradicts"
  | "transitions_to"
  | "activates"
  | "competes_with"
  | "recovers_via";

// ─── Visual mappings (for React Flow rendering) ────────────────────────────

export const NODE_TYPE_LABELS: Record<GraphNodeType, string> = {
  disease:    "Diagnostic Hypothesis",
  feature:    "Clinical Feature",
  signal:     "Symbolic Signal",
  fsm_state:  "Reasoning State",
  case:       "Patient Case",
  trajectory: "Trajectory",
};

export const EDGE_TYPE_COLORS: Record<GraphEdgeType, string> = {
  supports:       "#059669",  // emerald — supporting evidence
  contradicts:    "#dc2626",  // red — contradicting evidence
  transitions_to: "#94a3b8",  // slate — FSM transitions
  activates:      "#7c3aed",  // violet — signal activation
  competes_with:  "#d97706",  // amber — hypothesis competition
  recovers_via:   "#0891b2",  // cyan — recovery pathway
};

export const EDGE_TYPE_LABELS: Record<GraphEdgeType, string> = {
  supports:       "Supports",
  contradicts:    "Contradicts",
  transitions_to: "Transitions to",
  activates:      "Activates",
  competes_with:  "Competes with",
  recovers_via:   "Recovers via",
};

// ─── Schema types ────────────────────────────────────────────────────────────

export interface GraphNode {
  schema_version: typeof GRAPH_CONTRACT_VERSION;
  node_id: string;
  node_type: GraphNodeType;
  label: string;
  properties: Record<string, unknown>;
}

export interface GraphEdge {
  schema_version: typeof GRAPH_CONTRACT_VERSION;
  edge_id: string;
  source_id: string;
  target_id: string;
  edge_type: GraphEdgeType;
  weight: number;   // [0, 1]
  properties: Record<string, unknown>;
}

export interface GraphSnapshot {
  schema_version: typeof GRAPH_CONTRACT_VERSION;
  snapshot_id: string;
  case_id?: string;
  step?: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: Record<string, unknown>;
}
