/**
 * services/backend-service.ts
 * =============================
 * Backend integration layer for the CASDRE workstation.
 *
 * Provides a unified interface over:
 *   - Mock data (development / demonstration mode)
 *   - Future real backend API (production)
 *
 * All methods return data conforming to frozen contract schemas v1.0.0.
 */

import type {
  ReplayCaseRecord,
  ReasoningOutput,
  GraphSnapshot,
  CaseListItem,
} from "@/types";
import type { FeatureVector } from "@/types/clinical-input";
import {
  MOCK_CASES,
  MOCK_CASE_LIST,
  type CaseBundle,
} from "@/data/mock-cases";

// ─── Configuration ────────────────────────────────────────────────────────────

const USE_MOCK = true;  // Set to false when real backend is available
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

// ─── Schema validation ────────────────────────────────────────────────────────

function assertSchemaVersion(data: { schema_version: string }, expected: string): void {
  if (data.schema_version !== expected) {
    console.warn(
      `Schema version mismatch: expected ${expected}, got ${data.schema_version}`
    );
  }
}

// ─── Service errors ───────────────────────────────────────────────────────────

export class BackendServiceError extends Error {
  constructor(
    public readonly code: string,
    message: string
  ) {
    super(message);
    this.name = "BackendServiceError";
  }
}

// ─── Backend service ──────────────────────────────────────────────────────────

export const BackendService = {
  /**
   * Returns the list of available cases for the case selector.
   */
  async listCases(): Promise<CaseListItem[]> {
    if (USE_MOCK) {
      return MOCK_CASE_LIST;
    }
    const res = await fetch(`${API_BASE}/cases`);
    if (!res.ok) throw new BackendServiceError("LIST_CASES_FAILED", res.statusText);
    return res.json();
  },

  /**
   * Loads a complete case bundle (replay + reasoning + graph).
   */
  async loadCase(caseId: string): Promise<CaseBundle> {
    if (USE_MOCK) {
      const bundle = MOCK_CASES[caseId];
      if (!bundle) {
        throw new BackendServiceError(
          "CASE_NOT_FOUND",
          `Case ${caseId} not found in mock data.`
        );
      }
      return bundle;
    }
    const [replayRes, reasoningRes, graphRes] = await Promise.all([
      fetch(`${API_BASE}/cases/${caseId}/replay`),
      fetch(`${API_BASE}/cases/${caseId}/reasoning`),
      fetch(`${API_BASE}/cases/${caseId}/graph`),
    ]);
    if (!replayRes.ok)    throw new BackendServiceError("REPLAY_LOAD_FAILED",    replayRes.statusText);
    if (!reasoningRes.ok) throw new BackendServiceError("REASONING_LOAD_FAILED", reasoningRes.statusText);
    if (!graphRes.ok)     throw new BackendServiceError("GRAPH_LOAD_FAILED",     graphRes.statusText);

    const [replay, reasoning, graph] = await Promise.all([
      replayRes.json()    as Promise<ReplayCaseRecord>,
      reasoningRes.json() as Promise<ReasoningOutput>,
      graphRes.json()     as Promise<GraphSnapshot>,
    ]);

    // Validate schemas
    assertSchemaVersion(replay,    "1.0.0");
    assertSchemaVersion(reasoning, "1.0.0");
    assertSchemaVersion(graph,     "1.0.0");

    return { replay, reasoning, graph };
  },

  /**
   * Runs diagnostic reasoning for a supplied feature vector.
   * In mock mode, selects the closest matching demo case using weighted feature scoring.
   */
  async runReasoning(features: FeatureVector): Promise<CaseBundle> {
    if (USE_MOCK) {
      // Simulate processing delay
      await new Promise(r => setTimeout(r, 1400));

      // Score each demo case against the input feature profile
      const scores: Record<string, number> = {
        "PSO-001":
          (features.erythema          ?? 0) * 1.5 +
          (features.scaling           ?? 0) * 1.5 +
          (features.scalp_involvement ?? 0) * 2.0 +
          (features.koebner_phenomenon ?? 0) * 2.0 +
          (features.family_history    ?? 0) * 1.5 +
          (features.knee_elbow_involv ?? 0) * 1.5 +
          (features.definite_borders  ?? 0) * 0.5,

        "PRP-001":
          (features.follicular_papules ?? 0) * 5.0 +
          (features.scaling            ?? 0) * 1.0 +
          (features.erythema           ?? 0) * 0.5 +
          (features.knee_elbow_involv  ?? 0) * 0.5,

        "LP-001":
          (features.polygonal_papules  ?? 0) * 5.0 +
          (features.oral_involvement   ?? 0) * 3.0 +
          (features.itching            ?? 0) * 1.5 +
          (features.melanin_incontinence ?? 0) * 2.0 +
          (features.definite_borders   ?? 0) * 0.5,
      };

      const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
      const bundle = MOCK_CASES[best];
      if (!bundle) throw new BackendServiceError("CASE_NOT_FOUND", `Matched case ${best} not found.`);
      return bundle;
    }

    const res = await fetch(`${API_BASE}/reason`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ features }),
    });
    if (!res.ok) throw new BackendServiceError("REASONING_FAILED", res.statusText);
    const data = await res.json() as { replay: ReplayCaseRecord; reasoning: ReasoningOutput; graph: GraphSnapshot };
    assertSchemaVersion(data.reasoning, "1.0.0");
    return data;
  },

  /**
   * Validates a reasoning output against the frozen contract.
   * Returns null if valid, or an error message.
   */
  validateReasoningOutput(output: ReasoningOutput): string | null {
    if (output.contradiction.overall_load > 0.40) {
      return `Contradiction load ${output.contradiction.overall_load.toFixed(3)} exceeds ceiling 0.40`;
    }
    if (output.requires_biopsy && output.is_safe_triage) {
      return "requires_biopsy and is_safe_triage cannot both be true";
    }
    const leadingCount = output.differential.filter((d) => d.is_leading).length;
    if (leadingCount !== 1) {
      return `Expected exactly 1 leading diagnosis, found ${leadingCount}`;
    }
    return null;
  },
};
