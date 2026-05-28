/**
 * data/mock-cases.ts
 * ====================
 * Representative clinical mock cases for workstation demonstration.
 * All data conforms to frozen backend contracts v1.0.0.
 *
 * Cases:
 *   PSO-001 — Psoriasis, high-certainty convergence, no escalation
 *   PRP-001 — Pityriasis rubra pilaris, contradiction-driven escalation
 *   LP-001  — Lichen planus, symbolic recovery from discriminative ambiguity
 */

import type {
  ReplayCaseRecord,
  ReasoningOutput,
  GraphSnapshot,
} from "@/types";

const V = "1.0.0" as const;

// ──────────────────────────────────────────────────────────────────────────────
// CASE 1 — PSO-001: Psoriasis — clear convergence, no biopsy
// ──────────────────────────────────────────────────────────────────────────────

export const PSO_001_REPLAY: ReplayCaseRecord = {
  schema_version: V,
  case_id: "PSO-001",
  true_label: "psoriasis",
  final_diagnosis: "psoriasis",
  total_steps: 8,
  converged: true,
  requires_biopsy: false,
  is_safe_triage: true,
  final_certainty: 0.91,
  events: [
    {
      schema_version: V, case_id: "PSO-001", step: 0,
      event_type: "case_start",
      fsm_state: "INITIAL",
      certainty: 0.17, ambiguity_bits: 2.58, leading_diagnosis: "psoriasis",
      contradiction_load: 0.00, payload: {},
    },
    {
      schema_version: V, case_id: "PSO-001", step: 1,
      event_type: "clinical_eval",
      fsm_state: "CLINICAL_ASSESSMENT",
      certainty: 0.34, ambiguity_bits: 2.12, leading_diagnosis: "psoriasis",
      contradiction_load: 0.06,
      payload: { features_activated: ["erythema", "scaling", "koebner_phenomenon"] },
    },
    {
      schema_version: V, case_id: "PSO-001", step: 2,
      event_type: "symbolic_enrichment",
      fsm_state: "SYMBOLIC_ENRICHMENT",
      certainty: 0.52, ambiguity_bits: 1.74, leading_diagnosis: "psoriasis",
      contradiction_load: 0.09,
      payload: { rules_activated: ["PSO_001", "PSO_002"] },
    },
    {
      schema_version: V, case_id: "PSO-001", step: 3,
      event_type: "contradiction_check",
      fsm_state: "CONTRADICTION_CHECK",
      certainty: 0.52, ambiguity_bits: 1.74, leading_diagnosis: "psoriasis",
      contradiction_load: 0.09,
      payload: { tier: "MINOR", n_contradicting: 1 },
    },
    {
      schema_version: V, case_id: "PSO-001", step: 4,
      event_type: "trajectory_step",
      fsm_state: "AMBIGUITY_RESOLUTION",
      certainty: 0.67, ambiguity_bits: 1.31, leading_diagnosis: "psoriasis",
      contradiction_load: 0.07,
      payload: {},
    },
    {
      schema_version: V, case_id: "PSO-001", step: 5,
      event_type: "trajectory_step",
      fsm_state: "AMBIGUITY_RESOLUTION",
      certainty: 0.79, ambiguity_bits: 0.94, leading_diagnosis: "psoriasis",
      contradiction_load: 0.05,
      payload: {},
    },
    {
      schema_version: V, case_id: "PSO-001", step: 6,
      event_type: "escalation_decision",
      fsm_state: "ESCALATION_REVIEW",
      certainty: 0.79, ambiguity_bits: 0.94, leading_diagnosis: "psoriasis",
      contradiction_load: 0.05,
      payload: { escalated: false, reason: "Certainty above stabilisation threshold" },
    },
    {
      schema_version: V, case_id: "PSO-001", step: 7,
      event_type: "final_decision",
      fsm_state: "FINAL_DECISION",
      certainty: 0.91, ambiguity_bits: 0.43, leading_diagnosis: "psoriasis",
      contradiction_load: 0.04,
      payload: { requires_biopsy: false, is_safe_triage: true },
    },
  ],
};

export const PSO_001_REASONING: ReasoningOutput = {
  schema_version: V,
  case_id: "PSO-001",
  fsm_state: "FINAL_DECISION",
  leading_diagnosis: "psoriasis",
  certainty: 0.91,
  ambiguity_bits: 0.43,
  requires_biopsy: false,
  is_safe_triage: true,
  contradiction: {
    overall_load: 0.04,
    tier: "MINOR",
    n_contradicting_signals: 1,
    escalation_triggered_by_contradiction: false,
  },
  differential: [
    { disease: "psoriasis",              certainty: 0.91, rank: 1, is_leading: true  },
    { disease: "seborrheic_dermatitis",  certainty: 0.04, rank: 2, is_leading: false },
    { disease: "lichen_planus",          certainty: 0.02, rank: 3, is_leading: false },
    { disease: "pityriasis_rosea",       certainty: 0.01, rank: 4, is_leading: false },
    { disease: "chronic_dermatitis",     certainty: 0.01, rank: 5, is_leading: false },
    { disease: "pityriasis_rubra_pilaris", certainty: 0.01, rank: 6, is_leading: false },
  ],
  symbolic_signals: [
    { name: "scaling × erythema",       value: 0.92, activated: true,  contribution_weight: 0.28 },
    { name: "knee/elbow involvement",   value: 0.88, activated: true,  contribution_weight: 0.22 },
    { name: "scalp involvement",        value: 0.76, activated: true,  contribution_weight: 0.18 },
    { name: "Koebner phenomenon",       value: 0.71, activated: true,  contribution_weight: 0.14 },
    { name: "family history",           value: 0.60, activated: true,  contribution_weight: 0.10 },
    { name: "oral involvement",         value: 0.04, activated: false, contribution_weight: 0.02 },
    { name: "follicular papules",       value: 0.08, activated: false, contribution_weight: 0.02 },
    { name: "melanin incontinence",     value: 0.12, activated: false, contribution_weight: 0.04 },
  ],
  trajectory: [
    { step: 0, certainty: 0.17, ambiguity_bits: 2.58, leading_disease: "psoriasis",             fsm_state: "INITIAL"              },
    { step: 1, certainty: 0.34, ambiguity_bits: 2.12, leading_disease: "psoriasis",             fsm_state: "CLINICAL_ASSESSMENT"  },
    { step: 2, certainty: 0.52, ambiguity_bits: 1.74, leading_disease: "psoriasis",             fsm_state: "SYMBOLIC_ENRICHMENT"  },
    { step: 3, certainty: 0.52, ambiguity_bits: 1.74, leading_disease: "psoriasis",             fsm_state: "CONTRADICTION_CHECK"  },
    { step: 4, certainty: 0.67, ambiguity_bits: 1.31, leading_disease: "psoriasis",             fsm_state: "AMBIGUITY_RESOLUTION" },
    { step: 5, certainty: 0.79, ambiguity_bits: 0.94, leading_disease: "psoriasis",             fsm_state: "AMBIGUITY_RESOLUTION" },
    { step: 6, certainty: 0.79, ambiguity_bits: 0.94, leading_disease: "psoriasis",             fsm_state: "ESCALATION_REVIEW"    },
    { step: 7, certainty: 0.91, ambiguity_bits: 0.43, leading_disease: "psoriasis",             fsm_state: "FINAL_DECISION"       },
  ],
};

export const PSO_001_GRAPH: GraphSnapshot = {
  schema_version: V,
  snapshot_id: "PSO-001-snap",
  case_id: "PSO-001",
  step: 7,
  nodes: [
    { schema_version: V, node_id: "d_pso", node_type: "disease", label: "Psoriasis",                 properties: { certainty: 0.91, rank: 1, is_leading: true  } },
    { schema_version: V, node_id: "d_seb", node_type: "disease", label: "Seborrheic Dermatitis",     properties: { certainty: 0.04, rank: 2, is_leading: false } },
    { schema_version: V, node_id: "d_lp",  node_type: "disease", label: "Lichen Planus",             properties: { certainty: 0.02, rank: 3, is_leading: false } },
    { schema_version: V, node_id: "d_pr",  node_type: "disease", label: "Pityriasis Rosea",          properties: { certainty: 0.01, rank: 4, is_leading: false } },
    { schema_version: V, node_id: "d_chr", node_type: "disease", label: "Chronic Dermatitis",        properties: { certainty: 0.01, rank: 5, is_leading: false } },
    { schema_version: V, node_id: "d_prp", node_type: "disease", label: "Pityriasis Rubra Pilaris",  properties: { certainty: 0.01, rank: 6, is_leading: false } },
    { schema_version: V, node_id: "f_ery", node_type: "feature", label: "Erythema",                  properties: { value: 3, activated: true  } },
    { schema_version: V, node_id: "f_sca", node_type: "feature", label: "Scaling",                   properties: { value: 3, activated: true  } },
    { schema_version: V, node_id: "f_koe", node_type: "feature", label: "Koebner Phenomenon",        properties: { value: 2, activated: true  } },
    { schema_version: V, node_id: "f_kne", node_type: "feature", label: "Knee/Elbow Involvement",    properties: { value: 2, activated: true  } },
    { schema_version: V, node_id: "f_scp", node_type: "feature", label: "Scalp Involvement",         properties: { value: 2, activated: true  } },
    { schema_version: V, node_id: "f_fam", node_type: "feature", label: "Family History",             properties: { value: 1, activated: true  } },
    { schema_version: V, node_id: "f_ora", node_type: "feature", label: "Oral Involvement",           properties: { value: 0, activated: false } },
    { schema_version: V, node_id: "s_pso1", node_type: "signal", label: "PSO: Scaling Triad",        properties: { rule: "PSO_001", activated: true,  weight: 0.28 } },
    { schema_version: V, node_id: "s_pso2", node_type: "signal", label: "PSO: Koebner+Erythema",     properties: { rule: "PSO_002", activated: true,  weight: 0.22 } },
  ],
  edges: [
    { schema_version: V, edge_id: "e1",  source_id: "f_sca", target_id: "d_pso", edge_type: "supports",    weight: 0.28, properties: {} },
    { schema_version: V, edge_id: "e2",  source_id: "f_ery", target_id: "d_pso", edge_type: "supports",    weight: 0.22, properties: {} },
    { schema_version: V, edge_id: "e3",  source_id: "f_koe", target_id: "d_pso", edge_type: "supports",    weight: 0.18, properties: {} },
    { schema_version: V, edge_id: "e4",  source_id: "f_kne", target_id: "d_pso", edge_type: "supports",    weight: 0.16, properties: {} },
    { schema_version: V, edge_id: "e5",  source_id: "f_scp", target_id: "d_pso", edge_type: "supports",    weight: 0.14, properties: {} },
    { schema_version: V, edge_id: "e6",  source_id: "f_fam", target_id: "d_pso", edge_type: "supports",    weight: 0.10, properties: {} },
    { schema_version: V, edge_id: "e7",  source_id: "f_sca", target_id: "d_seb", edge_type: "supports",    weight: 0.08, properties: {} },
    { schema_version: V, edge_id: "e8",  source_id: "f_ery", target_id: "d_seb", edge_type: "supports",    weight: 0.06, properties: {} },
    { schema_version: V, edge_id: "e9",  source_id: "f_ora", target_id: "d_pso", edge_type: "contradicts", weight: 0.04, properties: {} },
    { schema_version: V, edge_id: "e10", source_id: "d_pso", target_id: "d_seb", edge_type: "competes_with", weight: 0.04, properties: {} },
    { schema_version: V, edge_id: "e11", source_id: "s_pso1", target_id: "d_pso", edge_type: "activates",  weight: 0.28, properties: {} },
    { schema_version: V, edge_id: "e12", source_id: "s_pso2", target_id: "d_pso", edge_type: "activates",  weight: 0.22, properties: {} },
    { schema_version: V, edge_id: "e13", source_id: "f_sca", target_id: "s_pso1", edge_type: "activates",  weight: 0.92, properties: {} },
    { schema_version: V, edge_id: "e14", source_id: "f_koe", target_id: "s_pso2", edge_type: "activates",  weight: 0.71, properties: {} },
  ],
  metadata: { case_id: "PSO-001", final_diagnosis: "psoriasis" },
};

// ──────────────────────────────────────────────────────────────────────────────
// CASE 2 — PRP-001: Pityriasis rubra pilaris — escalation due to contradiction
// ──────────────────────────────────────────────────────────────────────────────

export const PRP_001_REPLAY: ReplayCaseRecord = {
  schema_version: V,
  case_id: "PRP-001",
  true_label: "pityriasis_rubra_pilaris",
  final_diagnosis: "pityriasis_rubra_pilaris",
  total_steps: 9,
  converged: false,
  requires_biopsy: true,
  is_safe_triage: false,
  final_certainty: 0.54,
  events: [
    {
      schema_version: V, case_id: "PRP-001", step: 0,
      event_type: "case_start",
      fsm_state: "INITIAL",
      certainty: 0.14, ambiguity_bits: 2.81, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.00, payload: {},
    },
    {
      schema_version: V, case_id: "PRP-001", step: 1,
      event_type: "clinical_eval",
      fsm_state: "CLINICAL_ASSESSMENT",
      certainty: 0.22, ambiguity_bits: 2.64, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.08,
      payload: { features_activated: ["follicular_papules", "scaling", "knee_elbow_involvement"] },
    },
    {
      schema_version: V, case_id: "PRP-001", step: 2,
      event_type: "symbolic_enrichment",
      fsm_state: "SYMBOLIC_ENRICHMENT",
      certainty: 0.36, ambiguity_bits: 2.47, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.14,
      payload: { rules_activated: ["PRP_001"] },
    },
    {
      schema_version: V, case_id: "PRP-001", step: 3,
      event_type: "contradiction_check",
      fsm_state: "CONTRADICTION_CHECK",
      certainty: 0.36, ambiguity_bits: 2.47, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.22,
      payload: { tier: "MODERATE", n_contradicting: 3, competing: "psoriasis" },
    },
    {
      schema_version: V, case_id: "PRP-001", step: 4,
      event_type: "trajectory_step",
      fsm_state: "AMBIGUITY_RESOLUTION",
      certainty: 0.40, ambiguity_bits: 2.31, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.24,
      payload: {},
    },
    {
      schema_version: V, case_id: "PRP-001", step: 5,
      event_type: "recovery_attempt",
      fsm_state: "RECOVERY_ATTEMPT",
      certainty: 0.44, ambiguity_bits: 2.18, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.26,
      payload: { mechanism: "CONTRADICTION", success: false },
    },
    {
      schema_version: V, case_id: "PRP-001", step: 6,
      event_type: "trajectory_step",
      fsm_state: "AMBIGUITY_RESOLUTION",
      certainty: 0.48, ambiguity_bits: 2.02, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.28,
      payload: {},
    },
    {
      schema_version: V, case_id: "PRP-001", step: 7,
      event_type: "escalation_decision",
      fsm_state: "ESCALATION_REVIEW",
      certainty: 0.48, ambiguity_bits: 2.02, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.28,
      payload: { escalated: true, reason: "Persistent contradiction — rare class — biopsy warranted" },
    },
    {
      schema_version: V, case_id: "PRP-001", step: 8,
      event_type: "final_decision",
      fsm_state: "BIOPSY_REQUIRED",
      certainty: 0.54, ambiguity_bits: 1.74, leading_diagnosis: "pityriasis_rubra_pilaris",
      contradiction_load: 0.28,
      payload: { requires_biopsy: true, is_safe_triage: false },
    },
  ],
};

export const PRP_001_REASONING: ReasoningOutput = {
  schema_version: V,
  case_id: "PRP-001",
  fsm_state: "BIOPSY_REQUIRED",
  leading_diagnosis: "pityriasis_rubra_pilaris",
  certainty: 0.54,
  ambiguity_bits: 1.74,
  requires_biopsy: true,
  is_safe_triage: false,
  contradiction: {
    overall_load: 0.28,
    tier: "MODERATE",
    n_contradicting_signals: 3,
    escalation_triggered_by_contradiction: true,
  },
  differential: [
    { disease: "pityriasis_rubra_pilaris", certainty: 0.54, rank: 1, is_leading: true  },
    { disease: "psoriasis",               certainty: 0.30, rank: 2, is_leading: false },
    { disease: "seborrheic_dermatitis",   certainty: 0.07, rank: 3, is_leading: false },
    { disease: "lichen_planus",           certainty: 0.05, rank: 4, is_leading: false },
    { disease: "pityriasis_rosea",        certainty: 0.02, rank: 5, is_leading: false },
    { disease: "chronic_dermatitis",      certainty: 0.02, rank: 6, is_leading: false },
  ],
  symbolic_signals: [
    { name: "follicular papules × scaling", value: 0.86, activated: true,  contribution_weight: 0.30 },
    { name: "knee/elbow × follicular pap.", value: 0.79, activated: true,  contribution_weight: 0.25 },
    { name: "scalp + follicular papules",   value: 0.62, activated: true,  contribution_weight: 0.18 },
    { name: "scaling × erythema (PSO)",     value: 0.71, activated: true,  contribution_weight: 0.20 },
    { name: "Koebner phenomenon (PSO)",     value: 0.44, activated: true,  contribution_weight: 0.12 },
    { name: "oral involvement (anti-PRP)",  value: 0.05, activated: false, contribution_weight: 0.02 },
    { name: "polygonal papules (anti-PRP)", value: 0.08, activated: false, contribution_weight: 0.02 },
  ],
  trajectory: [
    { step: 0, certainty: 0.14, ambiguity_bits: 2.81, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "INITIAL"              },
    { step: 1, certainty: 0.22, ambiguity_bits: 2.64, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "CLINICAL_ASSESSMENT"  },
    { step: 2, certainty: 0.36, ambiguity_bits: 2.47, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "SYMBOLIC_ENRICHMENT"  },
    { step: 3, certainty: 0.36, ambiguity_bits: 2.47, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "CONTRADICTION_CHECK"  },
    { step: 4, certainty: 0.40, ambiguity_bits: 2.31, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "AMBIGUITY_RESOLUTION" },
    { step: 5, certainty: 0.44, ambiguity_bits: 2.18, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "RECOVERY_ATTEMPT"     },
    { step: 6, certainty: 0.48, ambiguity_bits: 2.02, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "AMBIGUITY_RESOLUTION" },
    { step: 7, certainty: 0.48, ambiguity_bits: 2.02, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "ESCALATION_REVIEW"    },
    { step: 8, certainty: 0.54, ambiguity_bits: 1.74, leading_disease: "pityriasis_rubra_pilaris", fsm_state: "BIOPSY_REQUIRED"      },
  ],
  recovery_mechanism: "CONTRADICTION",
  recovery_successful: false,
};

export const PRP_001_GRAPH: GraphSnapshot = {
  schema_version: V,
  snapshot_id: "PRP-001-snap",
  case_id: "PRP-001",
  step: 8,
  nodes: [
    { schema_version: V, node_id: "d_prp", node_type: "disease", label: "Pityriasis Rubra Pilaris", properties: { certainty: 0.54, rank: 1, is_leading: true  } },
    { schema_version: V, node_id: "d_pso", node_type: "disease", label: "Psoriasis",                properties: { certainty: 0.30, rank: 2, is_leading: false } },
    { schema_version: V, node_id: "d_seb", node_type: "disease", label: "Seborrheic Dermatitis",    properties: { certainty: 0.07, rank: 3, is_leading: false } },
    { schema_version: V, node_id: "d_lp",  node_type: "disease", label: "Lichen Planus",            properties: { certainty: 0.05, rank: 4, is_leading: false } },
    { schema_version: V, node_id: "d_pr",  node_type: "disease", label: "Pityriasis Rosea",         properties: { certainty: 0.02, rank: 5, is_leading: false } },
    { schema_version: V, node_id: "d_chr", node_type: "disease", label: "Chronic Dermatitis",       properties: { certainty: 0.02, rank: 6, is_leading: false } },
    { schema_version: V, node_id: "f_fol", node_type: "feature", label: "Follicular Papules",       properties: { value: 3, activated: true  } },
    { schema_version: V, node_id: "f_sca", node_type: "feature", label: "Scaling",                  properties: { value: 2, activated: true  } },
    { schema_version: V, node_id: "f_kne", node_type: "feature", label: "Knee/Elbow Involvement",   properties: { value: 2, activated: true  } },
    { schema_version: V, node_id: "f_ery", node_type: "feature", label: "Erythema",                 properties: { value: 2, activated: true  } },
    { schema_version: V, node_id: "f_koe", node_type: "feature", label: "Koebner Phenomenon",       properties: { value: 1, activated: true  } },
    { schema_version: V, node_id: "f_ora", node_type: "feature", label: "Oral Involvement",          properties: { value: 0, activated: false } },
    { schema_version: V, node_id: "s_prp1", node_type: "signal", label: "PRP: Follicular Triad",    properties: { rule: "PRP_001", activated: true,  weight: 0.30 } },
    { schema_version: V, node_id: "s_pso1", node_type: "signal", label: "PSO: Scaling Triad",       properties: { rule: "PSO_001", activated: true,  weight: 0.20 } },
  ],
  edges: [
    { schema_version: V, edge_id: "e1",  source_id: "f_fol", target_id: "d_prp", edge_type: "supports",    weight: 0.30, properties: {} },
    { schema_version: V, edge_id: "e2",  source_id: "f_kne", target_id: "d_prp", edge_type: "supports",    weight: 0.25, properties: {} },
    { schema_version: V, edge_id: "e3",  source_id: "f_sca", target_id: "d_prp", edge_type: "supports",    weight: 0.18, properties: {} },
    { schema_version: V, edge_id: "e4",  source_id: "f_sca", target_id: "d_pso", edge_type: "supports",    weight: 0.20, properties: {} },
    { schema_version: V, edge_id: "e5",  source_id: "f_ery", target_id: "d_pso", edge_type: "supports",    weight: 0.18, properties: {} },
    { schema_version: V, edge_id: "e6",  source_id: "f_koe", target_id: "d_pso", edge_type: "supports",    weight: 0.12, properties: {} },
    { schema_version: V, edge_id: "e7",  source_id: "f_fol", target_id: "d_pso", edge_type: "contradicts", weight: 0.15, properties: { note: "Follicular papules uncommon in psoriasis" } },
    { schema_version: V, edge_id: "e8",  source_id: "f_koe", target_id: "d_prp", edge_type: "contradicts", weight: 0.12, properties: { note: "Koebner uncommon in PRP" } },
    { schema_version: V, edge_id: "e9",  source_id: "d_prp", target_id: "d_pso", edge_type: "competes_with", weight: 0.30, properties: {} },
    { schema_version: V, edge_id: "e10", source_id: "s_prp1", target_id: "d_prp", edge_type: "activates",  weight: 0.30, properties: {} },
    { schema_version: V, edge_id: "e11", source_id: "s_pso1", target_id: "d_pso", edge_type: "activates",  weight: 0.20, properties: {} },
    { schema_version: V, edge_id: "e12", source_id: "f_fol", target_id: "s_prp1", edge_type: "activates",  weight: 0.86, properties: {} },
    { schema_version: V, edge_id: "e13", source_id: "f_sca", target_id: "s_pso1", edge_type: "activates",  weight: 0.71, properties: {} },
  ],
  metadata: { case_id: "PRP-001", final_diagnosis: "pityriasis_rubra_pilaris", requires_biopsy: true },
};

// ──────────────────────────────────────────────────────────────────────────────
// CASE 3 — LP-001: Lichen planus — symbolic recovery from discriminative ambiguity
// ──────────────────────────────────────────────────────────────────────────────

export const LP_001_REPLAY: ReplayCaseRecord = {
  schema_version: V,
  case_id: "LP-001",
  true_label: "lichen_planus",
  final_diagnosis: "lichen_planus",
  total_steps: 8,
  converged: true,
  requires_biopsy: false,
  is_safe_triage: true,
  final_certainty: 0.82,
  events: [
    {
      schema_version: V, case_id: "LP-001", step: 0,
      event_type: "case_start",
      fsm_state: "INITIAL",
      certainty: 0.15, ambiguity_bits: 2.73, leading_diagnosis: "lichen_planus",
      contradiction_load: 0.00, payload: {},
    },
    {
      schema_version: V, case_id: "LP-001", step: 1,
      event_type: "clinical_eval",
      fsm_state: "CLINICAL_ASSESSMENT",
      certainty: 0.28, ambiguity_bits: 2.41, leading_diagnosis: "chronic_dermatitis",
      contradiction_load: 0.05,
      payload: { features_activated: ["itching", "erythema"] },
    },
    {
      schema_version: V, case_id: "LP-001", step: 2,
      event_type: "symbolic_enrichment",
      fsm_state: "SYMBOLIC_ENRICHMENT",
      certainty: 0.21, ambiguity_bits: 2.62, leading_diagnosis: "lichen_planus",
      contradiction_load: 0.10,
      payload: { rules_activated: ["LP_001", "LP_002"], note: "Symbolic recovery from chronic_dermatitis" },
    },
    {
      schema_version: V, case_id: "LP-001", step: 3,
      event_type: "contradiction_check",
      fsm_state: "CONTRADICTION_CHECK",
      certainty: 0.21, ambiguity_bits: 2.62, leading_diagnosis: "lichen_planus",
      contradiction_load: 0.14,
      payload: { tier: "MINOR", n_contradicting: 2 },
    },
    {
      schema_version: V, case_id: "LP-001", step: 4,
      event_type: "recovery_attempt",
      fsm_state: "RECOVERY_ATTEMPT",
      certainty: 0.45, ambiguity_bits: 2.08, leading_diagnosis: "lichen_planus",
      contradiction_load: 0.12,
      payload: { mechanism: "SIGNATURE_MATCH", success: true, rule: "LP_001" },
    },
    {
      schema_version: V, case_id: "LP-001", step: 5,
      event_type: "trajectory_step",
      fsm_state: "AMBIGUITY_RESOLUTION",
      certainty: 0.60, ambiguity_bits: 1.58, leading_diagnosis: "lichen_planus",
      contradiction_load: 0.09,
      payload: {},
    },
    {
      schema_version: V, case_id: "LP-001", step: 6,
      event_type: "escalation_decision",
      fsm_state: "ESCALATION_REVIEW",
      certainty: 0.60, ambiguity_bits: 1.58, leading_diagnosis: "lichen_planus",
      contradiction_load: 0.09,
      payload: { escalated: false, reason: "Pathognomonic rule LP_001 activated — confident stabilisation" },
    },
    {
      schema_version: V, case_id: "LP-001", step: 7,
      event_type: "final_decision",
      fsm_state: "FINAL_DECISION",
      certainty: 0.82, ambiguity_bits: 0.78, leading_diagnosis: "lichen_planus",
      contradiction_load: 0.07,
      payload: { requires_biopsy: false, is_safe_triage: true },
    },
  ],
};

export const LP_001_REASONING: ReasoningOutput = {
  schema_version: V,
  case_id: "LP-001",
  fsm_state: "FINAL_DECISION",
  leading_diagnosis: "lichen_planus",
  certainty: 0.82,
  ambiguity_bits: 0.78,
  requires_biopsy: false,
  is_safe_triage: true,
  contradiction: {
    overall_load: 0.07,
    tier: "MINOR",
    n_contradicting_signals: 2,
    escalation_triggered_by_contradiction: false,
  },
  differential: [
    { disease: "lichen_planus",          certainty: 0.82, rank: 1, is_leading: true  },
    { disease: "chronic_dermatitis",     certainty: 0.09, rank: 2, is_leading: false },
    { disease: "psoriasis",              certainty: 0.04, rank: 3, is_leading: false },
    { disease: "pityriasis_rosea",       certainty: 0.02, rank: 4, is_leading: false },
    { disease: "seborrheic_dermatitis",  certainty: 0.02, rank: 5, is_leading: false },
    { disease: "pityriasis_rubra_pilaris", certainty: 0.01, rank: 6, is_leading: false },
  ],
  symbolic_signals: [
    { name: "polygonal pap. × oral inv.",  value: 0.91, activated: true,  contribution_weight: 0.35 },
    { name: "Koebner × polygonal papules", value: 0.74, activated: true,  contribution_weight: 0.22 },
    { name: "oral involvement",            value: 0.82, activated: true,  contribution_weight: 0.18 },
    { name: "itching + erythema (CHR)",    value: 0.63, activated: true,  contribution_weight: 0.12 },
    { name: "scalp exclusion signal",      value: 0.12, activated: false, contribution_weight: 0.04 },
    { name: "follicular papules (anti-LP)",value: 0.06, activated: false, contribution_weight: 0.02 },
    { name: "family history",              value: 0.10, activated: false, contribution_weight: 0.02 },
  ],
  trajectory: [
    { step: 0, certainty: 0.15, ambiguity_bits: 2.73, leading_disease: "lichen_planus",      fsm_state: "INITIAL"              },
    { step: 1, certainty: 0.28, ambiguity_bits: 2.41, leading_disease: "chronic_dermatitis", fsm_state: "CLINICAL_ASSESSMENT"  },
    { step: 2, certainty: 0.21, ambiguity_bits: 2.62, leading_disease: "lichen_planus",      fsm_state: "SYMBOLIC_ENRICHMENT"  },
    { step: 3, certainty: 0.21, ambiguity_bits: 2.62, leading_disease: "lichen_planus",      fsm_state: "CONTRADICTION_CHECK"  },
    { step: 4, certainty: 0.45, ambiguity_bits: 2.08, leading_disease: "lichen_planus",      fsm_state: "RECOVERY_ATTEMPT"     },
    { step: 5, certainty: 0.60, ambiguity_bits: 1.58, leading_disease: "lichen_planus",      fsm_state: "AMBIGUITY_RESOLUTION" },
    { step: 6, certainty: 0.60, ambiguity_bits: 1.58, leading_disease: "lichen_planus",      fsm_state: "ESCALATION_REVIEW"    },
    { step: 7, certainty: 0.82, ambiguity_bits: 0.78, leading_disease: "lichen_planus",      fsm_state: "FINAL_DECISION"       },
  ],
  recovery_mechanism: "SIGNATURE_MATCH",
  recovery_successful: true,
};

export const LP_001_GRAPH: GraphSnapshot = {
  schema_version: V,
  snapshot_id: "LP-001-snap",
  case_id: "LP-001",
  step: 7,
  nodes: [
    { schema_version: V, node_id: "d_lp",  node_type: "disease", label: "Lichen Planus",            properties: { certainty: 0.82, rank: 1, is_leading: true  } },
    { schema_version: V, node_id: "d_chr", node_type: "disease", label: "Chronic Dermatitis",        properties: { certainty: 0.09, rank: 2, is_leading: false } },
    { schema_version: V, node_id: "d_pso", node_type: "disease", label: "Psoriasis",                 properties: { certainty: 0.04, rank: 3, is_leading: false } },
    { schema_version: V, node_id: "d_pr",  node_type: "disease", label: "Pityriasis Rosea",          properties: { certainty: 0.02, rank: 4, is_leading: false } },
    { schema_version: V, node_id: "d_seb", node_type: "disease", label: "Seborrheic Dermatitis",     properties: { certainty: 0.02, rank: 5, is_leading: false } },
    { schema_version: V, node_id: "d_prp", node_type: "disease", label: "Pityriasis Rubra Pilaris",  properties: { certainty: 0.01, rank: 6, is_leading: false } },
    { schema_version: V, node_id: "f_pol", node_type: "feature", label: "Polygonal Papules",         properties: { value: 3, activated: true  } },
    { schema_version: V, node_id: "f_ora", node_type: "feature", label: "Oral Involvement",           properties: { value: 3, activated: true  } },
    { schema_version: V, node_id: "f_koe", node_type: "feature", label: "Koebner Phenomenon",        properties: { value: 2, activated: true  } },
    { schema_version: V, node_id: "f_itch",node_type: "feature", label: "Itching",                   properties: { value: 2, activated: true  } },
    { schema_version: V, node_id: "f_ery", node_type: "feature", label: "Erythema",                  properties: { value: 1, activated: true  } },
    { schema_version: V, node_id: "f_sca", node_type: "feature", label: "Scaling",                   properties: { value: 1, activated: true  } },
    { schema_version: V, node_id: "s_lp1", node_type: "signal",  label: "LP: Pathognomonic (poly+oral)", properties: { rule: "LP_001", activated: true,  weight: 0.35, pathognomonic: true } },
    { schema_version: V, node_id: "s_lp2", node_type: "signal",  label: "LP: Koebner+Poly",          properties: { rule: "LP_002", activated: true,  weight: 0.22 } },
  ],
  edges: [
    { schema_version: V, edge_id: "e1",  source_id: "f_pol",  target_id: "d_lp",  edge_type: "supports",    weight: 0.35, properties: {} },
    { schema_version: V, edge_id: "e2",  source_id: "f_ora",  target_id: "d_lp",  edge_type: "supports",    weight: 0.28, properties: {} },
    { schema_version: V, edge_id: "e3",  source_id: "f_koe",  target_id: "d_lp",  edge_type: "supports",    weight: 0.22, properties: {} },
    { schema_version: V, edge_id: "e4",  source_id: "f_itch", target_id: "d_chr", edge_type: "supports",    weight: 0.20, properties: {} },
    { schema_version: V, edge_id: "e5",  source_id: "f_ery",  target_id: "d_chr", edge_type: "supports",    weight: 0.12, properties: {} },
    { schema_version: V, edge_id: "e6",  source_id: "f_itch", target_id: "d_lp",  edge_type: "contradicts", weight: 0.07, properties: { note: "Itching less specific in LP" } },
    { schema_version: V, edge_id: "e7",  source_id: "d_lp",   target_id: "d_chr", edge_type: "competes_with", weight: 0.09, properties: {} },
    { schema_version: V, edge_id: "e8",  source_id: "s_lp1",  target_id: "d_lp",  edge_type: "activates",   weight: 0.35, properties: {} },
    { schema_version: V, edge_id: "e9",  source_id: "s_lp2",  target_id: "d_lp",  edge_type: "activates",   weight: 0.22, properties: {} },
    { schema_version: V, edge_id: "e10", source_id: "f_pol",  target_id: "s_lp1", edge_type: "activates",   weight: 0.91, properties: {} },
    { schema_version: V, edge_id: "e11", source_id: "f_ora",  target_id: "s_lp1", edge_type: "activates",   weight: 0.82, properties: {} },
    { schema_version: V, edge_id: "e12", source_id: "f_koe",  target_id: "s_lp2", edge_type: "activates",   weight: 0.74, properties: {} },
    { schema_version: V, edge_id: "e13", source_id: "d_chr",  target_id: "d_lp",  edge_type: "recovers_via",weight: 0.45, properties: { mechanism: "SIGNATURE_MATCH" } },
  ],
  metadata: { case_id: "LP-001", final_diagnosis: "lichen_planus", recovery: "SIGNATURE_MATCH" },
};

// ──────────────────────────────────────────────────────────────────────────────
// Catalog export
// ──────────────────────────────────────────────────────────────────────────────

export interface CaseBundle {
  replay: ReplayCaseRecord;
  reasoning: ReasoningOutput;
  graph: GraphSnapshot;
}

export const MOCK_CASES: Record<string, CaseBundle> = {
  "PSO-001": { replay: PSO_001_REPLAY, reasoning: PSO_001_REASONING, graph: PSO_001_GRAPH },
  "PRP-001": { replay: PRP_001_REPLAY, reasoning: PRP_001_REASONING, graph: PRP_001_GRAPH },
  "LP-001":  { replay: LP_001_REPLAY,  reasoning: LP_001_REASONING,  graph: LP_001_GRAPH  },
};

export const MOCK_CASE_LIST = Object.values(MOCK_CASES).map((b) => ({
  case_id:          b.replay.case_id,
  final_diagnosis:  b.replay.final_diagnosis,
  requires_biopsy:  b.replay.requires_biopsy,
  final_certainty:  b.replay.final_certainty,
  total_steps:      b.replay.total_steps,
  converged:        b.replay.converged,
  true_label:       b.replay.true_label,
}));
