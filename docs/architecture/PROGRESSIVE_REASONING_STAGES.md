# Progressive Reasoning Stages Specification
## Six-Stage Symbolic Clinical Inference Pipeline

**Document type:** Subsystem Architecture Reference  
**Subsystem:** Full symbolic reasoning pipeline  
**Role:** Defines the processing contract, inputs, outputs, and state transitions for each of the six reasoning stages

---

## Overview

The symbolic engine processes clinical evidence in six ordered stages. Each stage targets a distinct category of diagnostic knowledge: graded observation, pathognomonic patterns, supportive clusters, exclusionary contradictions, pairwise discriminators, and finally certainty stabilization with safety gating.

The multi-stage design is architecturally necessary because different categories of evidence should carry different epistemic weight:
- A **pathognomonic** sign (Tier A) alone justifies forming a hypothesis
- A **supportive cluster** (Tier B) reinforces an existing hypothesis
- An **exclusionary contradiction** (Stage 3) actively penalizes a hypothesis
- A **pairwise discriminator** (Tier D) resolves confusion between adjacent diseases
- **Certainty stabilization** (Stage 5) is a computational act, not a clinical evidence evaluation

Mixing all evidence types in a single-pass scoring function would lose the epistemic ordering that gives the system its clinical interpretability.

---

## Stage 0 — Clinical Feature Grading

**Subsystem:** `ClinicalGradingModule`  
**File:** `src/symbolic_engine/clinical_grading.py`  
**Processing scope:** Pre-reasoning feature transformation

### Purpose

Convert the raw feature vector into a graded, fuzzy-compatible representation that rules can evaluate against consistently regardless of the raw scale of each feature.

### Input

```
raw_feature_vector: Dict[feature_name → raw_value]
  - Ordinal features (0–3): erythema, scaling, definite_borders, itching
  - Binary features (0/1): koebner_phenomenon, polygonal_papules,
                            follicular_papules, oral_mucosal_involvement,
                            knee_and_elbow_involvement, scalp_involvement,
                            family_history
  - Continuous (float): age
```

### Processing

**Ordinal features → fuzzy membership grades:**

| Raw value | Grade | Interpretation |
|---|---|---|
| 0 | 0.00 | Absent |
| 1 | 0.33 | Mild (present but clinically insignificant) |
| 2 | 0.67 | Moderate (clinically significant) |
| 3 | 1.00 | Severe / strongly present |

The rule threshold `condition: "gte"` with `threshold: 2` activates on grades ≥ 0.67, implementing the "clinically significant present" criterion from dermatology literature without hard binary cutoffs.

**Binary features:** Passed through as 0.0 or 1.0.

**Age:** Normalized to percentile over the UCI dataset distribution. Stored in graded_features but used only for auxiliary rules (none in the primary rule set are age-dependent in the initial version).

**Feature completeness computation:**

```
completeness_score = present_features / total_features (12)
critical_feature_check:
  critical_features = [koebner_phenomenon, polygonal_papules,
                       follicular_papules, oral_mucosal_involvement]
  critical_missing = [f for f in critical_features if value is null]
```

If `completeness_score < 0.50`, the pipeline issues an INSUFFICIENT_EVIDENCE warning and outputs S0 state without proceeding to Stage 1.

### Output

```
graded_features: Dict[feature_name → float [0.0, 1.0]]
completeness_score: float
critical_features_missing: List[feature_name]
missing_feature_flags: Dict[feature_name → bool]
```

### State Transitions

- `DiagnosticStateTracker` remains in S0
- If `completeness_score < 0.30`: terminal at S0, INSUFFICIENT_EVIDENCE

### Trace Entry

```json
{
  "stage": 0,
  "module": "ClinicalGradingModule",
  "graded_features": {"erythema": 0.67, "koebner_phenomenon": 1.0, ...},
  "completeness_score": 1.0,
  "critical_missing": [],
  "warnings": []
}
```

---

## Stage 1 — Pathognomonic Pattern Detection

**Subsystem:** `DiagnosticEvidenceEvaluator` (evidence_tier: A)  
**File:** `src/symbolic_engine/evidence_evaluator.py`  
**Processing scope:** Tier A rules only

### Purpose

Detect pathognomonic or near-pathognomonic clinical signs — features that, when present, strongly and specifically suggest a single disease. These carry the highest epistemic weight and are evaluated first to anchor the hypothesis formation process.

### Tier A Rules (pathognomonic)

| Rule ID | Disease | Feature(s) | Clinical basis |
|---|---|---|---|
| PSO_001 | Psoriasis | koebner_phenomenon=1 | Isomorphic response; 25–50% of psoriasis cases |
| LP_001 | Lichen planus | polygonal_papules=1 | The "4 Ps" — pathognomonic |
| LP_002 | Lichen planus | oral_mucosal_involvement=1 | Wickham's striae; highly specific |
| PRP_001 | PRP | follicular_papules=1 | Keratotic follicular papules; pathognomonic |
| PR_001 | Pityriasis rosea | definite_borders≥2 + oral_mucosal=0 | Herald patch morphology |

### Processing

For each Tier A rule:
1. Evaluate `supporting_features` against `graded_features`
2. Compute `rule_activation ∈ [0.0, 1.0]` using activation_logic
3. Multiply by `confidence_weight` to get `weighted_activation`
4. Accumulate: `raw_score[disease] += weighted_activation`

**Activation logic types:**

- `binary`: activation = graded_features[feature] (0.0 or 1.0 only)
- `threshold`: activation = 1.0 if graded_features[feature] >= threshold else 0.0
- `fuzzy`: activation = graded_features[feature] (continuous)
- `composite`: activation = AND/OR combination of multiple feature grades

### Output

```
raw_scores_stage1: Dict[disease → float]
activated_rules_stage1: List[rule_id]
activated_rule_count: int
```

### State Transitions

- `activated_rule_count >= 2`: S0 → S1 (HYPOTHESIS_FORMING)
- `activated_rule_count < 2`: remain S0; proceed to Stage 2 before re-evaluating

### Trace Entry

```json
{
  "stage": 1,
  "module": "DiagnosticEvidenceEvaluator",
  "tier": "A",
  "rules_evaluated": [
    {"rule_id": "PSO_001", "activation": 0.85, "weighted": 0.72, "triggered": true},
    {"rule_id": "LP_001", "activation": 0.00, "weighted": 0.00, "triggered": false}
  ],
  "raw_scores": {"psoriasis": 0.72, "lichen_planus": 0.00, ...},
  "state_transition": "S0 → S1"
}
```

---

## Stage 2 — Supportive Evidence Integration

**Subsystem:** `DiagnosticEvidenceEvaluator` (evidence_tier: B)  
**File:** `src/symbolic_engine/evidence_evaluator.py`  
**Processing scope:** Tier B rules only

### Purpose

Evaluate supportive cluster rules — co-occurring clinical patterns that, when combined with other signs, substantially elevate a disease hypothesis without being individually pathognomonic. These are the workhorses of clinical reasoning under uncertainty.

### Tier B Rule Examples (supportive clusters)

| Rule ID | Disease | Feature cluster | Clinical basis |
|---|---|---|---|
| PSO_002 | Psoriasis | knee_and_elbow=1 | Extensor surface distribution; 80% of cases |
| PSO_003 | Psoriasis | scalp_involvement=1 | Scalp affected in ~80% of psoriasis |
| PSO_004 | Psoriasis | family_history=1 | Polygenic inheritance; HLA-Cw6 association |
| SD_001 | Seborrheic derm. | scalp=1, koebner=0 | Primary site; absence of Koebner distinguishes |
| LP_003 | Lichen planus | itching≥2, koebner=1 | Intense pruritus + isomorphic response in LP |
| PR_002 | Pityriasis rosea | oral_mucosal=0, knee_elbow=0 | PR spares mucosa and extremities |
| CD_001 | Chronic derm. | itching≥2, koebner=0 | Itch without Koebner distinguishes from LP/PSO |
| PRP_002 | PRP | erythema≥2, koebner=0 | Diffuse salmon erythema without Koebner |

### Processing

Same evaluation loop as Stage 1. Scores are **accumulated on top of** Stage 1 scores, not restarted.

```
raw_score[disease] += sum(weighted_activation for rules in Tier B targeting disease)
```

### State Transitions

- `activated_rules_per_leading_disease >= 3` (across Stages 1+2 combined): S1 → S2 (REINFORCING)
- if leading disease raw_score rises with `certainty_gap_estimate >= 0.20`: proceed toward S6 consideration

### Trace Entry

```json
{
  "stage": 2,
  "module": "DiagnosticEvidenceEvaluator",
  "tier": "B",
  "rules_evaluated": [...],
  "cumulative_raw_scores": {"psoriasis": 3.10, ...},
  "leading_disease": "psoriasis",
  "state_transition": "S1 → S2"
}
```

---

## Stage 3 — Exclusionary and Contradiction Analysis

**Subsystem:** `DiagnosticConflictAnalyzer`  
**File:** `src/symbolic_engine/conflict_analyzer.py`  
**Processing scope:** Contradiction matrix + contradiction features per rule

### Purpose

Detect active contradicting clinical signs and apply weighted penalties to the affected disease hypotheses. This is the mechanism by which the system acknowledges that clinical presentations can simultaneously suggest one disease and contradict another.

### Contradiction Architecture

The contradiction matrix is defined at two levels:

**Level 1 — Rule-embedded contradictions:** Each rule in YAML includes a `contradiction_features` list with per-feature penalties. These are evaluated against the leading hypothesis whenever that rule is active.

**Level 2 — Global contradiction matrix** (`rules/contradiction_matrix.yaml`): Disease-pair level conflict relationships that fire regardless of which specific rules are active.

### Processing

For each leading disease hypothesis (score > 0.15):

```
For each contradiction_feature in rule.contradiction_features:
    if graded_features[contradiction_feature.feature] >= contradiction_feature.threshold:
        penalty = contradiction_feature.penalty
        raw_score[disease] *= (1 - penalty)
        contradiction_load += penalty
        log_contradiction_event(...)
```

Then check global contradiction matrix for additional cross-disease signals.

### Safety Escalation Trigger

If `contradiction_load >= 0.40`:
- Immediately trigger `ClinicalSafetyGate`
- `DiagnosticStateTracker`: ANY → S8 (BIOPSY_ESCALATED)
- Pipeline continues to Stage 6 only to generate reasoning trace

### State Transitions

- Contradiction penalty > 0.20 detected: current state → S3 (CONTRADICTION_EMERGED)
- No contradictions detected: state remains S2 (REINFORCING) or S1; proceed to Stage 4
- contradiction_load >= 0.40: → S8 (BIOPSY_ESCALATED) via safety gate

### Trace Entry

```json
{
  "stage": 3,
  "module": "DiagnosticConflictAnalyzer",
  "contradictions": [
    {
      "feature": "oral_mucosal_involvement",
      "value": 1.0,
      "target_hypothesis": "psoriasis",
      "penalty": 0.30,
      "competing_disease": "lichen_planus",
      "rationale": "Oral mucosal involvement (Wickham striae) pathognomonic for LP",
      "source": "Le Cleach & Chosidow, NEJM 2012"
    }
  ],
  "contradiction_load": 0.30,
  "post_penalty_scores": {"psoriasis": 2.17, "lichen_planus": 0.90, ...},
  "state_transition": "S2 → S3"
}
```

---

## Stage 4 — Pairwise Discriminator Activation

**Subsystem:** `DiagnosticEvidenceEvaluator` (evidence_tier: D)  
**File:** `src/symbolic_engine/evidence_evaluator.py`  
**Processing scope:** Tier D (discriminating) cross-disease rules only

### Purpose

When two diseases have similar certainty scores (diagnostic tension), activate targeted pairwise discriminator rules that encode the specific clinical features distinguishing them. This stage is the system's final opportunity to resolve ambiguity from clinical evidence before resorting to biopsy escalation.

### Active Confusion Pairs (requiring discriminators)

Based on the UCI Dermatology dataset confusion profile and dermatology literature:

| Confusion Pair | Key Discriminating Features | Rule IDs |
|---|---|---|
| Psoriasis ↔ Seborrheic Dermatitis | definite_borders, family_history | PSO_SD_001 |
| Psoriasis ↔ Lichen Planus | oral_mucosal_involvement, polygonal_papules | PSO_LP_001 |
| Lichen Planus ↔ Pityriasis Rosea | oral_mucosal_involvement, polygonal_papules | LP_PR_001 |
| Seborrheic Derm. ↔ Chronic Derm. | koebner_phenomenon, scalp_involvement | SD_CD_001 |
| PRP ↔ Psoriasis | follicular_papules, koebner_phenomenon | PRP_PSO_001 |
| Chronic Derm. ↔ Lichen Planus | polygonal_papules, definite_borders | CD_LP_001 |

### Processing

Stage 4 only activates discriminator rules for pairs where:
`certainty_gap between pair < 0.30` (i.e., genuine diagnostic tension exists)

```
For each confusion_pair in KNOWN_CONFUSION_PAIRS:
    if certainty_gap(pair) < 0.30:
        activate discriminator rules for this pair
        apply score delta to winning disease
        update certainty_gap
```

### State Transitions

- `certainty_gap >= 0.25` after Stage 4: S3/S4 → S6 (CERTAINTY_STABILIZING)
- `certainty_gap < 0.20` AND `ambiguity_index > 1.5` after Stage 4: S4 → S5 (AMBIGUITY_ESCALATED) → S8
- `certainty_gap ∈ [0.20, 0.25)`: S4 → S6 (borderline; subject to Stage 5 safety gate)

### Trace Entry

```json
{
  "stage": 4,
  "module": "DiagnosticEvidenceEvaluator",
  "tier": "D",
  "active_tension_pairs": [["psoriasis", "lichen_planus"]],
  "discriminators_activated": [
    {"rule_id": "PSO_LP_001", "delta_to_psoriasis": 0.15, "delta_to_lp": -0.15}
  ],
  "updated_certainty_gap": 0.38,
  "state_transition": "S3 → S6"
}
```

---

## Stage 5 — Certainty Computation and Safety Gate

**Subsystems:** `HypothesisCertaintyPropagator` + `ClinicalSafetyGate`  
**Files:** `src/symbolic_engine/certainty_propagator.py`, `src/symbolic_engine/safety_gate.py`  
**Processing scope:** Final certainty computation; safety invariant evaluation

### Purpose

Convert accumulated raw scores into calibrated certainty probabilities, compute all certainty metrics, and evaluate the full safety gate battery before allowing triage output.

### Certainty Computation

```python
# Softmax over all 6 disease scores
exp_scores = {d: exp(score) for d, score in penalized_scores.items()}
total = sum(exp_scores.values())
certainty = {d: v / total for d, v in exp_scores.items()}

# Certainty metrics
max_certainty    = max(certainty.values())
top2             = sorted(certainty.values(), reverse=True)[:2]
certainty_gap    = top2[0] - top2[1]
ambiguity_index  = -sum(p * log2(p) for p in certainty.values() if p > 0)  # Shannon entropy
contradiction_load = accumulated during Stage 3
```

### Safety Gate Battery

Three invariants (always evaluated, cannot be bypassed):

**Invariant 1 — Contradiction Safety:**
```
IF contradiction_load >= 0.40:
    TRIGGER → S8 (BIOPSY_ESCALATED)
    max_recommendation = BIOPSY_ADVISED
```

**Invariant 2 — Evidence Sufficiency:**
```
IF activated_rule_count < 2:
    TRIGGER → S8 (BIOPSY_ESCALATED)
    max_recommendation = AMBIGUOUS_CASE (at best)
```

**Invariant 3 — Entropy Ceiling:**
```
IF ambiguity_index > 1.5:
    TRIGGER → S8 (BIOPSY_ESCALATED)
    max_recommendation = BIOPSY_ADVISED
```

Five gates (evaluated if invariants pass):

**Gate 1 — Single-Source Dominance:**
```
IF any_single_rule_contributes > 0.60 of max_certainty:
    FLAG "single_point_of_failure"
    cap at MODERATE_CERTAINTY
```

**Gate 2 — Pathognomonic Absence under High Certainty:**
```
IF pathognomonic_rule_count == 0 AND max_certainty > 0.75:
    FLAG "statistically_derived_certainty"
    cap at MODERATE_CERTAINTY
```

**Gate 3 — Critical Feature Missingness:**
```
IF critical_features_missing >= 3:
    FLAG "critical_data_gap"
    cap at AMBIGUOUS_CASE
```

**Gate 4 — Confusion Zone Proximity:**
```
IF leading_pair in KNOWN_CONFUSION_PAIRS AND certainty_gap < 0.30:
    apply confusion_zone_penalty = 0.15 to certainty
    re-evaluate thresholds
```

**Gate 5 — Overconfidence Prevention:**
```
IF max_certainty > 0.92 AND contradiction_load > 0.10:
    FLAG "suspect_overconfidence"
    cap at MODERATE_CERTAINTY
```

### Output

```
certainty_distribution: Dict[disease → float]
max_certainty: float
certainty_gap: float
ambiguity_index: float
contradiction_load: float
safety_gate_results: Dict[gate_id → {status, flag_raised, cap_applied}]
safety_flags: List[str]
final_state: DiagnosticState
```

### State Transitions

- All invariants pass, certainty thresholds met: S6 → S7 (CERTAINTY_STABILIZED)
- Any invariant or gate triggers escalation: → S8 (BIOPSY_ESCALATED)

---

## Stage 6 — Triage Decision and Narrative Generation

**Subsystems:** `ClinicalEscalationEngine` + `DiagnosticNarrativeGenerator`  
**Files:** `src/symbolic_engine/escalation_engine.py`, `src/symbolic_engine/narrative_generator.py`  
**Processing scope:** Final output generation

### Purpose

Map the final diagnostic state and certainty metrics to a biopsy triage recommendation, and generate the complete reasoning trace in both machine-readable and human-readable formats.

### Triage Decision Matrix

| Final State | Conditions | Recommendation |
|---|---|---|
| S7 | max_certainty ≥ 0.82, gap ≥ 0.40, load < 0.20 | SAFE_BIOPSY_FREE |
| S7 | max_certainty ≥ 0.65, gap ≥ 0.35 (below SAFE thresholds) | MODERATE_CERTAINTY |
| S4 | gap ∈ [0.10, 0.20) after Stage 4 | AMBIGUOUS_CASE |
| S5 | ambiguity_index > 1.5 | BIOPSY_ADVISED |
| S8 | Any safety gate / invariant triggered | BIOPSY_ADVISED |
| S0 | completeness_score < 0.30 | INSUFFICIENT_EVIDENCE |

### Reasoning Trace Structure

```
JSON trace: Full structured audit log (stored in outputs/traces/)
Clinician summary: Natural language report (stored in outputs/reports/)
```

The clinician summary includes:
1. Clinical features observed (with grades)
2. Activated diagnostic rules (with evidence tier, weight, literature source)
3. Contradictions detected (with clinical rationale)
4. Certainty evolution (max_certainty, certainty_gap, ambiguity_index)
5. Safety gate results (all passed / flags raised)
6. Final diagnostic state
7. Biopsy triage recommendation with justification
8. Suggested follow-up actions (where applicable)

### Stage I/O Contract Summary

| Stage | Primary Input | Primary Output | State Transition Domain |
|---|---|---|---|
| 0 | Raw features | Graded features + completeness | S0 (unchanged) |
| 1 | Graded features | Tier A activations + raw scores | S0 → S1 |
| 2 | Graded features | Tier B activations + cumulative scores | S1 → S2 or S6 |
| 3 | Cumulative scores | Penalized scores + contradiction trace | S2 → S3 or S8 |
| 4 | Penalized scores | Discriminated scores + gap update | S3/S4 → S5/S6 or S8 |
| 5 | Final scores | Certainty dist. + safety gate results | S6 → S7 or S8 |
| 6 | Final state + metrics | Triage recommendation + trace | Terminal (S7/S8/S0) |
