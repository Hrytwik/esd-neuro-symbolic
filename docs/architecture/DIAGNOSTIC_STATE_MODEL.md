# Formal Diagnostic State Model
## DiagnosticStateTracker — State Machine Specification

**Document type:** Subsystem Architecture Reference  
**Subsystem:** `DiagnosticStateTracker` (`src/symbolic_engine/state_tracker.py`)  
**Role:** Maintains diagnostic state throughout the 6-stage reasoning pipeline; governs transitions; prevents unsafe state jumps

---

## 1. Overview

The **Diagnostic State Machine** models how a case evolves from sparse evidence through progressive reasoning stages toward either a certainty-stabilized diagnosis or a biopsy escalation. It is a deterministic finite automaton with 9 states and guarded transitions.

Each state encodes:
- The current quality of evidence (sparse / forming / reinforcing / contradicted / ambiguous / stabilized)
- The safety envelope of diagnostic confidence
- The appropriate biopsy triage mapping

The state machine is **monotonically escalating toward certainty or escalation** — it cannot cycle backward (e.g., from CERTAINTY_STABILIZED back to HYPOTHESIS_FORMING). If evidence degrades (e.g., contradiction emerges after reinforcement), the state transitions forward to CONTRADICTION_EMERGED, not backward.

---

## 2. State Definitions

### S0 — EVIDENCE_SPARSE

**Formal definition:** Fewer than 2 diagnostic rules have activated above their minimum activation threshold across all disease targets.

**Entry condition:** Initial state; OR feature completeness score < 0.30; OR no features provided above baseline.

**Meaning:** Patient presentation is non-specific or critically incomplete. No stable diagnostic hypothesis can be formed.

**Triage mapping:** Cannot recommend biopsy-free or biopsy-advised. Outputs "INSUFFICIENT_EVIDENCE" and requests additional features.

**Clinical interpretation:** The 12 clinical features currently observable are insufficient to discriminate among the six diseases. Additional examination or patient history is required before diagnostic reasoning can proceed.

---

### S1 — HYPOTHESIS_FORMING

**Formal definition:** ≥2 rules have activated above threshold; initial probability mass is distributing across disease targets; no dominant hypothesis yet established.

**Entry condition:** From S0 when `activated_rule_count >= 2` after Stage 1 processing.

**Meaning:** The clinical presentation is beginning to suggest a diagnostic direction, but evidence is preliminary and non-discriminating.

**Triage mapping:** AMBIGUOUS_CASE (if forced to output); preferred response is to continue to Stage 2.

**Clinical interpretation:** Early differential is forming. Additional features — particularly Koebner phenomenon, follicular papule status, or oral mucosal involvement — would substantially narrow the hypothesis space.

---

### S2 — REINFORCING

**Formal definition:** ≥3 rules from the same target disease have activated; the leading hypothesis is accumulating certainty above the competing hypotheses.

**Entry condition:** From S1 when `activated_rules_per_leading_disease >= 3` after Stage 2 processing.

**Meaning:** A dominant disease hypothesis is forming with supportive evidence across multiple independent clinical signs.

**Triage mapping:** MODERATE_CERTAINTY (provisional; subject to contradiction analysis in Stage 3).

**Clinical interpretation:** The clinical presentation is beginning to converge on a single disease. Stage 3 contradiction analysis may suppress this hypothesis if conflicting signs are present.

---

### S3 — CONTRADICTION_EMERGED

**Formal definition:** One or more contradiction features have activated against the leading hypothesis, applying a combined penalty of > 0.20 to its certainty score.

**Entry condition:** From S1, S2, or S6 when any contradiction feature activates with `penalty > 0.20`.

**Meaning:** The clinical picture contains conflicting signs that challenge the leading hypothesis. Diagnostic tension is present. Certainty is damped.

**Triage mapping:** MODERATE_CERTAINTY downgraded; may become AMBIGUOUS_CASE if penalties are severe (contradiction_load > 0.30).

**Clinical interpretation:** Contradicting signs are present. For example: strong psoriasis evidence (Koebner positive, knee/elbow involvement) simultaneously with oral mucosal involvement (which strongly suggests lichen planus). The system acknowledges the conflict explicitly.

---

### S4 — DIAGNOSTIC_TENSION

**Formal definition:** Two competing disease hypotheses exist with `certainty_gap < 0.20` and `second_disease_score > 0.30`; the system cannot confidently rank one above the other from current evidence.

**Entry condition:** From S3 when `certainty_gap < 0.20 AND second_disease_score > 0.30` after contradiction analysis.

**Meaning:** The case is genuinely at the intersection of two diseases. Stage 4 pairwise discriminators are the last opportunity to resolve the tension before ambiguity escalation.

**Triage mapping:** AMBIGUOUS_CASE.

**Clinical interpretation:** The observable clinical features do not currently provide sufficient discrimination between the two competing diagnoses. This is a clinically recognized diagnostic challenge for the erythemato-squamous group.

---

### S5 — AMBIGUITY_ESCALATED

**Formal definition:** Shannon entropy of the disease certainty distribution exceeds 1.5 bits; no stable dominant hypothesis is achievable from the current feature set; Stage 4 discriminators failed to resolve diagnostic tension.

**Entry condition:** From S4 when `ambiguity_index > 1.5` after Stage 4 processing.

**Meaning:** The case has passed through all discrimination stages and clinical reasoning cannot converge. The evidence is genuinely insufficient for safe differential diagnosis.

**Triage mapping:** BIOPSY_ADVISED (strongly).

**Clinical interpretation:** Symbolic clinical reasoning has exhausted all available evidence without achieving a safe diagnostic conclusion. Histopathological analysis is clinically warranted.

**Terminal state:** Yes. Transitions only to S8 (BIOPSY_ESCALATED) for final output.

---

### S6 — CERTAINTY_STABILIZING

**Formal definition:** The leading hypothesis is pulling ahead with `certainty_gap >= 0.20` and `max_certainty >= 0.55`, but has not yet reached the stabilization threshold. The separation is trending toward safety.

**Entry condition:** From S2 when `max_certainty >= 0.55 AND certainty_gap >= 0.20`; or from S3 when the leading hypothesis survives contradiction penalties with sufficient separation.

**Meaning:** Evidence is converging toward a single diagnosis. Contradictions are present but not dominant. The system is approaching a safe diagnostic conclusion.

**Triage mapping:** MODERATE_CERTAINTY; approaching SAFE_BIOPSY_FREE threshold.

**Clinical interpretation:** The clinical picture is becoming clearer. Stage 4 discriminators and final safety gate processing will determine whether biopsy-free diagnosis is achievable.

---

### S7 — CERTAINTY_STABILIZED

**Formal definition:** `max_certainty >= 0.65 AND certainty_gap >= 0.35 AND contradiction_load < 0.40`. The dominant hypothesis has established sufficient separation and certainty for a confident diagnostic conclusion.

**Entry condition:** From S6 when all stabilization thresholds are met after Stage 5 processing.

**Meaning:** The symbolic reasoning pipeline has converged on a dominant diagnosis with sufficient certainty and evidence separation. Biopsy-free diagnosis may be safe.

**Triage mapping:**  
  - `max_certainty >= 0.82 AND certainty_gap >= 0.40 AND contradiction_load < 0.20` → **SAFE_BIOPSY_FREE**  
  - Otherwise → **MODERATE_CERTAINTY**

**Clinical interpretation:** The clinical evidence consistently supports a single diagnosis across multiple independent reasoning stages. Biopsy-free diagnosis is potentially safe, subject to final safety gate clearance.

**Terminal state:** Yes. Maps to SAFE_BIOPSY_FREE or MODERATE_CERTAINTY.

---

### S8 — BIOPSY_ESCALATED

**Formal definition:** Terminal safety escalation triggered by any of: safety invariant violation, contradiction_load ≥ 0.40, ambiguity_index > 1.5 after Stage 4, or clinical safety gate activation.

**Entry condition:** From any state when any safety invariant or gate is triggered (overrides all other transitions). Also from S5 as normal terminal path.

**Meaning:** The symbolic reasoning system cannot safely recommend biopsy-free diagnosis under current conditions. Histopathological confirmation is required.

**Triage mapping:** **BIOPSY_ADVISED** (non-negotiable; cannot be overridden).

**Clinical interpretation:** Symbolic reasoning was unable to achieve a safe diagnostic conclusion. This may indicate: genuine diagnostic ambiguity between diseases, contradicting clinical signs, insufficient observable features, or a presentation atypical for the training evidence base. Biopsy remains the definitive standard.

**Terminal state:** Yes. Always maps to BIOPSY_ADVISED.

---

## 3. State Transition Graph

```
                     ┌───────────────────────────────────────────────────┐
                     │           SAFETY GATE (overrides all)              │
                     │   contradiction_load >= 0.40  OR                  │
                     │   invariant violation         → S8                 │
                     └───────────────────────────────────────────────────┘
                                        │ [from any state]
                                        ▼
 [initial]
    │
    ▼
  ┌─────┐   activated_rule_count >= 2      ┌─────┐
  │ S0  │ ─────────────────────────────▶  │ S1  │
  └─────┘                                  └─────┘
EVIDENCE_SPARSE                          HYPOTHESIS
                                          FORMING
                                            │   │
              activated_rules_per_          │   │  contradiction detected
              leading_disease >= 3          │   │  penalty > 0.20
                                            ▼   ▼
                                         ┌─────┐   ┌─────┐
                                         │ S2  │   │ S3  │
                                         └─────┘   └─────┘
                                        REINFORCING  CONTRADICTION
                                            │        EMERGED
                                            │           │
                          contradiction     │           │ gap < 0.20
                          detected          │           │ second > 0.30
                                            │           ▼
                          max >= 0.55       │        ┌─────┐
                          gap >= 0.20       │        │ S4  │
                                 │          │        └─────┘
                                 │          │     DIAGNOSTIC
                                 │          │       TENSION
                                 │          │         │   │
                                 ▼          │         │   │ gap < 0.20
                              ┌─────┐       │         │   │ entropy > 1.5
                              │ S6  │◀──────┘         │   ▼
                              └─────┘    (survives     │ ┌─────┐
                           CERTAINTY      penalties,   │ │ S5  │
                           STABILIZING    gap >= 0.25) │ └─────┘
                                │                      │ AMBIGUITY
                                │ max >= 0.65          │ ESCALATED
                                │ gap >= 0.35          │    │
                                │ load < 0.40          │    │
                                ▼                      │    ▼
                             ┌─────┐                   │ ┌─────┐
                             │ S7  │                   └▶│ S8  │
                             └─────┘                     └─────┘
                          CERTAINTY                     BIOPSY
                          STABILIZED                   ESCALATED
                               │                           │
                    ┌──────────┴──────────┐               │
                    ▼                     ▼               ▼
            SAFE_BIOPSY_FREE    MODERATE_CERTAINTY   BIOPSY_ADVISED
```

---

## 4. Transition Guard Conditions

| Transition | Guard Condition | Processing Stage |
|---|---|---|
| S0 → S1 | `activated_rule_count >= 2` | After Stage 1 |
| S1 → S2 | `activated_rules_per_leading_disease >= 3` | After Stage 2 |
| S1 → S3 | `max(contradiction_penalty) > 0.20` | During Stage 3 |
| S2 → S3 | Any contradiction feature activates | During Stage 3 |
| S2 → S6 | `max_certainty >= 0.55 AND certainty_gap >= 0.20` | After Stage 2 |
| S3 → S4 | `certainty_gap < 0.20 AND second_disease_score > 0.30` | After Stage 3 |
| S3 → S6 | `max_certainty >= 0.55 AND certainty_gap >= 0.25` (survives penalties) | After Stage 3 |
| S3 → S8 | `contradiction_load >= 0.40` [SAFETY GATE] | Any time during Stage 3 |
| S4 → S5 | `certainty_gap < 0.20 AND ambiguity_index > 1.5` after Stage 4 | After Stage 4 |
| S4 → S6 | `certainty_gap >= 0.25` after Stage 4 discriminators | After Stage 4 |
| S4 → S8 | `contradiction_load >= 0.40` [SAFETY GATE] | Any time during Stage 4 |
| S5 → S8 | Terminal path (ambiguity irresolvable) | After Stage 5 |
| S6 → S7 | `max_certainty >= 0.65 AND certainty_gap >= 0.35 AND contradiction_load < 0.40` | After Stage 5 |
| S6 → S4 | `certainty_gap < 0.20` (regression under continued contradiction) | After Stage 4 |
| ANY → S8 | Any safety invariant or safety gate triggered [OVERRIDES ALL] | Any stage |

---

## 5. State → Biopsy Triage Mapping

| Terminal State | Condition | Triage Recommendation |
|---|---|---|
| S7 | `max_certainty >= 0.82 AND certainty_gap >= 0.40 AND contradiction_load < 0.20` | **SAFE_BIOPSY_FREE** |
| S7 | `max_certainty >= 0.65 AND certainty_gap >= 0.35` (does not meet SAFE threshold) | **MODERATE_CERTAINTY** |
| S4 | `certainty_gap in [0.10, 0.20)` (tension not resolved by Stage 4) | **AMBIGUOUS_CASE** |
| S5 | `ambiguity_index > 1.5` | **AMBIGUOUS_CASE** → routes to S8 |
| S8 | Any (safety escalation terminal) | **BIOPSY_ADVISED** |
| S0 | Feature completeness too low | **INSUFFICIENT_EVIDENCE** (special output) |

---

## 6. State History and Trace Recording

The `DiagnosticStateTracker` records a state history log for every case:

```json
{
  "case_id": "UCI_001",
  "state_history": [
    {"stage": 0, "state": "EVIDENCE_SPARSE", "activated_rules": 0},
    {"stage": 1, "state": "HYPOTHESIS_FORMING", "activated_rules": 2, "leading": "psoriasis"},
    {"stage": 2, "state": "REINFORCING", "activated_rules": 4, "leading": "psoriasis", "certainty": 0.61},
    {"stage": 3, "state": "CONTRADICTION_EMERGED", "contradiction": "oral_mucosal_involvement", "penalty": 0.30, "certainty": 0.48},
    {"stage": 4, "state": "CERTAINTY_STABILIZING", "discriminator": "PSO_LP_001", "certainty": 0.67, "gap": 0.28},
    {"stage": 5, "state": "CERTAINTY_STABILIZED", "max_certainty": 0.71, "gap": 0.33, "contradiction_load": 0.30},
    {"stage": 6, "state": "CERTAINTY_STABILIZED", "triage": "MODERATE_CERTAINTY"}
  ],
  "final_state": "CERTAINTY_STABILIZED",
  "biopsy_triage": "MODERATE_CERTAINTY",
  "safety_flags": []
}
```

The state history is included in both the JSON trace and the clinician-readable summary.

---

## 7. Invariant Properties of the State Machine

**Invariant A — Forward monotonicity:** The state sequence is non-decreasing in the diagnostic commitment axis. A case cannot transition from CERTAINTY_STABILIZED to HYPOTHESIS_FORMING.

**Invariant B — Safety gate supremacy:** A safety gate trigger unconditionally routes to S8 regardless of current state or certainty values. No other transition can override it.

**Invariant C — Terminal state completeness:** Every case terminates in exactly one of {S7, S8, S0} (insufficient evidence). No case exits the pipeline in a non-terminal state.

**Invariant D — Triage monotonicity under contradiction:** As `contradiction_load` increases, the triage recommendation can only move toward BIOPSY_ADVISED. Contradiction accumulation never improves the triage outcome.
