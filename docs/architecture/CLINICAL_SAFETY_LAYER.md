# Clinical Safety Layer Specification
## ClinicalSafetyGate — Formal Safety Invariants and Gate Definitions

**Document type:** Subsystem Architecture Reference  
**Subsystem:** `ClinicalSafetyGate` (`src/symbolic_engine/safety_gate.py`)  
**Critical property:** Escalation-only — the safety layer can only move the recommendation toward BIOPSY_ADVISED, never toward SAFE_BIOPSY_FREE

---

## 1. Purpose and Design Rationale

The Clinical Safety Layer exists because symbolic reasoning over sparse clinical evidence carries the risk of overconfident diagnosis. In a clinical context, an incorrect SAFE_BIOPSY_FREE recommendation is qualitatively worse than an overcautious BIOPSY_ADVISED recommendation:

- **False SAFE_BIOPSY_FREE** → disease misdiagnosis; delayed treatment; potential patient harm
- **False BIOPSY_ADVISED** → unnecessary procedure; additional cost; but no patient harm from misdiagnosis

The safety layer is therefore **asymmetrically conservative**: it preferentially escalates toward caution when conditions for confident diagnosis are not clearly met.

### Design Constraints

1. The safety layer cannot lower a triage recommendation — only raise it toward BIOPSY_ADVISED
2. Safety gate triggers are **non-negotiable**: no certainty score, regardless of magnitude, overrides a triggered invariant
3. The safety layer operates on final certainty metrics — it does not re-evaluate raw features
4. All gate evaluations are logged to the reasoning trace, including passed gates

---

## 2. Formal Safety Invariants

Invariants are structural guarantees of the system. Violating an invariant triggers immediate escalation to S8 (BIOPSY_ESCALATED) regardless of current diagnostic state or certainty values.

---

### Invariant I — Contradiction Safety Ceiling

**Formal statement:**
```
∀ case c:
    IF contradiction_load(c) ≥ 0.40
    THEN recommendation(c) ≤ BIOPSY_ADVISED
    AND recommendation(c) ≠ SAFE_BIOPSY_FREE
    AND recommendation(c) ≠ MODERATE_CERTAINTY
```

**Threshold:** `contradiction_load ≥ 0.40`

**Rationale:** A contradiction load of 0.40 means that ≥40% penalty weight is being applied against the leading hypothesis by contradicting clinical signs. At this level, the clinical picture is genuinely ambiguous — a high raw certainty score reflects evidence accumulation before penalties, not post-penalty confidence. Allowing SAFE_BIOPSY_FREE under these conditions would be epistemically unjustified.

**Clinical example:** A case with strong psoriasis evidence (Koebner positive, knee/elbow involvement, scalp involvement) simultaneously presenting with oral mucosal involvement (Wickham's striae-like pattern). The combined contradiction load exceeds 0.40; biopsy is needed to confirm whether this is LP with Koebner-like response or psoriasis with atypical mucosal involvement.

**Implementation:**
```python
if contradiction_load >= 0.40:
    self.trigger_invariant("CONTRADICTION_SAFETY_CEILING",
                            f"contradiction_load={contradiction_load:.3f} >= 0.40")
    return EscalationDecision(state=DiagnosticState.BIOPSY_ESCALATED,
                              recommendation=TriageRecommendation.BIOPSY_ADVISED)
```

---

### Invariant II — Evidence Sufficiency Floor

**Formal statement:**
```
∀ case c:
    IF activated_rule_count(c) < 2
    THEN recommendation(c) ≤ AMBIGUOUS_CASE
    AND recommendation(c) ∉ {SAFE_BIOPSY_FREE, MODERATE_CERTAINTY}
```

**Threshold:** `activated_rule_count < 2`

**Rationale:** A single activated rule is insufficient grounds for any diagnostic recommendation, regardless of the rule's confidence weight. Clinical reasoning requires corroborating evidence. Single-rule activation may reflect a non-specific sign present in multiple conditions, or a coincidental finding.

**Clinical example:** A patient presenting only with itching (grade 3), with all other features absent or mild. Itching is present in multiple erythemato-squamous diseases; without co-occurring signs, no inference is justified.

**Implementation:**
```python
if self.activated_rule_count < 2:
    self.trigger_invariant("EVIDENCE_SUFFICIENCY_FLOOR",
                            f"activated_rule_count={self.activated_rule_count} < 2")
    return EscalationDecision(state=DiagnosticState.BIOPSY_ESCALATED,
                              recommendation=TriageRecommendation.BIOPSY_ADVISED)
```

---

### Invariant III — Entropy Escalation Ceiling

**Formal statement:**
```
∀ case c:
    IF ambiguity_index(c) > 1.5 bits
    THEN recommendation(c) = BIOPSY_ADVISED
```

**Threshold:** `ambiguity_index > 1.5` (Shannon entropy in bits)

**Rationale:** Shannon entropy of 1.5 bits over a 6-class distribution means the probability mass is spread across approximately 3 diseases with near-equal weight. No single disease has sufficient separation for clinical confidence. This is a disease-agnostic measure of diagnostic indeterminacy.

**Calibration note:** Maximum entropy over 6 classes is log₂(6) ≈ 2.58 bits. A threshold of 1.5 bits corresponds approximately to three near-equal contenders, which is irresolvable from clinical signs alone.

**Implementation:**
```python
if self.ambiguity_index > 1.5:
    self.trigger_invariant("ENTROPY_ESCALATION_CEILING",
                            f"ambiguity_index={self.ambiguity_index:.3f} > 1.5")
    return EscalationDecision(state=DiagnosticState.BIOPSY_ESCALATED,
                              recommendation=TriageRecommendation.BIOPSY_ADVISED)
```

---

## 3. Safety Gates

Safety gates are contextual checks that apply soft-to-hard caps on the triage recommendation depending on specific risk conditions. Unlike invariants (which trigger unconditionally on threshold breach), gates evaluate conditions and apply recommendation caps — the recommendation remains within the gate's maximum but the exact level is still determined by the certainty metrics.

---

### Gate 1 — Single-Source Dominance Prevention

**Condition:**
```
IF (max_rule_certainty_contribution / max_certainty) > 0.60:
    FLAG "single_point_of_failure"
    cap recommendation at MODERATE_CERTAINTY
```

**Rationale:** If a single rule accounts for more than 60% of the leading hypothesis's total certainty, the reasoning is resting on a single clinical observation. Clinical diagnosis should be corroborated by multiple independent signs. A single observation — even a pathognomonic one — is insufficient for SAFE_BIOPSY_FREE recommendation because pathognomonic signs can be mimicked by atypical presentations of adjacent diseases.

**Effect:** SAFE_BIOPSY_FREE → MODERATE_CERTAINTY. No effect on lower recommendations.

**Flag raised in trace:** `"single_point_of_failure"`

---

### Gate 2 — Pathognomonic Absence under High Certainty

**Condition:**
```
IF pathognomonic_rule_count == 0 AND max_certainty > 0.75:
    FLAG "statistically_derived_certainty"
    cap recommendation at MODERATE_CERTAINTY
```

**Rationale:** If certainty exceeds 0.75 without any pathognomonic (Tier A) rule having activated, the certainty is derived entirely from the accumulation of non-specific supportive signs. Statistically-derived high certainty without pathognomonic anchoring is architecturally suspect — it suggests the system is over-weighting a supportive cluster. Clinical confidence above 0.75 should require at least one sign specific enough to serve as a diagnostic anchor.

**Effect:** SAFE_BIOPSY_FREE → MODERATE_CERTAINTY.

**Flag raised in trace:** `"statistically_derived_certainty"`

---

### Gate 3 — Critical Feature Missingness

**Condition:**
```
IF critical_features_missing >= 3:
    FLAG "critical_data_gap"
    cap recommendation at AMBIGUOUS_CASE
```

**Critical features:** `koebner_phenomenon`, `polygonal_papules`, `follicular_papules`, `oral_mucosal_involvement`

**Rationale:** These four features are the most diagnostically discriminating in the erythemato-squamous group. Their presence or absence is what distinguishes the highest-confusion disease pairs (psoriasis/LP, LP/PR, PRP/psoriasis). If three or more are missing, the symbolic engine is operating on a severely truncated feature set and cannot be trusted to produce reliable discrimination.

**Effect:** SAFE_BIOPSY_FREE → AMBIGUOUS_CASE; MODERATE_CERTAINTY → AMBIGUOUS_CASE.

**Flag raised in trace:** `"critical_data_gap"`

---

### Gate 4 — Confusion Zone Proximity

**Condition:**
```
IF (leading_disease, second_disease) in KNOWN_CONFUSION_PAIRS
   AND certainty_gap < 0.30:
    apply confusion_zone_penalty = 0.15 to max_certainty
    re-evaluate all thresholds with penalized certainty
```

**Known confusion pairs:**
- Psoriasis ↔ Seborrheic Dermatitis (shared: scalp involvement)
- Psoriasis ↔ Lichen Planus (shared: Koebner phenomenon)
- Lichen Planus ↔ Pityriasis Rosea (shared: scaling, borders)
- Seborrheic Dermatitis ↔ Chronic Dermatitis (shared: itching, no Koebner)
- PRP ↔ Psoriasis (shared: erythema, no Koebner)

**Rationale:** These pairs are documented in dermatology literature as the most commonly confused erythemato-squamous presentations. When the leading disease and second contender form one of these documented confusion pairs, an additional penalty is warranted to prevent overconfidence in a genuinely difficult clinical scenario.

**Effect:** May reduce SAFE_BIOPSY_FREE to MODERATE_CERTAINTY or MODERATE to AMBIGUOUS depending on exact post-penalty certainty.

**Flag raised in trace:** `"known_confusion_zone"` (if penalty applied)

---

### Gate 5 — Overconfidence Prevention

**Condition:**
```
IF max_certainty > 0.92 AND contradiction_load > 0.10:
    FLAG "suspect_overconfidence"
    cap recommendation at MODERATE_CERTAINTY
```

**Rationale:** A certainty of > 0.92 (essentially 92% confidence) combined with any active contradiction (load > 0.10) is epistemically suspicious. In clinical reasoning, genuine contradiction should reduce confidence, not coexist with near-certainty. This combination can arise when the contradiction penalty calculation underrepresents the severity of a conflicting sign. The gate prevents this from translating into a SAFE_BIOPSY_FREE recommendation.

**Note:** This gate does not apply when contradiction_load = 0 (contradiction-free high certainty is expected and valid for clear presentations).

**Effect:** SAFE_BIOPSY_FREE → MODERATE_CERTAINTY.

**Flag raised in trace:** `"suspect_overconfidence"`

---

## 4. Escalation-Only Property

The safety layer is architecturally constrained to only **increase** the severity of the triage recommendation (toward BIOPSY_ADVISED). It cannot:
- Improve a BIOPSY_ADVISED recommendation to AMBIGUOUS_CASE
- Improve an AMBIGUOUS_CASE to MODERATE_CERTAINTY
- Reduce a contradiction_load score
- Re-evaluate clinical feature evidence

The escalation order is:
```
INSUFFICIENT_EVIDENCE < SAFE_BIOPSY_FREE < MODERATE_CERTAINTY < AMBIGUOUS_CASE < BIOPSY_ADVISED
```

Each safety gate and invariant may only push the recommendation rightward on this scale.

---

## 5. Gate Evaluation Order and Short-Circuit Logic

Gates are evaluated in order: Invariant I → Invariant II → Invariant III → Gate 1 → Gate 2 → Gate 3 → Gate 4 → Gate 5.

Invariants use **short-circuit evaluation**: if any invariant triggers, evaluation stops immediately and no further gates are evaluated. The triggered invariant's result is final.

Gates are **cumulative**: multiple gates can trigger in the same case. The final recommendation is the maximum (most cautious) across all triggered gate caps.

---

## 6. Safety Gate Output Schema

```json
{
  "safety_evaluation": {
    "invariants": {
      "CONTRADICTION_SAFETY_CEILING": {"status": "pass", "value": 0.00, "threshold": 0.40},
      "EVIDENCE_SUFFICIENCY_FLOOR":   {"status": "pass", "value": 4, "threshold": 2},
      "ENTROPY_ESCALATION_CEILING":   {"status": "pass", "value": 0.73, "threshold": 1.5}
    },
    "gates": {
      "SINGLE_SOURCE_DOMINANCE":   {"status": "pass", "flag": null},
      "PATHOGNOMONIC_ABSENCE":     {"status": "pass", "flag": null},
      "CRITICAL_MISSINGNESS":      {"status": "pass", "flag": null, "missing": 0},
      "CONFUSION_ZONE_PROXIMITY":  {"status": "pass", "flag": null, "gap": 0.786},
      "OVERCONFIDENCE_PREVENTION": {"status": "pass", "flag": null}
    },
    "all_passed": true,
    "flags_raised": [],
    "pre_gate_recommendation": "SAFE_BIOPSY_FREE",
    "post_gate_recommendation": "SAFE_BIOPSY_FREE",
    "final_state": "CERTAINTY_STABILIZED"
  }
}
```

---

## 7. Threshold Calibration Notes

The numerical thresholds in the safety layer (0.40 for contradiction_load, 0.82 for max_certainty SAFE threshold, 1.5 bits for entropy, etc.) are **initial design values** established from clinical reasoning principles. They will be empirically calibrated against the UCI Dermatology dataset during the evaluation phase using:

- `BiopsyTriageValidator`: measures empirical correctness rate of SAFE_BIOPSY_FREE cases
- `CertaintyCalibrationAnalyzer`: measures ECE and reliability of certainty scores
- Threshold sweep experiments to identify operating points that maximize triage correctness while maintaining safety

The thresholds are stored in `configs/biopsy_thresholds.yaml` for easy adjustment without modifying source code. The calibration methodology is documented in `docs/evaluation/THRESHOLD_CALIBRATION.md` (produced during evaluation phase).

---

## 8. Relationship to Clinical Ethics

The safety layer encodes a specific **clinical risk preference**: false negatives (missed diagnoses due to incorrect SAFE recommendation) are weighted more seriously than false positives (unnecessary biopsies). This preference is:

- **Justified by asymmetric harm:** incorrect diagnosis carries downstream treatment risks; unnecessary biopsy carries procedural inconvenience
- **Consistent with clinical conservative practice** in uncertain presentation scenarios
- **Explicitly documented** so future clinical validation studies can evaluate whether the risk preference is appropriately calibrated for the specific patient population and resource context

The system is not designed to minimize total errors. It is designed to **preferentially minimize potentially harmful errors** while maintaining a useful level of diagnostic support.
