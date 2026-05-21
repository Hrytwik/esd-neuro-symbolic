# Novelty Boundaries
## Technical and Clinical Contribution Demarcation

**Document type:** Research Reference  
**Purpose:** Define the precise boundaries of novelty for this system relative to all known prior art; establish claims that may support future publication, IP documentation, and research positioning

---

## 1. Primary Novelty Claim

**Certainty-Aware Biopsy Triage Through Structured Symbolic Clinical Reasoning for Erythemato-Squamous Disease Differential Diagnosis**

A computational clinical reasoning framework that employs structured, literature-grounded symbolic rules, six-stage progressive evidence processing, a formal diagnostic state machine, and a clinical safety gate to determine whether biopsy-free differential diagnosis of erythemato-squamous diseases is clinically safe — producing a four-tier biopsy triage recommendation as its primary output, rather than a disease classification label.

**What makes this primary claim novel:**

1. No prior computational system for this disease domain produces biopsy triage as its primary output
2. No prior system combines symbolic rule-based reasoning with explicit contradiction propagation and a formal certainty model for this problem
3. The concept of "certainty-awareness" here is architecturally embodied — certainty evolves through structured reasoning stages, is subject to contradiction penalties, and is gated by formal safety invariants — not merely appended as a calibrated probability score to a classifier output

---

## 2. Secondary Novelty Claims

### N2 — Multi-Stage Progressive Symbolic Reasoning Pipeline

**Claim:** A six-stage symbolic reasoning architecture that processes evidence categories in a defined epistemic order: (0) clinical grading, (1) pathognomonic pattern detection, (2) supportive evidence integration, (3) exclusionary and contradiction analysis, (4) pairwise discrimination, (5) certainty stabilization and safety gating, (6) triage decision and narrative generation.

**Why novel:**
- Classical expert systems (MYCIN and successors) use single-pass inference with certainty factor combination
- No prior biopsy-free dermatological system processes evidence in ordered tiers with distinct epistemic weight
- The ordering is clinically grounded: a pathognomonic sign should anchor a hypothesis before supportive signs reinforce it; contradictions should be detected before certainty is finalized

**Prior art that does not address this:**
- MYCIN (1984): single-pass backward chaining
- DXplain (1987): probabilistic symptom-disease scoring, single-pass
- Cipriano et al. (2025): single-pass classifier with engineered features
- All Class 1–4 systems (Section 2 of Prior Art Analysis): single-pass

---

### N3 — Formal Diagnostic State Machine with Clinical Guard Conditions

**Claim:** A nine-state finite automaton (EVIDENCE_SPARSE → HYPOTHESIS_FORMING → REINFORCING → CONTRADICTION_EMERGED → DIAGNOSTIC_TENSION → AMBIGUITY_ESCALATED → CERTAINTY_STABILIZING → CERTAINTY_STABILIZED → BIOPSY_ESCALATED) with formally specified guard conditions, transition actions, and invariant properties governing the progression of diagnostic reasoning.

**Why novel:**
- No prior medical expert system models diagnostic reasoning as an explicit state machine with guard-conditioned transitions
- The state machine is not a post-hoc characterization of outputs — it is a real-time governance mechanism that constrains the reasoning process while it occurs
- The states are clinically interpretable and directly mappable to the four triage outcomes
- The invariant properties (forward monotonicity, safety gate supremacy, terminal state completeness, triage monotonicity under contradiction) provide formal guarantees about system behavior

**Prior art that does not address this:**
- No prior dermatological or general medical DSS has published a formal diagnostic state machine specification

---

### N4 — Contradiction-Aware Evidence Propagation with Penalty Architecture

**Claim:** An explicit contradiction matrix encoding clinically documented conflicting sign relationships across all six erythemato-squamous diseases, with a penalty propagation mechanism that reduces hypothesis certainty when contradicting features are active — producing contradiction traces as part of the per-case reasoning output.

**Why novel:**
- MYCIN uses combined CFs with a formula that implicitly handles some contradiction, but does not maintain an explicit contradiction matrix or generate contradiction traces
- No prior biopsy-free dermatological system maintains a contradiction state or detects contradiction emergence as a discrete clinical event
- The contradiction matrix is explicitly derived from published dermatology literature (Fitzpatrick, Habif, Andrews) — not learned from data
- The system generates contradiction events as first-class trace entries with clinical rationale and literature citations

**Specific contradiction relationships encoded (novel encodings for this domain):**
- Koebner phenomenon + oral mucosal involvement → psoriasis/lichen planus diagnostic tension
- Follicular papules + any other disease → PRP pathognomonic conflict
- Definite borders + scalp involvement without Koebner → psoriasis/seborrheic dermatitis discrimination
- Oral mucosal involvement + knee/elbow involvement → LP/psoriasis simultaneous multi-disease signal

---

### N5 — ClinicalSafetyGate: Escalation-Only Architectural Safety Layer

**Claim:** A formal safety layer enforcing three binary invariants (Contradiction Safety Ceiling, Evidence Sufficiency Floor, Entropy Escalation Ceiling) and five contextual gates (Single-Source Dominance, Pathognomonic Absence under High Certainty, Critical Feature Missingness, Confusion Zone Proximity, Overconfidence Prevention) — with the architectural constraint that the layer is escalation-only and cannot improve a triage recommendation.

**Why novel:**
- No prior clinical DSS or expert system formalizes a safety layer as an architectural component with explicit invariants and gate conditions
- The escalation-only property is a novel architectural guarantee: the safety layer introduces an asymmetric risk preference (avoiding false SAFE recommendations is weighted more heavily than avoiding false BIOPSY_ADVISED recommendations) as a structural property, not a tunable parameter
- The concept of "pathognomonic absence under high certainty" as a safety signal (Gate 2) is not present in any prior system

---

### N6 — Standalone Symbolic Inference Mode Without Statistical Dependency

**Claim:** The symbolic engine functions as a complete, self-contained diagnostic reasoning subsystem — producing disease hypothesis distributions, certainty scores, and biopsy triage recommendations from 12 clinical features without requiring any statistical classifier.

**Why novel:**
- In existing hybrid diagnostic systems, the symbolic component adds post-hoc constraints to a statistical classifier output; the statistical classifier is the primary inference mechanism
- In this system, the symbolic engine is the primary inference mechanism; the statistical classifier (StatisticalRefinementAdjunct) is the optional supplement
- Standalone mode allows the system to operate in contexts where training data is unavailable or unsuitable (new disease presentations not in training corpus), because the reasoning is rule-based rather than data-fitted
- Standalone mode's interpretability is fully intrinsic: the reasoning trace is the complete audit of the inference process

---

### N7 — DiagnosticNarrativeGenerator: Intrinsic Per-Case Reasoning Trace

**Claim:** A subsystem that assembles, during the inference process (not after), a structured per-case reasoning trace documenting: activated rules with evidence tier and literature source; contradiction events with clinical rationale; certainty evolution through stages; safety gate evaluation results; final state; and biopsy triage justification — generating both a machine-readable JSON audit log and a clinician-readable natural language summary.

**Why novel:**
- SHAP and LIME produce post-hoc mathematical attributions; they do not produce clinical reasoning narratives
- RuleFit extracts population-level rules; it does not produce per-case reasoning traces
- Classical expert systems produce explanations limited to rule firings; they do not include certainty evolution, state transitions, or safety gate evaluations
- The reasoning trace includes literature citations for each activated rule — connecting the computational inference directly to the published clinical evidence base

---

## 3. Boundary Conditions — What Is Not Claimed as Novel

**Not novel:** The use of clinical features from the UCI Dermatology dataset for erythemato-squamous classification. This is well-established.

**Not novel:** The use of symbolic rules derived from dermatology literature. Expert systems have done this since the 1980s.

**Not novel:** Fuzzy logic for ordinal clinical feature processing. This has been applied in medical systems for decades.

**Not novel:** SHAP analysis on statistical classifiers in a clinical context. This is standard practice.

**Not novel:** Evaluation using macro F1, per-class F1, and Wilcoxon signed-rank tests. These are standard evaluation methods.

**Not novel:** The statistical comparison models (Model A: XGBoost on 34 features; Model B: XGBoost on 12 features). These are replications of existing benchmarks.

---

## 4. Technical Claims Summary

The following technical claims define the system's novel contributions in terms amenable to research publication:

**Claim T1:** A symbolic clinical inference system for erythemato-squamous disease differential diagnosis that produces biopsy triage recommendations (SAFE_BIOPSY_FREE / MODERATE_CERTAINTY / AMBIGUOUS_CASE / BIOPSY_ADVISED) as primary output, derived from a structured symbolic reasoning process over 12 non-invasive clinical features.

**Claim T2:** A six-stage progressive symbolic reasoning pipeline in which evidence categories (pathognomonic, supportive, exclusionary, discriminating) are evaluated in clinically motivated order, each stage capable of transitioning a formal nine-state diagnostic state machine.

**Claim T3:** An explicit contradiction propagation mechanism encoding documented clinical contradictions as a weighted penalty matrix, reducing hypothesis certainty when contradicting signs are active and generating clinician-readable contradiction events.

**Claim T4:** A clinical safety gate enforcing escalation-only constraints on triage recommendations through three formal safety invariants and five contextual safety gates, preventing overconfident safe-diagnosis recommendations under evidence-sparse or high-contradiction conditions.

**Claim T5:** A standalone symbolic inference mode producing complete diagnostic outputs (hypothesis distribution + certainty metrics + biopsy triage + reasoning trace) without statistical classifier dependency, enabling interpretation of inference as clinical reasoning rather than statistical prediction.

---

## 5. Clinical Claims Summary

**Claim C1:** Structured symbolic clinical reasoning measurably narrows the diagnostic accuracy gap between biopsy-free (12 clinical features) and biopsy-assisted (34 features) erythemato-squamous disease diagnosis, as evaluated by macro F1 improvement of Model C over Model B with p < 0.05 under Wilcoxon signed-rank test.

**Claim C2:** The symbolic inference engine achieves a statistically reliable SAFE_BIOPSY_FREE recommendation rate (measured by empirical correctness: correct diagnoses / total SAFE cases), demonstrating that the certainty threshold and safety gate design are appropriately calibrated for clinical confidence.

**Claim C3:** The system's contradiction detection mechanism identifies cases at elevated misclassification risk (measured by correlation between active contradiction events and actual classification errors), providing a clinically interpretable signal for diagnostic caution.

**Claim C4:** The system's certainty metrics (ambiguity_index, certainty_gap, contradiction_load) are correlated with classification confidence in the direction predicted by clinical reasoning theory — high ambiguity and high contradiction load predict lower classification accuracy, validating the certainty model as clinically meaningful rather than arbitrary.

---

## 6. Publication Positioning

The system's novelty is best positioned at the intersection of:

1. **Clinical decision support systems** — contributing a new architecture for certainty-aware biopsy triage
2. **Symbolic clinical reasoning** — demonstrating multi-stage evidence propagation with formal state machine governance
3. **Biomedical informatics** — providing interpretable, literature-grounded diagnostic reasoning with per-case audit trails
4. **Low-resource clinical computing** — demonstrating robustness under feature sparsity relevant to primary care settings

Target publication venues may include:
- Journal of Biomedical Informatics
- Computer Methods and Programs in Biomedicine
- MEDINFO (International Congress on Medical Informatics)
- Methods of Information in Medicine
- npj Digital Medicine
- Journal of the American Medical Informatics Association (JAMIA)

Primary differentiator for all venues: the combination of standalone symbolic reasoning + biopsy triage as primary output + formal safety layer + diagnostic state machine is not present in any published work in this domain.
