# Patent Claim Mapping
## Subsystem-to-Claim Demarcation for Intellectual Property Documentation

**Document type:** Research Reference  
**Purpose:** Map each system subsystem to its potential patent claim territory; document the technical novelty, clinical novelty, and systems novelty of each component; establish differentiation from prior art for each claim

---

> **Important disclaimer:** This document is a technical reference for novelty demarcation in the research and publication context. It is not a legal patent filing. Patent applications require formal claim drafting by a registered patent attorney under applicable patent law. This document identifies the technical terrain — the specific novel contributions — that may support future IP documentation.

---

## 1. Claim Taxonomy

Claims are organized into five categories:

| Category | Abbreviation | Description |
|---|---|---|
| Technical | TC | Novel technical mechanism or algorithm |
| Clinical | CC | Novel clinical application or clinical reasoning capability |
| Systems | SC | Novel system architecture or subsystem integration |
| Safety | SF | Novel safety constraint or clinical risk management mechanism |
| Interpretability | IC | Novel interpretability or transparency mechanism |

Each subsystem is mapped to claims in one or more categories.

---

## 2. Subsystem-to-Claim Mappings

---

### 2.1 SymbolicClinicalInferenceSystem (Model C — Primary)

**Subsystem file:** `src/models/symbolic_inference.py`

#### TC-01 — Certainty-Aware Biopsy Triage via Symbolic Clinical Inference

**Claim territory:** A method for determining whether biopsy-free differential diagnosis of erythemato-squamous diseases is clinically safe, comprising:
- receiving a set of non-invasive clinical feature observations
- propagating the observations through a structured symbolic reasoning pipeline comprising ordered evidence evaluation stages
- computing a certainty distribution over a set of disease hypotheses
- applying a clinical safety gate to the certainty distribution
- generating a four-tier biopsy triage recommendation as primary output

**Technical novelty:** The method produces a biopsy triage recommendation (not a disease classification label) as its primary output, derived from structured symbolic reasoning (not statistical pattern matching).

**Prior art distinction:**
- All prior systems produce disease labels; biopsy triage as primary output is unaddressed in prior art (see PRIOR_ART_ANALYSIS §8, Gap 1)
- No prior clinical DSS generates a four-tier biopsy triage decision through symbolic rule-based reasoning

**Dependent claim territory:**
- The symbolic reasoning pipeline comprising exactly six ordered stages (TC-02)
- The certainty distribution computed via softmax over penalized hypothesis scores (TC-04)
- The clinical safety gate comprising three invariants and five gates (SF-01)

---

#### CC-01 — Biopsy Triage as Primary Clinical Output

**Claim territory:** A clinical decision support method wherein the primary actionable output is a structured biopsy triage recommendation (SAFE_BIOPSY_FREE / MODERATE_CERTAINTY / AMBIGUOUS_CASE / BIOPSY_ADVISED) derived from symbolic clinical inference over non-invasive observable features.

**Clinical novelty:** Prior clinical decision support systems for erythemato-squamous diseases produce ranked disease hypotheses or classification labels. The clinical question — whether biopsy is warranted — is left to the clinician. This system makes that clinical question the primary output.

---

### 2.2 DiagnosticEvidenceEvaluator (Multi-Stage Rule Activation)

**Subsystem file:** `src/symbolic_engine/evidence_evaluator.py`

#### TC-02 — Multi-Stage Progressive Symbolic Reasoning with Ordered Evidence Tiers

**Claim territory:** A computational method for clinical diagnostic reasoning comprising:
- evaluation of evidence in ordered stages corresponding to distinct clinical evidence tiers
- wherein pathognomonic evidence (Tier A) is evaluated before supportive evidence (Tier B)
- wherein supportive evidence is evaluated before exclusionary/contradiction analysis
- wherein pairwise discrimination rules (Tier D) are evaluated after contradiction analysis
- wherein each stage is capable of transitioning a formal diagnostic state machine independently

**Technical novelty:** No prior clinical reasoning system processes evidence in discrete ordered tiers with distinct epistemic priority. Multi-stage ordered inference with tier-based evidence weighting is not present in classical expert systems or prior DSS.

**Prior art distinction:**
- MYCIN (1984): single-pass backward chaining with CF combination
- DXplain (1987): single-pass symptom-disease scoring
- All Class 2–4 systems: single-pass statistical classification

**Dependent claim territory:**
- Evidence tiers comprising: A (pathognomonic), B (supportive), C (auxiliary), D (discriminating)
- Stage state machine interaction (TC-03)

---

### 2.3 DiagnosticStateTracker (Formal State Machine)

**Subsystem file:** `src/symbolic_engine/state_tracker.py`

#### TC-03 — Formal Diagnostic State Machine for Clinical Reasoning

**Claim territory:** A system for tracking the epistemic state of a clinical diagnostic reasoning process, comprising:
- a finite automaton with nine states representing distinct phases of diagnostic certainty
- formal guard conditions governing transitions between states
- invariant properties guaranteeing forward monotonicity, safety gate supremacy, and terminal state completeness
- mapping of terminal states to clinical triage recommendations

**Nine states:** EVIDENCE_SPARSE → HYPOTHESIS_FORMING → REINFORCING → CONTRADICTION_EMERGED → DIAGNOSTIC_TENSION → AMBIGUITY_ESCALATED → CERTAINTY_STABILIZING → CERTAINTY_STABILIZED → BIOPSY_ESCALATED

**Technical novelty:** No prior medical expert system or clinical DSS implements a formal state machine with guard-conditioned transitions as an internal governance mechanism for diagnostic reasoning.

**Systems novelty:** The state machine is not a post-hoc characterization of outputs; it governs the reasoning process in real time as evidence is processed.

---

#### SC-01 — Dual State Machine Architecture (Backend + Frontend)

**Claim territory:** A clinical reasoning system architecture wherein:
- a backend diagnostic state machine governs clinical reasoning logic
- a frontend interaction state machine governs interface behavior
- both machines operate in parallel with synchronized transitions

**Systems novelty:** The mapping of clinical epistemic states (CERTAINTY_STABILIZED) to interface interaction states (IS-5: CERTAINTY_STABILIZATION) with formal synchronization protocol is a novel architectural pattern for clinical computing systems.

---

### 2.4 DiagnosticConflictAnalyzer (Contradiction Engine)

**Subsystem file:** `src/symbolic_engine/conflict_analyzer.py`

#### TC-04 — Contradiction-Aware Evidence Propagation with Structured Penalty Architecture

**Claim territory:** A method for propagating contradicting clinical evidence in a diagnostic reasoning system, comprising:
- maintaining an explicit contradiction matrix encoding known conflicting sign relationships between clinical features and disease hypotheses
- detecting active contradictions by evaluating observed feature values against the contradiction matrix
- applying weighted penalties to affected hypothesis certainty scores
- computing an aggregate contradiction load metric
- generating contradiction trace entries with clinical rationale and literature citations
- routing inference to BIOPSY_ADVISED when contradiction load exceeds a safety threshold

**Technical novelty:** Explicit contradiction matrices with weighted penalty propagation are not present in any prior dermatological diagnostic system. MYCIN's certainty factor combination implicitly handles some contradiction but does not maintain an explicit contradiction matrix or generate contradiction traces.

**Clinical novelty:** The contradiction trace — recording which feature contradicted which hypothesis, with what penalty, citing which literature — provides a clinician-readable account of diagnostic conflict.

**Dependent claim territory:**
- Contradiction load as a continuous aggregate metric (vs. binary conflict detection)
- Safety threshold on contradiction load triggering mandatory BIOPSY_ADVISED (SF-01)
- Contradiction event as a first-class reasoning trace entry

---

#### CC-02 — Contradiction-Aware Biopsy Escalation

**Claim territory:** A clinical decision method wherein the presence of contradicting clinical signs above a defined cumulative penalty threshold automatically generates a BIOPSY_ADVISED recommendation regardless of disease hypothesis certainty values.

**Clinical novelty:** This implements a clinically justified principle: when evidence genuinely contradicts itself, no amount of certainty from supporting signs can make biopsy-free diagnosis safe.

---

### 2.5 ClinicalSafetyGate (Safety Layer)

**Subsystem file:** `src/symbolic_engine/safety_gate.py`

#### SF-01 — Escalation-Only Clinical Safety Gate with Formal Safety Invariants

**Claim territory:** A safety constraint system for clinical diagnostic inference, comprising:
- three formal safety invariants enforced unconditionally:
  1. Contradiction Safety Ceiling: if contradiction_load ≥ threshold, prohibit SAFE_BIOPSY_FREE
  2. Evidence Sufficiency Floor: if activated_rule_count < minimum, prohibit MODERATE_CERTAINTY
  3. Entropy Escalation Ceiling: if ambiguity_index > threshold, mandate BIOPSY_ADVISED
- five contextual safety gates applying recommendation caps under specific risk conditions
- the architectural property of escalation-only: the safety layer can only increase recommendation severity toward BIOPSY_ADVISED, never decrease it

**Safety novelty:** The escalation-only property is a novel architectural guarantee. It encodes an asymmetric risk preference (false SAFE is worse than false BIOPSY_ADVISED) as a structural property of the system, not a configurable parameter.

**Prior art distinction:**
- No prior clinical DSS formalizes safety as an architectural layer with invariant guarantees
- No prior system implements escalation-only as a structural property

**Dependent claim territory:**
- Gate 2 (Pathognomonic Absence under High Certainty): certainty above 0.75 without any pathognomonic rule is flagged as "statistically-derived certainty" (SF-02)
- Gate 5 (Overconfidence Prevention): max_certainty > 0.92 with non-zero contradiction_load triggers overconfidence flag (SF-02)

---

#### SF-02 — Pathognomonic Absence Detection as Safety Signal

**Claim territory:** A safety mechanism for diagnostic inference systems wherein high certainty in the absence of any pathognomonic (Tier A) evidence is detected and treated as a safety risk signal, capping the diagnostic recommendation below the SAFE threshold.

**Safety novelty:** The distinction between "statistically-derived certainty" (from accumulation of non-specific supportive signs) and "pathognomically-anchored certainty" (supported by at least one disease-specific sign) is not present in any prior clinical reasoning system.

---

### 2.6 HypothesisCertaintyPropagator (Certainty Engine)

**Subsystem file:** `src/symbolic_engine/certainty_propagator.py`

#### TC-05 — Multi-Metric Certainty Propagation with Ambiguity Quantification

**Claim territory:** A method for computing diagnostic certainty in a symbolic reasoning system, comprising:
- propagating weighted rule activations and contradiction penalties through a disease hypothesis layer
- computing a certainty distribution via softmax over penalized hypothesis scores
- computing a certainty gap (top-2 hypothesis separation) as a decision confidence metric
- computing a Shannon entropy-based ambiguity index over the certainty distribution
- computing a contradiction load metric as aggregate active penalty weight
- using these four metrics jointly to determine biopsy triage recommendation tier

**Technical novelty:** Prior systems report a single confidence score. This system computes four independent certainty metrics (max_certainty, certainty_gap, ambiguity_index, contradiction_load) and uses their joint values to determine triage tier — enabling discrimination between "high certainty with no ambiguity" and "high certainty with competing hypotheses."

---

### 2.7 ReasoningGraphEngine (Computational Graph)

**Subsystem file:** `src/symbolic_engine/reasoning_graph.py`

#### TC-06 — Directed Reasoning Graph as Primary Computational Representation

**Claim territory:** A computational architecture for clinical diagnostic inference wherein:
- the inference process is represented as a directed weighted graph
- nodes represent: clinical features, diagnostic rules, contradiction events, disease hypotheses, certainty state, safety state, triage decision
- edge types represent: evidence support, weighted activation, contradiction, penalty, certainty propagation, safety evaluation, escalation
- inference is equivalent to forward propagation through this graph
- the graph state at each reasoning stage is stored as a temporal snapshot
- graph snapshots form a replayable diagnostic trajectory

**Systems novelty:** Prior symbolic reasoning systems (expert systems) use rule chains or networks with fixed structure. The dynamic insertion of ContradictionNodes during Stage 3 — making contradiction events first-class graph nodes that emerge during inference — is a novel architectural feature.

---

### 2.8 DiagnosticNarrativeGenerator (Reasoning Trace)

**Subsystem file:** `src/symbolic_engine/narrative_generator.py`

#### IC-01 — Intrinsic Per-Case Reasoning Trace with Literature Citations

**Claim territory:** A method for generating an interpretable audit record of a clinical diagnostic reasoning process, comprising:
- recording rule activation events during inference (not after) with rule_id, activation score, evidence tier, and literature source
- recording contradiction events with trigger feature, affected hypothesis, penalty, competing disease, and clinical rationale
- recording certainty evolution at each reasoning stage
- recording safety gate evaluation results
- assembling the above into both a machine-readable JSON audit trace and a human-readable clinician summary
- the trace is generated intrinsically as part of the reasoning process, not by post-hoc attribution to a trained model

**Interpretability novelty:** Post-hoc attribution methods (SHAP, LIME) explain a trained model's behavior. This trace documents the reasoning process as it occurs — it is not an approximation of reasoning, it is reasoning.

**Prior art distinction:**
- SHAP (Lundberg & Lee, 2017): post-hoc mathematical attribution, not intrinsic reasoning record
- LIME (Ribeiro et al., 2016): local approximation of classifier behavior
- Classical expert system explanations: limited to rule firings, no certainty evolution or state transitions

---

### 2.9 CaseReplaySystem (Trajectory Replay)

**Subsystem file:** `src/symbolic_engine/reasoning_graph.py` (trajectory) + `app/frontend/`

#### IC-02 — Replayable Diagnostic Trajectory with Stage-Level Scrubbing

**Claim territory:** A system for recording and replaying the complete temporal evolution of a clinical diagnostic reasoning process, comprising:
- storing a sequence of graph state snapshots (one per reasoning stage)
- enabling stage-level scrubbing and step-by-step playback
- synchronizing all interface components to the scrubbed stage state
- supporting snapshot comparison (two-stage side-by-side)
- supporting counterfactual analysis (feature modification and trajectory re-computation)
- supporting multi-case trajectory comparison

**Systems novelty:** No prior clinical decision support system stores and replays reasoning as a replayable trajectory with stage-level inspection. The "diagnostic flight recorder" concept applied to clinical symbolic reasoning is novel.

---

### 2.10 ClinicalSafetyGate + ClinicalEscalationEngine (Triage Decision)

**Subsystem files:** `safety_gate.py` + `escalation_engine.py`

#### SC-02 — Four-Tier Biopsy Triage Architecture with Certainty Thresholds

**Claim territory:** A biopsy triage decision system comprising:
- four triage tiers: SAFE_BIOPSY_FREE, MODERATE_CERTAINTY, AMBIGUOUS_CASE, BIOPSY_ADVISED
- threshold conditions defined over max_certainty, certainty_gap, ambiguity_index, and contradiction_load jointly
- terminal diagnostic states mapped to triage tiers via formal state machine transitions
- safety gate modifications to triage tier (escalation only)

**Clinical novelty:** A four-tier biopsy triage output for erythemato-squamous diseases derived from structured symbolic reasoning is not present in any prior computational diagnostic system.

---

## 3. Claim Interaction Map

Some claims are interdependent — enabling a dependent claim requires the independent claim:

```
TC-01 (Biopsy Triage Method)
  └── requires TC-02 (Multi-Stage Pipeline)
        └── requires TC-03 (State Machine)
              └── enables CC-01 (Clinical Biopsy Output)
  └── requires TC-04 (Contradiction Propagation)
        └── enables CC-02 (Contradiction Escalation)
  └── requires SF-01 (Safety Gate)
        └── enables SF-02 (Pathognomonic Absence Detection)
  └── requires TC-05 (Certainty Propagation)
  └── enabled by TC-06 (Reasoning Graph)
        └── enables IC-02 (Trajectory Replay)
  └── documented by IC-01 (Reasoning Trace)
  └── output defined by SC-02 (Four-Tier Triage)
```

---

## 4. Prior Art Distinctions by Claim

| Claim | MYCIN (1984) | Cipriano 2025 | SHAP+XGBoost | This System |
|---|---|---|---|---|
| TC-01 (Biopsy triage) | No | No | No | **Yes** |
| TC-02 (Multi-stage tiers) | No (single-pass) | No | No | **Yes** |
| TC-03 (State machine) | No | No | No | **Yes** |
| TC-04 (Contradiction matrix) | Partial (CF) | No | No | **Yes (explicit)** |
| TC-05 (Multi-metric certainty) | No (CF only) | No | No | **Yes** |
| TC-06 (Reasoning graph) | No | No | No | **Yes** |
| SF-01 (Escalation-only gate) | No | No | No | **Yes** |
| SF-02 (Pathognomonic absence) | No | No | No | **Yes** |
| IC-01 (Intrinsic trace) | Partial | No | No | **Yes (full)** |
| IC-02 (Trajectory replay) | No | No | No | **Yes** |
| SC-01 (Dual state machine) | No | No | No | **Yes** |
| SC-02 (Four-tier triage) | No | No | No | **Yes** |

---

## 5. Freedom-to-Operate Notes

The following aspects of the system rely on well-established prior art and are NOT claimed as novel:

- Use of fuzzy logic for ordinal clinical feature processing (extensively prior-arted)
- Use of YAML for rule storage and dynamic loading (engineering practice)
- Use of softmax for probability normalization (mathematical standard)
- Use of Shannon entropy as an uncertainty metric (information-theoretic standard)
- SHAP analysis on statistical classifiers (extensively prior-arted)
- Stratified cross-validation with Wilcoxon testing (statistical standard)
- XGBoost for statistical classification (well-established)
- React Flow graph visualization (open-source library)

---

## 6. Next Steps for IP Documentation

The following sequence is recommended if formal patent protection is pursued:

1. Retain a registered patent attorney with biomedical computing or clinical decision support expertise
2. Provide this document as a technical briefing
3. Begin with provisional application covering TC-01, TC-04, SF-01 (highest novelty density)
4. File formal claims within 12 months of provisional
5. Document reduction-to-practice by running the full pipeline on UCI Dermatology dataset and archiving results with timestamps
6. Maintain version control of all YAML rules and source code as evidence of inventive date
