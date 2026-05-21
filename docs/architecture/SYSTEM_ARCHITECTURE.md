# System Architecture Specification
## Certainty-Aware Symbolic Clinical Inference Framework
### Computational Differential Diagnosis for Erythemato-Squamous Diseases

**Document type:** Master Architecture Reference  
**Status:** Design-stabilized — pre-implementation  
**Dataset:** UCI Dermatology (id=33), 366 patients, 34 features, CC BY 4.0  
**Clinical domain:** Erythemato-squamous disease differential diagnosis

---

## 1. System Identity and Purpose

This system is a **Certainty-Aware Symbolic Dermatological Reasoning Engine (CASDRE)** — a computational clinical reasoning infrastructure that determines whether erythemato-squamous diseases can be safely diagnosed without biopsy under low-resource clinical conditions.

The system is **not** a predictive classifier. Its primary output is a **biopsy triage recommendation** derived from structured symbolic reasoning over non-invasive clinical features. Disease hypothesis scores are intermediate computation products; the actionable output is:

```
SAFE_BIOPSY_FREE    — diagnosis achievable with acceptable certainty; biopsy not warranted
MODERATE_CERTAINTY  — probable diagnosis; biopsy optional; clinical follow-up recommended
AMBIGUOUS_CASE      — competing hypotheses irresolvable from clinical evidence alone
BIOPSY_ADVISED      — symbolic reasoning cannot safely differentiate; histopathology required
```

---

## 2. Design Philosophy

**Principle 1 — Interpretability is intrinsic, not post-hoc.**  
Every inference step is recorded as it occurs. The reasoning trace is generated during inference, not explained after the fact by an external attribution method.

**Principle 2 — Uncertainty is first-class.**  
The system does not produce a point estimate and append a confidence score. Uncertainty propagates through every reasoning stage and governs the final triage recommendation.

**Principle 3 — Clinical safety constrains inference.**  
Overconfident inference under contradiction-heavy conditions is architecturally prevented by the Clinical Safety Gate. Safety is a structural property, not a post-processing filter.

**Principle 4 — Evidence propagates through structured stages, not statistical transforms.**  
Pathognomonic signals, supportive clusters, exclusionary findings, and pairwise discriminators are processed in discrete, ordered reasoning stages. Each stage can transition the diagnostic state machine independently.

**Principle 5 — The symbolic engine is standalone.**  
The symbolic inference engine produces complete diagnostic outputs (disease hypothesis distribution + certainty scores + biopsy triage + reasoning trace) from 12 clinical features without requiring any downstream statistical classifier. The statistical refinement layer is supplementary, not primary.

**Principle 6 — Rules are literature-grounded, not learned.**  
All diagnostic rules are derived from peer-reviewed dermatology literature (Fitzpatrick, Habif, Andrews, Lancet, NEJM) and stored in structured YAML. No rule is an arbitrary heuristic.

---

## 3. System Boundaries

### 3.1 Inputs

| Input | Source | Used by |
|---|---|---|
| 12 non-invasive clinical features | Patient examination | Symbolic engine + Models B and C |
| 22 histopathological features | Biopsy results | Model A (reference only) |
| All 34 features | Combined | Model A (upper bound) |

### 3.2 Primary Outputs (Symbolic Engine)

| Output | Type | Description |
|---|---|---|
| `biopsy_triage` | Enum[4] | Primary recommendation: SAFE / MODERATE / AMBIGUOUS / ADVISED |
| `leading_diagnosis` | String | Most probable disease hypothesis |
| `disease_certainty` | Dict[disease→float] | Certainty score per disease (sum to 1.0) |
| `max_certainty` | Float [0,1] | Certainty of leading hypothesis |
| `certainty_gap` | Float [0,1] | Separation between top-2 hypotheses |
| `ambiguity_index` | Float [0,∞] | Shannon entropy of certainty distribution |
| `contradiction_load` | Float [0,1] | Aggregate active contradiction penalty |
| `diagnostic_state` | Enum[9] | Final state from diagnostic state machine |
| `reasoning_trace` | JSON + Text | Full reasoning log; clinician-readable summary |
| `safety_flags` | List[String] | Active safety gate triggers |

### 3.3 Secondary Outputs (Evaluation + Explainability)

- Macro F1 / per-class F1 across Models A, B, C
- Certainty calibration curves (ECE)
- Robustness profiles under missingness and noise
- Ablation results (per-component contribution)
- Confusion profiles per disease pair
- SHAP attribution (statistical layers only)
- Publication-quality visualizations

---

## 4. Subsystem Inventory

The system is organized into six architectural layers. All subsystem names use clinical terminology.

### Layer 1 — Clinical Data Infrastructure

| Module (file) | Class | Responsibility |
|---|---|---|
| `data/loader.py` | `ClinicalDataLoader` | UCI fetch, feature separation (clinical vs. histopathological), train/test split |
| `data/preprocessing.py` | `ClinicalDataPreprocessor` | Ordinal normalization, missing value flagging, feature standardization |
| `data/feature_registry.py` | `ClinicalFeatureRegistry` | Metadata store for all 34 features: type, scale, domain relevance, biopsy dependency |

### Layer 2 — Diagnostic Knowledge Base

| Module (file) | Class | Responsibility |
|---|---|---|
| `symbolic_engine/rule_registry.py` | `DiagnosticRuleRepository` | YAML rule loader, schema validation, rule indexing by disease and evidence tier |
| `symbolic_engine/rule_compiler.py` | `ClinicalRuleCompiler` | Translates YAML rule definitions into callable activation functions; validates completeness |
| `rules/contradiction_matrix.yaml` | `ContradictionKnowledgeBase` | Structured encoding of all known feature-level contradictions across disease pairs |

### Layer 3 — Symbolic Reasoning Engine (Primary Contribution)

This layer constitutes the novel contribution of the system. It operates in standalone mode without any statistical classifier.

| Module (file) | Class | Responsibility |
|---|---|---|
| `symbolic_engine/clinical_grading.py` | `ClinicalGradingModule` | Converts ordinal features (0–3) to fuzzy membership values; handles partial activation; computes feature completeness |
| `symbolic_engine/evidence_evaluator.py` | `DiagnosticEvidenceEvaluator` | Evaluates each diagnostic rule against current feature grades; computes weighted activation scores per disease |
| `symbolic_engine/conflict_analyzer.py` | `DiagnosticConflictAnalyzer` | Detects active contradiction features; computes penalty per hypothesis; generates contradiction trace; updates diagnostic tension state |
| `symbolic_engine/certainty_propagator.py` | `HypothesisCertaintyPropagator` | Aggregates evidence into disease certainty distribution; computes max_certainty, certainty_gap, ambiguity_index, contradiction_load |
| `symbolic_engine/state_tracker.py` | `DiagnosticStateTracker` | Maintains and transitions the 9-state diagnostic state machine; logs state history |
| `symbolic_engine/safety_gate.py` | `ClinicalSafetyGate` | Enforces 3 formal safety invariants and 5 safety gates; escalation-only property |
| `symbolic_engine/escalation_engine.py` | `ClinicalEscalationEngine` | Maps final diagnostic state + certainty metrics to biopsy triage recommendation |
| `symbolic_engine/narrative_generator.py` | `DiagnosticNarrativeGenerator` | Assembles per-case reasoning trace; generates JSON audit log and clinician-readable summary |

### Layer 4 — Reference and Comparison Systems

| Module (file) | Class | Role | Feature Set |
|---|---|---|---|
| `models/reference_system.py` | `BiopsyAssistedReferenceSystem` | Model A: upper-bound reference | All 34 features |
| `models/clinical_baseline.py` | `ClinicalOnlyInferenceBaseline` | Model B: low-resource baseline | 12 clinical only |
| `models/symbolic_inference.py` | `SymbolicClinicalInferenceSystem` | Model C: primary contribution | 12 clinical → symbolic engine |
| `models/statistical_adjunct.py` | `StatisticalRefinementAdjunct` | Model C hybrid: symbolic scores → statistical refinement | Symbolic feature vector |

> **Architecture note:** Model C in **standalone mode** uses only the symbolic engine. In **hybrid mode**, the symbolic engine's output scores are used as a feature vector by the `StatisticalRefinementAdjunct`. Both modes are evaluated separately. The standalone mode is the primary contribution; hybrid mode is a secondary validation.

### Layer 5 — Evaluation Infrastructure

| Module (file) | Class | Responsibility |
|---|---|---|
| `evaluation/performance.py` | `DiagnosticPerformanceEvaluator` | Stratified 10-fold CV; macro F1; per-class F1; Wilcoxon B vs. C |
| `evaluation/robustness.py` | `LowResourceRobustnessAnalyzer` | Feature masking (1–5 features); ordinal noise injection; partial examination simulation |
| `evaluation/calibration.py` | `CertaintyCalibrationAnalyzer` | ECE computation; reliability diagrams; Platt scaling validation |
| `evaluation/ablation.py` | `ComponentAblationStudy` | Per-component contribution: rules only vs. + contradiction vs. + safety gate vs. full |
| `evaluation/confusion_analysis.py` | `DiseaseConfusionProfiler` | Per-disease confusion matrices; confusion pair identification; triage error analysis |
| `evaluation/triage_validation.py` | `BiopsyTriageValidator` | Validates triage accuracy: when SAFE_BIOPSY_FREE is recommended, measures empirical correctness rate |

### Layer 6 — Explainability and Clinical Reporting

| Module (file) | Class | Responsibility |
|---|---|---|
| `explainability/feature_analyzer.py` | `PostHocFeatureAnalyzer` | SHAP TreeExplainer on Models A, B, and C hybrid; beeswarm plots; per-class SHAP bars |
| `explainability/rule_extractor.py` | `RuleContributionExtractor` | RuleFit extraction on Model C hybrid; rule validation against published criteria |
| `explainability/pathway_visualizer.py` | `ReasoningPathwayVisualizer` | Symbolic activation graphs; contradiction maps; certainty evolution visualizations |
| `explainability/clinical_reporter.py` | `ClinicalReportGenerator` | Full per-case PDF-style clinical report; population-level diagnostic summary |

---

## 5. Standalone Symbolic Engine Architecture

This is the architectural heart of the system. The symbolic engine runs as a fully self-contained pipeline:

```
Input: 12 clinical feature values (ordinal 0–3, binary 0/1, continuous)
         │
         ▼
Stage 0: ClinicalGradingModule
  - Ordinal → fuzzy membership
  - Feature completeness scoring
  - Missing feature flagging
         │
         ▼
Stage 1: DiagnosticEvidenceEvaluator (Tier A — Pathognomonic)
  - Activate pathognomonic rules (evidence_tier: A)
  - Initial disease hypothesis formation
  - DiagnosticStateTracker: S0 → S1
         │
         ▼
Stage 2: DiagnosticEvidenceEvaluator (Tier B — Supportive)
  - Activate supportive cluster rules
  - Refine hypothesis scores
  - DiagnosticStateTracker: S1 → S2 (if reinforcing) or remain S1
         │
         ▼
Stage 3: DiagnosticConflictAnalyzer
  - Check contradiction features for each hypothesis
  - Apply weighted contradiction penalties
  - Generate contradiction trace entries
  - DiagnosticStateTracker: → S3 (contradiction emerged) or continue
         │
         ▼
Stage 4: DiagnosticEvidenceEvaluator (Tier D — Discriminating)
  - Activate pairwise cross-disease discriminators
  - Target highest-confusion disease pairs
  - Update certainty separation
  - DiagnosticStateTracker: → S4/S5/S6 based on gap and entropy
         │
         ▼
Stage 5: HypothesisCertaintyPropagator + ClinicalSafetyGate
  - Compute final certainty distribution (softmax)
  - Compute: max_certainty, certainty_gap, ambiguity_index, contradiction_load
  - Run all 3 safety invariants + 5 safety gates
  - DiagnosticStateTracker: → S7 (stabilized) or S8 (escalated)
         │
         ▼
Stage 6: ClinicalEscalationEngine + DiagnosticNarrativeGenerator
  - Map final diagnostic state → biopsy triage recommendation
  - Assemble reasoning trace (JSON + text)
  - Generate clinician summary
         │
         ▼
Output: {
  biopsy_triage, leading_diagnosis, disease_certainty,
  max_certainty, certainty_gap, ambiguity_index,
  contradiction_load, diagnostic_state, safety_flags,
  reasoning_trace, clinician_summary
}
```

---

## 6. Rule Schema

Rules are stored in YAML files under `rules/`. Each rule follows this schema:

```yaml
- rule_id: PSO_001
  disease_target: psoriasis
  rule_name: "Koebner Isomorphic Response — Psoriasis Indicator"
  evidence_tier: A                    # A=pathognomonic, B=supportive, C=auxiliary, D=discriminating
  activation_logic: binary            # binary | fuzzy | threshold | composite
  confidence_weight: 0.85
  supporting_features:
    - feature: koebner_phenomenon
      condition: eq
      threshold: 1
      fuzzy_range: null
      partial_weight: 1.0
  contradiction_features:
    - feature: oral_mucosal_involvement
      condition: eq
      threshold: 1
      penalty: 0.30
      competing_disease: lichen_planus
      rationale: "Wickham's striae pathognomonic for LP, incompatible with isolated psoriasis"
    - feature: follicular_papules
      condition: eq
      threshold: 1
      penalty: 0.45
      competing_disease: pityriasis_rubra_pilaris
      rationale: "Follicular papules are pathognomonic for PRP"
  clinical_rationale: >
    The Koebner (isomorphic) phenomenon — development of lesions at sites of skin trauma —
    occurs in 25–50% of psoriasis patients and represents one of the most clinically
    specific observable signs. Its presence significantly elevates psoriasis probability
    in the erythemato-squamous differential.
  literature_source: "Griffiths CEM & Barker JNWN, Lancet 2007;370:263–271"
  min_activation_threshold: 0.50
  max_single_rule_certainty_contribution: 0.60
```

Rule YAML files per disease:
- `rules/psoriasis.yaml` — 6–8 rules
- `rules/seborrheic_dermatitis.yaml` — 4–5 rules
- `rules/lichen_planus.yaml` — 5–6 rules
- `rules/pityriasis_rosea.yaml` — 4–5 rules
- `rules/chronic_dermatitis.yaml` — 4–5 rules
- `rules/pityriasis_rubra_pilaris.yaml` — 3–4 rules
- `rules/discriminators.yaml` — 6–8 cross-disease pairwise rules
- `rules/contradiction_matrix.yaml` — structured contradiction definitions

Estimated total: **36–46 literature-traceable rules** across all files.

---

## 7. Three-System Comparison Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPARISON FRAMEWORK                         │
│                                                                 │
│  Model A: BiopsyAssistedReferenceSystem                        │
│  ─────────────────────────────────────                         │
│  Features: all 34 (12 clinical + 22 histopathological)         │
│  Architecture: statistical classifier (XGBoost)                │
│  Purpose: upper-bound reference; clinically infeasible          │
│  Expected macro F1: ~0.97                                       │
│  Primary use: sanity check and gap quantification               │
│                                                                 │
│  Model B: ClinicalOnlyInferenceBaseline                        │
│  ─────────────────────────────────────                         │
│  Features: 12 clinical only                                     │
│  Architecture: statistical classifier (XGBoost)                │
│  Purpose: low-resource baseline; standard approach              │
│  Expected macro F1: ~0.83–0.87 (Cipriano 2025: 0.86)          │
│  Primary use: establishes gap that Model C must narrow          │
│                                                                 │
│  Model C: SymbolicClinicalInferenceSystem (PRIMARY)            │
│  ──────────────────────────────────────────────────            │
│  Features: 12 clinical → symbolic engine → certainty scores    │
│  Architecture: 6-stage symbolic reasoning pipeline             │
│  Standalone mode: symbolic engine alone                         │
│  Hybrid mode: symbolic scores → StatisticalRefinementAdjunct   │
│  Purpose: primary contribution; certainty-aware biopsy triage   │
│  Primary use: demonstrate symbolic reasoning narrows the gap    │
│  Primary output: biopsy triage recommendation (not label)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Complete Project Structure

```
D:\esd-neuro-symbolic\
│
├── src/
│   ├── data/
│   │   ├── loader.py                    # ClinicalDataLoader
│   │   ├── preprocessing.py             # ClinicalDataPreprocessor
│   │   └── feature_registry.py          # ClinicalFeatureRegistry
│   │
│   ├── symbolic_engine/
│   │   ├── rule_registry.py             # DiagnosticRuleRepository
│   │   ├── rule_compiler.py             # ClinicalRuleCompiler
│   │   ├── clinical_grading.py          # ClinicalGradingModule
│   │   ├── evidence_evaluator.py        # DiagnosticEvidenceEvaluator
│   │   ├── conflict_analyzer.py         # DiagnosticConflictAnalyzer
│   │   ├── certainty_propagator.py      # HypothesisCertaintyPropagator
│   │   ├── state_tracker.py             # DiagnosticStateTracker
│   │   ├── safety_gate.py               # ClinicalSafetyGate
│   │   ├── escalation_engine.py         # ClinicalEscalationEngine
│   │   └── narrative_generator.py       # DiagnosticNarrativeGenerator
│   │
│   ├── models/
│   │   ├── reference_system.py          # BiopsyAssistedReferenceSystem (Model A)
│   │   ├── clinical_baseline.py         # ClinicalOnlyInferenceBaseline (Model B)
│   │   ├── symbolic_inference.py        # SymbolicClinicalInferenceSystem (Model C)
│   │   └── statistical_adjunct.py       # StatisticalRefinementAdjunct (C hybrid)
│   │
│   ├── evaluation/
│   │   ├── performance.py               # DiagnosticPerformanceEvaluator
│   │   ├── robustness.py                # LowResourceRobustnessAnalyzer
│   │   ├── calibration.py               # CertaintyCalibrationAnalyzer
│   │   ├── ablation.py                  # ComponentAblationStudy
│   │   ├── confusion_analysis.py        # DiseaseConfusionProfiler
│   │   └── triage_validation.py         # BiopsyTriageValidator
│   │
│   ├── explainability/
│   │   ├── feature_analyzer.py          # PostHocFeatureAnalyzer
│   │   ├── rule_extractor.py            # RuleContributionExtractor
│   │   ├── pathway_visualizer.py        # ReasoningPathwayVisualizer
│   │   └── clinical_reporter.py         # ClinicalReportGenerator
│   │
│   └── visualization/
│       ├── diagnostic_plots.py
│       ├── certainty_charts.py
│       └── clinical_dashboards.py
│
├── rules/
│   ├── psoriasis.yaml
│   ├── seborrheic_dermatitis.yaml
│   ├── lichen_planus.yaml
│   ├── pityriasis_rosea.yaml
│   ├── chronic_dermatitis.yaml
│   ├── pityriasis_rubra_pilaris.yaml
│   ├── discriminators.yaml
│   └── contradiction_matrix.yaml
│
├── configs/
│   ├── features.yaml                    # feature registry config
│   ├── biopsy_thresholds.yaml           # triage decision thresholds
│   ├── model_params.yaml                # hyperparameter grids
│   └── evaluation.yaml                  # CV folds, metrics, seeds
│
├── outputs/
│   ├── figures/                         # publication-quality plots
│   ├── tables/                          # CSV result tables
│   ├── traces/                          # per-case reasoning traces (JSON)
│   └── reports/                         # extracted rules, validation notes
│
├── docs/
│   ├── architecture/
│   │   ├── SYSTEM_ARCHITECTURE.md       # this document
│   │   ├── DIAGNOSTIC_STATE_MODEL.md
│   │   ├── SUBSYSTEM_INTERACTION_DIAGRAM.md
│   │   ├── PROGRESSIVE_REASONING_STAGES.md
│   │   └── CLINICAL_SAFETY_LAYER.md
│   └── research/
│       ├── PRIOR_ART_ANALYSIS.md
│       └── NOVELTY_BOUNDARIES.md
│
├── tests/
│   ├── test_clinical_grading.py
│   ├── test_evidence_evaluator.py
│   ├── test_conflict_analyzer.py
│   ├── test_certainty_propagator.py
│   ├── test_state_tracker.py
│   ├── test_safety_gate.py
│   └── test_escalation_engine.py
│
├── main.py                              # full pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## 9. Evaluation Methodology Summary

### Primary Comparison

| Comparison | Metric | Method | Purpose |
|---|---|---|---|
| B vs. C (macro F1) | Macro F1 improvement | Wilcoxon signed-rank on 10-fold CV | Does symbolic reasoning improve diagnostic accuracy? |
| A vs. C (diagnostic gap) | Gap = F1(A) − F1(C) | Direct comparison | How much does biopsy-free symbolic reasoning narrow the biopsy-dependent upper bound? |
| Triage accuracy | SAFE_BIOPSY_FREE correctness rate | Empirical: correct diagnoses / total SAFE cases | When system recommends biopsy-free, how often is it right? |

### Secondary Analyses

| Analysis | Module | Output |
|---|---|---|
| Certainty calibration | `CertaintyCalibrationAnalyzer` | ECE, reliability diagrams |
| Low-resource robustness | `LowResourceRobustnessAnalyzer` | F1 vs. missingness curves |
| Component ablation | `ComponentAblationStudy` | Per-component F1 delta |
| Contradiction detection utility | `DiseaseConfusionProfiler` | Contradiction precision vs. classification error rate |
| Ambiguity-accuracy correlation | `CertaintyCalibrationAnalyzer` | Spearman ρ (ambiguity_index, classification_error) |

---

## 10. Clinical Framing

The system does not replace clinical judgment. It provides structured computational support for the clinical question: **"Based on the observable signs, is a biopsy necessary for safe differential diagnosis of this erythemato-squamous presentation?"**

The biopsy triage output is designed to inform — not replace — the clinician's decision. The reasoning trace is designed to be auditable, challengeable, and educationally useful to clinicians in training or resource-constrained settings.

The system is evaluated against histopathological ground truth (Model A) specifically to quantify the safety margin of biopsy-free recommendations — not merely to maximize accuracy.
