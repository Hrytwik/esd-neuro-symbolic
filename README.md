# Computational Differential Diagnosis System for Erythemato-Squamous Diseases
## Certainty-Aware Symbolic Clinical Inference Framework

---

## Clinical Problem

Erythemato-squamous diseases — psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, and pityriasis rubra pilaris — share overlapping surface presentations (erythema, scaling, border characteristics) that are conventionally resolved through skin biopsy. Biopsy is invasive, specialist-gated, and inaccessible in resource-limited primary care settings.

Every existing computational diagnostic system for this disease group achieves high accuracy (96–99%) by incorporating histopathological features that are **only available after biopsy has already been performed**. These systems cannot inform the clinical decision of whether biopsy is warranted in the first place.

This project addresses the prior clinical question:

> **Given only the 12 non-invasive observable clinical signs, can a structured clinical reasoning system determine whether biopsy-free differential diagnosis is safe — and when it is not?**

---

## System Identity

This system is a **Certainty-Aware Symbolic Dermatological Reasoning Engine (CASDRE)** — a computational clinical reasoning infrastructure, not a statistical classifier.

**Primary output:** A biopsy triage recommendation derived from structured symbolic reasoning:

```
SAFE_BIOPSY_FREE    — clinical evidence sufficient for confident biopsy-free diagnosis
MODERATE_CERTAINTY  — probable diagnosis; clinical follow-up recommended; biopsy optional
AMBIGUOUS_CASE      — competing hypotheses irresolvable from observable signs alone
BIOPSY_ADVISED      — symbolic reasoning cannot safely differentiate; histopathology required
```

**Disease hypothesis scores** are intermediate computation products. The triage recommendation is the actionable clinical output.

---

## Architecture Overview

```
12 Non-Invasive Clinical Features
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│            SYMBOLIC REASONING ENGINE                    │
│                                                         │
│  Stage 0: ClinicalGradingModule                        │
│           (ordinal → fuzzy membership)                  │
│                     │                                   │
│  Stage 1: DiagnosticEvidenceEvaluator [Tier A]         │
│           (pathognomonic pattern detection)             │
│                     │                                   │
│  Stage 2: DiagnosticEvidenceEvaluator [Tier B]         │
│           (supportive evidence integration)             │
│                     │                                   │
│  Stage 3: DiagnosticConflictAnalyzer                   │
│           (contradiction detection + penalty)           │
│                     │                                   │
│  Stage 4: DiagnosticEvidenceEvaluator [Tier D]         │
│           (pairwise disease discriminators)             │
│                     │                                   │
│  Stage 5: HypothesisCertaintyPropagator                │
│           + ClinicalSafetyGate                         │
│           (certainty propagation + safety invariants)   │
│                     │                                   │
│  Stage 6: ClinicalEscalationEngine                     │
│           + DiagnosticNarrativeGenerator               │
│           (biopsy triage + reasoning trace)             │
└─────────────────────────────────────────────────────────┘
              │
              ▼
    Biopsy Triage Recommendation
    + Disease Certainty Distribution
    + Reasoning Trace (JSON + clinician summary)
```

The symbolic engine operates in **standalone mode** — no statistical classifier is required for basic diagnostic reasoning.

---

## Three-System Comparison

| System | Feature Set | Architecture | Purpose |
|---|---|---|---|
| **Model A** — BiopsyAssistedReferenceSystem | All 34 features (12 clinical + 22 histopathological) | Statistical classifier | Upper-bound reference; quantifies biopsy-dependent advantage |
| **Model B** — ClinicalOnlyInferenceBaseline | 12 clinical features only | Statistical classifier | Low-resource baseline; establishes gap to narrow |
| **Model C** — SymbolicClinicalInferenceSystem | 12 clinical features → symbolic engine | 6-stage symbolic reasoning pipeline | **Primary contribution**; certainty-aware biopsy triage |

The central research question: how much of the diagnostic gap between Model B (biopsy-free baseline) and Model A (biopsy-assisted reference) can be closed by structured symbolic clinical reasoning?

---

## Primary Novelty

The system makes the following novel contributions:

1. **Certainty-aware biopsy triage** as primary output (not disease label)
2. **Six-stage progressive symbolic reasoning** with ordered evidence processing tiers
3. **Formal nine-state diagnostic state machine** with guard-conditioned transitions
4. **Contradiction-aware evidence propagation** via explicit contradiction matrix and penalty architecture
5. **Clinical Safety Gate** (3 invariants + 5 gates) enforcing escalation-only safety constraints
6. **Standalone symbolic inference** mode without statistical classifier dependency
7. **Intrinsic per-case reasoning trace** with literature citations (not post-hoc attribution)

See [`docs/research/NOVELTY_BOUNDARIES.md`](docs/research/NOVELTY_BOUNDARIES.md) for complete novelty demarcation.

---

## Dataset

**UCI Dermatology Dataset** (Ilter & Güvenir, 1998)  
Repository ID: 33 | License: CC BY 4.0  
366 patient records | 34 features | 6 disease classes

**Feature partition:**
- 12 non-invasive clinical features: used by Models B and C (standalone)
- 22 histopathological features (biopsy-derived): used by Model A only
- All 34 features: Model A upper-bound reference

**Disease distribution:**

| Disease | Patients | Class |
|---|---|---|
| Psoriasis | 112 | 1 |
| Seborrheic dermatitis | 61 | 2 |
| Lichen planus | 72 | 3 |
| Pityriasis rosea | 49 | 4 |
| Chronic dermatitis | 52 | 5 |
| Pityriasis rubra pilaris | 20 | 6 |

---

## Diagnostic Rule Base

The symbolic engine's rule base is derived from peer-reviewed dermatology literature:

- Fitzpatrick TB et al., *Dermatology in General Medicine*, 8th ed., McGraw-Hill 2012
- Habif TP, *Clinical Dermatology*, 6th ed., Elsevier 2016
- Andrews' *Diseases of the Skin*, 13th ed., Elsevier 2019
- Griffiths CEM & Barker JNWN, *Lancet* 2007;370:263–271
- Le Cleach L & Chosidow O, *NEJM* 2012;366:723–732
- Stulberg DL & Wolfrey J, *Am Fam Physician* 2004;69:87–91

Rules are stored in structured YAML (`rules/`) with: rule ID, evidence tier, activation logic, confidence weight, contradiction features, clinical rationale, and literature citation. No rule is an arbitrary heuristic.

Estimated rule base: **36–46 literature-traceable rules** across 6 disease YAML files + cross-disease discriminators + contradiction matrix.

---

## Documentation Index

### Architecture

| Document | Contents |
|---|---|
| [`docs/architecture/SYSTEM_ARCHITECTURE.md`](docs/architecture/SYSTEM_ARCHITECTURE.md) | Master system specification; all subsystems; data flow |
| [`docs/architecture/DIAGNOSTIC_STATE_MODEL.md`](docs/architecture/DIAGNOSTIC_STATE_MODEL.md) | Formal 9-state machine; guard conditions; triage mapping |
| [`docs/architecture/SUBSYSTEM_INTERACTION_DIAGRAM.md`](docs/architecture/SUBSYSTEM_INTERACTION_DIAGRAM.md) | ASCII flow diagrams; worked examples; trace generation |
| [`docs/architecture/PROGRESSIVE_REASONING_STAGES.md`](docs/architecture/PROGRESSIVE_REASONING_STAGES.md) | Per-stage I/O contracts; rule tiers; state transitions |
| [`docs/architecture/CLINICAL_SAFETY_LAYER.md`](docs/architecture/CLINICAL_SAFETY_LAYER.md) | Safety invariants; gate specifications; escalation logic |

### Research

| Document | Contents |
|---|---|
| [`docs/research/PRIOR_ART_ANALYSIS.md`](docs/research/PRIOR_ART_ANALYSIS.md) | 5-class prior art analysis; 8-dimensional differentiation table |
| [`docs/research/NOVELTY_BOUNDARIES.md`](docs/research/NOVELTY_BOUNDARIES.md) | 7 novelty claims; technical and clinical claim statements |

---

## Project Structure

```
esd-neuro-symbolic/
├── src/
│   ├── data/                        # Clinical data infrastructure
│   ├── symbolic_engine/             # Primary contribution (standalone)
│   ├── models/                      # Reference systems A, B, C
│   ├── evaluation/                  # Performance, robustness, calibration
│   ├── explainability/              # Post-hoc analysis (Models A/B/C hybrid)
│   └── visualization/               # Publication-quality outputs
├── rules/                           # YAML diagnostic rule base
├── configs/                         # Feature registry, thresholds, hyperparams
├── outputs/                         # Figures, tables, traces, reports
├── docs/architecture/               # Architecture specifications
├── docs/research/                   # Research framing and novelty
└── tests/                           # Unit tests for symbolic subsystems
```

---

## Evaluation Framework

**Primary metrics:**
- Macro F1, per-class F1 — 10-fold stratified cross-validation
- Wilcoxon signed-rank test (Model B vs. C macro F1 over 10 folds)
- Biopsy triage correctness rate (SAFE_BIOPSY_FREE empirical accuracy)
- Diagnostic gap: F1(A) − F1(C) — how much symbolic reasoning narrows biopsy dependency

**Secondary analyses:**
- Certainty calibration (Expected Calibration Error)
- Low-resource robustness (feature masking + ordinal noise injection)
- Component ablation (rules only → + contradiction → + safety gate → full)
- Contradiction detection utility (correlation with misclassification risk)

**Success criterion:**
> Structured symbolic clinical reasoning measurably narrows the diagnostic gap between biopsy-free and biopsy-assisted differential diagnosis while preserving interpretability, transparency, and certainty-awareness.

Accuracy maximization is not the primary objective.

---

## Dependencies

```
ucimlrepo          — dataset retrieval
pandas, numpy      — data manipulation
scikit-learn       — cross-validation, metrics, calibration
xgboost            — statistical reference classifiers (Models A, B, C hybrid)
shap               — post-hoc feature attribution (statistical models only)
imodels            — RuleFit rule extraction (Model C hybrid validation)
pyyaml             — rule base loading
scipy              — Wilcoxon signed-rank test, calibration
matplotlib, seaborn — visualization
```

---

## Ethical and Clinical Constraints

- This system operates exclusively on the UCI Dermatology dataset (CC BY 4.0)
- No real patient data is used
- No synthetic data generation
- The system is a research prototype; it is not validated for clinical deployment
- All biopsy triage recommendations are research outputs, not medical advice
- The rule base is derived from published literature but has not been prospectively validated by dermatologists

---

## Status

**Current phase:** Architecture stabilization — pre-implementation  
See [`docs/architecture/SYSTEM_ARCHITECTURE.md`](docs/architecture/SYSTEM_ARCHITECTURE.md) for complete design specification before implementation begins.
