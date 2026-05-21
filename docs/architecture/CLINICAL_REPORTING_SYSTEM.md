# Clinical Reporting System Specification
## ClinicalReportGenerator — Report Architecture and Output Formats

**Document type:** Subsystem Architecture Reference  
**Subsystem:** `src/explainability/clinical_reporter.py` + `app/frontend/src/components/workspaces/ClinicalReportViewer.tsx`  
**Role:** Generate structured clinical diagnostic reports for clinician review, research dissemination, and audit

---

## 1. Purpose

The Clinical Reporting System transforms the symbolic engine's reasoning trace into structured, human-readable reports. Every output serves a specific audience and purpose:

| Report Type | Primary Audience | Format | Trigger |
|---|---|---|---|
| Concise Clinical Summary | Clinician (point of care) | PDF, 1 page | After Stage 6 |
| Detailed Diagnostic Reasoning Report | Clinical reviewer, researcher | PDF, multi-page | On-demand |
| Publication Visual Appendix | Peer review, conference | PDF (figures only) | On-demand |
| Structured JSON Audit Trace | System audit, reproducibility | JSON | Auto (all cases) |
| Reproducibility Package | Research replication | ZIP archive | On-demand |

---

## 2. Report Type Specifications

### 2.1 Concise Clinical Summary (1 Page)

**Purpose:** Deliver the essential diagnostic conclusions to a clinician in a format readable in under 90 seconds.

**Sections:**

```
══════════════════════════════════════════════════════════
  DIAGNOSTIC REASONING SUMMARY
  Case ID: [id] | Date: [date] | Features Observed: [n/12]
══════════════════════════════════════════════════════════

CLINICAL PRESENTATION
─────────────────────────────────────────────────────────
  [Table: Observed features with grades]

BIOPSY TRIAGE RECOMMENDATION
─────────────────────────────────────────────────────────
  ┌─────────────────────────────────────────────────────┐
  │  SAFE BIOPSY-FREE DIAGNOSIS                         │
  │  Primary: Psoriasis (84.7% certainty)               │
  └─────────────────────────────────────────────────────┘

KEY METRICS
─────────────────────────────────────────────────────────
  Diagnostic Certainty:   84.7%
  Certainty Gap:          78.6%  (Psoriasis vs Lichen Planus)
  Ambiguity Index:         0.73 bits
  Contradiction Load:      0.00
  Diagnostic State:        CERTAINTY_STABILIZED
  Safety Gate:             All criteria passed

SUPPORTING EVIDENCE
─────────────────────────────────────────────────────────
  ✓ Koebner isomorphic response [pathognomonic, w=0.85]
  ✓ Knee/elbow extensor involvement [supportive, w=0.80]
  ✓ Scalp involvement [supportive, w=0.75]
  ✓ Family history (polygenic) [supportive, w=0.70]

CONTRADICTIONS DETECTED
─────────────────────────────────────────────────────────
  None detected.

SAFETY STATUS
─────────────────────────────────────────────────────────
  All safety invariants and gates: PASSED
  No flags raised.

NOTE: This report is a research output from a computational
clinical reasoning framework evaluated on the UCI Dermatology
dataset. It does not constitute medical advice.
══════════════════════════════════════════════════════════
```

---

### 2.2 Detailed Diagnostic Reasoning Report (Multi-Page)

**Purpose:** Full forensic account of the reasoning process — for clinical reviewers, researchers, and quality audit.

**Sections:**

**Section 1 — Case Profile**
- Case identifier, feature completeness, date/time
- Feature input table with all 12 features and their observed values/grades

**Section 2 — Reasoning Pipeline Summary**
- Table: Stage | Module | Key Actions | State Transition | Duration
- Annotated state history (from DiagnosticStateTracker)

**Section 3 — Rule Activation Detail**
- Per-stage rule activation tables:
  - Stage 1 (Tier A): Rule ID | Disease Target | Activation | Status | Literature
  - Stage 2 (Tier B): Same format
  - Stage 4 (Tier D): Same format
- Rule activation counts per disease

**Section 4 — Contradiction Analysis**
- Contradiction events table: Feature | Value | Target Disease | Competing Disease | Penalty | Rationale | Source
- Contradiction load evolution (Stage 0–6)
- Clinical interpretation of contradiction pattern

**Section 5 — Certainty Propagation**
- Disease certainty at each stage (tabular)
- Embedded certainty evolution chart (from VISUAL_REASONING_SPECIFICATION)
- Final certainty distribution (all 6 diseases)

**Section 6 — Safety Gate Evaluation**
- Full gate evaluation table: Gate ID | Condition | Value | Threshold | Status | Flag
- Pre-gate vs. post-gate recommendation comparison
- Safety flags if any (with clinical interpretation)

**Section 7 — Biopsy Triage Determination**
- Final triage recommendation with full rationale
- Mapping from Diagnostic State to recommendation
- Comparison with Model A (biopsy-assisted reference) outcome if available

**Section 8 — Differential Diagnosis Rankings**
- Final certainty distribution table (ranked 1–6)
- Per-disease evidence count and contradiction count
- Known confusion pair identification

**Section 9 — Reproducibility Metadata**
- Case ID and feature vector hash (SHA-256)
- Rule base version (YAML hash)
- Configuration parameters (threshold values used)
- Engine version
- Timestamp

---

### 2.3 Publication Visual Appendix

**Purpose:** High-resolution figures for peer review, conference posters, and publication submissions.

**Contents:**
- Figure 1: Activation propagation graph (full case)
- Figure 2: Certainty evolution timeline (annotated)
- Figure 3: Disease differential at each stage (bar progression)
- Figure 4: Safety gate evaluation summary
- Figure 5: Contradiction detail visualization (if contradictions present)
- Figure 6: Diagnostic tension map (if Stage 4 tension detected)
- All figures at 300 DPI PDF/PNG per VISUAL_REASONING_SPECIFICATION

---

### 2.4 Structured JSON Audit Trace

**Purpose:** Complete machine-readable audit log for every case; primary source of truth for reproducibility.

```json
{
  "schema_version": "1.0",
  "case_id": "UCI_001",
  "created_at": "2026-05-22T14:32:11Z",
  "feature_vector": {
    "erythema": 2, "scaling": 2, "definite_borders": 2,
    "itching": 3, "koebner_phenomenon": 1, "polygonal_papules": 0,
    "follicular_papules": 0, "oral_mucosal_involvement": 0,
    "knee_and_elbow_involvement": 1, "scalp_involvement": 1,
    "family_history": 1, "age": 35
  },
  "feature_grades": {
    "erythema": 0.67, "scaling": 0.67, "definite_borders": 0.67,
    "itching": 1.0, "koebner_phenomenon": 1.0, "polygonal_papules": 0.0,
    ...
  },
  "completeness_score": 1.0,
  "critical_features_missing": [],
  "reasoning_stages": [
    {
      "stage": 0,
      "module": "ClinicalGradingModule",
      "state": "EVIDENCE_SPARSE",
      "actions": ["feature_grading_complete"],
      "trace": {}
    },
    {
      "stage": 1,
      "module": "DiagnosticEvidenceEvaluator",
      "tier": "A",
      "state": "HYPOTHESIS_FORMING",
      "rules_evaluated": [
        {"rule_id": "PSO_001", "activation": 0.85, "weighted": 0.72, "status": "activated"},
        {"rule_id": "LP_001", "activation": 0.0, "weighted": 0.0, "status": "dormant"}
      ],
      "raw_scores": {"psoriasis": 0.72},
      "state_transition": "S0 → S1"
    }
  ],
  "contradiction_events": [],
  "safety_evaluation": {
    "invariants": {...},
    "gates": {...},
    "all_passed": true,
    "flags_raised": []
  },
  "final_result": {
    "biopsy_triage": "SAFE_BIOPSY_FREE",
    "leading_diagnosis": "psoriasis",
    "disease_certainty": {"psoriasis": 0.847, "lichen_planus": 0.061, ...},
    "max_certainty": 0.847,
    "certainty_gap": 0.786,
    "ambiguity_index": 0.73,
    "contradiction_load": 0.00,
    "diagnostic_state": "CERTAINTY_STABILIZED"
  },
  "graph_trajectory": {
    "snapshot_count": 7,
    "snapshots": ["..."]
  },
  "reproducibility": {
    "feature_hash": "sha256:a3f9...",
    "rule_base_hash": "sha256:7c2e...",
    "config_hash": "sha256:11b4...",
    "engine_version": "0.1.0",
    "python_version": "3.11.x",
    "seed": 42
  }
}
```

---

### 2.5 Reproducibility Package

**Purpose:** Enable exact replication of any reported case by another researcher.

**Contents (ZIP archive):**
```
reproducibility_UCI_001_20260522/
├── feature_vector.json          # Input feature values
├── config_snapshot.yaml         # All threshold/parameter values used
├── rule_base_snapshot/          # Copy of all YAML rules at runtime
│   ├── psoriasis.yaml
│   ├── ... (all 8 rule files)
├── reasoning_trace.json         # Full audit trace
├── report_concise.pdf
├── report_detailed.pdf
├── figures/                     # All publication figures
├── REPRODUCE.md                 # Step-by-step replication instructions
└── checksums.sha256             # File integrity verification
```

`REPRODUCE.md` includes:
1. Environment setup (Python version, package versions)
2. Data retrieval command (`ucimlrepo fetch id=33`)
3. Exact run command to reproduce this case
4. Expected output checksums

---

## 3. Report Generation Architecture

### 3.1 ClinicalReportGenerator (Python Backend)

```python
class ClinicalReportGenerator:

    def generate_concise_summary(
        self,
        case_id: str,
        reasoning_trace: ReasoningTrace,
        graph_trajectory: CaseTrajectory
    ) -> bytes:  # PDF bytes
        ...

    def generate_detailed_report(
        self,
        case_id: str,
        reasoning_trace: ReasoningTrace,
        graph_trajectory: CaseTrajectory,
        include_figures: bool = True
    ) -> bytes:  # PDF bytes
        ...

    def generate_json_trace(
        self,
        case_id: str,
        reasoning_trace: ReasoningTrace,
        graph_trajectory: CaseTrajectory
    ) -> dict:
        ...

    def generate_reproducibility_package(
        self,
        case_id: str,
        reasoning_trace: ReasoningTrace
    ) -> bytes:  # ZIP bytes
        ...
```

**PDF generation:** `reportlab` (primary) with `weasyprint` fallback for HTML-to-PDF.

**Figure inclusion:** Figures are generated first by `ReasoningPathwayVisualizer` and `src/visualization/diagnostic_plots.py`, then embedded as high-resolution PNG into the PDF.

---

### 3.2 ClinicalReportViewer (Frontend Component)

The frontend report viewer renders reports in-application without requiring PDF download:

**Tabs:**
1. **Summary** — Concise summary rendered as structured React components (not PDF embed)
2. **Full Report** — Detailed report sections rendered as collapsible React components
3. **Visualizations** — Interactive versions of all publication figures
4. **Raw Trace** — Formatted JSON with syntax highlighting and search
5. **Reproducibility** — Metadata panel with copy/download controls

**Export controls** (top-right of viewer):
- Download Summary PDF
- Download Full Report PDF
- Download JSON Trace
- Download Reproducibility Package
- Download Visualization Bundle

---

## 4. Report Metadata Standard

Every report must include a reproducibility metadata block:

```
REPORT METADATA
───────────────────────────────────────────────────────
Case ID:             UCI_001
Generated:           2026-05-22 14:32:11 UTC
Feature hash:        sha256:a3f9c2d8...
Rule base version:   rules_v0.1.0 (sha256:7c2e...)
Configuration:       configs/biopsy_thresholds.yaml v1.0
Engine version:      CASDRE v0.1.0
Dataset:             UCI Dermatology (id=33, CC BY 4.0)
Reproducibility:     See reproducibility package
───────────────────────────────────────────────────────
This report is a research output. It does not constitute
medical advice and has not been prospectively validated.
```

This block appears as a footer on every page of every report.

---

## 5. Output Directory Structure

```
outputs/
├── figures/
│   ├── fig_activation_graph_{case_id}.pdf
│   ├── fig_certainty_timeline_{case_id}.pdf
│   ├── fig_confusion_modelA_6class.pdf
│   ├── fig_confusion_modelB_6class.pdf
│   ├── fig_confusion_modelC_6class.pdf
│   ├── fig_calibration_curves_ABC.pdf
│   ├── fig_contradiction_heatmap.pdf
│   ├── fig_biopsy_dependency_analysis.pdf
│   └── fig_rule_activation_sankey.pdf
├── tables/
│   ├── cv_summary.csv               # macro F1 for A, B, C
│   ├── per_class_f1.csv             # per-disease F1
│   ├── b_vs_c_wilcoxon.txt          # Wilcoxon test result
│   ├── triage_accuracy.csv          # biopsy triage correctness
│   └── shap_top10_features.csv
├── traces/
│   └── {case_id}_trace.json         # per-case reasoning trace
├── reports/
│   ├── {case_id}_summary.pdf
│   ├── {case_id}_detailed.pdf
│   └── reproducibility_{case_id}/
└── rules/
    ├── extracted_rules.csv
    └── rulefit_validation.md
```
