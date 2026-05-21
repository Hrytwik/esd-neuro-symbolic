# Demonstration Workflow
## Conference, Publication, and Clinical Demonstration Architecture

**Document type:** Operational Reference  
**Purpose:** Define the structured demonstration flow for conferences, peer reviewers, judges, clinicians, and patent evaluators  
**Key message:** Symbolic clinical reasoning provides interpretable, safety-aware biopsy triage — not a prediction label

---

## 1. Demonstration Objectives

Every demonstration, regardless of audience, must communicate three things:

1. **What the system does:** Determines whether biopsy-free diagnosis of erythemato-squamous diseases is safe, using structured clinical reasoning
2. **How it does it:** Through transparent, stage-by-stage symbolic inference — not a black box
3. **Why it matters:** It narrows the clinical gap between "observable signs only" and "biopsy-confirmed diagnosis" while preserving interpretability and safety

**What the demonstration must never communicate:**
- "We built a classifier that achieves X% accuracy"
- "Our model predicts which disease you have"
- "See this probability score from our trained model"

---

## 2. Demonstration Cases

Four UCI Dermatology cases are pre-selected for demonstration. Each is chosen to illustrate a specific system capability:

### Demo Case A — "Clear Psoriasis" (SAFE_BIOPSY_FREE showcase)

**Clinical profile:** Koebner positive, knee/elbow involvement, scalp involvement, family history, no contradicting signs  
**Purpose:** Show the system reaching a confident, safe biopsy-free diagnosis with clean reasoning  
**Highlights:** Full rule activation, zero contradictions, high certainty gap, all safety gates passed  
**Triage output:** SAFE_BIOPSY_FREE

### Demo Case B — "LP-PSO Tension" (Contradiction showcase)

**Clinical profile:** Koebner positive (psoriasis signal) AND oral mucosal involvement (LP signal) simultaneously  
**Purpose:** Demonstrate contradiction detection and diagnostic tension  
**Highlights:** Contradiction emergence at Stage 3, certainty dampening, diagnostic tension, Stage 4 discrimination attempt, MODERATE_CERTAINTY or BIOPSY_ADVISED  
**Triage output:** BIOPSY_ADVISED (demonstrates safety gate)

### Demo Case C — "Lichen Planus via Discrimination" (Discrimination showcase)

**Clinical profile:** Polygonal papules, oral mucosal involvement, itching grade 2, Koebner present — but no psoriasis-specific features  
**Purpose:** Show multi-stage reasoning and pairwise discriminator activation  
**Highlights:** Pathognomonic LP rule fires (Stage 1), discrimination of LP vs. PR at Stage 4, SAFE_BIOPSY_FREE  
**Triage output:** SAFE_BIOPSY_FREE

### Demo Case D — "Ambiguous Presentation" (Ambiguity escalation showcase)

**Clinical profile:** Non-specific presentation — moderate erythema, moderate scaling, no distinctive signs, multiple mild features across several diseases  
**Purpose:** Demonstrate that the system appropriately escalates to BIOPSY_ADVISED when evidence is insufficient  
**Highlights:** High entropy after Stage 2, discriminators fail to resolve, safety gate (Entropy Escalation Ceiling) triggers  
**Triage output:** BIOPSY_ADVISED (ambiguity escalation path)

---

## 3. Eight-Step Demonstration Flow

The following structured demonstration flow takes approximately 12–15 minutes for a conference or live demo, and 6–8 minutes for an accelerated version.

---

### Step 1 — Clinical Context Setting (90 seconds)

**Presenter script:** "Erythemato-squamous diseases share overlapping clinical presentation. Every existing computational approach achieves high accuracy by using biopsy-derived histopathological features — features you can only access after the biopsy is already done. These systems cannot help you decide whether to perform biopsy in the first place. This system addresses that prior question."

**Display:** The differential diagnosis challenge diagram — six diseases with overlapping features circled, biopsy icon marked as "required by prior systems"

**Key point:** The problem is triage, not classification.

---

### Step 2 — Clinical Feature Entry (90 seconds)

**Action:** Enter Demo Case A feature values into the ClinicalInputWorkspace.

**Presenter script:** "We begin by observing twelve non-invasive clinical signs — everything a clinician can see and measure without laboratory equipment or tissue sampling."

**Display:** 
- ClinicalInputWorkspace: feature controls filling in one by one
- Feature completeness indicator rising: 0/12 → 12/12
- Reasoning graph begins pre-populating with feature nodes

**Key point:** No biopsy features, no histopathological data. These are bedside observations.

---

### Step 3 — Evidence Accumulation (Stage 1 + 2) (90 seconds)

**Action:** Trigger inference. Play at 0.75× speed through Stages 1 and 2.

**Presenter script:** "The reasoning engine activates diagnostic rules in tiers. First, pathognomonic signs — single features that strongly suggest specific diseases. Then, supportive clusters — co-occurring patterns that reinforce an emerging hypothesis."

**Display:**
- Reasoning graph: rule nodes illuminate in sequence
- Activation edges pulse from feature nodes → rule nodes → hypothesis nodes
- SymbolicRuleWorkspace: rule table populates (Stage 1 then Stage 2)
- CertaintyTimeline: psoriasis line rises
- DifferentialDiagnosisPanel: psoriasis takes rank 1

**Key point:** You can see exactly which rules activated and why. This is intrinsic interpretability — generated during reasoning, not explained afterward.

---

### Step 4 — Contradiction Analysis (Stage 3) (60 seconds)

**Action:** Continue through Stage 3. (For Demo Case A, no contradictions emerge — narrate the absence as significant.)

**Presenter script:** "Stage 3 checks for contradicting clinical signs — features that would challenge the leading hypothesis. The system maintains an explicit contradiction matrix derived from published dermatology literature. For this case: no contradictions. For a case where oral mucosal involvement and Koebner phenomenon co-occur, a contradiction emerges here."

**Display (Demo Case A):** ContradictionViewer: "No contradictions detected" in green  
**Optional (switch to Demo Case B):** Show contradiction node emerging — pause and inspect.

**Key point:** Contradiction detection is not statistical smoothing — it is explicit clinical reasoning about conflicting signs.

---

### Step 5 — Certainty Stabilization (Stage 4 + 5) (90 seconds)

**Action:** Continue through Stages 4 and 5.

**Presenter script:** "Stage 4 applies pairwise discriminators for the highest-confusion disease pairs. Stage 5 computes final certainty metrics and runs the clinical safety gate — a formal battery of safety checks that can only escalate toward biopsy-advised, never the other direction."

**Display:**
- CertaintyTimeline: psoriasis certainty rises above SAFE threshold line (0.82)
- SafetyMonitorPanel: all gates showing green
- BiopsyTriageWorkspace: transitioning to SAFE_BIOPSY_FREE (green badge)

**Key point:** Safety is architectural, not optional. No certainty score, however high, overrides a safety gate trigger.

---

### Step 6 — Biopsy Triage Output (60 seconds)

**Action:** Stage 6 completes. Pause on the final state.

**Presenter script:** "The system's primary output is not a disease label with a confidence score. It is a biopsy triage recommendation: SAFE BIOPSY-FREE DIAGNOSIS. The system additionally reports certainty metrics, the diagnostic state, and the complete reasoning trace."

**Display:**
- BiopsyTriageWorkspace: SAFE_BIOPSY_FREE in clinical green, psoriasis 84.7%
- Certainty gap: 78.6%
- Safety gate: all passed
- Reasoning trace: first stage expanded

**Key point:** The clinical question — "should we biopsy?" — is answered, not just "what disease is this?"

---

### Step 7 — Reasoning Replay (120 seconds)

**Action:** Switch to CaseReplaySystem. Play case from Stage 0 at 1.5× speed.

**Presenter script:** "Every diagnostic reasoning trajectory is recorded — every rule that fired, every contradiction that emerged, every certainty update. Clinicians, reviewers, or students can replay any case and inspect the reasoning at any stage."

**Display:**
- Replay from Stage 0: graph rebuilds step by step
- Pause at Stage 1: show pathognomonic rule activation
- Scrub to Stage 3: show (or show absence of) contradictions
- Scrub to Stage 5: show safety gate evaluation

**Key point:** Complete auditability. No black box. The reasoning trace is the explanation.

---

### Step 8 — Clinician Report Generation (60 seconds)

**Action:** Click "Generate Report". Show ClinicalReportViewer.

**Presenter script:** "Finally, the system produces a structured clinical report — a concise summary for the point of care, a detailed reasoning report for review, and a reproducibility package for research replication. Every report includes the activated rules, contradictions detected, certainty evolution, and biopsy triage rationale."

**Display:**
- ClinicalReportViewer: Summary tab showing formatted report
- Export controls: PDF / JSON / Reproducibility Package
- Page 1 of detailed report: reasoning trace section

**Key point:** Publication-grade outputs. Reproducible. Auditable. Clinician-readable.

---

## 4. Audience-Specific Variations

### 4.1 Conference (12–15 minutes, live demo)

Full eight-step flow. Emphasize Steps 3 (contradiction), 5 (safety gate), and 8 (reporting). Use Demo Case B (contradiction showcase) as a second case to demonstrate safety escalation. End with side-by-side comparison of Model B (baseline) vs. Model C (symbolic) macro F1 and triage accuracy.

### 4.2 Peer Review Submission

No live demo. Use static figures from outputs/figures/:
- Fig 1: Activation propagation graph (Demo Case A)
- Fig 2: Certainty evolution timeline (Demo Case A)
- Fig 3: Contradiction emergence visualization (Demo Case B)
- Fig 4: Disease confusion heatmaps (Models A, B, C)
- Fig 5: Certainty calibration curves (A, B, C)
- Fig 6: Biopsy triage accuracy per disease

Supplementary: Demo Case A and B full reasoning traces (JSON).

### 4.3 Clinical Audience

Emphasize Step 6 (biopsy triage output) and Step 7 (replay). Use plain-language framing. Show the Concise Clinical Summary report (1 page). Emphasize: "This is a research prototype — it shows what structured clinical reasoning can tell you from bedside observations alone, and when it can't safely conclude."

### 4.4 Patent Evaluators

Emphasize novelty boundaries throughout:
- Step 3: "No prior system maintains an explicit contradiction matrix and detects contradiction as a distinct diagnostic phase" [N4]
- Step 5: "The safety gate is escalation-only — an architectural property that cannot be configured away" [N5]
- Step 6: "Biopsy triage as primary output does not exist in any prior clinical decision support system for this disease group" [N1]
- Step 7: "Replayable diagnostic trajectories with stage-by-stage state reconstruction" [N6, N7]

### 4.5 Judges / Competition

Six-minute version: Steps 2, 3, 6, 7 only. Lead with the clinical problem (30 sec), show Demo Case A (2 min), Demo Case B contradiction (2 min), finish with triage output and report (90 sec). Close with one slide: Model A vs. B vs. C comparison.

---

## 5. Demo Environment Setup

### 5.1 Pre-Demo Checklist

```
□ Four demo cases pre-loaded in CaseLibrary
□ FastAPI backend running: http://localhost:8000
□ Frontend running: http://localhost:3000
□ WebSocket connection verified
□ outputs/figures/ populated (pre-generated)
□ Screen resolution: 1920×1080 minimum
□ Browser: Chrome (React Flow performs best)
□ Demo cases tested: confirm all produce expected triage outputs
□ Offline mode ready (local dataset, no internet required)
```

### 5.2 Demo Data Preparation Script

```bash
python main.py --mode demo --cases A,B,C,D --precompute
```

This pre-runs all four demo cases, stores trajectories, and generates all figures. The live demo then loads from stored trajectories (no live computation delay).

### 5.3 Fallback Plan

If live system unavailable: a pre-recorded screen capture of the full demonstration workflow is stored in `docs/demo/demo_recording.mp4`. Static figures and the detailed reasoning report for Demo Case A and B are in `docs/demo/`.

---

## 6. Key Messages Reference

| Audience question | Correct response |
|---|---|
| "What accuracy does it achieve?" | "Accuracy is not the primary metric. The primary metric is biopsy triage correctness — when the system says SAFE_BIOPSY_FREE, how often is the diagnosis correct? That's what matters clinically." |
| "How does it compare to prior methods?" | "Prior methods produce disease labels. This system produces a biopsy recommendation. They address different clinical questions." |
| "Can you explain why it made this prediction?" | "Every rule that activated is visible in the trace, with its literature source. Every contradiction is recorded. There is no prediction — there is reasoning." |
| "What happens with ambiguous cases?" | "The system escalates to BIOPSY_ADVISED. The safety layer is escalation-only — it cannot make the system more confident, only more cautious." |
| "Is this validated on real patients?" | "This is a research prototype evaluated on the UCI Dermatology dataset (366 patients, CC BY 4.0). Prospective clinical validation is future work." |
