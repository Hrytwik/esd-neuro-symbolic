# Frontend Architecture Specification
## Clinical Diagnostic Reasoning Workstation

**Document type:** Subsystem Architecture Reference  
**Subsystem:** `app/frontend/`  
**Role:** Interactive clinical reasoning environment; core platform subsystem — not a visualization layer

---

## 1. Identity and Design Mandate

The frontend is a **Clinical Diagnostic Reasoning Workstation** — a purpose-built computational environment for driving and observing the symbolic inference pipeline. Its design mandate derives from the same clinical framing as the backend:

**The interface must function as:**
- A biomedical inference environment where clinical evidence is entered and reasoning is observed
- A reasoning transparency interface where every inference step is navigable and auditable
- A computational differential diagnosis platform producing clinician-readable outputs
- A diagnostic evolution playback system enabling retrospective case analysis

**The interface must never resemble:**
- A prediction dashboard with accuracy metrics as primary display
- A generic admin template with tables and form inputs
- A demo application with a submit button and output label
- A chatbot or conversational query interface

Every design decision is governed by clinical professionalism and reasoning transparency — not feature richness or visual novelty.

---

## 2. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Framework | Next.js 14+ (App Router) | Server/client component separation; streaming; API routes |
| Language | TypeScript (strict mode) | Type safety across API contract + component interface |
| Styling | Tailwind CSS | Utility-first; consistent clinical design tokens |
| Animation | Framer Motion | State-transition animations; certainty evolution playback |
| Graph rendering | React Flow | Programmatic reasoning graph with custom nodes/edges |
| Secondary graph | Cytoscape.js | Complex network layouts for disease relationship maps |
| Charts | Recharts + D3.js | Certainty timelines, calibration curves, heatmaps |
| UI components | shadcn/ui | Accessible, unstyled-first component primitives |
| State management | Zustand | Lightweight stores per workspace domain |
| API communication | Axios + React Query | Typed REST calls; caching; background refetch |
| Real-time updates | WebSocket (native) | Live reasoning propagation during inference |
| Report export | PDF: react-pdf; JSON: native | Clinical report generation |

---

## 3. Application Layout

The workstation is organized as a **multi-panel clinical workspace** rather than a single-page dashboard. Panels are dockable and context-sensitive — they activate, dim, and animate based on the current diagnostic state.

```
╔════════════════════════════════════════════════════════════════════════════╗
║  CLINICAL DIAGNOSTIC REASONING WORKSTATION                [Case: —] [New] ║
╠═══════════════╦═══════════════════════════╦══════════════════════════════╣
║               ║                           ║                              ║
║  CLINICAL     ║   REASONING GRAPH         ║   BIOPSY TRIAGE WORKSPACE    ║
║  INPUT        ║   (live propagation)      ║                              ║
║  WORKSPACE    ║                           ║   ┌──────────────────────┐   ║
║               ║                           ║   │ SAFE_BIOPSY_FREE     │   ║
║  [12 feature  ║   [React Flow graph       ║   │ ████████████ 84.7%   │   ║
║   entry       ║    with animated          ║   └──────────────────────┘   ║
║   panels]     ║    propagation]           ║                              ║
║               ║                           ║   Certainty Gap:  78.6%      ║
║               ║                           ║   Contradiction:  0.00       ║
║               ║                           ║   Entropy:        0.73 bits  ║
╠═══════════════╩═══════════════════════════╩══════════════════════════════╣
║                                                                           ║
║  CERTAINTY EVOLUTION TIMELINE        │  SYMBOLIC RULE ACTIVATION          ║
║  ────────────────────────────────    │  ─────────────────────────────     ║
║  [multi-disease certainty chart]     │  [rule firing log, per stage]      ║
║                                      │                                    ║
╠══════════════════════════════════════╪════════════════════════════════════╣
║                                      │                                    ║
║  DIFFERENTIAL DIAGNOSIS PANEL        │  CONTRADICTION EMERGENCE VIEWER    ║
║  ─────────────────────────────       │  ──────────────────────────────    ║
║  1. Psoriasis        84.7%  ██████   │  [contradiction events, rationale] ║
║  2. Lichen Planus     6.1%  █         │                                    ║
║  3. Seborrheic D.     4.2%           │                                    ║
╠══════════════════════════════════════╩════════════════════════════════════╣
║  REASONING TRACE NAVIGATOR  │  CLINICAL SAFETY MONITOR  │  [EXPORT] [REPLAY] ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 4. Workspace Panel Specifications

### 4.1 Clinical Input Workspace

**Purpose:** Structured evidence entry — the starting point of every diagnostic session.

**Component:** `ClinicalInputWorkspace.tsx`  
**State:** Active during `EVIDENCE_ENTRY` interaction state; dimmed during `REPLAY_MODE`

**Layout:**
- 12 feature entry controls, organized by feature type:
  - **Ordinal features (0–3):** Segmented control with clinical labels (Absent / Mild / Moderate / Severe)
  - **Binary features (0/1):** Toggle with clinical descriptors (Not present / Present)
  - **Continuous (age):** Numeric input with range validation
- Feature completeness indicator (progress bar, 0/12 → 12/12)
- Critical feature warning badges (when pathognomonic features are left unset)
- "Begin Reasoning" button — triggers symbolic engine API call
- Feature group labels: Morphological Signs | Distribution Pattern | Historical Factors

**Behavior:**
- As features are entered, the reasoning graph animates preliminary node states
- Critical features (koebner_phenomenon, polygonal_papules, follicular_papules, oral_mucosal_involvement) are visually distinguished
- Partial input triggers `PARTIAL_REASONING` state (engine runs up to available evidence)

---

### 4.2 Reasoning Graph Panel

**Purpose:** Live visualization of the symbolic inference graph — the central epistemic display.

**Component:** `ReasoningGraph.tsx` (React Flow)  
**State:** Active from `PARTIAL_REASONING` through `REPORT_GENERATION`

**Graph structure:**
```
[Feature Nodes]  →  [Rule Nodes]  →  [Hypothesis Nodes]  →  [Triage Node]
                          ↑
                  [Contradiction Nodes]
                          ↓
                  [Safety State Node]
```

**Node visual encodings:**
- **Feature nodes:** Clinical blue (#1A3A5C); fill opacity proportional to feature grade
- **Rule nodes (dormant):** Gray (#6B7280); small
- **Rule nodes (activated):** Clinical teal (#2A7A6F); enlarged; pulsing on activation
- **Rule nodes (suppressed by contradiction):** Clinical red (#8B1A1A); strikethrough styling
- **Hypothesis nodes:** Size proportional to certainty score; color shifts from gray → teal → green as certainty rises
- **Contradiction nodes:** Emerge dynamically in red; connected by dashed penalty edges
- **Safety gate node:** Amber ring if any gate flagged; green if all clear
- **Triage node:** Large; color maps to triage recommendation (green/amber/slate/red)

**Animation behavior:**
- Rule activation: node expands and brightens over 300ms
- Edge activation: directional pulse (Framer Motion path animation) from source to target
- Contradiction emergence: red node fades in; penalty edges animate leftward (toward hypothesis, dampening it)
- Certainty stabilization: hypothesis node settles to final size; subtle glow

**Interactivity:**
- Click any rule node: opens Rule Detail Drawer (rule_id, rationale, literature source, activation score)
- Click any contradiction node: opens Contradiction Detail Drawer
- Click hypothesis node: shows disease certainty detail
- Hover edges: tooltip with edge weight, propagation type

---

### 4.3 Symbolic Rule Activation Workspace

**Purpose:** Detailed log of every rule evaluated across all six reasoning stages.

**Component:** `SymbolicRuleWorkspace.tsx`  
**State:** Active from Stage 1 onward; expandable

**Layout:**
- Stage tabs: Stage 0 | Stage 1 (Tier A) | Stage 2 (Tier B) | Stage 3 (Contradiction) | Stage 4 (Tier D) | Stage 5 (Safety) | Stage 6 (Triage)
- Per-stage rule table:
  - Rule ID | Disease Target | Evidence Tier | Activation Score | Weight | Status
  - Status: ACTIVATED (teal) | DORMANT (gray) | SUPPRESSED (red)
- Literature reference badge (hover for full citation)
- Stage summary metrics (rules activated, leading disease after stage, certainty estimate)

---

### 4.4 Contradiction Emergence Viewer

**Purpose:** Real-time display of all detected contradictions with clinical rationale.

**Component:** `ContradictionViewer.tsx`  
**State:** Activates on `CONTRADICTION_EMERGENCE`; pulses on new contradictions

**Layout:**
- Empty state: "No contradictions detected" with subtle green indicator
- Active state: chronological list of contradiction events
- Per-contradiction card:
  - Trigger feature (value observed)
  - Affected hypothesis (disease penalized)
  - Competing disease (disease suggested by the feature)
  - Penalty applied (percentage)
  - Clinical rationale (one sentence)
  - Literature citation
  - Certainty before/after penalty (delta display)
- Contradiction load meter (0.0 → 0.4 threshold line → 1.0)
- Safety threshold warning when load approaches 0.40

---

### 4.5 Certainty Evolution Timeline

**Purpose:** Temporal display of disease certainty across all six reasoning stages.

**Component:** `CertaintyTimeline.tsx` (Recharts LineChart)  
**State:** Updates after each stage completes

**Chart specification:**
- X-axis: Reasoning stages (0 → 6), labeled with stage names
- Y-axis: Certainty score (0.0 → 1.0)
- One line per disease (6 lines, disease-coded colors)
- Threshold lines: SAFE (0.82, green dashed), MODERATE (0.65, amber dashed), AMBIGUOUS (0.45, slate dashed)
- Annotation points: contradiction events (red markers on affected disease line at Stage 3)
- Shaded regions: certainty gap between top-2 diseases

**Interactivity:**
- Hover: tooltip with exact certainty values at that stage
- Click stage label: snap reasoning graph to that stage's state (feeds into replay system)

---

### 4.6 Biopsy Triage Workspace

**Purpose:** Primary output display — the clinical decision output of the system.

**Component:** `BiopsyTriageWorkspace.tsx`  
**State:** Updates to final state at Stage 6; prominent throughout

**Layout:**
- **Triage recommendation badge** (large, color-coded):
  - SAFE_BIOPSY_FREE: clinical green (#2D6A4F) background
  - MODERATE_CERTAINTY: clinical amber (#D4A017) background
  - AMBIGUOUS_CASE: slate (#6B7280) background
  - BIOPSY_ADVISED: clinical red (#8B1A1A) background
- Leading disease + certainty percentage
- Key metrics row: max_certainty | certainty_gap | contradiction_load | ambiguity_index
- Safety gate summary: all passed (green) or flags raised (amber/red list)
- Triage rationale (one paragraph, auto-generated from reasoning trace)
- Diagnostic state badge (current FSM state)

---

### 4.7 Differential Diagnosis Panel

**Purpose:** Ranked list of all six disease hypotheses with certainty visualization.

**Component:** `DifferentialDiagnosisPanel.tsx`  
**State:** Updates after each stage

**Layout:**
- Ranked list (1–6) of diseases with:
  - Disease name | Certainty bar (proportional fill) | Certainty percentage
  - Certainty bar color: green (leading) → amber (competitive) → gray (unlikely)
- Certainty gap annotation between rank 1 and rank 2
- Disease evidence count (how many rules active for each disease)
- Confusion zone indicator (badge on rank 1-2 pair if they form a known confusion pair)

---

### 4.8 Clinical Safety Monitoring Panel

**Purpose:** Real-time safety gate status display throughout the inference process.

**Component:** `SafetyMonitorPanel.tsx`  
**State:** Always visible; updates at Stage 5

**Layout:**
- Three invariant status indicators:
  - Contradiction Safety Ceiling | Evidence Sufficiency Floor | Entropy Escalation Ceiling
  - Status: CLEAR (green) | APPROACHING (amber) | TRIGGERED (red)
- Five gate status indicators:
  - Single-Source Dominance | Pathognomonic Absence | Critical Missingness | Confusion Zone | Overconfidence
  - Status: CLEAR | FLAGGED
- Active safety flags list (if any triggered)
- Escalation-only property indicator (permanent reminder)
- Pre/post-gate recommendation comparison (if gates modified the recommendation)

---

### 4.9 Reasoning Trace Navigator

**Purpose:** Interactive exploration of the per-case JSON reasoning trace.

**Component:** `ReasoningTraceNavigator.tsx`  
**State:** Active from Stage 1 onward; fully navigable after Stage 6

**Layout:**
- Vertical timeline with one entry per stage
- Per-stage expandable section: module name | key actions | state transition | trace excerpt
- JSON raw view toggle (formatted with syntax highlighting)
- Search/filter across trace entries
- "Explain this entry" inline expansion (structured annotation of each trace field)
- Export controls: Export JSON | Copy to clipboard | Include in report

---

### 4.10 Case Replay System

**Purpose:** Playback interface for the diagnostic trajectory replay engine.

**Component:** `CaseReplaySystem.tsx`  
**State:** `REPLAY_MODE`

**Layout:**
- Timeline scrubber (0 → 6 stages, snappable)
- Play / Pause / Step Forward / Step Backward controls
- Playback speed control (0.5× | 1× | 2×)
- Stage annotation panel (what happened in this stage)
- Graph synchronization (reasoning graph updates to show state at scrubbed stage)
- Snapshot comparison: show two stages side by side
- Case comparison: load second case for parallel trajectory display

---

### 4.11 Publication Export System

**Purpose:** Export clinical reports, reasoning traces, and visualizations for research dissemination.

**Component:** `PublicationExportSystem.tsx`  
**State:** Available after Stage 6 completes

**Export types:**
- **Clinical Summary PDF:** One-page formatted report
- **Detailed Reasoning Report PDF:** Full multi-page clinical report
- **JSON Reasoning Trace:** Complete audit log
- **Visualization Bundle:** PNG exports of all panels at publication resolution (300 DPI)
- **Reproducibility Package:** Feature values + configuration + trace (reproducible run)

---

### 4.12 Clinician Report Viewer

**Purpose:** In-application rendering of the clinical reasoning report.

**Component:** `ClinicalReportViewer.tsx`  
**State:** `REPORT_GENERATION`

**Layout:** Formatted clinical report with sections matching `CLINICAL_REPORTING_SYSTEM.md` specification. Tabbed view: Summary | Full Report | Appendix (visuals) | Raw Data.

---

## 5. Application Routes

```
/                           → WorkbenchLayout (new case entry)
/case/[id]                  → DiagnosticWorkspace (active case)
/case/[id]/replay           → ReplayWorkspace
/case/[id]/report           → ReportViewer
/cases                      → CaseLibrary (historical cases)
/about                      → Platform Information
```

---

## 6. Backend API Interface

All communication with the FastAPI backend follows this contract:

```typescript
// POST /api/inference/run
type InferenceRequest = {
  case_id: string
  features: ClinicalFeatureVector
}

type InferenceResponse = {
  case_id: string
  biopsy_triage: TriageRecommendation
  leading_diagnosis: string
  disease_certainty: Record<Disease, number>
  max_certainty: number
  certainty_gap: number
  ambiguity_index: number
  contradiction_load: number
  diagnostic_state: DiagnosticState
  safety_flags: string[]
  reasoning_trace: ReasoningTrace
  graph_snapshots: GraphSnapshot[]   // one per stage — for replay
}

// WebSocket /ws/inference/stream
// Streams stage-by-stage updates during inference
type StageUpdate = {
  stage: number
  state: DiagnosticState
  partial_scores: Record<Disease, number>
  activated_rules: RuleActivation[]
  contradiction_events: ContradictionEvent[]
  graph_delta: GraphDelta
}
```

---

## 7. State Management Architecture

Each workspace domain has its own Zustand store to prevent cross-domain re-renders:

```typescript
// stores/inferenceStore.ts      — active inference state
// stores/graphStore.ts          — reasoning graph state (nodes, edges, layout)
// stores/traceStore.ts          — reasoning trace navigation state
// stores/replayStore.ts         — replay position and playback state
// stores/reportStore.ts         — report generation and export state
// stores/caseStore.ts           — case library and case metadata
```

---

## 8. Directory Structure

```
app/frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                         # Root workbench
│   │   ├── case/[id]/page.tsx
│   │   ├── case/[id]/replay/page.tsx
│   │   ├── case/[id]/report/page.tsx
│   │   └── cases/page.tsx
│   ├── components/
│   │   ├── workspaces/
│   │   │   ├── ClinicalInputWorkspace.tsx
│   │   │   ├── SymbolicRuleWorkspace.tsx
│   │   │   ├── ContradictionViewer.tsx
│   │   │   ├── CertaintyTimeline.tsx
│   │   │   ├── BiopsyTriageWorkspace.tsx
│   │   │   ├── DifferentialDiagnosisPanel.tsx
│   │   │   ├── SafetyMonitorPanel.tsx
│   │   │   ├── ReasoningTraceNavigator.tsx
│   │   │   ├── CaseReplaySystem.tsx
│   │   │   ├── PublicationExportSystem.tsx
│   │   │   └── ClinicalReportViewer.tsx
│   │   ├── graph/
│   │   │   ├── ReasoningGraph.tsx
│   │   │   ├── nodes/
│   │   │   │   ├── FeatureNode.tsx
│   │   │   │   ├── RuleNode.tsx
│   │   │   │   ├── ContradictionNode.tsx
│   │   │   │   ├── HypothesisNode.tsx
│   │   │   │   └── TriageNode.tsx
│   │   │   └── edges/
│   │   │       ├── ActivationEdge.tsx
│   │   │       └── ContradictionEdge.tsx
│   │   ├── layout/
│   │   │   ├── WorkbenchLayout.tsx
│   │   │   └── PanelGrid.tsx
│   │   └── shared/
│   │       ├── CertaintyBar.tsx
│   │       ├── TiageBadge.tsx
│   │       ├── StateBadge.tsx
│   │       └── LiteratureCitation.tsx
│   ├── lib/
│   │   ├── api/
│   │   │   ├── client.ts
│   │   │   └── inference.ts
│   │   ├── graph/
│   │   │   ├── graphBuilder.ts
│   │   │   └── graphAnimator.ts
│   │   ├── stores/
│   │   │   ├── inferenceStore.ts
│   │   │   ├── graphStore.ts
│   │   │   ├── traceStore.ts
│   │   │   ├── replayStore.ts
│   │   │   └── caseStore.ts
│   │   └── types/
│   │       ├── inference.ts
│   │       ├── graph.ts
│   │       └── clinical.ts
│   └── styles/
│       └── globals.css
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.ts
```
