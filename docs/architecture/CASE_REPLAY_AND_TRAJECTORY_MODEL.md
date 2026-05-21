# Case Replay and Trajectory Model
## Diagnostic Flight Recorder — Architecture Specification

**Document type:** Subsystem Architecture Reference  
**Subsystem:** `src/symbolic_engine/` (trajectory recording) + `app/frontend/src/components/workspaces/CaseReplaySystem.tsx`  
**Role:** Persistent, replayable record of every diagnostic reasoning trajectory — enabling stepwise case review, counterfactual analysis, and clinical education

---

## 1. Conceptual Identity

The case trajectory system functions as a **diagnostic flight recorder** — analogous to an aviation flight data recorder, it captures the complete internal state of the reasoning engine at every decision point throughout the diagnostic process.

A flight recorder serves two purposes:
1. **Postmortem analysis** — understand exactly what happened and why
2. **Training** — educate on what correct (and incorrect) reasoning looks like

The diagnostic trajectory system serves the same purposes for clinical reasoning:
1. **Case review** — auditors, clinicians, and researchers can replay any case and inspect every reasoning decision
2. **Clinical education** — trainees can step through the reasoning process, observe contradiction emergence, and understand why biopsy triage recommendations are generated

---

## 2. Trajectory Data Model

A complete case trajectory consists of seven ordered graph snapshots (one per reasoning stage) plus a final output record.

### 2.1 CaseTrajectory

```python
@dataclass
class CaseTrajectory:
    # Identity
    trajectory_id: str              # UUID
    case_id: str                    # e.g., "UCI_001" or user-assigned
    created_at: str                 # ISO 8601 timestamp
    
    # Input record
    feature_vector: dict            # raw input feature values
    feature_grades: dict            # graded values after Stage 0
    completeness_score: float
    critical_features_missing: list[str]
    
    # Trajectory
    snapshots: list[GraphSnapshot]  # ordered: stage 0 → 6
    
    # Summary record (derived from final snapshot)
    final_state: str                # DiagnosticState
    biopsy_triage: str              # TriageRecommendation
    leading_diagnosis: str
    disease_certainty: dict
    max_certainty: float
    contradiction_load: float
    ambiguity_index: float
    safety_flags: list[str]
    
    # Metadata
    rule_base_version: str          # YAML rule hash
    config_version: str             # threshold config hash
    engine_version: str
    feature_vector_hash: str        # SHA-256 for reproducibility
    
    # Counterfactual data (populated on demand)
    counterfactual_analyses: list[CounterfactualAnalysis] = field(default_factory=list)
```

### 2.2 GraphSnapshot (per-stage)

```python
@dataclass
class GraphSnapshot:
    snapshot_id: str                # "{trajectory_id}_stage_{n}"
    trajectory_id: str
    stage: int                      # 0–6
    diagnostic_state: str           # DiagnosticState at this stage
    
    # Graph state
    nodes: list[dict]               # serialized node states
    edges: list[dict]               # serialized edge states
    
    # Aggregate state at this stage
    partial_scores: dict            # {disease: float}
    activated_rule_count: int
    activated_rules: list[str]      # rule_ids active at this stage
    contradiction_count: int
    contradiction_events: list[dict]
    contradiction_load: float       # cumulative at this stage
    
    # State machine
    state_transition: str           # "S2 → S3" or "S2 → S2 (no change)"
    guard_condition_met: str        # which guard triggered this transition
    
    # Timing
    stage_duration_ms: float
    timestamp: str
```

---

## 3. Trajectory Recording Architecture

Trajectories are recorded by the `ReasoningGraphEngine` as a side effect of normal inference. No additional instrumentation is required — the graph snapshot is taken automatically after each stage completes.

```python
class ReasoningGraphEngine:

    def propagate_stage(self, stage: int) -> GraphSnapshot:
        # ... reasoning logic ...
        snapshot = self._take_snapshot(stage)
        self.trajectory.snapshots.append(snapshot)
        return snapshot  # also returned for real-time WebSocket streaming
    
    def get_trajectory(self) -> CaseTrajectory:
        # Returns the complete trajectory after Stage 6 completes
        return self.trajectory
```

### 3.1 Trajectory Storage

Trajectories are stored in two locations:

**Backend (persistent):**  
`outputs/traces/{case_id}_trajectory.json` — full trajectory serialized to JSON

**Frontend (session):**  
`replayStore` (Zustand) — trajectory loaded into memory for replay; cleared on new case

**API endpoint:**  
`GET /api/trajectories/{case_id}` — retrieves stored trajectory for replay loading

---

## 4. Replay Engine Architecture

### 4.1 TrajectoryReplayer (Frontend)

```typescript
class TrajectoryReplayer {
  private trajectory: CaseTrajectory
  private currentStage: number = 0
  private isPlaying: boolean = false
  private playbackSpeed: number = 1.0  // 0.5x | 1x | 2x
  private intervalMs: number = 1200   // time between stage advances at 1x speed

  constructor(trajectory: CaseTrajectory) {
    this.trajectory = trajectory
  }

  // Playback controls
  play(): void
  pause(): void
  stepForward(): void
  stepBackward(): void
  scrubToStage(stage: number): void
  setPlaybackSpeed(speed: 0.5 | 1 | 2): void

  // State access
  getCurrentSnapshot(): GraphSnapshot
  getSnapshotAtStage(stage: number): GraphSnapshot

  // Events emitted (consumed by stores)
  onStageChange: (stage: number, snapshot: GraphSnapshot) => void
}
```

### 4.2 TemporalGraphReconstructor (Frontend)

When the scrubber jumps to stage N, the reasoning graph must be reconstructed to match that stage's state:

```typescript
function reconstructGraphAtStage(
  trajectory: CaseTrajectory,
  targetStage: number
): ReactFlowGraph {
  const snapshot = trajectory.snapshots[targetStage]
  return {
    nodes: snapshot.nodes.map(applyVisualEncoding),
    edges: snapshot.edges.map(applyEdgeStyling)
  }
}
```

This is applied directly to the `graphStore`, causing React Flow to re-render to the historical state.

### 4.3 Replay Synchronization Protocol

When the scrubber moves to stage N, all panels synchronize:

| Panel | Synchronized to stage N |
|---|---|
| ReasoningGraph | `reconstructGraphAtStage(trajectory, N)` |
| CertaintyTimeline | Scrubber position highlighted at stage N; lines up to N shown |
| SymbolicRuleWorkspace | Shows `snapshot.activated_rules` at stage N |
| ContradictionViewer | Shows `snapshot.contradiction_events` up to stage N |
| DifferentialDiagnosisPanel | Shows `snapshot.partial_scores` at stage N |
| SafetyMonitorPanel | Shows safety state at stage N (N=5/6 only) |
| BiopsyTriageWorkspace | Shows provisional triage at stage N |

---

## 5. Replay Interface Features

### 5.1 Timeline Scrubber

```
Stage: [0] [1] [2] [3] [4] [5] [6]
        ●───────────────────────────
       Feature  Path.  Supp.  Contra.  Disc.  Stab.  Triage
       Grading  (A)    (B)    (det.)   (D)    +Gate  Decision

        ◀◀  ◀  ▶/⏸  ▶▶    Speed: [0.5x] [1x] [2x]
```

- Stages are labeled with clinical names (not "Stage N")
- Active stage is highlighted
- Contradiction stages are marked with red indicator on timeline
- Safety gate stage (5) is marked with shield icon

### 5.2 Stage Inspection Panel

When the scrubber is paused at a stage, the Stage Inspection Panel shows:

```
STAGE 3 — Contradiction Analysis
────────────────────────────────────────────
Module: DiagnosticConflictAnalyzer
State Transition: S2 → S3 (CONTRADICTION_EMERGED)
Guard Condition: oral_mucosal_involvement = 1 (penalty > 0.20)

Contradiction detected:
  Feature: oral_mucosal_involvement = 1
  Affects: Psoriasis (−30% certainty)
  Suggests: Lichen Planus
  Rationale: Oral mucosal involvement (Wickham's striae) is
             pathognomonic for LP; inconsistent with isolated psoriasis
  Source: Le Cleach & Chosidow, NEJM 2012;366:723–732

Certainty at this stage:
  Psoriasis: 48.2% (was 68.1% before penalty)
  Lichen Planus: 22.0% (rising)

Stage duration: 3.2ms
────────────────────────────────────────────
```

### 5.3 Snapshot Comparison (Two-Stage View)

Users can split the reasoning graph panel into two panes showing two different stages simultaneously:

```
[Stage 2: REINFORCING]          [Stage 3: CONTRADICTION_EMERGED]
   PSO certainty: 68.1%            PSO certainty: 48.2%
   No contradictions               oral_mucosal contradiction active
```

This comparison mode is particularly valuable for educational purposes — showing exactly how a contradiction event changes the reasoning state.

---

## 6. Counterfactual Analysis

The trajectory system supports **counterfactual analysis** — answering "what if this feature had been different?"

### 6.1 CounterfactualAnalysis Data Model

```python
@dataclass
class CounterfactualAnalysis:
    analysis_id: str
    base_trajectory_id: str
    
    # Counterfactual modification
    feature_modified: str           # e.g., "oral_mucosal_involvement"
    original_value: float
    counterfactual_value: float
    
    # Counterfactual trajectory
    counterfactual_trajectory: CaseTrajectory
    
    # Comparison
    original_triage: str
    counterfactual_triage: str
    triage_changed: bool
    
    certainty_delta: dict           # {disease: delta} change in certainty
    state_delta: str                # e.g., "CONTRADICTION_EMERGED → CERTAINTY_STABILIZED"
    stages_affected: list[int]      # which stages changed
```

### 6.2 Counterfactual Use Cases

**Use case 1 — Teaching:** Show a student what would have happened if oral_mucosal_involvement had been absent: "Without the contradicting sign, the psoriasis diagnosis would have reached SAFE_BIOPSY_FREE."

**Use case 2 — Sensitivity analysis:** For a published case, evaluate how robust the triage recommendation is to single-feature changes.

**Use case 3 — Clinical reasoning education:** Present a case where one feature makes the difference between SAFE and BIOPSY_ADVISED — illustrating the clinical significance of that feature.

---

## 7. Trajectory Comparison (Multi-Case)

The replay system supports side-by-side comparison of two different cases:

```
CASE A (UCI_001)               CASE B (UCI_047)
──────────────────             ──────────────────
Triage: SAFE_BIOPSY_FREE       Triage: BIOPSY_ADVISED
PSO: 84.7%                     LP: 38.2%, PSO: 33.1%
Contradictions: 0              Contradictions: 2
Entropy: 0.73 bits             Entropy: 1.67 bits

[Aligned timeline scrubber — both cases advance together]
```

This enables researchers to understand why clinically similar presentations can lead to different triage outcomes.

---

## 8. Storage and API

### 8.1 Server-Side Storage

```
outputs/
└── traces/
    ├── {case_id}_trajectory.json    # Full trajectory
    └── {case_id}_snapshots/
        ├── stage_0.json
        ├── stage_1.json
        ├── stage_2.json
        ├── stage_3.json
        ├── stage_4.json
        ├── stage_5.json
        └── stage_6.json
```

### 8.2 API Endpoints

```
GET  /api/trajectories                           → list all stored trajectories
GET  /api/trajectories/{case_id}                 → full trajectory JSON
GET  /api/trajectories/{case_id}/snapshots/{n}   → single stage snapshot
POST /api/trajectories/{case_id}/counterfactual  → run counterfactual analysis
GET  /api/trajectories/{case_id}/compare/{case_id2} → two-case comparison
```

---

## 9. Educational Use of Trajectories

The trajectory system is designed to support three educational modes in the frontend:

**Mode 1 — Guided Replay:** The system plays a case trajectory with annotated explanations at each stage. Suitable for clinical education.

**Mode 2 — Inspection Mode:** User can pause and inspect any stage in detail. Suitable for clinical reviewers.

**Mode 3 — Comparative Analysis:** Two cases shown side by side. Suitable for researchers studying diagnostic difficulty.

These modes are accessible from the CaseLibrary via the "Educational Replay" button on any stored case.
