# Interaction State Model
## Frontend Diagnostic Interaction State Machine

**Document type:** Subsystem Architecture Reference  
**Subsystem:** Frontend — `lib/stores/inferenceStore.ts`, `components/layout/WorkbenchLayout.tsx`  
**Role:** Governs which workspaces are active, what actions are available, and how panels animate in response to the diagnostic state machine's output

---

## 1. Overview

The frontend maintains its own **Interaction State Machine** — a nine-state automaton that mirrors the backend's Diagnostic State Machine but governs UI behavior rather than clinical reasoning. The two state machines are synchronized: the frontend interaction state updates when the backend emits a `StageUpdate` event via WebSocket.

The distinction is important:
- **Backend Diagnostic State Machine** (in `DiagnosticStateTracker`) governs clinical reasoning logic
- **Frontend Interaction State Machine** governs visual behavior, panel activation, and available user actions

Both machines have 9 states with parallel semantics, but the frontend machine adds states for UI-specific concerns (REPORT_GENERATION, REPLAY_MODE) that the backend does not need.

---

## 2. State Definitions

### IS-0: IDLE

**Description:** No case is loaded. The workstation is ready to receive clinical feature input.

**Active panels:**
- ClinicalInputWorkspace (prominent, centered)
- CaseLibrary access (sidebar)

**Dimmed/hidden panels:**
- ReasoningGraph (shows empty state placeholder)
- All inference output panels

**Available actions:**
- Enter feature values
- Load historical case from library
- Enter REPLAY_MODE for a historical case

**Graph state:** Empty — no nodes rendered

**Transition triggers:**
- First feature value entered → IS-1 (EVIDENCE_ENTRY)
- Historical case loaded → IS-8 (REPLAY_MODE)

---

### IS-1: EVIDENCE_ENTRY

**Description:** Clinician is entering clinical feature observations. No inference has been triggered yet.

**Active panels:**
- ClinicalInputWorkspace (full attention)
- ReasoningGraph (pre-populated with feature nodes in waiting state)
- SymbolicRuleWorkspace (shows rule list in dormant state)

**Graph state:**
- Feature nodes rendered (dim, unfilled)
- Rule nodes rendered (dim, dormant)
- No edges animated yet

**Available actions:**
- Enter/modify feature values
- Trigger inference ("Begin Reasoning")
- Clear all features

**Transition triggers:**
- "Begin Reasoning" clicked → IS-2 (PARTIAL_REASONING)
- All features cleared → IS-0 (IDLE)

**UI behavior:**
- Feature completeness progress bar animates as features are entered
- Critical feature warning badges appear for unset pathognomonic features
- "Begin Reasoning" button activates once completeness_score >= 0.30

---

### IS-2: PARTIAL_REASONING

**Description:** Inference engine is running. Stage-by-stage updates are arriving via WebSocket. The workstation is in active diagnostic computation mode.

**Active panels:**
- ReasoningGraph (active — nodes activating in real time)
- SymbolicRuleWorkspace (rules firing per stage)
- CertaintyTimeline (updating)
- DifferentialDiagnosisPanel (updating)
- BiopsyTriageWorkspace (showing provisional state)
- SafetyMonitorPanel (showing EVALUATING state)

**Graph state:**
- Feature nodes illuminated proportional to grade
- Rule nodes activating as stages progress
- Activation edges pulsing directionally

**Available actions:**
- Observe reasoning (no feature editing)
- Pause/inspect (pause-on-stage mode — Stage inspection mode)
- Abort inference (returns to IS-1)

**Transition triggers (backend-driven):**
- Stage 3 produces contradiction events → IS-3 (CONTRADICTION_EMERGENCE)
- Stage 4 produces high entropy (ambiguity_index > 1.3) → IS-4 (AMBIGUITY_ESCALATION)
- Stage 5 produces certainty_gap >= 0.20 → IS-5 (CERTAINTY_STABILIZATION)
- Safety gate triggered → IS-6 (BIOPSY_ESCALATION)
- Inference completes → appropriate terminal state

**Animation:** Loading indicator in triage workspace ("Reasoning in progress...")

---

### IS-3: CONTRADICTION_EMERGENCE

**Description:** One or more contradiction events have been detected during Stage 3. The workstation highlights the contradiction visually and alerts the clinician.

**Active panels:**
- All IS-2 panels remain active
- ContradictionViewer (newly activated; pulsed animation on entry)
- ReasoningGraph (contradiction nodes animate into view; penalty edges appear)
- SafetyMonitorPanel (updating — contradiction_load meter rising)

**Graph state:**
- Contradiction nodes appear (red, fade-in animation)
- Penalty edges animate from contradiction node toward hypothesis node
- Hypothesis node slightly contracts (visual certainty dampening)

**Available actions:**
- Expand ContradictionViewer for full contradiction detail
- Click contradiction node in graph for detail drawer
- Continue observing (inference continues automatically)

**Transition triggers:**
- contradiction_load >= 0.40 → IS-6 (BIOPSY_ESCALATION, safety gate triggered)
- Stage 4 resolves tension → IS-5 (CERTAINTY_STABILIZATION)
- Stage 5 produces high entropy → IS-4 (AMBIGUITY_ESCALATION)

**UI alert:** Subtle amber banner: "Contradicting clinical signs detected — reasoning continues"

---

### IS-4: AMBIGUITY_ESCALATION

**Description:** The certainty distribution is broad (high entropy). No dominant hypothesis is emerging. The workstation signals diagnostic difficulty.

**Active panels:**
- All previous panels active
- DifferentialDiagnosisPanel highlighted (showing near-equal disease probabilities)
- BiopsyTriageWorkspace (showing AMBIGUOUS_CASE provisional)
- CertaintyTimeline (showing flat/diverging lines)

**Graph state:**
- Multiple hypothesis nodes with similar sizes
- No clear dominant node
- Ambiguity indicated by unsettled/oscillating node states

**Available actions:**
- Observe (inference continues to Stage 4 discriminators)
- If Stage 4 fails to resolve: inference routes to BIOPSY_ADVISED

**Transition triggers:**
- Stage 4 discriminators resolve tension (gap >= 0.25) → IS-5
- Stage 5 produces entropy > 1.5 → IS-6 (BIOPSY_ESCALATION)

**UI indicator:** "Diagnostic ambiguity detected — activating pairwise discriminators"

---

### IS-5: CERTAINTY_STABILIZATION

**Description:** A dominant hypothesis is emerging. Certainty is trending toward the stabilization threshold. The workstation shifts toward a calm, convergent visual state.

**Active panels:**
- BiopsyTriageWorkspace (updating with MODERATE_CERTAINTY provisional, trending toward SAFE)
- DifferentialDiagnosisPanel (rank 1 pulling ahead)
- CertaintyTimeline (clear separation between rank 1 and rank 2 lines)
- ReasoningGraph (leading hypothesis node growing; others shrinking)

**Graph state:**
- Leading hypothesis node grows and brightens
- Competing hypothesis nodes dim
- Certainty gap visually widening

**Transition triggers:**
- Stage 5 confirms stability (max_certainty >= 0.65, gap >= 0.35) → IS-7 (terminal)
- Stage 5 safety gate triggers → IS-6

---

### IS-6: BIOPSY_ESCALATION

**Description:** Safety gate or invariant has triggered. Biopsy recommendation is final. The workstation enters a safety-alert terminal state.

**Active panels:**
- BiopsyTriageWorkspace (BIOPSY_ADVISED, prominent)
- SafetyMonitorPanel (showing triggered gates/invariants)
- ContradictionViewer (if contradiction caused escalation)
- ReasoningTraceNavigator (full trace available for review)

**Graph state:**
- Safety gate node activates (amber/red ring)
- Escalation edges animate toward triage node
- Triage node displays BIOPSY_ADVISED in red

**Available actions:**
- Review safety flags and reasoning trace
- Generate report (moves to IS-7 report state)
- Replay case trajectory

**UI alert:** Red banner: "Biopsy required — [reason: contradiction load / insufficient evidence / high ambiguity]"

**Transition triggers:**
- "Generate Report" clicked → IS-7 (REPORT_GENERATION)
- "Replay" clicked → IS-8 (REPLAY_MODE)

---

### IS-7: REPORT_GENERATION / TERMINAL

**Description:** Inference complete. Report is being generated or has been generated. All panels show final state.

**Active panels:**
- All panels showing final values (static)
- ClinicalReportViewer (new panel activated)
- PublicationExportSystem (available)
- ReasoningTraceNavigator (full trace)

**Graph state:** Final — no animation; graph is frozen at terminal state

**Available actions:**
- View report tabs (Summary / Full / Appendix / Raw)
- Export (PDF / JSON / Bundle)
- Replay case trajectory
- Begin new case

**Transition triggers:**
- "Replay" clicked → IS-8 (REPLAY_MODE)
- "New Case" clicked → IS-0 (IDLE)

---

### IS-8: REPLAY_MODE

**Description:** Case trajectory is being replayed. All panels synchronize to the scrubber position.

**Active panels:**
- CaseReplaySystem (prominent — playback controls)
- ReasoningGraph (synchronized to scrubbed stage)
- CertaintyTimeline (scrubber position highlighted)
- SymbolicRuleWorkspace (showing state at scrubbed stage)
- ContradictionViewer (showing contradictions detected up to scrubbed stage)

**Graph state:** Reconstructed from stored GraphSnapshot at the scrubbed stage

**Available actions:**
- Scrub timeline
- Play/Pause/Step
- Compare snapshots (two-stage comparison)
- Exit replay (return to IS-7)

**Transition triggers:**
- "Exit Replay" → IS-7 (TERMINAL)

---

## 3. State Transition Summary

```
IS-0: IDLE
  → IS-1 (feature entry begins)
  → IS-8 (historical case loaded)

IS-1: EVIDENCE_ENTRY
  → IS-2 (inference triggered)
  → IS-0 (features cleared)

IS-2: PARTIAL_REASONING
  → IS-3 (contradiction detected)
  → IS-4 (high entropy after Stage 2/3)
  → IS-5 (certainty trending stable)
  → IS-6 (safety gate triggered)

IS-3: CONTRADICTION_EMERGENCE
  → IS-6 (contradiction_load >= 0.40)
  → IS-4 (high entropy)
  → IS-5 (gap resolved by Stage 4)

IS-4: AMBIGUITY_ESCALATION
  → IS-5 (discriminators resolve)
  → IS-6 (irresolvable; entropy > 1.5)

IS-5: CERTAINTY_STABILIZATION
  → IS-7 (stable threshold met)
  → IS-6 (safety gate overrides)

IS-6: BIOPSY_ESCALATION
  → IS-7 (report generated)
  → IS-8 (replay triggered)

IS-7: REPORT_GENERATION / TERMINAL
  → IS-0 (new case)
  → IS-8 (replay)

IS-8: REPLAY_MODE
  → IS-7 (exit replay)
```

---

## 4. UI Panel Activation Matrix

| Panel | IS-0 | IS-1 | IS-2 | IS-3 | IS-4 | IS-5 | IS-6 | IS-7 | IS-8 |
|---|---|---|---|---|---|---|---|---|---|
| ClinicalInputWorkspace | ● | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| ReasoningGraph | ○ | ◔ | ● | ● | ● | ● | ● | ● | ● |
| SymbolicRuleWorkspace | ○ | ◔ | ● | ● | ● | ● | ● | ● | ● |
| ContradictionViewer | ○ | ○ | ○ | ● | ◔ | ◔ | ● | ● | ● |
| CertaintyTimeline | ○ | ○ | ● | ● | ● | ● | ● | ● | ● |
| BiopsyTriageWorkspace | ○ | ○ | ◔ | ◔ | ◔ | ◔ | ● | ● | ● |
| DifferentialDiagnosisPanel | ○ | ○ | ● | ● | ● | ● | ● | ● | ● |
| SafetyMonitorPanel | ○ | ○ | ◔ | ◔ | ◔ | ◔ | ● | ● | ◔ |
| ReasoningTraceNavigator | ○ | ○ | ◔ | ◔ | ◔ | ◔ | ● | ● | ● |
| CaseReplaySystem | ○ | ○ | ○ | ○ | ○ | ○ | ◔ | ◔ | ● |
| ClinicalReportViewer | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ● | ○ |
| PublicationExportSystem | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ● | ○ |

● Active / ◔ Partial (updating) / ○ Dimmed or hidden

---

## 5. WebSocket Synchronization Protocol

The frontend subscribes to the backend WebSocket at `/ws/inference/stream`. Stage updates arrive as the symbolic engine completes each stage:

```typescript
interface StageUpdate {
  stage: 0 | 1 | 2 | 3 | 4 | 5 | 6
  diagnostic_state: DiagnosticState
  partial_scores: Record<Disease, number>
  activated_rules: RuleActivation[]
  contradiction_events: ContradictionEvent[]
  safety_updates: SafetyGateUpdate[]
  graph_delta: GraphDelta           // nodes/edges added/modified at this stage
  interaction_state_target: InteractionState  // backend suggests target IS
}
```

The frontend `inferenceStore` processes each `StageUpdate`:
1. Apply `graph_delta` to `graphStore` (updates React Flow graph)
2. Append to `traceStore` (updates ReasoningTraceNavigator)
3. Update partial certainty scores (updates CertaintyTimeline + DifferentialDiagnosisPanel)
4. Evaluate contradiction events (triggers IS-3 if new contradictions)
5. Evaluate `diagnostic_state` → map to `InteractionState` → dispatch transition

---

## 6. State Persistence

Case state is persisted to the backend via `POST /api/cases/{id}/checkpoint` after each stage completes. This enables:
- Browser refresh recovery
- Multi-session case continuation
- Replay system (trajectory stored server-side)
- Reproducibility (checkpoint includes feature values + configuration hash)
