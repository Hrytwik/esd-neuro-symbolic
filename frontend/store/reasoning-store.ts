/**
 * store/reasoning-store.ts
 * ==========================
 * Global state management for the CASDRE clinical reasoning workstation.
 * Uses Zustand for lightweight, predictable state.
 */

"use client";

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import type {
  ReplayCaseRecord,
  ReplayEvent,
  ReasoningOutput,
  GraphSnapshot,
  CaseListItem,
} from "@/types";

// ─── Store state shape ────────────────────────────────────────────────────────

export interface ReasoningState {
  // ── Case catalog ────────────────────────────────────────────────────────
  availableCases: CaseListItem[];
  currentCaseId: string | null;
  currentCaseRecord: ReplayCaseRecord | null;
  currentReasoning: ReasoningOutput | null;
  currentGraphSnapshot: GraphSnapshot | null;

  // ── Replay playback state ───────────────────────────────────────────────
  currentStep: number;
  totalSteps: number;
  isPlaying: boolean;
  playbackSpeed: number;  // ms per step (500 / 1000 / 2000)

  // ── Graph interaction state ─────────────────────────────────────────────
  selectedNodeId: string | null;
  highlightedEdgeTypes: string[];

  // ── UI layout state ─────────────────────────────────────────────────────
  activePanel: "graph" | "trajectory" | null;
  showContradictions: boolean;
  showRulePanel: boolean;
}

// ─── Store actions shape ──────────────────────────────────────────────────────

export interface ReasoningActions {
  // Case management
  setAvailableCases: (cases: CaseListItem[]) => void;
  loadCase: (
    caseRecord: ReplayCaseRecord,
    reasoning: ReasoningOutput,
    graphSnapshot: GraphSnapshot
  ) => void;
  clearCase: () => void;

  // Replay controls
  setStep: (step: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  play: () => void;
  pause: () => void;
  setPlaybackSpeed: (ms: number) => void;

  // Graph interaction
  selectNode: (nodeId: string | null) => void;
  setHighlightedEdgeTypes: (types: string[]) => void;

  // UI
  setActivePanel: (panel: "graph" | "trajectory" | null) => void;
  toggleContradictions: () => void;
  toggleRulePanel: () => void;
}

// ─── Derived helpers ──────────────────────────────────────────────────────────

/** Returns the replay event at the given step (or the last event if out of bounds). */
function getEventAtStep(
  events: ReplayEvent[],
  step: number
): ReplayEvent | null {
  const sorted = [...events].sort((a, b) => a.step - b.step);
  // Return the most recent event at or before this step
  const candidates = sorted.filter((e) => e.step <= step);
  return candidates.length > 0 ? candidates[candidates.length - 1] : null;
}

// ─── Store definition ─────────────────────────────────────────────────────────

const initialState: ReasoningState = {
  availableCases: [],
  currentCaseId: null,
  currentCaseRecord: null,
  currentReasoning: null,
  currentGraphSnapshot: null,
  currentStep: 0,
  totalSteps: 0,
  isPlaying: false,
  playbackSpeed: 1000,
  selectedNodeId: null,
  highlightedEdgeTypes: [],
  activePanel: "graph",
  showContradictions: true,
  showRulePanel: true,
};

export const useReasoningStore = create<ReasoningState & ReasoningActions>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    // ── Case management ────────────────────────────────────────────────────

    setAvailableCases: (cases) => set({ availableCases: cases }),

    loadCase: (caseRecord, reasoning, graphSnapshot) =>
      set({
        currentCaseId: caseRecord.case_id,
        currentCaseRecord: caseRecord,
        currentReasoning: reasoning,
        currentGraphSnapshot: graphSnapshot,
        currentStep: 0,
        totalSteps: caseRecord.total_steps,
        isPlaying: false,
        selectedNodeId: null,
      }),

    clearCase: () =>
      set({
        currentCaseId: null,
        currentCaseRecord: null,
        currentReasoning: null,
        currentGraphSnapshot: null,
        currentStep: 0,
        totalSteps: 0,
        isPlaying: false,
      }),

    // ── Replay controls ────────────────────────────────────────────────────

    setStep: (step) => {
      const { totalSteps } = get();
      const clamped = Math.max(0, Math.min(step, totalSteps - 1));
      set({ currentStep: clamped });
    },

    stepForward: () => {
      const { currentStep, totalSteps } = get();
      if (currentStep < totalSteps - 1) {
        set({ currentStep: currentStep + 1 });
      } else {
        set({ isPlaying: false });
      }
    },

    stepBackward: () => {
      const { currentStep } = get();
      if (currentStep > 0) {
        set({ currentStep: currentStep - 1 });
      }
    },

    play: () => set({ isPlaying: true }),
    pause: () => set({ isPlaying: false }),

    setPlaybackSpeed: (ms) => set({ playbackSpeed: ms }),

    // ── Graph interaction ──────────────────────────────────────────────────

    selectNode: (nodeId) => set({ selectedNodeId: nodeId }),

    setHighlightedEdgeTypes: (types) => set({ highlightedEdgeTypes: types }),

    // ── UI layout ──────────────────────────────────────────────────────────

    setActivePanel: (panel) => set({ activePanel: panel }),

    toggleContradictions: () =>
      set((s) => ({ showContradictions: !s.showContradictions })),

    toggleRulePanel: () =>
      set((s) => ({ showRulePanel: !s.showRulePanel })),
  }))
);

// ─── Derived selectors ────────────────────────────────────────────────────────

/** Current replay event (the most recent event at currentStep). */
export function selectCurrentEvent(state: ReasoningState): ReplayEvent | null {
  if (!state.currentCaseRecord) return null;
  return getEventAtStep(state.currentCaseRecord.events, state.currentStep);
}

/** Events up to and including currentStep (for the trajectory chart). */
export function selectEventsUpToStep(state: ReasoningState): ReplayEvent[] {
  if (!state.currentCaseRecord) return [];
  return state.currentCaseRecord.events
    .filter((e) => e.step <= state.currentStep)
    .sort((a, b) => a.step - b.step);
}

/** Certainty series for the trajectory chart. */
export function selectCertaintySeries(
  state: ReasoningState
): Array<{ step: number; certainty: number; ambiguity: number; disease: string }> {
  if (!state.currentCaseRecord) return [];
  return state.currentCaseRecord.events
    .filter((e) => e.step <= state.currentStep)
    .sort((a, b) => a.step - b.step)
    .map((e) => ({
      step: e.step,
      certainty: e.certainty,
      ambiguity: e.ambiguity_bits,
      disease: e.leading_diagnosis,
    }));
}
