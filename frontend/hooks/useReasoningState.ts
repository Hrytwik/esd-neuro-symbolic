/**
 * hooks/useReasoningState.ts
 * ============================
 * Exposes derived reasoning state for components, with memoisation.
 */

"use client";

import { useMemo } from "react";
import {
  useReasoningStore,
  selectCurrentEvent,
  selectCertaintySeries,
  selectEventsUpToStep,
} from "@/store/reasoning-store";
import { activatedSignals, suppressedSignals, competitionMargin } from "@/lib/reasoning-utils";
import type { ReasoningOutput } from "@/types";

export function useReasoningState() {
  const store = useReasoningStore();

  const currentEvent = useMemo(
    () => selectCurrentEvent(store),
    [store.currentCaseRecord, store.currentStep]
  );

  const certaintySeries = useMemo(
    () => selectCertaintySeries(store),
    [store.currentCaseRecord, store.currentStep]
  );

  const eventsUpToStep = useMemo(
    () => selectEventsUpToStep(store),
    [store.currentCaseRecord, store.currentStep]
  );

  // Reasoning snapshot at current step (use step-adjusted data if available)
  const stepReasoning = useMemo<ReasoningOutput | null>(() => {
    if (!store.currentReasoning || !store.currentCaseRecord) return null;
    const event = currentEvent;
    if (!event) return store.currentReasoning;

    // Patch the final reasoning with step-specific certainty/fsm data
    return {
      ...store.currentReasoning,
      certainty: event.certainty,
      ambiguity_bits: event.ambiguity_bits,
      fsm_state: event.fsm_state as ReasoningOutput["fsm_state"],
      leading_diagnosis: event.leading_diagnosis,
      contradiction: {
        ...store.currentReasoning.contradiction,
        overall_load: event.contradiction_load,
      },
    };
  }, [store.currentReasoning, store.currentCaseRecord, currentEvent]);

  const signals = useMemo(
    () => (stepReasoning ? activatedSignals(stepReasoning) : []),
    [stepReasoning]
  );

  const suppressedSignalsList = useMemo(
    () => (stepReasoning ? suppressedSignals(stepReasoning) : []),
    [stepReasoning]
  );

  const margin = useMemo(
    () => (stepReasoning ? competitionMargin(stepReasoning) : 0),
    [stepReasoning]
  );

  return {
    stepReasoning,
    currentEvent,
    certaintySeries,
    eventsUpToStep,
    signals,
    suppressedSignals: suppressedSignalsList,
    competitionMargin: margin,
    isLoaded: !!store.currentReasoning,
    requiresBiopsy: store.currentReasoning?.requires_biopsy ?? false,
    isSafeTriage: store.currentReasoning?.is_safe_triage ?? false,
  };
}
