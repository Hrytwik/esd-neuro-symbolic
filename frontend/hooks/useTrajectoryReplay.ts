/**
 * hooks/useTrajectoryReplay.ts
 * ==============================
 * Drives automatic step-by-step playback of the trajectory replay.
 */

"use client";

import { useEffect, useRef } from "react";
import { useReasoningStore } from "@/store/reasoning-store";

export function useTrajectoryReplay() {
  const { isPlaying, playbackSpeed, stepForward, totalSteps, currentStep } =
    useReasoningStore();
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        stepForward();
      }, playbackSpeed);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isPlaying, playbackSpeed, stepForward]);

  // Auto-stop at last step
  useEffect(() => {
    if (currentStep >= totalSteps - 1 && isPlaying) {
      useReasoningStore.getState().pause();
    }
  }, [currentStep, totalSteps, isPlaying]);
}
