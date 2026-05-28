/**
 * components/clinical/ClinicalWorkflow.tsx
 * ==========================================
 * Top-level clinical workflow orchestrator.
 *
 * Phases:
 *   input    → Doctor enters clinical findings (ClinicalInputForm)
 *   running  → System processes the feature vector
 *   results  → Simplified diagnosis results (ClinicalResultsView)
 *
 * Advanced reasoning (graph, replay, contradictions, rules, trajectory)
 * is accessible from the results view via AdvancedReasoningDrawer — it
 * is NOT shown by default.
 */

"use client";

import { useState, useCallback } from "react";
import { ClinicalInputForm } from "./ClinicalInputForm";
import { ClinicalResultsView } from "./ClinicalResultsView";
import { AdvancedReasoningDrawer } from "@/components/advanced/AdvancedReasoningDrawer";
import { BackendService } from "@/services/backend-service";
import { useReasoningStore } from "@/store/reasoning-store";
import type { FeatureVector } from "@/types/clinical-input";
import type { ReasoningOutput } from "@/types";
import type { ReplayCaseRecord } from "@/types";
import { clsx } from "clsx";
import { Activity } from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

type WorkflowPhase = "input" | "running" | "results";

interface ResultState {
  reasoning:  ReasoningOutput;
  caseRecord: ReplayCaseRecord;
}

// ─── Loading screen ───────────────────────────────────────────────────────────

const ANALYSIS_STEPS = [
  "Evaluating clinical feature profile…",
  "Resolving competing diagnostic hypotheses…",
  "Checking for contradicting patterns…",
  "Applying symbolic clinical criteria…",
  "Generating differential assessment…",
];

function RunningScreen() {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center px-6">
      <div className="text-center max-w-sm">

        {/* Animated icon */}
        <div className="w-16 h-16 rounded-full bg-blue-50 border-2 border-blue-100 flex items-center justify-center mx-auto mb-6">
          <Activity size={24} className="text-blue-600 animate-pulse" />
        </div>

        <h2 className="text-lg font-bold text-slate-800 mb-2">
          Analysing Clinical Presentation
        </h2>
        <p className="text-sm text-slate-500 mb-8">
          Processing entered findings through the diagnostic reasoning engine.
        </p>

        {/* Step sequence — animated stagger via CSS */}
        <div className="space-y-2.5 text-left">
          {ANALYSIS_STEPS.map((step, i) => (
            <div
              key={step}
              className={clsx(
                "flex items-center gap-2.5 text-sm opacity-0",
                "animate-[fadeIn_0.4s_ease-out_forwards]"
              )}
              style={{ animationDelay: `${i * 240}ms` }}
            >
              <div className="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
              <span className="text-slate-600">{step}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Workflow ─────────────────────────────────────────────────────────────────

export function ClinicalWorkflow() {
  const [phase, setPhase]           = useState<WorkflowPhase>("input");
  const [result, setResult]         = useState<ResultState | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [error, setError]           = useState<string | null>(null);

  const { loadCase } = useReasoningStore();

  const runReasoning = useCallback(async (features: FeatureVector) => {
    setPhase("running");
    setError(null);
    try {
      const bundle = await BackendService.runReasoning(features);

      // Populate the Zustand store so advanced panels have data
      loadCase(bundle.replay, bundle.reasoning, bundle.graph);

      setResult({
        reasoning:  bundle.reasoning,
        caseRecord: bundle.replay,
      });
      setPhase("results");
    } catch (err) {
      console.error("Reasoning failed:", err);
      setError("Diagnostic reasoning could not complete. Please try again.");
      setPhase("input");
    }
  }, [loadCase]);

  const handleBack = useCallback(() => {
    setPhase("input");
    setResult(null);
    setDrawerOpen(false);
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────────

  if (phase === "running") {
    return <RunningScreen />;
  }

  if (phase === "results" && result) {
    return (
      <>
        <ClinicalResultsView
          reasoning={result.reasoning}
          caseRecord={result.caseRecord}
          onBack={handleBack}
          onAdvanced={() => setDrawerOpen(true)}
        />
        <AdvancedReasoningDrawer
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
        />
      </>
    );
  }

  // Phase = "input"
  return (
    <>
      {error && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700 shadow-lg">
          {error}
        </div>
      )}
      <ClinicalInputForm
        onSubmit={runReasoning}
        loading={false}
      />
    </>
  );
}
