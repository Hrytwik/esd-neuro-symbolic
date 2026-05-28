/**
 * components/workspace/ClinicalReasoningWorkspace.tsx
 * =====================================================
 * Main four-panel clinical reasoning workstation layout.
 *
 * Layout:
 *   ┌─────────┬───────────────────────────┬────────────────┐
 *   │ LEFT    │ CENTER                    │ RIGHT          │
 *   │ 260px   │ flex                      │ 300px          │
 *   │         │                           │                │
 *   │ Feature │ Reasoning Graph           │ Certainty      │
 *   │ Input   │ (React Flow)              │ Evolution      │
 *   │         │                           │                │
 *   │ Rule    │                           │ Escalation     │
 *   │ Panel   │                           │ Reasoning      │
 *   │         │                           │                │
 *   │         │                           │ Contradiction  │
 *   ├─────────┴───────────────────────────┴────────────────┤
 *   │ BOTTOM 220px                                         │
 *   │ Trajectory Replay                                    │
 *   └──────────────────────────────────────────────────────┘
 */

"use client";

import { useEffect, useState, useCallback } from "react";
import { useReasoningStore } from "@/store/reasoning-store";
import { BackendService } from "@/services/backend-service";
import { FeatureInputPanel } from "@/components/panels/FeatureInputPanel";
import { ReasoningGraphViewer } from "@/components/graph/ReasoningGraphViewer";
import { CertaintyEvolutionPanel } from "@/components/panels/CertaintyEvolutionPanel";
import { TrajectoryReplayViewer } from "@/components/trajectory/TrajectoryReplayViewer";
import { EscalationReasoningPanel } from "@/components/escalation/EscalationReasoningPanel";
import { ContradictionPanel } from "@/components/contradictions/ContradictionPanel";
import { SymbolicRulePanel } from "@/components/reasoning/SymbolicRulePanel";
import { ClinicalBadge } from "@/components/ui/ClinicalBadge";
import { caseStatusColor, caseStatusLabel, formatCertainty } from "@/lib/reasoning-utils";
import { diseaseLabel } from "@/lib/clinical-language";
import { clsx } from "clsx";
import { ChevronDown, Activity, LayoutGrid } from "lucide-react";

// ─── Right-panel tab type ─────────────────────────────────────────────────────

type RightTab = "certainty" | "escalation" | "contradiction" | "rules";

const RIGHT_TABS: { id: RightTab; label: string }[] = [
  { id: "certainty",    label: "Certainty"     },
  { id: "escalation",  label: "Triage"         },
  { id: "contradiction",label: "Contradiction" },
  { id: "rules",       label: "Rules"          },
];

// ─── Case selector ────────────────────────────────────────────────────────────

function CaseSelector() {
  const { availableCases, currentCaseId, loadCase, setAvailableCases } = useReasoningStore();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  // Load case list on mount
  useEffect(() => {
    BackendService.listCases().then(setAvailableCases);
  }, [setAvailableCases]);

  const handleSelectCase = useCallback(
    async (caseId: string) => {
      setOpen(false);
      setLoading(true);
      try {
        const bundle = await BackendService.loadCase(caseId);
        loadCase(bundle.replay, bundle.reasoning, bundle.graph);
      } catch (err) {
        console.error("Failed to load case:", err);
      } finally {
        setLoading(false);
      }
    },
    [loadCase]
  );

  const currentCase = availableCases.find(c => c.case_id === currentCaseId);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={clsx(
          "flex items-center gap-2 px-3 py-1.5 rounded border text-xs transition-colors",
          "border-slate-200 bg-white hover:bg-slate-50 text-slate-700",
          open && "bg-slate-50"
        )}
      >
        {loading ? (
          <span className="text-slate-400">Loading…</span>
        ) : currentCase ? (
          <>
            <span className="font-mono text-slate-500">{currentCase.case_id}</span>
            <span className="font-medium">{diseaseLabel(currentCase.final_diagnosis)}</span>
            {currentCase.requires_biopsy && (
              <ClinicalBadge variant="biopsy" size="sm">Biopsy</ClinicalBadge>
            )}
          </>
        ) : (
          <span className="text-slate-400">Select a case…</span>
        )}
        <ChevronDown size={12} className="text-slate-400 ml-1" />
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-clinical-lg z-50 min-w-[280px]">
          <div className="p-2 border-b border-slate-100 text-[10px] text-slate-400 font-semibold uppercase tracking-wide">
            Available Cases
          </div>
          {availableCases.map(c => (
            <button
              key={c.case_id}
              onClick={() => handleSelectCase(c.case_id)}
              className={clsx(
                "w-full text-left flex items-start gap-3 px-3 py-2 hover:bg-slate-50 transition-colors",
                c.case_id === currentCaseId && "bg-blue-50"
              )}
            >
              <span className="font-mono text-[10px] text-slate-400 w-16 shrink-0 pt-0.5">
                {c.case_id}
              </span>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium text-slate-700">
                  {diseaseLabel(c.final_diagnosis)}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {c.total_steps} steps · {formatCertainty(c.final_certainty)}
                  {!c.converged && " · Not converged"}
                </div>
              </div>
              <span className={clsx("text-[10px] shrink-0 mt-0.5", caseStatusColor(c as any).split(" ")[0])}>
                {c.requires_biopsy ? "Biopsy" : c.converged ? "Stable" : "Inconclusive"}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main workspace ───────────────────────────────────────────────────────────

export function ClinicalReasoningWorkspace() {
  const { currentReasoning, currentCaseRecord } = useReasoningStore();
  const [rightTab, setRightTab] = useState<RightTab>("certainty");

  return (
    <div className="flex flex-col h-screen bg-clinical-bg overflow-hidden">

      {/* ── Top header bar ─────────────────────────────────────────────────── */}
      <header className="shrink-0 h-12 bg-white border-b border-clinical-border flex items-center justify-between px-5">
        <div className="flex items-center gap-3">
          {/* Wordmark */}
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-blue-900 flex items-center justify-center">
              <span className="text-[9px] font-bold text-white tracking-tight">Rx</span>
            </div>
            <div>
              <div className="text-sm font-bold text-slate-800 tracking-tight leading-none">
                CASDRE
              </div>
              <div className="text-[9px] text-slate-400 leading-none mt-0.5">
                Dermatological Reasoning Workstation
              </div>
            </div>
          </div>

          <div className="w-px h-6 bg-slate-200 mx-1" />

          {/* Case selector */}
          <CaseSelector />
        </div>

        {/* Right header */}
        <div className="flex items-center gap-3">
          {currentReasoning && (
            <>
              <div className="flex items-center gap-1.5">
                <Activity size={12} className="text-slate-400" />
                <span className="text-xs text-slate-500 font-mono">
                  {currentReasoning.fsm_state.replace(/_/g, " ")}
                </span>
              </div>
              <ClinicalBadge
                variant={currentReasoning.requires_biopsy ? "biopsy" : "safe"}
                size="sm"
              >
                {currentReasoning.requires_biopsy ? "Biopsy Required" : "Safe Triage"}
              </ClinicalBadge>
            </>
          )}
          <div className="text-[10px] text-slate-300 font-mono">v1.0.0</div>
        </div>
      </header>

      {/* ── Main three-column body ─────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">

        {/* LEFT — Feature input + rule panel */}
        <div className="w-[260px] shrink-0 border-r border-clinical-border bg-white flex flex-col min-h-0">
          {/* Top half: Features */}
          <div className="flex-1 min-h-0 overflow-hidden border-b border-slate-100">
            <FeatureInputPanel />
          </div>
          {/* Bottom half: Symbolic rules */}
          <div className="h-[45%] min-h-0 overflow-hidden">
            <SymbolicRulePanel />
          </div>
        </div>

        {/* CENTER — Reasoning graph */}
        <div className="flex-1 min-w-0 relative bg-clinical-bg">
          <ReasoningGraphViewer />

          {/* Center overlay — when no case */}
          {!currentCaseRecord && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center">
                <LayoutGrid size={40} className="text-slate-200 mx-auto mb-3" />
                <div className="text-sm font-medium text-slate-400">
                  No case selected
                </div>
                <div className="text-xs text-slate-300 mt-1">
                  Use the case selector above to load a case
                </div>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT — Tabbed analysis panels */}
        <div className="w-[300px] shrink-0 border-l border-clinical-border bg-white flex flex-col min-h-0">
          {/* Tab bar */}
          <div className="flex shrink-0 border-b border-slate-100 bg-slate-50">
            {RIGHT_TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setRightTab(tab.id)}
                className={clsx(
                  "flex-1 py-2 text-[11px] font-medium transition-colors",
                  rightTab === tab.id
                    ? "text-blue-700 border-b-2 border-blue-600 bg-white"
                    : "text-slate-400 hover:text-slate-600"
                )}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 min-h-0 overflow-hidden">
            {rightTab === "certainty"     && <CertaintyEvolutionPanel />}
            {rightTab === "escalation"   && <EscalationReasoningPanel />}
            {rightTab === "contradiction" && <ContradictionPanel />}
            {rightTab === "rules"        && <SymbolicRulePanel />}
          </div>
        </div>
      </div>

      {/* ── BOTTOM — Trajectory replay ────────────────────────────────────── */}
      <div className="shrink-0 h-[220px] border-t border-clinical-border bg-white">
        <TrajectoryReplayViewer />
      </div>

    </div>
  );
}
