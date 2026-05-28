/**
 * components/reasoning/SymbolicRulePanel.tsx
 * ============================================
 * Displays activated symbolic rules and clinical signatures.
 * Critical for interpretability and clinician trust.
 */

"use client";

import { useReasoningState } from "@/hooks/useReasoningState";
import { useReasoningStore } from "@/store/reasoning-store";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ClinicalBadge } from "@/components/ui/ClinicalBadge";
import { diseaseLabel } from "@/lib/clinical-language";
import { clsx } from "clsx";
import { Zap, ZapOff, Star } from "lucide-react";

export function SymbolicRulePanel() {
  const { stepReasoning, signals, suppressedSignals } = useReasoningState();
  const { currentCaseRecord } = useReasoningStore();

  if (!stepReasoning || !currentCaseRecord) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400 text-xs p-4 text-center">
        Load a case to view symbolic rule activations
      </div>
    );
  }

  // Detect pathognomonic signals
  const isPathognomonic = (name: string) =>
    name.toLowerCase().includes("pathognomonic") ||
    name.toLowerCase().includes("triad") ||
    (name.toLowerCase().includes("pso") && name.includes("×")) ||
    (name.toLowerCase().includes("prp") && name.includes("×")) ||
    (name.toLowerCase().includes("lp") && name.includes("oral"));

  return (
    <div className="flex flex-col h-full">
      <SectionHeader
        title="Symbolic Rule Activations"
        subtitle={`${signals.length} active · ${suppressedSignals.length} suppressed`}
        size="sm"
      />

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">

        {/* Activated signals */}
        {signals.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <Zap size={11} className="text-emerald-500" />
              <span className="text-[11px] font-semibold text-slate-700">
                Supporting Findings
              </span>
            </div>
            <div className="space-y-1.5">
              {signals.map((sig) => {
                const patho = isPathognomonic(sig.name);
                return (
                  <div
                    key={sig.name}
                    className={clsx(
                      "flex items-center justify-between py-1.5 px-2.5 rounded border",
                      patho
                        ? "bg-violet-50 border-violet-200"
                        : "bg-emerald-50 border-emerald-100"
                    )}
                  >
                    <div className="flex items-start gap-1.5 min-w-0">
                      {patho ? (
                        <Star size={11} className="text-violet-500 shrink-0 mt-0.5" />
                      ) : (
                        <Zap size={11} className="text-emerald-500 shrink-0 mt-0.5" />
                      )}
                      <div className="min-w-0">
                        <div
                          className={clsx(
                            "text-[11px] font-medium leading-tight",
                            patho ? "text-violet-700" : "text-emerald-700"
                          )}
                        >
                          {sig.name}
                        </div>
                        {patho && (
                          <div className="text-[9px] font-semibold text-violet-500 uppercase tracking-wide mt-0.5">
                            Pathognomonic
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0 ml-2">
                      {/* Contribution bar */}
                      <div className="w-14 h-1.5 bg-slate-200 rounded-full">
                        <div
                          className={clsx(
                            "h-full rounded-full",
                            patho ? "bg-violet-400" : "bg-emerald-400"
                          )}
                          style={{ width: `${sig.contribution_weight * 100}%` }}
                        />
                      </div>
                      <span
                        className={clsx(
                          "text-[10px] font-mono",
                          patho ? "text-violet-600" : "text-emerald-600"
                        )}
                      >
                        {(sig.contribution_weight * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Suppressed / non-activated signals */}
        {suppressedSignals.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <ZapOff size={11} className="text-slate-400" />
              <span className="text-[11px] font-semibold text-slate-600">
                Absent or Suppressed Findings
              </span>
            </div>
            <div className="space-y-1">
              {suppressedSignals.map((sig) => (
                <div
                  key={sig.name}
                  className="flex items-center justify-between py-1 px-2.5 rounded border border-slate-100 bg-slate-50"
                >
                  <span className="text-[11px] text-slate-400 line-through truncate">
                    {sig.name}
                  </span>
                  <span className="text-[10px] font-mono text-slate-300 shrink-0 ml-2">
                    {(sig.value * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-slate-400 mt-1.5">
              These findings are absent or below activation threshold.
            </p>
          </div>
        )}

        {/* Differential summary */}
        <div className="border-t border-slate-100 pt-3">
          <div className="text-[11px] font-semibold text-slate-700 mb-2">
            Differential by Certainty
          </div>
          <div className="space-y-1.5">
            {[...stepReasoning.differential]
              .sort((a, b) => b.certainty - a.certainty)
              .map((entry) => (
                <div key={entry.disease} className="flex items-center gap-2">
                  <div className="w-20 shrink-0">
                    <span
                      className={clsx(
                        "text-[11px] truncate block",
                        entry.is_leading ? "font-semibold text-blue-700" : "text-slate-500"
                      )}
                    >
                      {diseaseLabel(entry.disease)}
                    </span>
                  </div>
                  <div className="flex-1 h-1.5 bg-slate-100 rounded-full">
                    <div
                      className={clsx(
                        "h-full rounded-full transition-all duration-500",
                        entry.is_leading ? "bg-blue-400" : "bg-slate-300"
                      )}
                      style={{ width: `${entry.certainty * 100}%` }}
                    />
                  </div>
                  <span
                    className={clsx(
                      "text-[10px] font-mono w-10 text-right shrink-0",
                      entry.is_leading ? "text-blue-600 font-semibold" : "text-slate-400"
                    )}
                  >
                    {(entry.certainty * 100).toFixed(1)}%
                  </span>
                  {entry.is_leading && (
                    <ClinicalBadge variant="active" size="sm">Lead</ClinicalBadge>
                  )}
                </div>
              ))}
          </div>
        </div>

      </div>
    </div>
  );
}
