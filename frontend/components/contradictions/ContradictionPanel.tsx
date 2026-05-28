/**
 * components/contradictions/ContradictionPanel.tsx
 * ==================================================
 * Displays localised contradictions and competing diagnostic evidence.
 * Helps clinicians understand WHY escalation occurred.
 */

"use client";

import { useReasoningState } from "@/hooks/useReasoningState";
import { useReasoningStore } from "@/store/reasoning-store";
import { CONTRADICTION_DESCRIPTIONS, diseaseLabel } from "@/lib/clinical-language";
import { ContradictionMeter } from "@/components/ui/ContradictionMeter";
import { ClinicalBadge } from "@/components/ui/ClinicalBadge";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { formatCertainty } from "@/lib/reasoning-utils";
import { AlertTriangle, Minus, ArrowLeftRight } from "lucide-react";
import { clsx } from "clsx";

export function ContradictionPanel() {
  const { stepReasoning, competitionMargin } = useReasoningState();
  const { currentCaseRecord } = useReasoningStore();

  if (!stepReasoning || !currentCaseRecord) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400 text-xs p-4 text-center">
        Load a case to view contradiction analysis
      </div>
    );
  }

  const contra = stepReasoning.contradiction;
  const diff   = [...stepReasoning.differential].sort((a, b) => b.certainty - a.certainty);
  const leader = diff[0];
  const second = diff[1];

  // Suppressed signals are treating as locally contradicting
  const contradictingSignals = stepReasoning.symbolic_signals.filter(
    (s) => !s.activated && s.value > 0.05
  );

  const competitionClose = competitionMargin < 0.20;

  return (
    <div className="flex flex-col h-full">
      <SectionHeader
        title="Contradiction Analysis"
        badge={
          contra.tier !== "NONE" ? (
            <ClinicalBadge
              variant={
                contra.tier === "CRITICAL" ? "biopsy" :
                contra.tier === "MODERATE" ? "warning" : "neutral"
              }
              size="sm"
            >
              {contra.tier}
            </ClinicalBadge>
          ) : undefined
        }
        size="sm"
      />

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">

        {/* Global load */}
        <div>
          <ContradictionMeter
            load={contra.overall_load}
            label="Global contradiction load"
            showCeiling
          />
          <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">
            {CONTRADICTION_DESCRIPTIONS[contra.tier]}
          </p>
        </div>

        {/* Hypothesis competition */}
        {leader && second && (
          <div className="border border-slate-100 rounded-lg p-3">
            <div className="flex items-center gap-1 mb-2.5">
              <ArrowLeftRight size={11} className="text-amber-500" />
              <span className="text-[11px] font-semibold text-slate-700">
                Hypothesis Competition
              </span>
              {competitionClose && (
                <ClinicalBadge variant="warning" size="sm">Close</ClinicalBadge>
              )}
            </div>

            {/* Leader bar */}
            <div className="mb-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-blue-700">
                  {diseaseLabel(leader.disease)}
                </span>
                <span className="text-xs font-mono text-blue-700">
                  {formatCertainty(leader.certainty)}
                </span>
              </div>
              <div className="w-full h-1.5 bg-slate-100 rounded-full">
                <div
                  className="h-full rounded-full bg-blue-400 transition-all duration-500"
                  style={{ width: `${leader.certainty * 100}%` }}
                />
              </div>
            </div>

            {/* Second bar */}
            <div className="mb-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-amber-600">
                  {diseaseLabel(second.disease)}
                </span>
                <span className="text-xs font-mono text-amber-600">
                  {formatCertainty(second.certainty)}
                </span>
              </div>
              <div className="w-full h-1.5 bg-slate-100 rounded-full">
                <div
                  className="h-full rounded-full bg-amber-400 transition-all duration-500"
                  style={{ width: `${second.certainty * 100}%` }}
                />
              </div>
            </div>

            {/* Margin */}
            <div className="flex items-center gap-1.5 text-[10px]">
              <Minus size={10} className="text-slate-400" />
              <span className="text-slate-400">Separation margin:</span>
              <span
                className={clsx(
                  "font-mono font-semibold",
                  competitionClose ? "text-amber-600" : "text-emerald-600"
                )}
              >
                {formatCertainty(competitionMargin)}
              </span>
              {competitionClose && (
                <span className="text-amber-500">
                  — close competition detected
                </span>
              )}
            </div>
          </div>
        )}

        {/* Conflicting signals */}
        {contradictingSignals.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <AlertTriangle size={11} className="text-red-400" />
              <span className="text-[11px] font-semibold text-slate-700">
                Conflicting Signals ({contradictingSignals.length})
              </span>
            </div>
            <div className="space-y-1.5">
              {contradictingSignals.map((sig) => (
                <div
                  key={sig.name}
                  className="flex items-center justify-between text-[11px] py-1.5 px-2.5 bg-red-50 border border-red-100 rounded"
                >
                  <span className="text-red-700">{sig.name}</span>
                  <span className="font-mono text-red-500 shrink-0 ml-2">
                    {(sig.value * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              These signals are present but inconsistent with the leading diagnostic hypothesis.
              They contribute to the overall contradiction load.
            </p>
          </div>
        )}

        {/* No contradictions */}
        {contra.tier === "NONE" && contradictingSignals.length === 0 && (
          <div className="text-center py-4">
            <div className="text-slate-300 text-2xl mb-2">✓</div>
            <div className="text-xs text-slate-500">
              Clinical findings are internally consistent.
              No significant conflicting signals detected.
            </div>
          </div>
        )}

        {/* Escalation trigger info */}
        {contra.escalation_triggered_by_contradiction && (
          <div className="border-t border-slate-100 pt-3">
            <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded p-2.5">
              <AlertTriangle size={13} className="text-red-500 shrink-0 mt-0.5" />
              <div className="text-[11px]">
                <div className="font-semibold text-red-700 mb-0.5">
                  Escalation triggered by contradiction
                </div>
                <div className="text-red-600 leading-relaxed">
                  The diagnostic contradiction was severe enough to require biopsy referral.
                  Competing inflammatory patterns could not be resolved through clinical assessment alone.
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
