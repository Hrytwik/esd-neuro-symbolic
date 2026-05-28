/**
 * components/escalation/EscalationReasoningPanel.tsx
 * =====================================================
 * Displays biopsy escalation decision with clinical reasoning rationale.
 * Language is always clinician-facing — no technical jargon.
 */

"use client";

import { useReasoningState } from "@/hooks/useReasoningState";
import { useReasoningStore } from "@/store/reasoning-store";
import {
  escalationRationale,
  certaintyClinicalLabel,
  ambiguityClinicalLabel,
  CONTRADICTION_DESCRIPTIONS,
  FSM_STATE_DESCRIPTIONS,
  RECOVERY_DESCRIPTIONS,
} from "@/lib/clinical-language";
import { ClinicalBadge } from "@/components/ui/ClinicalBadge";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { CertaintyBar } from "@/components/ui/CertaintyBar";
import { ContradictionMeter } from "@/components/ui/ContradictionMeter";
import { formatBits } from "@/lib/reasoning-utils";
import { AlertTriangle, CheckCircle2, Activity, ArrowRight } from "lucide-react";
import { clsx } from "clsx";

export function EscalationReasoningPanel() {
  const { stepReasoning } = useReasoningState();
  const { currentCaseRecord } = useReasoningStore();

  if (!stepReasoning || !currentCaseRecord) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400 text-xs p-4 text-center">
        Load a case to view escalation reasoning
      </div>
    );
  }

  const biopsy     = stepReasoning.requires_biopsy;
  const certainty  = stepReasoning.certainty;
  const ambiguity  = stepReasoning.ambiguity_bits;
  const contra     = stepReasoning.contradiction;
  const fsm        = stepReasoning.fsm_state;

  const rationale = escalationRationale(biopsy, contra.tier, certainty, ambiguity);

  return (
    <div className="flex flex-col h-full">
      <SectionHeader
        title="Triage Assessment"
        subtitle={`Step: ${FSM_STATE_DESCRIPTIONS[fsm].split(".")[0]}`}
        size="sm"
      />

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">

        {/* Decision banner */}
        <div
          className={clsx(
            "rounded-lg border p-3",
            biopsy
              ? "border-red-200 bg-red-50"
              : "border-emerald-200 bg-emerald-50"
          )}
        >
          <div className="flex items-start gap-2">
            {biopsy ? (
              <AlertTriangle size={15} className="text-red-500 shrink-0 mt-0.5" />
            ) : (
              <CheckCircle2 size={15} className="text-emerald-500 shrink-0 mt-0.5" />
            )}
            <div>
              <div
                className={clsx(
                  "text-xs font-semibold mb-1",
                  biopsy ? "text-red-700" : "text-emerald-700"
                )}
              >
                {biopsy
                  ? "Histopathological Confirmation Required"
                  : "Safe Clinical Triage"}
              </div>
              <p className="text-[11px] leading-relaxed text-slate-600">
                {rationale}
              </p>
            </div>
          </div>
        </div>

        {/* Certainty */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-semibold text-slate-700">
              Diagnostic Certainty
            </span>
            <ClinicalBadge
              variant={certainty >= 0.75 ? "safe" : certainty >= 0.55 ? "warning" : "biopsy"}
              size="sm"
            >
              {certaintyClinicalLabel(certainty)}
            </ClinicalBadge>
          </div>
          <CertaintyBar value={certainty} height="lg" showLabel animate />
        </div>

        {/* Ambiguity */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] font-semibold text-slate-700">
              Residual Ambiguity
            </span>
            <span className="text-[10px] font-mono text-slate-500">
              {formatBits(ambiguity)}
            </span>
          </div>
          <div className="text-[11px] text-slate-500">
            {ambiguityClinicalLabel(ambiguity)}
          </div>
        </div>

        {/* Contradiction load */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-semibold text-slate-700">
              Contradiction Load
            </span>
            <ClinicalBadge
              variant={
                contra.tier === "CRITICAL" ? "biopsy" :
                contra.tier === "MODERATE" ? "warning" :
                contra.tier === "MINOR"    ? "neutral" :
                "safe"
              }
              size="sm"
            >
              {contra.tier}
            </ClinicalBadge>
          </div>
          <ContradictionMeter load={contra.overall_load} showCeiling />
          <p className="text-[11px] text-slate-500 mt-1.5 leading-relaxed">
            {CONTRADICTION_DESCRIPTIONS[contra.tier]}
          </p>
          {contra.escalation_triggered_by_contradiction && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-red-600">
              <AlertTriangle size={11} />
              Escalation was triggered by contradiction
            </div>
          )}
          {contra.n_contradicting_signals > 0 && (
            <div className="mt-1 text-[10px] text-slate-400 font-mono">
              {contra.n_contradicting_signals} contradicting signal{contra.n_contradicting_signals > 1 ? "s" : ""} detected
            </div>
          )}
        </div>

        {/* Recovery (if applicable) */}
        {stepReasoning.recovery_mechanism && (
          <div className="border-t border-slate-100 pt-3">
            <div className="flex items-center gap-1.5 mb-1.5">
              <Activity size={12} className="text-cyan-500" />
              <span className="text-[11px] font-semibold text-slate-700">
                Recovery Attempt
              </span>
              <ClinicalBadge
                variant={stepReasoning.recovery_successful ? "safe" : "biopsy"}
                size="sm"
              >
                {stepReasoning.recovery_successful ? "Successful" : "Failed"}
              </ClinicalBadge>
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              {RECOVERY_DESCRIPTIONS[stepReasoning.recovery_mechanism]}
            </p>
          </div>
        )}

        {/* FSM state */}
        <div className="border-t border-slate-100 pt-3">
          <div className="flex items-center gap-1.5 mb-1">
            <ArrowRight size={11} className="text-slate-400" />
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wide">
              {fsm.replace(/_/g, " ")}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            {FSM_STATE_DESCRIPTIONS[fsm]}
          </p>
        </div>

      </div>
    </div>
  );
}
