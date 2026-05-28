/**
 * components/clinical/ClinicalResultsView.tsx
 * =============================================
 * Simplified, doctor-facing results view.
 * Shows: primary diagnosis, confidence, biopsy recommendation,
 * alternatives, supporting findings, contradiction summary, and
 * a plain-language clinical interpretation.
 * Advanced reasoning is accessible via a single disclosure button.
 */

"use client";

import { type ReasoningOutput } from "@/types";
import { type ReplayCaseRecord } from "@/types";
import {
  confidenceLevel,
  contradictionSummaryText,
  clinicalInterpretation,
  biopsyRationale,
  extractKeyFindings,
} from "@/lib/clinical-interpreter";
import { diseaseLabel } from "@/lib/clinical-language";
import { formatCertainty } from "@/lib/reasoning-utils";
import { clsx } from "clsx";
import {
  CheckCircle,
  AlertTriangle,
  ChevronRight,
  ArrowLeft,
  BookOpen,
  Microscope,
} from "lucide-react";

// ─── Confidence dots ──────────────────────────────────────────────────────────

function ConfidenceDots({ filled, tier }: { filled: number; tier: string }) {
  const color =
    tier === "high"         ? "bg-emerald-500" :
    tier === "moderate"     ? "bg-amber-400"   :
    tier === "low"          ? "bg-orange-400"  :
    "bg-red-400";

  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map(i => (
        <div
          key={i}
          className={clsx(
            "w-2.5 h-2.5 rounded-full transition-colors",
            i <= filled ? color : "bg-slate-200"
          )}
        />
      ))}
    </div>
  );
}

// ─── Biopsy banner ────────────────────────────────────────────────────────────

function BiopsyBanner({ required, rationale }: { required: boolean; rationale: string }) {
  return (
    <div className={clsx(
      "rounded-xl border p-4 flex items-start gap-3",
      required
        ? "bg-red-50 border-red-200"
        : "bg-emerald-50 border-emerald-200"
    )}>
      <div className={clsx(
        "w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5",
        required ? "bg-red-100" : "bg-emerald-100"
      )}>
        {required
          ? <Microscope size={15} className="text-red-600" />
          : <CheckCircle size={15} className="text-emerald-600" />}
      </div>
      <div>
        <div className={clsx(
          "text-sm font-bold",
          required ? "text-red-700" : "text-emerald-700"
        )}>
          {required ? "Biopsy Recommended" : "Biopsy Not Required"}
        </div>
        <div className={clsx(
          "text-[12px] mt-0.5 leading-relaxed",
          required ? "text-red-600" : "text-emerald-600"
        )}>
          {rationale}
        </div>
      </div>
    </div>
  );
}

// ─── Alternative diagnosis row ────────────────────────────────────────────────

function AlternativeRow({ disease, certainty, index }: {
  disease: string;
  certainty: number;
  index: number;
}) {
  return (
    <div className="flex items-center gap-3 py-2">
      <span className="text-[11px] font-mono text-slate-300 w-5 text-right shrink-0">
        #{index + 2}
      </span>
      <div className="flex-1 min-w-0">
        <div className="text-sm text-slate-600 truncate">
          {diseaseLabel(disease)}
        </div>
        <div className="mt-1 h-1 bg-slate-100 rounded-full">
          <div
            className="h-full bg-slate-300 rounded-full transition-all duration-500"
            style={{ width: `${certainty * 100}%` }}
          />
        </div>
      </div>
      <span className="text-xs font-mono text-slate-400 shrink-0 w-10 text-right">
        {formatCertainty(certainty)}
      </span>
    </div>
  );
}

// ─── Supporting finding row ───────────────────────────────────────────────────

function FindingRow({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2.5 py-1.5">
      <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 shrink-0" />
      <span className="text-sm text-slate-700 leading-snug">{text}</span>
    </div>
  );
}

// ─── Contradiction note ───────────────────────────────────────────────────────

function ContradictionNote({ tier, text }: {
  tier: "none" | "low" | "moderate" | "high";
  text: string;
}) {
  if (tier === "none" || tier === "low") {
    return (
      <div className="flex items-start gap-2 text-sm text-slate-500">
        <CheckCircle size={14} className="text-slate-300 mt-0.5 shrink-0" />
        <span>{text}</span>
      </div>
    );
  }
  return (
    <div className={clsx(
      "flex items-start gap-2.5 rounded-lg p-3 text-sm",
      tier === "high"
        ? "bg-red-50 border border-red-100 text-red-700"
        : "bg-amber-50 border border-amber-100 text-amber-700"
    )}>
      <AlertTriangle size={14} className="mt-0.5 shrink-0" />
      <span>{text}</span>
    </div>
  );
}

// ─── Main view ────────────────────────────────────────────────────────────────

interface ClinicalResultsViewProps {
  reasoning:    ReasoningOutput;
  caseRecord:   ReplayCaseRecord;
  onBack:       () => void;
  onAdvanced:   () => void;
}

export function ClinicalResultsView({
  reasoning,
  caseRecord,
  onBack,
  onAdvanced,
}: ClinicalResultsViewProps) {

  const conf         = confidenceLevel(reasoning.certainty);
  const altDiff      = [...reasoning.differential]
    .sort((a, b) => b.certainty - a.certainty)
    .filter(d => !d.is_leading)
    .slice(0, 3);
  const findings     = extractKeyFindings(reasoning);
  const contra       = contradictionSummaryText(reasoning.contradiction.tier);
  const interpretation = clinicalInterpretation(reasoning);
  const biopsyNote   = biopsyRationale(reasoning);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-700 transition-colors"
          >
            <ArrowLeft size={15} />
            New Assessment
          </button>
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-blue-900 flex items-center justify-center">
              <span className="text-[10px] font-bold text-white">Rx</span>
            </div>
            <span className="text-sm font-bold text-slate-800 tracking-tight">CASDRE</span>
          </div>
          <div className="text-[10px] text-slate-300 font-mono">v1.0.0</div>
        </div>
      </header>

      {/* ── Main content ───────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-8 space-y-6">

          {/* Page title */}
          <div>
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-1">
              Diagnostic Assessment
            </div>
            <h1 className="text-xl font-bold text-slate-800">
              Assessment Complete
            </h1>
            <p className="text-sm text-slate-500 mt-1">
              Based on the entered clinical findings
            </p>
          </div>

          {/* ── Primary diagnosis card ──────────────────────────────────────── */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">
              Primary Diagnosis
            </div>
            <div className="text-3xl font-bold text-slate-800 tracking-tight mb-4">
              {diseaseLabel(reasoning.leading_diagnosis)}
            </div>

            {/* Confidence */}
            <div className="flex items-center gap-3">
              <ConfidenceDots filled={conf.dots} tier={conf.tier} />
              <div>
                <span className={clsx(
                  "text-sm font-semibold",
                  conf.tier === "high"         ? "text-emerald-600" :
                  conf.tier === "moderate"     ? "text-amber-600"   :
                  conf.tier === "low"          ? "text-orange-600"  :
                  "text-red-600"
                )}>
                  {conf.label}
                </span>
                <span className="text-sm text-slate-400 ml-1.5">
                  ({formatCertainty(reasoning.certainty)})
                </span>
              </div>
            </div>
            <div className="text-xs text-slate-400 mt-1">{conf.sublabel}</div>
          </div>

          {/* ── Two-column: biopsy + alternatives ──────────────────────────── */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">

            {/* Biopsy */}
            <BiopsyBanner required={reasoning.requires_biopsy} rationale={biopsyNote} />

            {/* Alternatives */}
            {altDiff.length > 0 && (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
                  Alternative Diagnoses
                </div>
                <div className="divide-y divide-slate-100">
                  {altDiff.map((d, i) => (
                    <AlternativeRow
                      key={d.disease}
                      disease={d.disease}
                      certainty={d.certainty}
                      index={i}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* ── Supporting findings ─────────────────────────────────────────── */}
          {findings.length > 0 && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">
                Key Supporting Findings
              </div>
              <div className="divide-y divide-slate-50">
                {findings.map((f, i) => (
                  <FindingRow key={i} text={f} />
                ))}
              </div>
            </div>
          )}

          {/* ── Contradiction summary ───────────────────────────────────────── */}
          {reasoning.contradiction.tier !== "NONE" && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">
                Conflicting Findings
              </div>
              <ContradictionNote tier={contra.severity} text={contra.text} />
            </div>
          )}

          {/* ── Clinical interpretation ─────────────────────────────────────── */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">
              Clinical Interpretation
            </div>
            <p className="text-sm text-slate-700 leading-relaxed">
              {interpretation}
            </p>
          </div>

          {/* ── Advanced reasoning link ─────────────────────────────────────── */}
          <div className="border-t border-slate-200 pt-5">
            <button
              onClick={onAdvanced}
              className="group w-full flex items-center justify-between px-5 py-4 bg-white rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50/50 shadow-sm transition-all"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center group-hover:bg-blue-100 transition-colors">
                  <BookOpen size={15} className="text-blue-600" />
                </div>
                <div className="text-left">
                  <div className="text-sm font-semibold text-slate-700 group-hover:text-blue-700 transition-colors">
                    View Advanced Reasoning
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Reasoning graph · Trajectory · Symbolic activations · Contradiction analysis
                  </div>
                </div>
              </div>
              <ChevronRight size={16} className="text-slate-300 group-hover:text-blue-400 transition-colors" />
            </button>
          </div>

          <div className="h-4" />
        </div>
      </main>
    </div>
  );
}
