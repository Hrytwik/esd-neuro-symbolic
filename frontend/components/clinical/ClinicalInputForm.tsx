/**
 * components/clinical/ClinicalInputForm.tsx
 * ===========================================
 * Doctor-facing clinical assessment form.
 * Collects feature values via sliders (0–3) and presence toggles.
 */

"use client";

import { useState } from "react";
import {
  FEATURE_DEFINITIONS,
  FEATURE_CATEGORY_LABELS,
  defaultFeatureVector,
  type FeatureVector,
  type FeatureCategory,
} from "@/types/clinical-input";
import { clsx } from "clsx";
import { Activity, RotateCcw } from "lucide-react";

// ─── Severity labels ──────────────────────────────────────────────────────────

const SLIDER_LABELS = ["None", "Mild", "Moderate", "Severe"] as const;

const SLIDER_COLORS = [
  "bg-slate-200",   // 0 — none
  "bg-sky-300",     // 1 — mild
  "bg-amber-400",   // 2 — moderate
  "bg-red-400",     // 3 — severe
] as const;

// ─── Feature controls ─────────────────────────────────────────────────────────

function FeatureSlider({
  label,
  description,
  value,
  onChange,
}: {
  label: string;
  description: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div>
        <div className="text-sm font-medium text-slate-700">{label}</div>
        <div className="text-[11px] text-slate-400 mt-0.5">{description}</div>
      </div>
      <div>
        <input
          type="range"
          min={0}
          max={3}
          step={1}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-blue-600"
          style={{ background: `linear-gradient(to right, #2563eb ${(value / 3) * 100}%, #e2e8f0 ${(value / 3) * 100}%)` }}
        />
        <div className="flex justify-between mt-1">
          {SLIDER_LABELS.map((lbl, i) => (
            <span
              key={lbl}
              className={clsx(
                "text-[10px] transition-colors",
                value === i ? "font-semibold text-blue-600" : "text-slate-400"
              )}
            >
              {lbl}
            </span>
          ))}
        </div>
      </div>
      {/* Visual severity pip */}
      <div className="flex gap-1">
        {[1, 2, 3].map(pip => (
          <div
            key={pip}
            className={clsx(
              "h-1 flex-1 rounded-full transition-colors duration-200",
              value >= pip ? SLIDER_COLORS[pip] : "bg-slate-100"
            )}
          />
        ))}
      </div>
    </div>
  );
}

function FeatureToggle({
  label,
  description,
  value,
  onChange,
}: {
  label: string;
  description: string;
  value: number;
  onChange: (v: number) => void;
}) {
  const active = value > 0;
  return (
    <div className="space-y-1.5">
      <div>
        <div className="text-sm font-medium text-slate-700">{label}</div>
        <div className="text-[11px] text-slate-400 mt-0.5">{description}</div>
      </div>
      <button
        type="button"
        onClick={() => onChange(active ? 0 : 1)}
        className={clsx(
          "w-full flex items-center gap-2.5 px-3 py-2 rounded-lg border text-sm font-medium transition-all duration-150",
          active
            ? "bg-blue-50 border-blue-300 text-blue-700 shadow-sm"
            : "bg-slate-50 border-slate-200 text-slate-500 hover:border-slate-300 hover:bg-white"
        )}
      >
        {/* Toggle pill */}
        <div className={clsx(
          "relative w-8 h-4 rounded-full transition-colors duration-200 shrink-0",
          active ? "bg-blue-500" : "bg-slate-300"
        )}>
          <div className={clsx(
            "absolute top-0.5 w-3 h-3 bg-white rounded-full shadow transition-transform duration-200",
            active ? "translate-x-4" : "translate-x-0.5"
          )} />
        </div>
        <span>{active ? "Present" : "Absent"}</span>
      </button>
    </div>
  );
}

// ─── Form ─────────────────────────────────────────────────────────────────────

interface ClinicalInputFormProps {
  onSubmit: (features: FeatureVector) => void;
  loading?: boolean;
}

const CATEGORIES: FeatureCategory[] = [
  "primary",
  "secondary",
  "distribution",
  "history",
  "histological",
];

export function ClinicalInputForm({ onSubmit, loading = false }: ClinicalInputFormProps) {
  const [features, setFeatures] = useState<FeatureVector>(defaultFeatureVector());

  const setFeature = (key: string, value: number) =>
    setFeatures(prev => ({ ...prev, [key]: value }));

  const hasAnyInput = Object.values(features).some(v => v > 0);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-900 flex items-center justify-center shadow-sm">
              <span className="text-[11px] font-bold text-white tracking-tight">Rx</span>
            </div>
            <div>
              <div className="text-base font-bold text-slate-800 tracking-tight leading-none">
                CASDRE
              </div>
              <div className="text-[10px] text-slate-400 leading-none mt-0.5">
                Dermatological Reasoning Workstation
              </div>
            </div>
          </div>
          <div className="text-[10px] text-slate-300 font-mono">v1.0.0</div>
        </div>
      </header>

      {/* ── Form body ──────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-8">

          {/* Intro */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-slate-800 tracking-tight">
              Clinical Assessment
            </h1>
            <p className="text-slate-500 mt-2 text-sm leading-relaxed">
              Enter the clinical findings observed in this patient. The system will analyse
              the pattern and generate a differential diagnosis with supporting rationale.
            </p>
          </div>

          {/* Feature sections */}
          {CATEGORIES.map(cat => {
            const catFeatures = FEATURE_DEFINITIONS.filter(f => f.category === cat);
            return (
              <section key={cat} className="mb-8">
                <div className="flex items-center gap-2 mb-4">
                  <div className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                    {FEATURE_CATEGORY_LABELS[cat]}
                  </div>
                  <div className="flex-1 h-px bg-slate-200" />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  {catFeatures.map(feat => (
                    <div
                      key={feat.key}
                      className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm"
                    >
                      {feat.inputType === "slider" ? (
                        <FeatureSlider
                          label={feat.label}
                          description={feat.description}
                          value={features[feat.key] ?? 0}
                          onChange={v => setFeature(feat.key, v)}
                        />
                      ) : (
                        <FeatureToggle
                          label={feat.label}
                          description={feat.description}
                          value={features[feat.key] ?? 0}
                          onChange={v => setFeature(feat.key, v)}
                        />
                      )}
                    </div>
                  ))}
                </div>
              </section>
            );
          })}

          {/* Action bar */}
          <div className="sticky bottom-6 mt-4">
            <div className="bg-white border border-slate-200 rounded-2xl shadow-lg px-6 py-4 flex items-center justify-between gap-4">
              <div className="text-sm text-slate-500">
                {hasAnyInput
                  ? `${Object.values(features).filter(v => v > 0).length} findings entered`
                  : "Enter clinical findings above to begin"}
              </div>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setFeatures(defaultFeatureVector())}
                  disabled={!hasAnyInput || loading}
                  className={clsx(
                    "flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border transition-colors",
                    hasAnyInput && !loading
                      ? "border-slate-200 text-slate-500 hover:bg-slate-50"
                      : "border-slate-100 text-slate-300 cursor-not-allowed"
                  )}
                >
                  <RotateCcw size={13} />
                  Reset
                </button>
                <button
                  type="button"
                  onClick={() => onSubmit(features)}
                  disabled={!hasAnyInput || loading}
                  className={clsx(
                    "flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-all shadow-sm",
                    hasAnyInput && !loading
                      ? "bg-blue-700 text-white hover:bg-blue-800 active:scale-95"
                      : "bg-slate-200 text-slate-400 cursor-not-allowed"
                  )}
                >
                  <Activity size={14} />
                  Run Diagnostic Reasoning
                </button>
              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
