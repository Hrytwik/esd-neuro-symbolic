/**
 * components/panels/FeatureInputPanel.tsx
 * =========================================
 * Left panel: clinical feature display for the current case.
 * Shows raw feature values and their 0–3 severity scores.
 */

"use client";

import { useReasoningStore } from "@/store/reasoning-store";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ClinicalBadge } from "@/components/ui/ClinicalBadge";
import { diseaseLabel } from "@/lib/clinical-language";
import { certaintyCssClass, formatCertainty } from "@/lib/reasoning-utils";
import { clsx } from "clsx";

// Clinical feature definitions with human-readable labels
const CLINICAL_FEATURES = [
  { key: "erythema",            label: "Erythema",              category: "Primary" },
  { key: "scaling",             label: "Scaling",               category: "Primary" },
  { key: "definite_borders",    label: "Definite Borders",      category: "Primary" },
  { key: "itching",             label: "Itching",               category: "Primary" },
  { key: "koebner_phenomenon",  label: "Koebner Phenomenon",    category: "Secondary" },
  { key: "polygonal_papules",   label: "Polygonal Papules",     category: "Secondary" },
  { key: "follicular_papules",  label: "Follicular Papules",    category: "Secondary" },
  { key: "oral_involvement",    label: "Oral Involvement",      category: "Secondary" },
  { key: "knee_elbow_involv",   label: "Knee/Elbow Involvement", category: "Distribution" },
  { key: "scalp_involvement",   label: "Scalp Involvement",     category: "Distribution" },
  { key: "family_history",      label: "Family History",        category: "History" },
  { key: "melanin_incontinence",label: "Melanin Incontinence",  category: "Histological" },
] as const;

const SEVERITY_LABELS: Record<number, string> = {
  0: "Absent",
  1: "Mild",
  2: "Moderate",
  3: "Severe",
};

function SeverityPips({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3].map((pip) => (
        <div
          key={pip}
          className={clsx(
            "w-2 h-2 rounded-full",
            value >= pip
              ? pip <= 1 ? "bg-slate-400"
                : pip <= 2 ? "bg-amber-400"
                : "bg-red-400"
              : "bg-slate-100"
          )}
        />
      ))}
    </div>
  );
}

export function FeatureInputPanel() {
  const { currentReasoning, currentCaseRecord } = useReasoningStore();

  if (!currentReasoning || !currentCaseRecord) {
    return (
      <div className="flex items-center justify-center h-full p-4 text-center">
        <div>
          <div className="text-slate-300 text-3xl mb-2">⬡</div>
          <div className="text-xs text-slate-500 font-medium">Select a case</div>
          <div className="text-[11px] text-slate-400 mt-1">
            Clinical features will appear here
          </div>
        </div>
      </div>
    );
  }

  // Extract feature values from symbolic signals (use signal names as proxy for now)
  // In production, this would come from the case feature vector
  const signals = currentReasoning.symbolic_signals;

  // Build a rough feature map from activated signals
  const featureActivation: Record<string, boolean> = {
    erythema:            signals.some(s => s.name.toLowerCase().includes("erythema") && s.activated),
    scaling:             signals.some(s => s.name.toLowerCase().includes("scaling") && s.activated),
    koebner_phenomenon:  signals.some(s => s.name.toLowerCase().includes("koebner") && s.activated),
    polygonal_papules:   signals.some(s => s.name.toLowerCase().includes("polygonal") && s.activated),
    follicular_papules:  signals.some(s => s.name.toLowerCase().includes("follicular") && s.activated),
    oral_involvement:    signals.some(s => s.name.toLowerCase().includes("oral") && s.activated),
    knee_elbow_involv:   signals.some(s => (s.name.toLowerCase().includes("knee") || s.name.toLowerCase().includes("elbow")) && s.activated),
    scalp_involvement:   signals.some(s => s.name.toLowerCase().includes("scalp") && s.activated),
    family_history:      signals.some(s => s.name.toLowerCase().includes("family") && s.activated),
  };

  const categories = [...new Set(CLINICAL_FEATURES.map(f => f.category))];

  return (
    <div className="flex flex-col h-full">
      <SectionHeader
        title="Clinical Findings"
        subtitle={`Case ${currentCaseRecord.case_id}`}
        size="sm"
        badge={
          currentReasoning.is_safe_triage ? (
            <ClinicalBadge variant="safe" size="sm">Safe</ClinicalBadge>
          ) : (
            <ClinicalBadge variant="biopsy" size="sm">Biopsy</ClinicalBadge>
          )
        }
      />

      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3">
        {/* Leading diagnosis summary */}
        <div className="bg-blue-50 border border-blue-100 rounded-lg p-2.5">
          <div className="text-[10px] font-semibold text-blue-500 uppercase tracking-wide mb-0.5">
            Leading Hypothesis
          </div>
          <div className="text-sm font-semibold text-blue-800">
            {diseaseLabel(currentReasoning.leading_diagnosis)}
          </div>
          <div
            className={clsx(
              "text-xs font-mono mt-0.5",
              certaintyCssClass(currentReasoning.certainty)
            )}
          >
            {formatCertainty(currentReasoning.certainty)}
          </div>
        </div>

        {/* Feature groups */}
        {categories.map(cat => (
          <div key={cat}>
            <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
              {cat}
            </div>
            <div className="space-y-1">
              {CLINICAL_FEATURES.filter(f => f.category === cat).map(feat => {
                const active = featureActivation[feat.key] ?? false;
                return (
                  <div
                    key={feat.key}
                    className={clsx(
                      "flex items-center justify-between py-1 px-2 rounded",
                      active ? "bg-white border border-slate-200" : "bg-slate-50"
                    )}
                  >
                    <span
                      className={clsx(
                        "text-[11px] truncate mr-2",
                        active ? "text-slate-700 font-medium" : "text-slate-400"
                      )}
                    >
                      {feat.label}
                    </span>
                    <SeverityPips value={active ? 2 : 0} />
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        {/* Case metadata */}
        <div className="border-t border-slate-100 pt-2 text-[10px] text-slate-400 space-y-0.5">
          <div className="flex justify-between">
            <span>True diagnosis</span>
            <span className="font-medium text-slate-500">
              {currentCaseRecord.true_label
                ? diseaseLabel(currentCaseRecord.true_label)
                : "—"}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Steps</span>
            <span className="font-mono">{currentCaseRecord.total_steps}</span>
          </div>
          <div className="flex justify-between">
            <span>Converged</span>
            <span className={currentCaseRecord.converged ? "text-emerald-500" : "text-amber-500"}>
              {currentCaseRecord.converged ? "Yes" : "No"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
