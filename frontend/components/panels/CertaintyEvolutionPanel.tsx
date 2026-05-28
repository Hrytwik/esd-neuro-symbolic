/**
 * components/panels/CertaintyEvolutionPanel.tsx
 * ===============================================
 * Right-panel certainty evolution: bar chart + differential ranking.
 */

"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
  Tooltip,
} from "recharts";
import { useReasoningState } from "@/hooks/useReasoningState";
import { useReasoningStore } from "@/store/reasoning-store";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { CertaintyBar } from "@/components/ui/CertaintyBar";
import { diseaseLabel, certaintyClinicalLabel, ambiguityClinicalLabel } from "@/lib/clinical-language";
import { formatBits, certaintyCssClass } from "@/lib/reasoning-utils";
import { clsx } from "clsx";

const DISEASE_COLORS: Record<string, string> = {
  psoriasis:               "#2563eb",
  seborrheic_dermatitis:   "#7c3aed",
  lichen_planus:           "#0891b2",
  pityriasis_rosea:        "#059669",
  chronic_dermatitis:      "#d97706",
  pityriasis_rubra_pilaris:"#dc2626",
};

function DiffTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number; payload: { disease: string } }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const entry = payload[0];
  return (
    <div className="bg-white border border-slate-200 rounded shadow-clinical p-2 text-xs">
      <div className="font-medium text-slate-700">{diseaseLabel(entry.payload.disease)}</div>
      <div className="font-mono text-blue-600 mt-0.5">
        {(entry.value * 100).toFixed(1)}%
      </div>
    </div>
  );
}

export function CertaintyEvolutionPanel() {
  const { stepReasoning } = useReasoningState();
  const { currentCaseRecord } = useReasoningStore();

  if (!stepReasoning || !currentCaseRecord) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400 text-xs p-4 text-center">
        Load a case to view certainty evolution
      </div>
    );
  }

  const diff = [...stepReasoning.differential].sort((a, b) => b.certainty - a.certainty);

  const chartData = diff.map(d => ({
    disease: d.disease,
    label:   diseaseLabel(d.disease).split(" ").slice(0, 2).join(" "),
    certainty: d.certainty,
    is_leading: d.is_leading,
  }));

  return (
    <div className="flex flex-col h-full">
      <SectionHeader title="Diagnostic Certainty" size="sm" />

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">

        {/* Current certainty + ambiguity */}
        <div className="bg-slate-50 rounded-lg border border-slate-100 p-2.5 space-y-2">
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[11px] font-semibold text-slate-700">
                {diseaseLabel(stepReasoning.leading_diagnosis)}
              </span>
              <span className={clsx("text-xs font-mono font-bold", certaintyCssClass(stepReasoning.certainty))}>
                {(stepReasoning.certainty * 100).toFixed(1)}%
              </span>
            </div>
            <CertaintyBar value={stepReasoning.certainty} showLabel={false} height="lg" animate />
            <div className="text-[10px] text-slate-400 mt-1">
              {certaintyClinicalLabel(stepReasoning.certainty)}
            </div>
          </div>

          <div className="border-t border-slate-200 pt-2">
            <div className="flex justify-between items-center">
              <span className="text-[11px] text-slate-500">Residual ambiguity</span>
              <span className="text-[11px] font-mono text-amber-600">
                {formatBits(stepReasoning.ambiguity_bits)}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">
              {ambiguityClinicalLabel(stepReasoning.ambiguity_bits)}
            </div>
          </div>
        </div>

        {/* Differential bar chart */}
        <div>
          <div className="text-[11px] font-semibold text-slate-700 mb-2">
            Hypothesis Comparison
          </div>
          <div className="h-36">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 0, right: 4, bottom: 0, left: 0 }}
                barSize={10}
              >
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  tick={{ fontSize: 9, fill: "#94a3b8" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis
                  type="category"
                  dataKey="label"
                  tick={{ fontSize: 9, fill: "#64748b" }}
                  tickLine={false}
                  axisLine={false}
                  width={68}
                />
                <Tooltip content={<DiffTooltip />} />
                <Bar dataKey="certainty" radius={[0, 3, 3, 0]}>
                  {chartData.map((entry) => (
                    <Cell
                      key={entry.disease}
                      fill={DISEASE_COLORS[entry.disease] ?? "#94a3b8"}
                      opacity={entry.is_leading ? 1 : 0.45}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Ranked list */}
        <div>
          <div className="text-[11px] font-semibold text-slate-700 mb-2">
            Differential Ranking
          </div>
          <div className="space-y-1">
            {diff.map((entry, i) => (
              <div key={entry.disease} className="flex items-center gap-2">
                <span
                  className={clsx(
                    "text-[10px] font-mono w-4 shrink-0 text-right",
                    entry.is_leading ? "text-blue-600 font-bold" : "text-slate-300"
                  )}
                >
                  #{i + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <CertaintyBar
                    value={entry.certainty}
                    label={diseaseLabel(entry.disease)}
                    showLabel
                    height="sm"
                    animate
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
