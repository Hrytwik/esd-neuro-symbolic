/**
 * components/trajectory/TrajectoryReplayViewer.tsx
 * ==================================================
 * Certainty evolution chart with playback controls.
 * Displays certainty and ambiguity trajectories over reasoning steps.
 */

"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { useReasoningStore } from "@/store/reasoning-store";
import { useReasoningState } from "@/hooks/useReasoningState";
import { useTrajectoryReplay } from "@/hooks/useTrajectoryReplay";
import { FSM_STATE_LABELS, REPLAY_EVENT_LABELS, REPLAY_EVENT_COLORS } from "@/types";
import { diseaseLabel } from "@/lib/clinical-language";
import { clsx } from "clsx";
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight } from "lucide-react";

// ─── Custom tooltip ───────────────────────────────────────────────────────────

function ReasoningTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string | number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-clinical-md p-2.5 text-xs">
      <div className="text-slate-500 mb-1.5 font-medium">Step {label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center justify-between gap-3 mb-0.5">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full inline-block" style={{ background: p.color }} />
            <span className="text-slate-600">{p.name}</span>
          </span>
          <span className="font-mono text-slate-800 tabular-nums">
            {p.name === "Certainty" ? `${(p.value * 100).toFixed(1)}%` : `${p.value.toFixed(2)} bits`}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function TrajectoryReplayViewer() {
  // Drive playback
  useTrajectoryReplay();

  const {
    currentStep,
    totalSteps,
    isPlaying,
    playbackSpeed,
    currentCaseRecord,
    play,
    pause,
    stepForward,
    stepBackward,
    setStep,
    setPlaybackSpeed,
  } = useReasoningStore();

  const { certaintySeries, currentEvent, eventsUpToStep } = useReasoningState();

  if (!currentCaseRecord) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400 text-xs">
        No case loaded — select a case to view the trajectory
      </div>
    );
  }

  const allStepData = currentCaseRecord.events
    .sort((a, b) => a.step - b.step)
    .map((e) => ({
      step: e.step,
      certainty: e.certainty,
      ambiguity: e.ambiguity_bits,
      label: REPLAY_EVENT_LABELS[e.event_type] ?? e.event_type,
      disease: e.leading_diagnosis,
    }));

  const visibleData = allStepData.filter((d) => d.step <= currentStep);

  const canPlay  = !isPlaying && currentStep < totalSteps - 1;
  const canBack  = currentStep > 0;
  const canFwd   = currentStep < totalSteps - 1;

  return (
    <div className="flex flex-col h-full px-4 py-2 gap-3">
      {/* Header */}
      <div className="flex items-center justify-between shrink-0">
        <div>
          <span className="text-xs font-semibold text-slate-700">
            Diagnostic Trajectory
          </span>
          {currentEvent && (
            <span className="ml-2 text-xs text-slate-400">
              — {REPLAY_EVENT_LABELS[currentEvent.event_type]}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-slate-400 font-mono">
            {currentStep + 1} / {totalSteps}
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={visibleData}
            margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              axisLine={false}
              tickLine={false}
              label={{ value: "Step", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#94a3b8" }}
            />
            <YAxis
              yAxisId="certainty"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              width={36}
            />
            <YAxis
              yAxisId="ambiguity"
              orientation="right"
              domain={[0, 3.5]}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v.toFixed(1)}`}
              width={28}
            />
            <Tooltip content={<ReasoningTooltip />} />

            {/* Certainty stabilisation threshold */}
            <ReferenceLine
              yAxisId="certainty"
              y={0.75}
              stroke="#10b981"
              strokeDasharray="6 4"
              strokeWidth={1}
              label={{ value: "Stabilisation", position: "right", fontSize: 9, fill: "#10b981" }}
            />

            {/* Escalation trigger threshold */}
            <ReferenceLine
              yAxisId="certainty"
              y={0.50}
              stroke="#dc2626"
              strokeDasharray="4 4"
              strokeWidth={1}
              label={{ value: "Escalation", position: "right", fontSize: 9, fill: "#dc2626" }}
            />

            <Line
              yAxisId="certainty"
              type="monotone"
              dataKey="certainty"
              name="Certainty"
              stroke="#2563eb"
              strokeWidth={2}
              dot={{ r: 3, fill: "#2563eb", strokeWidth: 0 }}
              activeDot={{ r: 5, fill: "#2563eb" }}
            />
            <Line
              yAxisId="ambiguity"
              type="monotone"
              dataKey="ambiguity"
              name="Ambiguity (bits)"
              stroke="#d97706"
              strokeWidth={1.5}
              strokeDasharray="5 3"
              dot={{ r: 2, fill: "#d97706", strokeWidth: 0 }}
              activeDot={{ r: 4, fill: "#d97706" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Timeline scrubber */}
      <div className="shrink-0">
        <div className="flex items-center gap-1 mb-2">
          {currentCaseRecord.events
            .sort((a, b) => a.step - b.step)
            .map((e) => (
              <button
                key={e.step}
                onClick={() => setStep(e.step)}
                className={clsx(
                  "flex-1 h-6 rounded text-[9px] font-medium transition-all duration-150 truncate px-0.5",
                  e.step <= currentStep
                    ? "opacity-100"
                    : "opacity-30",
                  e.step === currentStep
                    ? "ring-2 ring-blue-400 ring-offset-1"
                    : ""
                )}
                style={{
                  backgroundColor:
                    e.step <= currentStep
                      ? REPLAY_EVENT_COLORS[e.event_type] + "20"
                      : "#f1f5f9",
                  borderColor: REPLAY_EVENT_COLORS[e.event_type],
                  borderWidth: 1,
                  color: REPLAY_EVENT_COLORS[e.event_type],
                }}
                title={REPLAY_EVENT_LABELS[e.event_type]}
              >
                {e.step}
              </button>
            ))}
        </div>

        {/* Current event label */}
        {currentEvent && (
          <div className="text-center text-[11px] text-slate-500 mb-2">
            <span className="font-medium text-slate-700">
              {REPLAY_EVENT_LABELS[currentEvent.event_type]}
            </span>
            {" — "}
            <span>{diseaseLabel(currentEvent.leading_diagnosis)}</span>
            {" — "}
            <span className="font-mono text-blue-600">
              {(currentEvent.certainty * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>

      {/* Playback controls */}
      <div className="flex items-center justify-between shrink-0">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setStep(0)}
            disabled={!canBack}
            className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 transition-colors"
            title="Go to start"
          >
            <SkipBack size={14} className="text-slate-600" />
          </button>
          <button
            onClick={stepBackward}
            disabled={!canBack}
            className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 transition-colors"
            title="Previous step"
          >
            <ChevronLeft size={14} className="text-slate-600" />
          </button>
          <button
            onClick={isPlaying ? pause : play}
            disabled={!canPlay && !isPlaying}
            className={clsx(
              "p-1.5 rounded transition-colors",
              isPlaying
                ? "bg-blue-600 hover:bg-blue-700 text-white"
                : "bg-blue-50 hover:bg-blue-100 text-blue-700",
              !canPlay && !isPlaying && "opacity-40"
            )}
            title={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button
            onClick={stepForward}
            disabled={!canFwd}
            className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 transition-colors"
            title="Next step"
          >
            <ChevronRight size={14} className="text-slate-600" />
          </button>
          <button
            onClick={() => setStep(totalSteps - 1)}
            disabled={!canFwd}
            className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 transition-colors"
            title="Go to end"
          >
            <SkipForward size={14} className="text-slate-600" />
          </button>
        </div>

        {/* Speed selector */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-slate-400 mr-1">Speed</span>
          {([2000, 1000, 500] as const).map((ms) => (
            <button
              key={ms}
              onClick={() => setPlaybackSpeed(ms)}
              className={clsx(
                "px-1.5 py-0.5 text-[10px] rounded transition-colors",
                playbackSpeed === ms
                  ? "bg-blue-100 text-blue-700 font-semibold"
                  : "text-slate-400 hover:bg-slate-100"
              )}
            >
              {ms === 2000 ? "0.5×" : ms === 1000 ? "1×" : "2×"}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
