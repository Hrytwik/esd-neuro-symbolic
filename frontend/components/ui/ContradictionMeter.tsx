/**
 * components/ui/ContradictionMeter.tsx
 * =======================================
 * Visual meter for contradiction load with ceiling indicator.
 */

import { clsx } from "clsx";
import { contradictionBarClass, formatLoad } from "@/lib/reasoning-utils";

interface ContradictionMeterProps {
  load: number;     // [0, 0.40]
  label?: string;
  showCeiling?: boolean;
  className?: string;
}

export function ContradictionMeter({
  load,
  label,
  showCeiling = true,
  className,
}: ContradictionMeterProps) {
  const CEILING = 0.40;
  const pct = Math.round((load / CEILING) * 100);

  return (
    <div className={clsx("w-full", className)}>
      {(label || showCeiling) && (
        <div className="flex items-center justify-between mb-1">
          {label && (
            <span className="text-xs text-slate-500">{label}</span>
          )}
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-xs font-mono text-slate-700 tabular-nums">
              {formatLoad(load)}
            </span>
            {showCeiling && (
              <span className="text-[10px] text-slate-400 font-mono">/ 40</span>
            )}
          </div>
        </div>
      )}

      {/* Bar with ceiling marker */}
      <div className="relative w-full h-2 bg-slate-100 rounded-full overflow-visible">
        {/* Fill */}
        <div
          className={clsx(
            "h-full rounded-full transition-all duration-500 ease-out",
            contradictionBarClass(load)
          )}
          style={{ width: `${pct}%` }}
        />
        {/* Ceiling marker at 100% */}
        {showCeiling && (
          <div
            className="absolute top-[-2px] h-[calc(100%+4px)] w-px bg-slate-400"
            style={{ right: 0 }}
            title="Safety ceiling: 0.40"
          />
        )}
      </div>
    </div>
  );
}
