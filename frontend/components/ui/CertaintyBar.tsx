/**
 * components/ui/CertaintyBar.tsx
 * ================================
 * Horizontal certainty bar with colour gradient.
 */

import { clsx } from "clsx";
import { certaintyBarClass, formatCertainty } from "@/lib/reasoning-utils";

interface CertaintyBarProps {
  value: number;       // [0, 1]
  label?: string;
  showLabel?: boolean;
  height?: "sm" | "md" | "lg";
  className?: string;
  animate?: boolean;
}

const HEIGHT_CLASSES = {
  sm: "h-1",
  md: "h-1.5",
  lg: "h-2",
};

export function CertaintyBar({
  value,
  label,
  showLabel = true,
  height = "md",
  className,
  animate = true,
}: CertaintyBarProps) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);

  return (
    <div className={clsx("w-full", className)}>
      {(showLabel || label) && (
        <div className="flex items-center justify-between mb-1">
          {label && (
            <span className="text-xs text-slate-500 truncate mr-2">{label}</span>
          )}
          {showLabel && (
            <span className="text-xs font-mono text-slate-700 tabular-nums shrink-0">
              {formatCertainty(value)}
            </span>
          )}
        </div>
      )}
      <div className={clsx("w-full bg-slate-100 rounded-full overflow-hidden", HEIGHT_CLASSES[height])}>
        <div
          className={clsx(
            "h-full rounded-full",
            certaintyBarClass(value),
            animate && "transition-all duration-500 ease-out"
          )}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}
