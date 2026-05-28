/**
 * components/ui/ClinicalBadge.tsx
 * =================================
 * Status badges for the clinical workstation.
 */

import { clsx } from "clsx";

type Variant = "safe" | "biopsy" | "warning" | "neutral" | "active" | "muted";

interface ClinicalBadgeProps {
  variant: Variant;
  children: React.ReactNode;
  className?: string;
  size?: "sm" | "md";
}

const VARIANT_CLASSES: Record<Variant, string> = {
  safe:    "bg-emerald-50 text-emerald-700 border border-emerald-200",
  biopsy:  "bg-red-50 text-red-700 border border-red-200",
  warning: "bg-amber-50 text-amber-700 border border-amber-200",
  neutral: "bg-slate-100 text-slate-600 border border-slate-200",
  active:  "bg-blue-50 text-blue-700 border border-blue-200",
  muted:   "bg-slate-50 text-slate-400 border border-slate-100",
};

const SIZE_CLASSES = {
  sm: "px-1.5 py-0.5 text-[10px] font-medium tracking-wide",
  md: "px-2 py-1 text-xs font-medium tracking-wide",
};

export function ClinicalBadge({
  variant,
  children,
  className,
  size = "md",
}: ClinicalBadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded uppercase",
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
        className
      )}
    >
      {children}
    </span>
  );
}
