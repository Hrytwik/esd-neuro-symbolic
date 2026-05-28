/**
 * components/ui/SectionHeader.tsx
 * =================================
 * Consistent section header for all panels.
 */

import { clsx } from "clsx";

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
  size?: "sm" | "md";
}

export function SectionHeader({
  title,
  subtitle,
  badge,
  action,
  className,
  size = "md",
}: SectionHeaderProps) {
  return (
    <div
      className={clsx(
        "flex items-start justify-between",
        size === "sm" ? "px-3 py-2" : "px-4 py-3",
        "border-b border-slate-100",
        className
      )}
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <h2
            className={clsx(
              "font-semibold text-slate-800 tracking-tight",
              size === "sm" ? "text-xs" : "text-sm"
            )}
          >
            {title}
          </h2>
          {badge}
        </div>
        {subtitle && (
          <p className="text-xs text-slate-400 mt-0.5 truncate">{subtitle}</p>
        )}
      </div>
      {action && <div className="shrink-0 ml-2">{action}</div>}
    </div>
  );
}
