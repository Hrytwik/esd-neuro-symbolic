/**
 * components/advanced/AdvancedReasoningDrawer.tsx
 * =================================================
 * Full-screen slide-over drawer containing all advanced reasoning panels.
 * Tabs: Graph | Replay | Certainty | Contradictions | Rules
 *
 * This panel is only opened on explicit request — it is NOT the default
 * interface. Clinicians who want to inspect the internal reasoning engine
 * can open it here without it interfering with the simple clinical workflow.
 */

"use client";

import { useState, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ReasoningGraphViewer } from "@/components/graph/ReasoningGraphViewer";
import { TrajectoryReplayViewer } from "@/components/trajectory/TrajectoryReplayViewer";
import { CertaintyEvolutionPanel } from "@/components/panels/CertaintyEvolutionPanel";
import { ContradictionPanel } from "@/components/contradictions/ContradictionPanel";
import { SymbolicRulePanel } from "@/components/reasoning/SymbolicRulePanel";
import { clsx } from "clsx";
import { X, GitBranch, PlayCircle, TrendingUp, AlertTriangle, Zap } from "lucide-react";

// ─── Tabs ─────────────────────────────────────────────────────────────────────

type DrawerTab = "graph" | "replay" | "certainty" | "contradictions" | "rules";

const TABS: { id: DrawerTab; label: string; Icon: React.ElementType; description: string }[] = [
  {
    id: "graph",
    label: "Reasoning Graph",
    Icon: GitBranch,
    description: "Hypothesis network and evidence connections",
  },
  {
    id: "replay",
    label: "Trajectory Replay",
    Icon: PlayCircle,
    description: "Step-by-step reasoning progression",
  },
  {
    id: "certainty",
    label: "Certainty Analysis",
    Icon: TrendingUp,
    description: "Diagnostic certainty evolution per hypothesis",
  },
  {
    id: "contradictions",
    label: "Contradiction Analysis",
    Icon: AlertTriangle,
    description: "Conflicting evidence and competition margins",
  },
  {
    id: "rules",
    label: "Symbolic Rules",
    Icon: Zap,
    description: "Activated clinical criteria and their weights",
  },
];

// ─── Drawer ───────────────────────────────────────────────────────────────────

interface AdvancedReasoningDrawerProps {
  open:    boolean;
  onClose: () => void;
}

export function AdvancedReasoningDrawer({ open, onClose }: AdvancedReasoningDrawerProps) {
  const [activeTab, setActiveTab] = useState<DrawerTab>("graph");

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Prevent body scroll when open
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Dark overlay */}
          <motion.div
            key="overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
            onClick={onClose}
          />

          {/* Drawer panel */}
          <motion.div
            key="drawer"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 32, stiffness: 320 }}
            className="fixed inset-y-0 right-0 w-[90vw] max-w-5xl bg-white shadow-2xl z-50 flex flex-col"
          >
            {/* ── Drawer header ─────────────────────────────────────────────── */}
            <div className="shrink-0 h-14 border-b border-slate-200 flex items-center justify-between px-5 bg-white">
              <div>
                <div className="text-sm font-bold text-slate-800">Advanced Reasoning</div>
                <div className="text-[10px] text-slate-400">
                  Expert inspection mode — symbolic reasoning internals
                </div>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-slate-100 transition-colors text-slate-500 hover:text-slate-700"
                aria-label="Close advanced reasoning"
              >
                <X size={16} />
              </button>
            </div>

            {/* ── Tab bar ───────────────────────────────────────────────────── */}
            <div className="shrink-0 flex border-b border-slate-200 bg-slate-50 overflow-x-auto">
              {TABS.map(tab => {
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={clsx(
                      "flex items-center gap-2 px-4 py-3 text-xs font-medium whitespace-nowrap border-b-2 transition-colors",
                      active
                        ? "border-blue-600 text-blue-700 bg-white"
                        : "border-transparent text-slate-500 hover:text-slate-700 hover:bg-white/60"
                    )}
                  >
                    <tab.Icon size={13} />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* ── Tab description ───────────────────────────────────────────── */}
            <div className="shrink-0 px-5 py-2 border-b border-slate-100 bg-white">
              <p className="text-[11px] text-slate-400">
                {TABS.find(t => t.id === activeTab)?.description}
              </p>
            </div>

            {/* ── Tab content ───────────────────────────────────────────────── */}
            <div className="flex-1 min-h-0 overflow-hidden">
              {activeTab === "graph" && (
                <div className="h-full">
                  <ReasoningGraphViewer />
                </div>
              )}

              {activeTab === "replay" && (
                <div className="h-full overflow-hidden">
                  <TrajectoryReplayViewer />
                </div>
              )}

              {activeTab === "certainty" && (
                <div className="h-full overflow-y-auto">
                  <CertaintyEvolutionPanel />
                </div>
              )}

              {activeTab === "contradictions" && (
                <div className="h-full overflow-y-auto">
                  <ContradictionPanel />
                </div>
              )}

              {activeTab === "rules" && (
                <div className="h-full overflow-y-auto">
                  <SymbolicRulePanel />
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
