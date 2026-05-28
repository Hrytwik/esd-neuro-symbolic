/**
 * components/graph/ReasoningGraphViewer.tsx
 * ===========================================
 * React Flow visualization of the reasoning graph.
 * Renders disease hypotheses, clinical features, symbolic signals and their
 * connections (support, contradiction, competition, activation).
 */

"use client";

import { useCallback, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type NodeTypes,
  type Node,
  useNodesState,
  useEdgesState,
  type NodeMouseHandler,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useGraphSync, type DiseaseNodeData, type FeatureNodeData, type SignalNodeData } from "@/hooks/useGraphSync";
import { useReasoningStore } from "@/store/reasoning-store";
import { CertaintyBar } from "@/components/ui/CertaintyBar";
import { formatCertainty } from "@/lib/reasoning-utils";
import { diseaseLabel } from "@/lib/clinical-language";
import { clsx } from "clsx";

// ─── Disease node ─────────────────────────────────────────────────────────────

function DiseaseNode({ data }: { data: DiseaseNodeData }) {
  const rank = data.rank;
  const isLeading = data.is_leading;

  return (
    <div
      className={clsx(
        "rounded-lg border px-3 py-2 min-w-[160px] max-w-[190px] cursor-pointer",
        "transition-all duration-200",
        isLeading
          ? "border-blue-300 bg-blue-50 shadow-clinical-md"
          : "border-slate-200 bg-white shadow-clinical",
        data.isSelected && "ring-2 ring-blue-400"
      )}
    >
      <div className="flex items-center justify-between mb-1">
        <span
          className={clsx(
            "text-[10px] font-mono px-1 py-0.5 rounded",
            isLeading ? "bg-blue-100 text-blue-700" : "bg-slate-100 text-slate-400"
          )}
        >
          #{rank}
        </span>
        {isLeading && (
          <span className="text-[10px] text-blue-600 font-semibold">Leading</span>
        )}
      </div>
      <div className="text-xs font-semibold text-slate-800 leading-tight mb-1.5">
        {diseaseLabel(data.label.toLowerCase().replace(/\s+/g, "_"))}
      </div>
      <CertaintyBar value={data.certainty} height="sm" showLabel={false} animate />
      <div className="text-[10px] font-mono text-slate-500 mt-1 text-right">
        {formatCertainty(data.certainty)}
      </div>
    </div>
  );
}

// ─── Feature node ─────────────────────────────────────────────────────────────

function FeatureNode({ data }: { data: FeatureNodeData }) {
  return (
    <div
      className={clsx(
        "rounded border px-2.5 py-1.5 min-w-[120px] max-w-[150px] cursor-pointer",
        "transition-all duration-150",
        data.activated
          ? "border-slate-300 bg-white"
          : "border-slate-100 bg-slate-50 opacity-60",
        data.isSelected && "ring-2 ring-blue-300"
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] text-slate-600 font-medium leading-tight truncate">
          {data.label}
        </span>
        <span
          className={clsx(
            "text-[10px] font-mono px-1 rounded shrink-0",
            data.activated ? "bg-slate-200 text-slate-700" : "bg-slate-100 text-slate-400"
          )}
        >
          {data.value}/3
        </span>
      </div>
    </div>
  );
}

// ─── Signal node ──────────────────────────────────────────────────────────────

function SignalNode({ data }: { data: SignalNodeData }) {
  return (
    <div
      className={clsx(
        "rounded border px-2 py-1.5 min-w-[130px] max-w-[165px] cursor-pointer",
        "transition-all duration-150",
        data.activated
          ? data.pathognomonic
            ? "border-violet-300 bg-violet-50"
            : "border-purple-200 bg-purple-50"
          : "border-slate-100 bg-slate-50 opacity-50",
        data.isSelected && "ring-2 ring-violet-300"
      )}
    >
      <div className="flex items-start justify-between gap-1 mb-1">
        {data.pathognomonic && (
          <span className="text-[9px] font-semibold text-violet-600 uppercase tracking-wide">
            Pathognomonic
          </span>
        )}
        <span
          className={clsx(
            "text-[9px] font-mono px-1 rounded ml-auto shrink-0",
            data.activated ? "bg-purple-100 text-purple-700" : "bg-slate-100 text-slate-400"
          )}
        >
          {(data.weight * 100).toFixed(0)}%
        </span>
      </div>
      <span className="text-[10px] text-slate-700 leading-tight block">
        {data.label}
      </span>
      {data.rule && (
        <span className="text-[9px] font-mono text-slate-400 mt-0.5 block">
          {data.rule}
        </span>
      )}
    </div>
  );
}

// ─── Node type registry ───────────────────────────────────────────────────────

const NODE_TYPES: NodeTypes = {
  diseaseNode:  DiseaseNode  as unknown as NodeTypes[string],
  featureNode:  FeatureNode  as unknown as NodeTypes[string],
  signalNode:   SignalNode   as unknown as NodeTypes[string],
};

// ─── Main viewer ──────────────────────────────────────────────────────────────

export function ReasoningGraphViewer() {
  const { nodes: syncedNodes, edges: syncedEdges } = useGraphSync();
  const { selectNode, selectedNodeId, currentCaseRecord } = useReasoningStore();

  const [nodes, , onNodesChange] = useNodesState(syncedNodes);
  const [edges, , onEdgesChange] = useEdgesState(syncedEdges);

  // Sync nodes/edges from store when they change
  const prevCaseRef = useRef<string | null>(null);

  const onNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      selectNode(node.id === selectedNodeId ? null : node.id);
    },
    [selectNode, selectedNodeId]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  if (!currentCaseRecord) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400 text-sm">
        <div className="text-center">
          <div className="text-4xl mb-3">⬡</div>
          <div className="font-medium text-slate-500">No case loaded</div>
          <div className="text-xs text-slate-400 mt-1">
            Select a case to view the reasoning graph
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={syncedNodes}
        edges={syncedEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.12 }}
        minZoom={0.4}
        maxZoom={2.0}
        proOptions={{ hideAttribution: true }}
        className="bg-clinical-bg"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#e2e8f0"
        />
        <Controls
          showInteractive={false}
          className="shadow-clinical border border-slate-200 rounded-lg overflow-hidden"
        />
        <MiniMap
          nodeColor={(n) => {
            if (n.type === "diseaseNode")  return (n.data as DiseaseNodeData).is_leading ? "#2563eb" : "#cbd5e1";
            if (n.type === "featureNode")  return (n.data as FeatureNodeData).activated  ? "#64748b" : "#e2e8f0";
            if (n.type === "signalNode")   return (n.data as SignalNodeData).activated   ? "#7c3aed" : "#e2e8f0";
            return "#e2e8f0";
          }}
          maskColor="rgba(240,243,247,0.7)"
          className="border border-slate-200 rounded-lg shadow-clinical"
        />
      </ReactFlow>

      {/* Edge legend */}
      <div className="absolute bottom-3 right-3 bg-white border border-slate-100 rounded-lg shadow-clinical p-2.5 text-[10px]">
        <div className="text-[9px] text-slate-400 font-semibold uppercase tracking-wider mb-1.5">
          Evidence Links
        </div>
        {[
          { color: "#059669", label: "Supports", dashed: false },
          { color: "#dc2626", label: "Contradicts", dashed: true },
          { color: "#d97706", label: "Competes with", dashed: true },
          { color: "#7c3aed", label: "Activates", dashed: false },
          { color: "#0891b2", label: "Recovers via", dashed: false },
        ].map(({ color, label, dashed }) => (
          <div key={label} className="flex items-center gap-1.5 mb-0.5">
            <svg width="18" height="8" viewBox="0 0 18 8">
              <line
                x1="0" y1="4" x2="18" y2="4"
                stroke={color} strokeWidth="2"
                strokeDasharray={dashed ? "4 3" : undefined}
              />
            </svg>
            <span className="text-slate-600">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
