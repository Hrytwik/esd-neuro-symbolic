/**
 * hooks/useGraphSync.ts
 * =======================
 * Synchronises graph state with the current replay step.
 * Derives React Flow node and edge arrays from the graph snapshot.
 */

"use client";

import { useMemo } from "react";
import { useReasoningStore } from "@/store/reasoning-store";
import { useReasoningState } from "./useReasoningState";
import type { Node, Edge } from "@xyflow/react";
import { EDGE_TYPE_COLORS } from "@/types";

// ─── React Flow node data shapes ──────────────────────────────────────────────

export interface DiseaseNodeData {
  label: string;
  certainty: number;
  rank: number;
  is_leading: boolean;
  isSelected: boolean;
  [key: string]: unknown;
}

export interface FeatureNodeData {
  label: string;
  value: number;
  activated: boolean;
  isSelected: boolean;
  [key: string]: unknown;
}

export interface SignalNodeData {
  label: string;
  rule?: string;
  activated: boolean;
  weight: number;
  pathognomonic?: boolean;
  isSelected: boolean;
  [key: string]: unknown;
}

// ─── Layout constants ─────────────────────────────────────────────────────────

const DISEASE_X  = 520;
const FEATURE_X  = 80;
const SIGNAL_X   = 290;
const ROW_SPACING = 90;
const DISEASE_ROW_SPACING = 80;

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useGraphSync() {
  const { currentGraphSnapshot, selectedNodeId } = useReasoningStore();
  const { stepReasoning } = useReasoningState();

  const { nodes, edges } = useMemo(() => {
    if (!currentGraphSnapshot) return { nodes: [], edges: [] };

    const snap = currentGraphSnapshot;

    // Apply step-specific certainty values from stepReasoning to disease nodes
    const diseaseCertaintyMap: Record<string, number> = {};
    if (stepReasoning) {
      for (const entry of stepReasoning.differential) {
        diseaseCertaintyMap[entry.disease] = entry.certainty;
      }
    }

    // Position counters
    let diseaseRow = 0;
    let featureRow = 0;
    let signalRow  = 0;

    // Build RF nodes
    const rfNodes: Node[] = snap.nodes.map((n) => {
      let position = { x: 0, y: 0 };
      let type = "diseaseNode";
      let data: DiseaseNodeData | FeatureNodeData | SignalNodeData;

      const isSelected = n.node_id === selectedNodeId;

      if (n.node_type === "disease") {
        const disease = n.label.toLowerCase().replace(/\s+/g, "_");
        const cert = diseaseCertaintyMap[disease] ?? (n.properties.certainty as number) ?? 0;
        position = { x: DISEASE_X, y: diseaseRow * DISEASE_ROW_SPACING };
        diseaseRow++;
        type = "diseaseNode";
        data = {
          label: n.label,
          certainty: cert,
          rank: (n.properties.rank as number) ?? 0,
          is_leading: (n.properties.is_leading as boolean) ?? false,
          isSelected,
        };
      } else if (n.node_type === "feature") {
        position = { x: FEATURE_X, y: featureRow * ROW_SPACING + 20 };
        featureRow++;
        type = "featureNode";
        data = {
          label: n.label,
          value: (n.properties.value as number) ?? 0,
          activated: (n.properties.activated as boolean) ?? false,
          isSelected,
        };
      } else if (n.node_type === "signal") {
        position = { x: SIGNAL_X, y: signalRow * ROW_SPACING + 20 };
        signalRow++;
        type = "signalNode";
        data = {
          label: n.label,
          rule: (n.properties.rule as string) ?? "",
          activated: (n.properties.activated as boolean) ?? false,
          weight: (n.properties.weight as number) ?? 0,
          pathognomonic: (n.properties.pathognomonic as boolean) ?? false,
          isSelected,
        };
      } else {
        // Fallback
        position = { x: 0, y: 0 };
        type = "featureNode";
        data = {
          label: n.label,
          value: 0,
          activated: false,
          isSelected,
        };
      }

      return {
        id: n.node_id,
        type,
        position,
        data,
        draggable: true,
      };
    });

    // Build RF edges
    const rfEdges: Edge[] = snap.edges.map((e) => {
      const color = EDGE_TYPE_COLORS[e.edge_type] ?? "#94a3b8";
      const dashed = e.edge_type === "contradicts" || e.edge_type === "competes_with";
      const animated = e.edge_type === "activates" || e.edge_type === "recovers_via";
      const strokeWidth = Math.max(1, Math.round(e.weight * 5));

      return {
        id: e.edge_id,
        source: e.source_id,
        target: e.target_id,
        type: "smoothstep",
        animated,
        style: {
          stroke: color,
          strokeWidth,
          strokeDasharray: dashed ? "6 4" : undefined,
          opacity: 0.85,
        },
        label: undefined,
        data: { edge_type: e.edge_type, weight: e.weight },
      };
    });

    return { nodes: rfNodes, edges: rfEdges };
  }, [currentGraphSnapshot, selectedNodeId, stepReasoning]);

  return { nodes, edges };
}
