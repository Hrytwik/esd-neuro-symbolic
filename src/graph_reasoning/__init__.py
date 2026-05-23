"""
src/graph_reasoning — Graph-based reasoning visibility and trajectory infrastructure.

Represents the diagnostic symbolic reasoning process as a typed, weighted
directed graph that evolves stage-by-stage through the reasoning pipeline.

The graph layer does not re-execute reasoning — it structures the outputs of
completed PipelineResult / DiagnosticTrajectory instances into a graph model
that supports:

  · Reasoning trajectory replay (step-by-step graph reconstruction)
  · Contradiction cluster visualisation (competing hypothesis tension)
  · Certainty evolution tracking (per-disease certainty series)
  · Graph export to React Flow, Cytoscape, and plain JSON formats
  · Propagation path analysis (evidence → rule → hypothesis chains)
  · Temporal delta analysis (stage-to-stage quantitative change)

Module hierarchy
----------------
  graph_nodes       — Typed node dataclasses (evidence, hypothesis, contradiction,
                      escalation, stage, instability)
  graph_edges       — Typed edge dataclasses (reinforcement, contradiction,
                      suppression, escalation, propagation, trajectory)
  graph_snapshot    — Immutable point-in-time graph state capture
  reasoning_graph   — Core mutable graph with builder API
  trajectory_graph  — Ordered stage-by-stage trajectory as graph snapshots
  contradiction_graph — Contradiction-focused subgraph with tension clusters
  certainty_graph   — Certainty evolution and entropy series
  replay_engine     — Stepwise graph replay infrastructure
  graph_serializer  — Low-level dict serialization of graph components
  graph_exporter    — High-level JSON / React Flow / Cytoscape export

Primary entry points
--------------------
  ReasoningGraph.from_result(result)          → full graph from PipelineResult
  TrajectoryGraph.from_result(result)         → ordered snapshot trajectory
  ReplayEngine(trajectory_graph)              → stepwise replay controller
  GraphExporter.to_react_flow(graph)          → React Flow compatible dict
  GraphExporter.to_cytoscape(graph)           → Cytoscape.js compatible dict
  GraphExporter.to_json(graph, path)          → JSON file export
"""
