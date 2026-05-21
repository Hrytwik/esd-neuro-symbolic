# Reasoning Graph Engine Specification
## ReasoningGraphEngine — Computational Subsystem Architecture

**Document type:** Subsystem Architecture Reference  
**Subsystem:** `src/symbolic_engine/reasoning_graph.py` (backend) + `app/frontend/src/lib/graph/` (frontend)  
**Role:** Internal computational representation of the diagnostic reasoning process as a directed, weighted graph with temporal state evolution

---

## 1. Architectural Status: Internal Computational Subsystem

The Reasoning Graph Engine is **not a visualization layer**. It is the internal data structure through which the symbolic reasoning pipeline propagates evidence, certainty, and contradiction signals.

The inference algorithm IS the graph propagation algorithm. Every operation performed by the six symbolic reasoning stages corresponds to a graph operation:

| Reasoning Operation | Graph Operation |
|---|---|
| Feature grading | Feature node weight assignment |
| Rule activation | Support edge activation; rule node illumination |
| Contradiction detection | Contradiction node insertion; penalty edge addition |
| Certainty propagation | Forward propagation from rule → hypothesis nodes |
| Safety gate evaluation | Safety state node computation |
| Triage decision | Escalation edge traversal to triage node |

The graph is the authoritative in-memory representation of the reasoning state. The JSON trace is derived from the graph. The frontend visualization is derived from the graph's serialization. The graph is primary; everything else is a view of it.

---

## 2. Node Taxonomy

### 2.1 FeatureNode

**Represents:** A single clinical feature in the input vector.

```python
@dataclass
class FeatureNode:
    node_id: str                    # "F:koebner_phenomenon"
    node_type: str = "feature"
    feature_name: str               # from ClinicalFeatureRegistry
    raw_value: float                # 0–3 (ordinal) / 0–1 (binary) / float (age)
    graded_value: float             # [0.0, 1.0] after ClinicalGradingModule
    feature_type: str               # ordinal | binary | continuous
    is_critical: bool               # true for pathognomonic discriminators
    is_missing: bool                # true if not observed
    stage_set: int                  # stage at which this node was initialized (0)
```

**Graph role:** Source nodes (no incoming edges from other graph nodes). All propagation originates at FeatureNodes.

---

### 2.2 RuleNode

**Represents:** A diagnostic rule from the YAML rule base.

```python
@dataclass
class RuleNode:
    node_id: str                    # "R:PSO_001"
    node_type: str = "rule"
    rule_id: str
    disease_target: str
    evidence_tier: str              # A | B | C | D
    activation_score: float         # [0.0, 1.0] computed in Stage 1/2/4
    confidence_weight: float        # from YAML rule definition
    weighted_activation: float      # activation_score × confidence_weight
    status: str                     # dormant | activated | suppressed
    stage_activated: int | None     # stage at which activation was computed
    literature_source: str
```

**Graph role:** Intermediate nodes. Receive SupportEdges from FeatureNodes; emit WeightedActivationEdges to HypothesisNodes.

---

### 2.3 ContradictionNode

**Represents:** An active contradiction event — emerges dynamically during Stage 3.

```python
@dataclass
class ContradictionNode:
    node_id: str                    # "C:oral_mucosal_involvement→psoriasis"
    node_type: str = "contradiction"
    trigger_feature: str            # feature that triggered the contradiction
    trigger_value: float
    target_hypothesis: str          # disease being penalized
    competing_disease: str          # disease suggested by the trigger feature
    penalty: float                  # [0.0, 1.0] penalty magnitude
    clinical_rationale: str
    literature_source: str
    stage_emerged: int              # stage at which this node was created (3)
```

**Graph role:** Emergent nodes — created dynamically during Stage 3. Emit PenaltyEdges to HypothesisNodes.

---

### 2.4 HypothesisNode

**Represents:** A disease hypothesis with its current certainty score.

```python
@dataclass
class HypothesisNode:
    node_id: str                    # "H:psoriasis"
    node_type: str = "hypothesis"
    disease_name: str
    raw_score: float                # sum of weighted rule activations (pre-contradiction)
    penalized_score: float          # raw_score × (1 - contradiction_penalties)
    certainty: float                # softmax(penalized_score) at Stage 5
    rank: int                       # 1 = leading hypothesis
    active_rule_count: int
    active_contradiction_count: int
    stage_updated: int              # last stage that modified this node
```

**Graph role:** Central aggregation nodes. Receive WeightedActivationEdges (from RuleNodes) and PenaltyEdges (from ContradictionNodes). Emit CertaintyEdges to CertaintyNode.

---

### 2.5 CertaintyNode

**Represents:** The aggregate certainty state of the reasoning process at a given stage.

```python
@dataclass
class CertaintyNode:
    node_id: str = "CERT"
    node_type: str = "certainty"
    max_certainty: float
    certainty_gap: float
    ambiguity_index: float          # Shannon entropy (bits)
    contradiction_load: float
    stage_computed: int             # 5
    certainty_distribution: dict    # {disease: certainty} for all 6 diseases
```

**Graph role:** Singleton aggregation node. Receives CertaintyEdges from all HypothesisNodes; emits SafetyEdge to SafetyStateNode and EscalationEdge to BiopsyTriageNode.

---

### 2.6 SafetyStateNode

**Represents:** The state of the Clinical Safety Gate after Stage 5 evaluation.

```python
@dataclass
class SafetyStateNode:
    node_id: str = "SAFETY"
    node_type: str = "safety_state"
    invariant_results: dict         # {invariant_id: {status, value, threshold}}
    gate_results: dict              # {gate_id: {status, flag, cap_applied}}
    all_passed: bool
    flags_raised: list[str]
    pre_gate_recommendation: str
    post_gate_recommendation: str
    stage_evaluated: int            # 5
```

**Graph role:** Constraint node. Modifies the EscalationEdge weight or target based on gate outcomes.

---

### 2.7 BiopsyTriageNode

**Represents:** The terminal triage decision node.

```python
@dataclass
class BiopsyTriageNode:
    node_id: str = "TRIAGE"
    node_type: str = "biopsy_triage"
    recommendation: str             # SAFE_BIOPSY_FREE | MODERATE_CERTAINTY | AMBIGUOUS_CASE | BIOPSY_ADVISED
    leading_diagnosis: str
    diagnostic_state: str           # final FSM state
    stage_finalized: int            # 6
    triage_rationale: str
```

**Graph role:** Terminal sink node. No outgoing edges within the reasoning graph.

---

## 3. Edge Taxonomy

### 3.1 SupportEdge

**Direction:** FeatureNode → RuleNode  
**Semantics:** Feature provides supporting evidence for this rule's activation

```python
@dataclass
class SupportEdge:
    edge_id: str
    edge_type: str = "support"
    source: str         # FeatureNode.node_id
    target: str         # RuleNode.node_id
    feature_weight: float           # how much this feature contributes to rule activation
    condition_met: bool             # whether the feature's threshold condition was satisfied
    partial_activation: float       # [0.0, 1.0] — degree to which condition is met (fuzzy)
```

---

### 3.2 WeightedActivationEdge

**Direction:** RuleNode → HypothesisNode  
**Semantics:** Activated rule contributes certainty weight to disease hypothesis

```python
@dataclass
class WeightedActivationEdge:
    edge_id: str
    edge_type: str = "weighted_activation"
    source: str         # RuleNode.node_id
    target: str         # HypothesisNode.node_id
    activation_weight: float        # rule.weighted_activation
    stage: int                      # stage at which edge became active
    is_active: bool                 # only true if rule is activated
```

---

### 3.3 ContradictionEdge

**Direction:** FeatureNode → ContradictionNode  
**Semantics:** Feature triggered this contradiction event

```python
@dataclass
class ContradictionEdge:
    edge_id: str
    edge_type: str = "contradiction"
    source: str         # FeatureNode.node_id
    target: str         # ContradictionNode.node_id
    trigger_value: float
```

---

### 3.4 PenaltyEdge

**Direction:** ContradictionNode → HypothesisNode  
**Semantics:** Contradiction applies certainty penalty to hypothesis

```python
@dataclass
class PenaltyEdge:
    edge_id: str
    edge_type: str = "penalty"
    source: str         # ContradictionNode.node_id
    target: str         # HypothesisNode.node_id
    penalty_magnitude: float        # fraction of certainty removed
    is_active: bool
```

---

### 3.5 CertaintyEdge

**Direction:** HypothesisNode → CertaintyNode  
**Semantics:** Hypothesis contributes its certainty score to aggregate computation

```python
@dataclass
class CertaintyEdge:
    edge_id: str
    edge_type: str = "certainty"
    source: str         # HypothesisNode.node_id
    target: str = "CERT"
    hypothesis_certainty: float
```

---

### 3.6 SafetyEdge

**Direction:** CertaintyNode → SafetyStateNode  
**Semantics:** Certainty metrics feed into safety gate evaluation

```python
@dataclass
class SafetyEdge:
    edge_id: str
    edge_type: str = "safety"
    source: str = "CERT"
    target: str = "SAFETY"
```

---

### 3.7 EscalationEdge

**Direction:** SafetyStateNode → BiopsyTriageNode  
**Semantics:** Final triage determination after safety gating

```python
@dataclass
class EscalationEdge:
    edge_id: str
    edge_type: str = "escalation"
    source: str = "SAFETY"
    target: str = "TRIAGE"
    pre_gate_recommendation: str
    post_gate_recommendation: str
    gates_applied: list[str]
```

---

## 4. Graph State Model

The reasoning graph exists as a sequence of **temporal snapshots** — one per reasoning stage. Each snapshot is a complete description of the graph's state at that stage:

```python
@dataclass
class GraphSnapshot:
    snapshot_id: str                # "{case_id}_stage_{n}"
    case_id: str
    stage: int
    diagnostic_state: str           # DiagnosticState FSM value at this stage
    nodes: list[GraphNode]          # all node states at this stage
    edges: list[GraphEdge]          # all edge states at this stage
    timestamp: float                # Unix timestamp
    partial_certainty: dict         # {disease: score} at this stage (pre-Stage 5)
    activated_rule_count: int
    contradiction_count: int
    safety_flags: list[str]
```

The full trajectory for a case is a `CaseTrajectory`:

```python
@dataclass
class CaseTrajectory:
    case_id: str
    feature_vector: dict            # original input features
    snapshots: list[GraphSnapshot]  # ordered by stage (0–6)
    final_snapshot: GraphSnapshot   # Stage 6 terminal state
    total_stages: int
```

---

## 5. Propagation Algorithm

The graph propagation algorithm executes in stage order:

```python
class ReasoningGraphEngine:

    def __init__(self, rule_repository, contradiction_matrix):
        self.graph = ReasoningGraph()
        self.snapshots = []

    def initialize(self, feature_vector: dict) -> None:
        # Stage 0: Initialize FeatureNodes + RuleNodes
        # SupportEdges created from FeatureNodes → RuleNodes based on rule definitions
        # HypothesisNodes, CertaintyNode, SafetyStateNode, BiopsyTriageNode pre-created (dormant)
        ...

    def propagate_stage(self, stage: int) -> GraphSnapshot:
        if stage == 1:
            self._activate_tier_rules("A")
        elif stage == 2:
            self._activate_tier_rules("B")
        elif stage == 3:
            self._propagate_contradictions()
        elif stage == 4:
            self._activate_tier_rules("D")
        elif stage == 5:
            self._propagate_certainty()
            self._evaluate_safety_gate()
        elif stage == 6:
            self._finalize_triage()

        snapshot = self._take_snapshot(stage)
        self.snapshots.append(snapshot)
        return snapshot

    def _activate_tier_rules(self, tier: str) -> None:
        # For each rule node matching tier:
        #   compute activation from support edges
        #   update rule node status + activation_score
        #   activate WeightedActivationEdge to target HypothesisNode
        #   accumulate raw_score on HypothesisNode
        ...

    def _propagate_contradictions(self) -> None:
        # For each feature node with graded_value > threshold:
        #   check contradiction_matrix for this feature
        #   if contradiction active:
        #     create ContradictionNode
        #     create ContradictionEdge (feature → contradiction)
        #     create PenaltyEdge (contradiction → hypothesis)
        #     apply penalty to HypothesisNode.penalized_score
        #     accumulate contradiction_load on CertaintyNode
        ...

    def _propagate_certainty(self) -> None:
        # Softmax over all HypothesisNode.penalized_score values
        # Update HypothesisNode.certainty for each disease
        # Update CertaintyNode: max_certainty, gap, entropy, contradiction_load
        # Activate CertaintyEdges (all HypothesisNodes → CertaintyNode)
        ...

    def _evaluate_safety_gate(self) -> None:
        # Evaluate 3 invariants + 5 gates from ClinicalSafetyGate
        # Update SafetyStateNode
        # Modify EscalationEdge if gates triggered
        ...

    def _finalize_triage(self) -> None:
        # Map final diagnostic state → TriageRecommendation
        # Update BiopsyTriageNode
        # EscalationEdge becomes active (SAFETY → TRIAGE)
        ...

    def serialize(self) -> dict:
        # Convert full CaseTrajectory to JSON-serializable dict
        # Used by DiagnosticNarrativeGenerator and API response
        ...

    def get_graph_delta(self, stage: int) -> GraphDelta:
        # Returns only nodes/edges that changed at this stage
        # Used by WebSocket StageUpdate events (efficient frontend update)
        ...
```

---

## 6. JSON Serialization Format

Each GraphSnapshot serializes to a JSON object consumable by the frontend React Flow renderer:

```json
{
  "snapshot_id": "UCI_001_stage_3",
  "case_id": "UCI_001",
  "stage": 3,
  "diagnostic_state": "CONTRADICTION_EMERGED",
  "nodes": [
    {
      "id": "F:koebner_phenomenon",
      "type": "feature",
      "data": {"graded_value": 1.0, "is_critical": true, "status": "active"},
      "position": {"x": 0, "y": 120}
    },
    {
      "id": "R:PSO_001",
      "type": "rule",
      "data": {"activation_score": 0.85, "disease_target": "psoriasis", "tier": "A", "status": "activated"},
      "position": {"x": 300, "y": 120}
    },
    {
      "id": "C:oral_mucosal_involvement→psoriasis",
      "type": "contradiction",
      "data": {"penalty": 0.30, "target_hypothesis": "psoriasis", "competing": "lichen_planus"},
      "position": {"x": 300, "y": 350}
    },
    {
      "id": "H:psoriasis",
      "type": "hypothesis",
      "data": {"penalized_score": 2.17, "certainty": null, "rank": 1, "status": "leading"},
      "position": {"x": 600, "y": 200}
    }
  ],
  "edges": [
    {"id": "e:F:koebner→R:PSO_001", "source": "F:koebner_phenomenon", "target": "R:PSO_001", "type": "support", "data": {"active": true}},
    {"id": "e:R:PSO_001→H:psoriasis", "source": "R:PSO_001", "target": "H:psoriasis", "type": "weighted_activation", "data": {"weight": 0.72, "active": true}},
    {"id": "e:C:oral→H:psoriasis", "source": "C:oral_mucosal_involvement→psoriasis", "target": "H:psoriasis", "type": "penalty", "data": {"penalty": 0.30, "active": true}}
  ],
  "partial_certainty": {"psoriasis": 0.62, "lichen_planus": 0.22, "seborrheic_dermatitis": 0.08, ...}
}
```

The frontend's `graphBuilder.ts` converts this format directly into React Flow node/edge objects with visual properties applied via `graphAnimator.ts`.

---

## 7. GraphDelta Protocol (WebSocket Efficiency)

To minimize WebSocket payload size, each `StageUpdate` carries only a `GraphDelta` — the **incremental change** from the previous snapshot:

```typescript
interface GraphDelta {
  stage: number
  nodes_added: GraphNodeSerialized[]
  nodes_updated: { id: string; data_patch: Partial<NodeData> }[]
  edges_added: GraphEdgeSerialized[]
  edges_activated: string[]           // edge IDs transitioning to active
  edges_deactivated: string[]
}
```

The frontend `graphStore` applies GraphDeltas via `immer` patches to maintain React Flow's state efficiently.

---

## 8. Graph Layout Strategy

Node positions are computed by a deterministic layout algorithm (not force-directed) to ensure stable, readable structure:

```
Column 0 (x=0):    Feature nodes (12 nodes, y-spaced)
Column 1 (x=280):  Rule nodes (36–46 nodes, y-spaced by disease)
Column 1.5 (x=420): Contradiction nodes (emergent, between rule and hypothesis columns)
Column 2 (x=560):  Hypothesis nodes (6 nodes, evenly spaced)
Column 3 (x=740):  CertaintyNode (centered)
Column 3.5 (x=860): SafetyStateNode
Column 4 (x=980):  BiopsyTriageNode (centered)
```

Rule nodes are grouped vertically by disease target. Within each disease group, Tier A rules appear at top, Tier B below, Tier D at bottom.

This layout ensures the visual propagation direction (left → right) matches the inference direction and is readable without force-directed chaos.
