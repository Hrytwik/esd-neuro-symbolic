# Visual Reasoning Specification
## Publication-Grade Visualization Standards

**Document type:** Design Reference  
**Applies to:** `src/visualization/`, `app/frontend/src/components/`, `outputs/figures/`  
**Role:** Establishes the visual language for all system outputs — backend publication figures and frontend interface elements

---

## 1. Design Identity

All visualizations communicate one of three things:
1. **Reasoning structure** — how evidence propagates through the inference graph
2. **Certainty evolution** — how diagnostic confidence accumulates and is modified
3. **Clinical decision** — what the system concludes and why it is or is not safe

The visual language must feel like:
- Clinical research data visualization (cardiology, genomics, pharmacology style)
- Biomedical informatics publication standards
- Diagnostic imaging report aesthetics

The visual language must never feel like:
- A startup product dashboard
- A futuristic AI visualization
- A gamified prediction interface
- A corporate BI tool

No glowing nodes. No particle effects. No neon gradients. No robot iconography.

---

## 2. Color System

### 2.1 Clinical Color Palette

All colors derive from clinically grounded anchors. The palette is adapted from biomedical publication standards.

**Primary clinical colors:**

| Token | Hex | Usage |
|---|---|---|
| `clinical-blue-900` | `#0F2744` | Primary headings, section labels |
| `clinical-blue-700` | `#1A3A5C` | Feature nodes, data labels, borders |
| `clinical-blue-500` | `#2A5F8E` | Supporting elements, secondary actions |
| `clinical-teal-700` | `#1A5C55` | Active/activated elements, positive states |
| `clinical-teal-500` | `#2A7A6F` | Rule nodes (activated), confirmatory signals |
| `certainty-green-700` | `#1B4332` | High certainty (SAFE threshold), SAFE_BIOPSY_FREE |
| `certainty-green-500` | `#2D6A4F` | Strong positive evidence, above SAFE threshold |
| `certainty-amber-700` | `#92400E` | MODERATE_CERTAINTY state |
| `certainty-amber-400` | `#D97706` | Moderate states, approaching contradictions |
| `ambiguity-slate-600` | `#475569` | AMBIGUOUS_CASE, uncertain states |
| `ambiguity-slate-400` | `#94A3B8` | Low-certainty elements |
| `contradiction-red-900` | `#450A0A` | BIOPSY_ADVISED, critical safety alerts |
| `contradiction-red-700` | `#7F1D1D` | Contradiction nodes, active contradictions |
| `contradiction-red-500` | `#B91C1C` | Contradiction edges, penalty markers |
| `neutral-100` | `#F1F5F9` | Panel backgrounds |
| `neutral-200` | `#E2E8F0` | Dividers, borders |
| `neutral-50` | `#F8FAFC` | Page/canvas background |

**Triage color mapping:**

| Recommendation | Background | Text | Border |
|---|---|---|---|
| SAFE_BIOPSY_FREE | `#DCFCE7` | `#14532D` | `#2D6A4F` |
| MODERATE_CERTAINTY | `#FEF9C3` | `#713F12` | `#D97706` |
| AMBIGUOUS_CASE | `#F1F5F9` | `#334155` | `#475569` |
| BIOPSY_ADVISED | `#FEE2E2` | `#450A0A` | `#7F1D1D` |

### 2.2 Disease Color Codes

Consistent across all visualizations (certainty timelines, differential panels, confusion matrices):

| Disease | Color | Hex |
|---|---|---|
| Psoriasis | Deep blue | `#1E40AF` |
| Seborrheic Dermatitis | Warm teal | `#0F766E` |
| Lichen Planus | Dusty purple | `#6D28D9` |
| Pityriasis Rosea | Sage green | `#166534` |
| Chronic Dermatitis | Burnt sienna | `#9A3412` |
| Pityriasis Rubra Pilaris | Deep slate | `#1E293B` |

---

## 3. Typography

| Element | Font | Weight | Size | Notes |
|---|---|---|---|---|
| Page title | IBM Plex Sans | 600 | 24px | No uppercase |
| Panel header | IBM Plex Sans | 500 | 14px | 0.08em letter-spacing |
| Section label | IBM Plex Sans | 400 | 12px | Uppercase, 0.12em spacing |
| Body text | IBM Plex Sans | 400 | 14px | Line-height 1.6 |
| Clinical label | IBM Plex Sans | 500 | 13px | Used for feature/disease names |
| Metric value | IBM Plex Mono | 500 | 16px | All certainty percentages, scores |
| Trace text | IBM Plex Mono | 400 | 12px | Reasoning trace, JSON |
| Citation | IBM Plex Sans | 400 | 11px | Italic, `#64748B` |
| Alert text | IBM Plex Sans | 600 | 13px | Safety alerts, warnings |

No decorative fonts. No script typefaces. Clinical context demands legibility and precision.

---

## 4. Visualization Specifications

### 4.1 Activation Propagation Graph

**Type:** Directed graph (React Flow — frontend; NetworkX export — publication)  
**Purpose:** Show evidence flow from feature observations through rule activations to disease hypotheses and triage decision

**Publication figure (`outputs/figures/graph_activation.pdf`):**
- Format: PDF vector (for publication); PNG 300 DPI (for submission)
- Size: Full page (A4) or two-column figure (86mm × 120mm)
- Layout: Left-to-right, 4-column structure (see Reasoning Graph Engine spec)
- Node styling: Circles with fill opacity proportional to activation/certainty
- Edge styling: Directed arrows; line weight proportional to edge weight; dashed for dormant
- Color: Per taxonomy (section 2)
- Annotations: Rule IDs, disease names, activation scores
- Legend: Node types (feature/rule/contradiction/hypothesis/triage) + edge types
- Caption format: "Activation propagation graph for Case [ID]. Feature observations (left) propagate through diagnostic rules to disease hypotheses (center-right). Contradiction events (red nodes) apply penalty edges to the affected hypothesis..."

---

### 4.2 Disease Confusion Heatmap

**Type:** Heatmap (Seaborn / Matplotlib)  
**Purpose:** Show prediction confusion patterns across six diseases for Models A, B, C

**Specification:**
- 6×6 matrix per model; three heatmaps side by side for A/B/C comparison
- Colormap: Sequential blue (0 → 1); diverging red-blue for A-vs-C difference map
- Diagonal: True positive rates (recall per class)
- Off-diagonal: Confusion fractions (normalized by true class)
- Annotation: Per-cell count and percentage (font: IBM Plex Mono, 10pt)
- Border: Thick border around known confusion pairs (psoriasis/LP, LP/PR, etc.)
- X-axis: Predicted disease (abbreviated); Y-axis: True disease
- Title: "Confusion Profile — Model [A/B/C] — [feature set]"
- Size: 6in × 6in per panel; 18in × 6in for three-panel comparison

---

### 4.3 Certainty Evolution Timeline

**Type:** Multi-line chart (Recharts — frontend; Matplotlib — publication)  
**Purpose:** Show how disease certainty evolves across six reasoning stages for a single case

**Specification:**
- X-axis: Reasoning stages (0–6), labeled with clinical names (not "Stage N")
  - 0: Feature Grading | 1: Pathognomonic | 2: Supportive | 3: Contradiction | 4: Discrimination | 5: Stabilization | 6: Triage
- Y-axis: Certainty [0.0 → 1.0], labeled as percentage
- 6 lines (one per disease, disease-coded colors from section 2.2)
- Threshold annotations: SAFE=0.82, MODERATE=0.65, AMBIGUOUS=0.45 (dashed horizontal lines, labeled)
- Contradiction event markers: Red ▼ symbols on the affected disease line at Stage 3
- Shaded region: Certainty gap between rank-1 and rank-2 lines (light gray fill)
- Safety gate marker: Star symbol at Stage 5 if gates triggered
- Font: IBM Plex Mono for axis tick labels; IBM Plex Sans for line labels
- Legend: Disease names with colored swatches (right of chart)
- Size (publication): 7in × 4in

---

### 4.4 Diagnostic Tension Map

**Type:** Force-directed bubble chart (D3.js — frontend; custom Matplotlib — publication)  
**Purpose:** Visualize pairwise diagnostic tension between competing disease hypotheses at Stage 4

**Specification:**
- Six disease bubbles; bubble radius proportional to certainty score
- Edges between bubbles represent diagnostic tension (edge thickness ∝ tension strength)
- Thicker edges for known confusion pairs
- Bubble position: Determined by pairwise similarity (more confused diseases are closer)
- Color: Disease-coded (section 2.2); fill opacity proportional to certainty
- Contradiction markers: Small red circle on bubble if contradiction is active for that disease
- Annotations: Disease name + certainty percentage inside bubble
- This visualization is case-specific (one per case at Stage 4)

---

### 4.5 Certainty Calibration Curves

**Type:** Reliability diagram (Matplotlib)  
**Purpose:** Assess whether the system's certainty scores are well-calibrated against empirical outcomes

**Specification:**
- X-axis: Mean predicted certainty (binned 0–1, 10 bins)
- Y-axis: Fraction of correct diagnoses within each bin (empirical accuracy)
- Perfect calibration diagonal: dashed gray line (y=x)
- Three calibration curves: Model A (blue), Model B (amber), Model C symbolic (green)
- Error bars: Wilson confidence intervals per bin
- Inset histogram: Distribution of predicted certainties for Model C
- Secondary panel: ECE (Expected Calibration Error) table for A, B, C
- Size: 6in × 5in

---

### 4.6 Contradiction Heatmap

**Type:** Symmetric heatmap (Seaborn)  
**Purpose:** Show the contradiction relationship structure across disease pairs and feature triggers

**Specification:**
- Two heatmaps:
  1. **Feature × Disease Contradiction Matrix:** Rows = 12 clinical features; Columns = 6 diseases; cells = maximum contradiction penalty if feature active against disease
  2. **Disease × Disease Tension Matrix:** Symmetric; cells = aggregate confusion difficulty score between each pair
- Colormap: White (no contradiction) → clinical red (maximum penalty)
- Annotations: Penalty values in cells (IBM Plex Mono, 9pt)
- Highlighted cells: Known confusion pairs marked with thick borders
- Size: 8in × 5in (feature×disease) / 4in × 4in (disease×disease)

---

### 4.7 Biopsy Dependency Analysis Chart

**Type:** Stacked bar chart + scatter overlay (Matplotlib)  
**Purpose:** Show how biopsy-dependency distributes across diseases and how Model C biopsy triage aligns with actual outcomes

**Specification:**
- X-axis: 6 diseases
- Y-axis (left): Per-disease classification F1 for Models A, B, C (three grouped bars)
- Y-axis (right): Biopsy triage rate for Model C (line overlay)
- Annotations: F1 values on bars; SAFE/MODERATE/AMBIGUOUS/ADVISED percentages per disease
- Reference lines: A=0.97 (approximate), B=0.86 (approximate Cipriano baseline)
- Color: Model A (clinical-blue-500), Model B (certainty-amber-400), Model C (certainty-green-500)

---

### 4.8 Rule Activation Pathway Diagram

**Type:** Sankey diagram (D3.js — frontend; custom Matplotlib — publication)  
**Purpose:** Show aggregate rule activation patterns across the UCI dataset — which features most commonly activate which rules for which diseases

**Specification:**
- Three columns: Features (left) → Rules (center) → Diseases (right)
- Flow width proportional to activation frequency across dataset
- Color coded by target disease
- Top 15 rules by total activation weight shown (all rules would create excessive density)
- Publication size: 10in × 7in

---

## 5. Animation Specifications (Frontend)

### 5.1 Rule Activation Animation

**Trigger:** Rule node transitions from dormant → activated  
**Duration:** 280ms  
**Easing:** `easeOut` cubic  
**Behavior:**
- Node border-color transitions from `neutral-200` → disease-appropriate teal
- Node scale: 1.0 → 1.15 → 1.0 (brief pulse)
- Node fill-opacity: 0.2 → 0.9
- Simultaneous: outgoing WeightedActivationEdge animates a directional pulse (SVG stroke-dashoffset animation)

### 5.2 Contradiction Emergence Animation

**Trigger:** ContradictionNode created at Stage 3  
**Duration:** 400ms  
**Easing:** `easeInOut`  
**Behavior:**
- Contradiction node fades in at `opacity: 0` → `opacity: 1`
- Node has red pulsing ring (Framer Motion `animate` prop cycling opacity 0.4→1.0, 1.2s loop)
- PenaltyEdge animates: appears as dashed red line drawing from contradiction node to hypothesis
- Affected HypothesisNode briefly contracts (scale 1.0 → 0.93 → 1.0, 300ms) and dims

### 5.3 Certainty Stabilization Animation

**Trigger:** DiagnosticState → CERTAINTY_STABILIZED  
**Duration:** 600ms  
**Easing:** Spring (stiffness 100, damping 20)  
**Behavior:**
- Leading HypothesisNode grows to final certainty-proportional size
- Competing HypothesisNodes shrink
- BiopsyTriageNode background transitions to recommendation color
- Triage text fades in

### 5.4 Safety Gate Trigger Animation

**Trigger:** Any safety invariant or gate triggers  
**Duration:** 500ms  
**Behavior:**
- SafetyStateNode border pulses amber/red (3 pulses, 200ms each)
- EscalationEdge color shifts from `clinical-teal-500` → `contradiction-red-700`
- BiopsyTriageNode transitions to BIOPSY_ADVISED red

### 5.5 Stage Replay Scrubbing

**Trigger:** Replay scrubber position changes  
**Duration:** 150ms per step  
**Behavior:**
- All node states lerp to snapshot values
- Edges appear/disappear based on snapshot edge set
- CertaintyTimeline scrubber position marker updates

---

## 6. Publication Figure Export Standards

All publication-quality figures generated by `src/visualization/` must meet these standards:

| Standard | Requirement |
|---|---|
| Resolution | 300 DPI minimum (600 DPI for heatmaps) |
| Format | PDF (vector) primary; PNG fallback |
| Color mode | RGB (for screens); CMYK profile embedded for print |
| Font embedding | All fonts embedded in PDF |
| Figure size | PLOS-compatible: max 6.5in width (single col: 3.27in) |
| Colorblind safety | All palettes tested against Deuteranopia and Protanopia simulations |
| Background | White (`#FFFFFF`) — no gray backgrounds in final figures |
| File naming | `fig_{type}_{model}_{descriptor}.pdf` (e.g., `fig_confusion_modelC_6class.pdf`) |
| DPI setting | `matplotlib.rcParams['figure.dpi'] = 300` |
| Font | `rcParams['font.family'] = 'IBM Plex Sans'` (or Liberation Sans fallback) |
| Output directory | `outputs/figures/` |

---

## 7. Accessibility Standards

All visualizations must be accessible:
- Color + shape encoding (never color alone) for distinguishing data series
- Minimum contrast ratio 4.5:1 for all text
- All interactive elements keyboard-navigable
- Alt text for all graph nodes in the frontend (screenreader label = clinical description)
- Colorblind-safe palette validated (no red/green alone for clinical decisions — always add shape/pattern)
