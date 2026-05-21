# Subsystem Interaction Diagrams
## Symbolic Reasoning Flow, Certainty Evolution, Contradiction Propagation, Triage Transitions

**Document type:** Architectural Interaction Reference  
**Scope:** All symbolic engine subsystems and their runtime data exchange

---

## Diagram 1 — Top-Level System Data Flow

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        CLINICAL DATA INFRASTRUCTURE                          ║
║                                                                               ║
║   UCI Dataset ──▶ ClinicalDataLoader ──▶ ClinicalDataPreprocessor            ║
║                                │                                              ║
║                       ClinicalFeatureRegistry                                 ║
║                  (metadata: type, scale, biopsy_dependency)                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                 │
              ┌──────────────────┼──────────────────┐
              │ 12 clinical      │ 12 clinical       │ all 34
              │ features         │ features           │ features
              ▼                  ▼                   ▼
╔════════════╗  ╔═══════════════════════════════╗  ╔═════════════════╗
║  Model B   ║  ║   SYMBOLIC REASONING ENGINE   ║  ║    Model A      ║
║ Clinical   ║  ║   (standalone — primary)      ║  ║  Biopsy-Assist  ║
║ Baseline   ║  ║                               ║  ║  Reference      ║
╚════════════╝  ║  ┌─────────────────────────┐  ║  ╚═════════════════╝
       │        ║  │ DiagnosticRuleRepository │  ║         │
       │        ║  │ ClinicalRuleCompiler     │  ║         │
       │        ║  └──────────┬──────────────┘  ║         │
       │        ║             │ compiled rules    ║         │
       │        ║             ▼                  ║         │
       │        ║  ┌─────────────────────────┐  ║         │
       │        ║  │  ClinicalGradingModule  │  ║         │
       │        ║  │  (ordinal→fuzzy grade)  │  ║         │
       │        ║  └──────────┬──────────────┘  ║         │
       │        ║             │ graded features   ║         │
       │        ║             ▼                  ║         │
       │        ║  ┌─────────────────────────┐  ║         │
       │        ║  │DiagnosticEvidenceEval.  │  ║         │
       │        ║  │Stages 1, 2, 4 (A/B/D)  │  ║         │
       │        ║  └──────────┬──────────────┘  ║         │
       │        ║             │ activation scores ║         │
       │        ║             ▼                  ║         │
       │        ║  ┌─────────────────────────┐  ║         │
       │        ║  │DiagnosticConflictAnalyz.│  ║         │
       │        ║  │Stage 3                  │  ║         │
       │        ║  └──────┬──────────────────┘  ║         │
       │        ║         │ penalized scores      ║         │
       │        ║         │ contradiction trace   ║         │
       │        ║         ▼                      ║         │
       │        ║  ┌──────────────────────────┐ ║         │
       │        ║  │HypothesisCertaintyProp.  │ ║         │
       │        ║  │max_cert / gap / entropy  │ ║         │
       │        ║  └──────────┬───────────────┘ ║         │
       │        ║             │ certainty metrics ║         │
       │        ║             ▼                  ║         │
       │        ║  ┌─────────────────────────┐  ║         │
       │        ║  │  ClinicalSafetyGate     │  ║         │
       │        ║  │  (3 invariants, 5 gates)│  ║         │
       │        ║  └──────────┬──────────────┘  ║         │
       │        ║             │ cleared / flagged ║         │
       │        ║             ▼                  ║         │
       │        ║  ┌─────────────────────────┐  ║         │
       │        ║  │  ClinicalEscalationEng. │  ║         │
       │        ║  │  (biopsy triage output) │  ║         │
       │        ║  └──────────┬──────────────┘  ║         │
       │        ║             │                  ║         │
       │        ║  ┌──────────▼──────────────┐  ║         │
       │        ║  │DiagnosticNarrativeGen.  │  ║         │
       │        ║  │(JSON trace + clinician  │  ║         │
       │        ║  │ summary)                │  ║         │
       │        ║  └─────────────────────────┘  ║         │
       │        ╚═══════════════════════════════╝         │
       │                     │ symbolic scores             │
       │                     ▼                             │
       │        ╔═══════════════════════════════╗         │
       │        ║  StatisticalRefinementAdjunct  ║         │
       │        ║  (Model C hybrid — optional)   ║         │
       │        ╚═══════════════════════════════╝         │
       │                     │                             │
       └─────────────────────┼─────────────────────────────┘
                             ▼
        ╔═══════════════════════════════════════════════════╗
        ║              EVALUATION INFRASTRUCTURE            ║
        ║                                                   ║
        ║  DiagnosticPerformanceEvaluator                   ║
        ║  LowResourceRobustnessAnalyzer                    ║
        ║  CertaintyCalibrationAnalyzer                     ║
        ║  ComponentAblationStudy                           ║
        ║  DiseaseConfusionProfiler                         ║
        ║  BiopsyTriageValidator                            ║
        ╚═══════════════════════════════════════════════════╝
```

---

## Diagram 2 — Symbolic Engine Internal Flow (Detailed)

```
INPUT: 12 clinical features
{erythema:2, scaling:2, definite_borders:2, itching:3,
 koebner_phenomenon:1, polygonal_papules:0,
 follicular_papules:0, oral_mucosal_involvement:0,
 knee_and_elbow_involvement:1, scalp_involvement:1,
 family_history:1, age:35}
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              STAGE 0: ClinicalGradingModule             │
│                                                         │
│  Ordinal features (0–3) → fuzzy membership [0.0–1.0]:  │
│    erythema=2        → grade=0.67  (moderate)           │
│    scaling=2         → grade=0.67  (moderate)           │
│    definite_borders=2→ grade=0.67  (moderate)           │
│    itching=3         → grade=1.00  (severe)             │
│                                                         │
│  Binary features:    passthrough as-is                  │
│  Continuous (age):   percentile normalization           │
│                                                         │
│  Feature completeness score = 12/12 = 1.00             │
│  Missing features = []                                  │
│  DiagnosticStateTracker → S0 remains                   │
└────────────────────────────┬────────────────────────────┘
                             │ graded_features{}
                             ▼
┌─────────────────────────────────────────────────────────┐
│    STAGE 1: DiagnosticEvidenceEvaluator (Tier A)        │
│              Pathognomonic Rule Activation              │
│                                                         │
│  Rule PSO_001 (koebner=1):      activation=0.85 ✓      │
│  Rule LP_001 (polygonal_pap=0): activation=0.00         │
│  Rule PRP_001 (follicular=0):   activation=0.00         │
│                                                         │
│  Leading hypothesis: psoriasis (raw_score=0.85)         │
│  Activated rule count: 1                                │
│  DiagnosticStateTracker: S0 → S0 (still < 2 rules)     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│    STAGE 2: DiagnosticEvidenceEvaluator (Tier B)        │
│              Supportive Evidence Integration            │
│                                                         │
│  Rule PSO_002 (knee/elbow=1):   activation=0.80 ✓      │
│  Rule PSO_003 (scalp=1):        activation=0.75 ✓      │
│  Rule PSO_004 (family_hist=1):  activation=0.70 ✓      │
│  Rule SD_001 (scalp=1, koebn=0):activation=0.00         │
│                                                         │
│  Psoriasis raw_score = 0.85 + 0.80 + 0.75 + 0.70       │
│                       = 3.10 (weighted)                 │
│  Activated rule count: 4                                │
│  DiagnosticStateTracker: S0 → S1 → S2 (REINFORCING)   │
└────────────────────────────┬────────────────────────────┘
                             │ raw_scores{disease→score}
                             ▼
┌─────────────────────────────────────────────────────────┐
│    STAGE 3: DiagnosticConflictAnalyzer                  │
│              Contradiction Detection + Penalty          │
│                                                         │
│  Checking contradiction features for psoriasis:         │
│    oral_mucosal_involvement = 0 → no LP contradiction   │
│    follicular_papules = 0       → no PRP contradiction  │
│                                                         │
│  Contradiction features active: NONE                    │
│  Contradiction load: 0.00                               │
│  Penalized psoriasis score: 3.10 × (1.0 - 0.00) = 3.10 │
│                                                         │
│  Contradiction trace: []                                │
│  DiagnosticStateTracker: S2 remains (no contradiction)  │
└────────────────────────────┬────────────────────────────┘
                             │ penalized_scores{}
                             ▼
┌─────────────────────────────────────────────────────────┐
│    STAGE 4: DiagnosticEvidenceEvaluator (Tier D)        │
│              Pairwise Discriminator Activation          │
│                                                         │
│  Rule PSO_SD_001 (definite_borders=2, family=1):        │
│    → Psoriasis +0.20 vs. Seborrheic Dermatitis          │
│                                                         │
│  Rule PSO_LP_001 (koebner=1, oral_mucosal=0):           │
│    → Psoriasis +0.15 vs. Lichen Planus                  │
│                                                         │
│  Updated psoriasis score: 3.45                          │
│  Certainty gap (psoriasis vs. next): 0.71               │
│  DiagnosticStateTracker: S2 → S6 (CERTAINTY_STABILIZING)│
└────────────────────────────┬────────────────────────────┘
                             │ discriminated_scores{}
                             ▼
┌─────────────────────────────────────────────────────────┐
│    STAGE 5: HypothesisCertaintyPropagator               │
│             + ClinicalSafetyGate                        │
│                                                         │
│  Softmax over all 6 disease scores:                     │
│    psoriasis:           0.847                           │
│    lichen_planus:       0.061                           │
│    seborrheic_derm:     0.042                           │
│    chronic_derm:        0.028                           │
│    pityriasis_rosea:    0.014                           │
│    pityriasis_rubra_p:  0.008                           │
│                                                         │
│  max_certainty:         0.847                           │
│  certainty_gap:         0.786  (0.847 - 0.061)          │
│  ambiguity_index:       0.73 bits                       │
│  contradiction_load:    0.00                            │
│                                                         │
│  SAFETY GATE:                                           │
│    Invariant 1 (contradiction): load=0.00 ✓ PASS        │
│    Invariant 2 (evidence suff): 4 rules ✓ PASS          │
│    Invariant 3 (entropy):       0.73 < 1.5 ✓ PASS       │
│    Gate 1 (single-source):      no single rule > 0.60   │
│    Gate 2 (pathognomonic):      1 pathognomonic ✓ PASS  │
│    Gate 3 (missingness):        0 critical missing ✓    │
│    Gate 4 (confusion zone):     gap=0.786 >> 0.30 ✓    │
│    Gate 5 (overconfidence):     0.847 < 0.92 ✓ PASS     │
│                                                         │
│  All gates: PASS                                        │
│  DiagnosticStateTracker: S6 → S7 (CERTAINTY_STABILIZED) │
└────────────────────────────┬────────────────────────────┘
                             │ certainty_metrics + safety_flags
                             ▼
┌─────────────────────────────────────────────────────────┐
│    STAGE 6: ClinicalEscalationEngine                    │
│             + DiagnosticNarrativeGenerator              │
│                                                         │
│  State S7, max_certainty=0.847 >= 0.82 ✓                │
│  certainty_gap=0.786 >= 0.40 ✓                          │
│  contradiction_load=0.00 < 0.20 ✓                       │
│                                                         │
│  BIOPSY TRIAGE: SAFE_BIOPSY_FREE                        │
│  Leading diagnosis: Psoriasis                           │
│                                                         │
│  Narrative: "Psoriasis diagnosis supported by:          │
│    Koebner phenomenon [PSO_001, tier A],                │
│    knee/elbow involvement [PSO_002, tier B],            │
│    scalp involvement [PSO_003, tier B],                 │
│    family history [PSO_004, tier B].                    │
│    No contradicting signs detected.                     │
│    Certainty: 84.7%; Gap: 78.6%; Entropy: 0.73 bits.   │
│    All safety criteria met.                             │
│    Recommendation: SAFE_BIOPSY_FREE"                    │
└─────────────────────────────────────────────────────────┘

OUTPUT:
{
  "biopsy_triage": "SAFE_BIOPSY_FREE",
  "leading_diagnosis": "psoriasis",
  "max_certainty": 0.847,
  "certainty_gap": 0.786,
  "ambiguity_index": 0.73,
  "contradiction_load": 0.00,
  "diagnostic_state": "CERTAINTY_STABILIZED",
  "safety_flags": [],
  "disease_certainty": {
    "psoriasis": 0.847, "lichen_planus": 0.061,
    "seborrheic_dermatitis": 0.042, "chronic_dermatitis": 0.028,
    "pityriasis_rosea": 0.014, "pityriasis_rubra_pilaris": 0.008
  }
}
```

---

## Diagram 3 — Certainty Evolution Through Reasoning Stages

```
Disease Certainty (psoriasis) — trajectory through 6 stages

Certainty
1.00 │
     │                                          ╔═══════╗
0.85 │                                    ──────║ 0.847 ║ SAFE
     │                               ──── S6    ╚═══════╝
0.70 │                         ──────  (0.718)
     │                    ─────
0.55 │               ──── S2
     │          ─── (0.550)
0.40 │     ──── S1
     │  ─── (no             no contradiction in this case;
0.25 │       certainty)     scores rise unimpeded
     │                      through all stages
0.00 └──┬───────┬───────┬───────┬───────┬───────┬──────▶
       S0      S1      S2      S3      S4      S5      S6
    SPARSE  HYPO.  REINF.  CONTRA  DISC.  CERTAIN TRIAGE
             FORM          (none)
                    Stage →

------------------------------------------------------------
Contradicted Case (oral_mucosal_involvement = 1 also active)
------------------------------------------------------------

Certainty
1.00 │
     │
0.70 │         ──── S2        ╔═══════════╗
     │    ─── (0.68)   ──────▶║ CONTRA.   ║ penalty=0.30
0.50 │  ──                    ╚═══════════╝
     │                    ──── (0.48 post-penalty)
0.35 │                              ──── S4
     │                                    ──── S6
0.25 │                 (gap narrows; S4 tension)   ──── 0.58
     │                                              S7 MODERATE
0.00 └──┬───────┬───────┬───────┬───────┬───────┬──────▶
       S0      S1      S2      S3      S4      S5      S6

Lichen planus certainty (competing):
0.30 │               ──────── S3 (0.30; rises as psoriasis penalized)
     │          ──── (0.18)        ──── S4 (0.22)
     │    ────                               ──── S6 (0.18)
0.00 └──┬───────┬───────┬───────┬───────┬───────┬──────▶
```

---

## Diagram 4 — Contradiction Propagation Flow

```
Clinical Feature Vector
         │
         │ oral_mucosal_involvement = 1
         ▼
┌──────────────────────────────────────────────────────────┐
│            DiagnosticConflictAnalyzer                    │
│                                                          │
│  Contradiction matrix lookup:                            │
│                                                          │
│  oral_mucosal_involvement=1 triggers:                    │
│    → PSO penalty:  −0.30  (competes with lichen planus)  │
│    → PR penalty:   −0.25  (PR has no mucosal involvement)│
│    → SD penalty:   −0.15  (weak; some overlap possible)  │
│    → LP boost:     +0.20  (Wickham's striae indicator)   │
│                                                          │
│  follicular_papules=1 triggers (if active):              │
│    → PSO penalty:  −0.45  (PRP pathognomonic)            │
│    → LP penalty:   −0.30  (LP has no follicular papules) │
│    → PR penalty:   −0.35                                 │
│    → PRP boost:    +0.30                                 │
│                                                          │
│  koebner=1 + oral_mucosal=1 SIMULTANEOUS triggers:       │
│    → LP-PSO ambiguity flag                               │
│    → contradiction_load += 0.30 + 0.25 = 0.55           │
│    → SAFETY GATE TRIGGERED (load >= 0.40)                │
│    → S8: BIOPSY_ESCALATED                               │
│    → BIOPSY_ADVISED                                      │
└──────────────────────────────────────────────────────────┘
         │
         ▼
Contradiction Trace Entry:
{
  "stage": 3,
  "feature": "oral_mucosal_involvement",
  "value": 1,
  "affected_hypothesis": "psoriasis",
  "penalty_applied": 0.30,
  "competing_disease": "lichen_planus",
  "rationale": "Oral mucosal involvement (Wickham striae) is pathognomonic for LP;
                incompatible with isolated psoriasis diagnosis",
  "literature": "Le Cleach & Chosidow, NEJM 2012;366:723-732",
  "pre_penalty_certainty": 0.68,
  "post_penalty_certainty": 0.48
}
```

---

## Diagram 5 — Biopsy Triage Decision Logic

```
                    Final Certainty Metrics
                           │
              ┌────────────▼────────────┐
              │   ClinicalSafetyGate   │
              │   (3 invariants +       │
              │    5 gates)             │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────────────────────────┐
              │  Any gate TRIGGERED?                         │
              └────────────────────────────────────────────┘
                    YES │                NO │
                        ▼                  ▼
              ┌─────────────────┐  ┌───────────────────────────┐
              │ S8: BIOPSY      │  │ Check final state          │
              │ ESCALATED       │  └───────────────────────────┘
              │ → BIOPSY_ADVISED│           │
              └─────────────────┘    ┌──────▼──────┐
                                     │  S7?        │
                                     └──────┬──────┘
                                       YES  │  NO
                              ┌────────────┘  └─────────────┐
                              ▼                             ▼
                   ┌──────────────────┐         ┌──────────────────┐
                   │ max_cert >= 0.82 │         │  S5 / S4?        │
                   │ AND gap >= 0.40  │         └──────────────────┘
                   │ AND load < 0.20? │                   │
                   └──────────┬───────┘         ┌─────────▼────────┐
                         YES  │  NO             │ ambiguity_index   │
                              │   │             │ > 1.5?            │
                              ▼   ▼             └──────────────────┘
                   ┌──────────┐  ┌─────────┐     YES │      NO │
                   │  SAFE_   │  │MODERATE_│         ▼         ▼
                   │BIOPSY_   │  │CERTAINTY│  ┌──────────┐ ┌──────────┐
                   │ FREE     │  │         │  │ BIOPSY_  │ │AMBIGUOUS │
                   └──────────┘  └─────────┘  │ ADVISED  │ │  _CASE   │
                                              └──────────┘ └──────────┘
```

---

## Diagram 6 — Reasoning Trace Generation Flow

```
Per-Stage Trace Events
│
│  [Stage 0] ClinicalGradingModule logs:
│    - feature values + fuzzy grades
│    - completeness score
│    - missing feature warnings
│
│  [Stage 1] DiagnosticEvidenceEvaluator logs (Tier A):
│    - each rule evaluated (rule_id, activation, weight)
│    - initial hypothesis scores
│    - state transition event
│
│  [Stage 2] DiagnosticEvidenceEvaluator logs (Tier B):
│    - each supportive rule evaluated
│    - updated hypothesis scores
│    - state transition event
│
│  [Stage 3] DiagnosticConflictAnalyzer logs:
│    - each contradiction feature checked
│    - penalties applied (with rationale + literature)
│    - contradiction_load
│    - state transition event
│
│  [Stage 4] DiagnosticEvidenceEvaluator logs (Tier D):
│    - each discriminator evaluated
│    - certainty gap update
│    - state transition event
│
│  [Stage 5] HypothesisCertaintyPropagator + ClinicalSafetyGate logs:
│    - final certainty distribution
│    - all gate evaluations (pass / triggered)
│    - safety flags raised
│    - final state determination
│
│  [Stage 6] ClinicalEscalationEngine + DiagnosticNarrativeGenerator:
│    - triage recommendation
│    - clinician-readable summary (natural language)
│    - JSON audit trace
│
▼
DiagnosticNarrativeGenerator assembles:
┌──────────────────────────────────────────────────────────────┐
│                    JSON REASONING TRACE                       │
│  {case_id, feature_grades, rule_activations,                  │
│   contradiction_events, certainty_evolution,                  │
│   state_history, safety_gate_results,                         │
│   final_state, biopsy_triage, disease_certainty}              │
└──────────────────────────────────────────────────────────────┘
                              +
┌──────────────────────────────────────────────────────────────┐
│                 CLINICIAN-READABLE SUMMARY                    │
│                                                               │
│  DIAGNOSTIC REASONING REPORT                                 │
│  Case ID: UCI_001 | Age: 35 | Date: 2026-05-22               │
│                                                               │
│  CLINICAL FEATURES OBSERVED:                                  │
│    Erythema: moderate (grade 2)                               │
│    Scaling: moderate (grade 2)                                │
│    Koebner phenomenon: present                                │
│    Knee/elbow involvement: present                            │
│    Scalp involvement: present                                 │
│    Family history: positive                                   │
│                                                               │
│  ACTIVATED DIAGNOSTIC RULES:                                  │
│    ✓ PSO_001 — Koebner isomorphic response [Tier A, w=0.85]  │
│    ✓ PSO_002 — Knee/elbow extensor pattern [Tier B, w=0.80]  │
│    ✓ PSO_003 — Scalp involvement [Tier B, w=0.75]            │
│    ✓ PSO_004 — Family history (HLA-Cw6) [Tier B, w=0.70]    │
│    ✓ PSO_SD_001 — Borders + family_hist discriminator [D]    │
│                                                               │
│  CONTRADICTIONS DETECTED: None                                │
│                                                               │
│  CERTAINTY METRICS:                                           │
│    Leading diagnosis: Psoriasis (84.7%)                       │
│    Certainty gap: 78.6% (Lichen planus: 6.1%)                 │
│    Ambiguity index: 0.73 bits (low)                           │
│    Contradiction load: 0.00                                   │
│                                                               │
│  SAFETY GATE: ALL CRITERIA PASSED                             │
│                                                               │
│  DIAGNOSTIC STATE: CERTAINTY_STABILIZED                       │
│  RECOMMENDATION: SAFE BIOPSY-FREE DIAGNOSIS                   │
│                                                               │
│  REASONING: Multiple independent clinical signs consistently  │
│  support psoriasis. No contradicting findings detected.       │
│  Evidence sufficient for biopsy-free clinical diagnosis.      │
└──────────────────────────────────────────────────────────────┘
```
