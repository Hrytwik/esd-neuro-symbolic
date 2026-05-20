# Neuro-Symbolic AI for Biopsy-Free Diagnosis of Erythemato-Squamous Diseases
**Spec date:** 2026-05-20  
**Team:** Two undergraduate CS students, VIT Vellore  
**Timeline:** 6–8 weeks  
**Dataset:** UCI Dermatology (id=33), 366 patients, 34 features, CC BY 4.0

---

## 1. Problem Statement

Erythemato-Squamous Diseases (ESD) — psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, pityriasis rubra pilaris — share identical surface symptoms (redness, scaling) and are conventionally separated by biopsy. Every existing ML paper achieves 96–99% accuracy by using the 22 biopsy-derived histopathological features, making them clinically useless in primary care settings without biopsy access. This project builds a biopsy-free classifier using only the 12 non-invasive clinical features, augmented with symbolic rules encoded from published dermatological criteria.

Baseline comparison: Cipriano et al. (2025) achieved 86% accuracy with Random Forest + SHAP on 12 clinical features, without domain knowledge encoding and without interpretable rule extraction. This work extends that by adding symbolic rule features and extracting IF-THEN diagnostic rules.

---

## 2. Project Structure

```
ESD/
├── data/
│   └── raw/                    # dataset via ucimlrepo fetch, or download notes
├── notebooks/
│   └── exploration.ipynb       # optional EDA only
├── results/
│   ├── figures/                # SHAP plots, confusion matrices
│   ├── tables/                 # CV summaries, per-class F1, Wilcoxon
│   └── rules/                  # extracted RuleFit rules + validation notes
├── src/
│   ├── data.py                 # load dataset, split clinical vs histo, preprocessing
│   ├── rules.py                # 8 scored symbolic rule definitions + encoding
│   ├── models.py               # Model A, B, C training functions
│   ├── evaluate.py             # CV, macro F1, per-class metrics, confusion matrices
│   ├── explain.py              # SHAP + RuleFit extraction
│   ├── visualize.py            # all plots and result visualizations
│   ├── config.py               # feature lists, seeds, CV folds, thresholds
│   └── utils.py                # path helpers, save-figure wrapper, logging
├── main.py                     # orchestrator: runs full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## 3. Feature Sets

### 12 Non-Invasive Clinical Features (available without biopsy)

| Feature | Type | Scale |
|---|---|---|
| erythema | ordinal | 0–3 |
| scaling | ordinal | 0–3 |
| definite_borders | ordinal | 0–3 |
| itching | ordinal | 0–3 |
| koebner_phenomenon | binary | 0/1 |
| polygonal_papules | binary | 0/1 |
| follicular_papules | binary | 0/1 |
| oral_mucosal_involvement | binary | 0/1 |
| knee_and_elbow_involvement | binary | 0/1 |
| scalp_involvement | binary | 0/1 |
| family_history | binary | 0/1 |
| age | continuous | years |

### 22 Histopathological Features (biopsy-derived — used in Model A only)
All remaining features in the UCI dataset (melanin incontinence, fibrosis, etc.).

### 8 Symbolic Rule Features (Model C only)
Derived deterministically from the 12 clinical features. See Section 5.

---

## 4. Three-Model Comparison

| Model | Feature set | Purpose |
|---|---|---|
| A | All 34 features | Biopsy-dependent sanity check — expected ~97% accuracy |
| B | 12 clinical features | Biopsy-free baseline — true comparison point vs. Cipriano 2025 |
| C | 12 clinical + 8 rule scores (20 features) | Proposed NSAI contribution |

**Classifier:** XGBoost (`XGBClassifier`) for all three models. Fixing the classifier across A/B/C ensures performance differences are attributable solely to the feature set, not model choice.

**Hyperparameter tuning:** Nested `GridSearchCV` on `n_estimators` ∈ {100, 200, 300} and `max_depth` ∈ {3, 5, 7} inside the CV loop to prevent optimistic bias.

---

## 5. Evaluation Framework

- **Cross-validation:** 10-fold stratified (stratification preserves PRP class at ~3.3% per fold)
- **Primary metric:** Macro F1 — averages F1 equally across all 6 classes, penalising poor performance on PRP (n=20) equally to psoriasis (n=111)
- **Secondary metrics:** Accuracy, per-class precision/recall/F1, confusion matrix
- **Statistical test:** Wilcoxon signed-rank test on the 10 fold-level macro F1 scores of B vs C — determines whether Model C's improvement is statistically significant

**Output artifacts:**
```
results/tables/cv_summary.csv          — mean ± std macro F1 for A, B, C
results/tables/per_class_f1.csv        — per-disease F1 for all three models
results/tables/b_vs_c_wilcoxon.txt     — p-value and effect size
results/figures/confusion_A.png
results/figures/confusion_B.png
results/figures/confusion_C.png
```

---

## 6. The 8 Symbolic Rules

All rules use only the 12 clinical features. For ordinal features, threshold ≥ 2 marks "clinically significant present" (0=absent, 1=mild, 2=moderate, 3=severe).

Implementation: pure pandas/NumPy arithmetic in `src/rules.py`. No external rule engine required.

---

### Rule 1 — Psoriasis Score (0–4)

```python
psoriasis_score = (
    koebner_phenomenon +
    knee_and_elbow_involvement +
    scalp_involvement +
    family_history
)
```

| Condition | Clinical rationale |
|---|---|
| koebner_phenomenon | Isomorphic response at trauma sites; highly specific to psoriasis |
| knee_and_elbow_involvement | Extensor surface plaque distribution: classic psoriasis pattern |
| scalp_involvement | Affected in ~80% of psoriasis cases |
| family_history | Strong polygenic inheritance (HLA-Cw6 association) |

**Sources:** Griffiths CEM & Barker JNWN, *Lancet* 2007;370:263–271. Habif TP, *Clinical Dermatology* 6th ed., Ch. 8, Elsevier 2016.

---

### Rule 2 — Seborrheic Dermatitis Score (0–3)

```python
seborrheic_score = (
    scalp_involvement +
    (1 - koebner_phenomenon) +
    (1 - polygonal_papules)
)
```

| Condition | Clinical rationale |
|---|---|
| scalp_involvement | Primary predilection site (seborrheic dandruff pattern) |
| koebner_phenomenon == 0 | Koebner absent; distinguishes from psoriasis |
| polygonal_papules == 0 | Absence of papules distinguishes from lichen planus |

**Sources:** Schwartz RA et al., *Am Fam Physician* 2006;74:125–130. Naldi L & Rebora A, *NEJM* 2009;360:387–396.

---

### Rule 3 — Lichen Planus Score (0–4)

```python
lp_score = (
    polygonal_papules +
    oral_mucosal_involvement +
    koebner_phenomenon +
    int(itching >= 2)
)
```

| Condition | Clinical rationale |
|---|---|
| polygonal_papules | Pathognomonic ("4 Ps: pruritic, planar, polygonal, purple papules") |
| oral_mucosal_involvement | Wickham's striae; highly specific to LP |
| koebner_phenomenon | Isomorphic response common in LP |
| itching ≥ 2 | Intense pruritus is a characteristic feature of LP |

**Sources:** Le Cleach L & Chosidow O, *NEJM* 2012;366:723–732. Fitzpatrick's *Dermatology in General Medicine* 8th ed., Ch. 26, McGraw-Hill 2012.

---

### Rule 4 — Pityriasis Rosea Score (0–3)

```python
pr_score = (
    int(definite_borders >= 2) +
    (1 - oral_mucosal_involvement) +
    (1 - knee_and_elbow_involvement)
)
```

| Condition | Clinical rationale |
|---|---|
| definite_borders ≥ 2 | Herald patch and satellite lesions have distinct, well-defined margins |
| oral_mucosal_involvement == 0 | Mucosal involvement absent in PR |
| knee_and_elbow_involvement == 0 | PR typically spares extremities |

**Sources:** Stulberg DL & Wolfrey J, *Am Fam Physician* 2004;69:87–91. Habif TP, *Clinical Dermatology* 6th ed., Ch. 11.

---

### Rule 5 — Chronic Dermatitis Score (0–4)

```python
cd_score = (
    int(itching >= 2) +
    (1 - koebner_phenomenon) +
    (1 - polygonal_papules) +
    (1 - follicular_papules)
)
```

| Condition | Clinical rationale |
|---|---|
| itching ≥ 2 | Chronic intense itch is the defining hallmark |
| koebner_phenomenon == 0 | Typically absent; distinguishes from psoriasis and LP |
| polygonal_papules == 0 | Distinguishes from lichen planus |
| follicular_papules == 0 | Distinguishes from pityriasis rubra pilaris |

**Sources:** Williams HC et al., *Br J Dermatol* 1994;131:383–396. Boguniewicz M & Leung DYM, *Immunol Rev* 2011;242:233–246.

---

### Rule 6 — Pityriasis Rubra Pilaris Score (0–3)

```python
prp_score = (
    follicular_papules +
    int(erythema >= 2) +
    (1 - koebner_phenomenon)
)
```

| Condition | Clinical rationale |
|---|---|
| follicular_papules | Keratotic follicular papules are pathognomonic for PRP |
| erythema ≥ 2 | Diffuse salmon-red erythema characteristic |
| koebner_phenomenon == 0 | Koebner typically absent in PRP |

**Sources:** Griffiths WAD, *Clin Exp Dermatol* 1980;5:105–112. Sehgal VN et al., *Int J Dermatol* 2013;52:775–791.

---

### Rule 7 — Psoriasis vs. Seborrheic Dermatitis Discriminator (0–2)

```python
ps_vs_sd_score = (
    int(definite_borders >= 2) +
    family_history
)
```

Fires positively for psoriasis. Both diseases affect the scalp; this rule captures the key differentiators.

| Condition | Clinical rationale |
|---|---|
| definite_borders ≥ 2 | Psoriasis: sharp, well-demarcated plaques; seb derm: diffuse, poorly defined |
| family_history | Genetic component strong in psoriasis; absent in seborrheic dermatitis |

**Source:** van de Kerkhof PCM & Vissers WHPM, *Skin Pharmacol Physiol* 2003;16:69–83.

---

### Rule 8 — Lichen Planus vs. Pityriasis Rosea Discriminator (0–2)

```python
lp_vs_pr_score = (
    oral_mucosal_involvement +
    polygonal_papules
)
```

Fires positively for lichen planus. Both diseases present with scaling and definite borders.

| Condition | Clinical rationale |
|---|---|
| oral_mucosal_involvement | Present in LP (Wickham's striae); absent in PR |
| polygonal_papules | Pathognomonic for LP; never present in PR |

**Sources:** Fitzpatrick's *Dermatology* 8th ed. Le Cleach & Chosidow, *NEJM* 2012.

---

## 7. Explainability Layer

### SHAP (`shap.TreeExplainer` on Model C)
- Beeswarm plot of top 15 features by mean |SHAP value| across all test folds
- Per-class SHAP bar charts for each of the 6 ESD diseases
- Success criterion: ≥ 3 of the 8 rule-score features must appear in the top 10 by mean |SHAP| — confirms symbolic rules are load-bearing, not decorative

```
results/figures/shap_beeswarm_C.png
results/figures/shap_per_class/shap_<disease>.png   (×6)
results/tables/shap_top10_features.csv
```

### RuleFit (`imodels.RuleFitClassifier`)
- Trained on full Model C feature set (20 features)
- Extracts sparse IF-THEN rules with non-zero coefficients
- Rules are filtered to those referencing only clinical or symbolic features (no raw tree-split indices)
- Each extracted rule is manually validated against the 6 published disease criteria in `results/rules/rulefit_validation.md`
- Success criterion: ≥ 4 of 6 extracted disease rules align with published diagnostic criteria

**Expected output format:**
```
IF koebner_phenomenon = 1 AND psoriasis_score >= 3
THEN → Psoriasis  (support: 0.28, confidence: 0.91)
← Matches: Griffiths & Barker 2007 ✓
```

```
results/rules/extracted_rules.csv
results/rules/rulefit_validation.md
```

---

## 8. Success Criteria

| Criterion | Threshold | Measurement |
|---|---|---|
| Model C macro F1 > Model B macro F1 | Any positive delta + p < 0.05 Wilcoxon | `cv_summary.csv` + `b_vs_c_wilcoxon.txt` |
| ≥ 3 rule features in SHAP top 10 | Count of rule-score features in top 10 | `shap_top10_features.csv` |
| ≥ 4 of 6 extracted rules match published criteria | Manual validation | `rulefit_validation.md` |
| ≥ 2 diseases with per-class F1 ≥ 0.90 in Model C | Per-disease F1 | `per_class_f1.csv` |
| ≥ 1 disease where biopsy remains necessary (F1 < 0.80) | Per-disease F1 | `per_class_f1.csv` |

---

## 9. Libraries

```
ucimlrepo          # dataset fetch
pandas, numpy      # data manipulation
scikit-learn       # stratified CV, metrics, GridSearchCV
xgboost            # classifier for A, B, C
shap               # TreeExplainer, beeswarm plots
imodels            # RuleFitClassifier
matplotlib, seaborn # visualization
scipy              # Wilcoxon signed-rank test
```

---

## 10. What This Project Does NOT Do

- No deployment, web app, or API
- No real patient data (UCI dataset only)
- No synthetic data generation
- No deep neuro-symbolic frameworks (LNN, Neural Theorem Provers, etc.)
- No binary collapse of the 6-class problem
- No fabricated clinical rules — every rule condition is traceable to a cited source
