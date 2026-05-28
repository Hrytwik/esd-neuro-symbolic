"""
symbolic_rule_refinement_v2.py
=================================
Version 2 symbolic rule refinement for the CASDRE clinical inference pipeline.

Refines and extends disease-specific symbolic rule sets to achieve:
  - higher discriminative power per rule
  - sharper disease-specific signatures
  - stronger differential competition
  - higher symbolic recoverability

All rules remain clinically grounded and interpretable.  No rules are added
that lack a published dermatological basis.

Target diseases for v2 refinement:
  - psoriasis       : scaling triad, koebner, knee/elbow/scalp involvement
  - lichen_planus   : polygonal papules, Koebner, oral mucosal, Wickham striae
  - seborrheic_dermatitis : scalp/facial distribution, melanin incontinence, fine scaling
  - chronic_dermatitis    : lichenification proxy, chronic itch pattern
  - pityriasis_rubra_pilaris : follicular hyperkeratosis, orange discolouration proxy
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class RuleStrength(str, Enum):
    PATHOGNOMONIC = "pathognomonic"   # near-definitive for one disease
    STRONG        = "strong"          # high discriminative power
    MODERATE      = "moderate"
    WEAK          = "weak"
    EXPLORATORY   = "exploratory"     # low evidence, monitor only


class RuleDirection(str, Enum):
    POSITIVE  = "positive"    # feature presence supports disease
    NEGATIVE  = "negative"    # feature absence supports disease
    COMPOSITE = "composite"   # combination of features


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolicRule:
    """One symbolic dermatological rule."""
    rule_id: str
    target_disease: str
    description: str
    strength: RuleStrength
    direction: RuleDirection
    features_involved: List[str]
    activation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    # fn(X_clinical) → activation score (n,) in [0, 1]
    clinical_reference: str = ""

    def activate(self, X_clinical: np.ndarray) -> np.ndarray:
        """Compute rule activation for each case in X_clinical."""
        if self.activation_fn is not None:
            return self.activation_fn(X_clinical)
        return np.zeros(X_clinical.shape[0])


@dataclass
class RulePerformanceMetrics:
    """Empirical performance of a rule on the dataset."""
    rule_id: str
    target_disease: str
    n_activated: int
    n_true_positive: int
    n_false_positive: int
    precision: float
    recall: float
    f1: float
    discriminative_power: float   # correlation with correct-class indicator [0,1]


@dataclass
class DiseaseRuleSet:
    """Collection of rules for one disease with aggregate metrics."""
    disease: str
    rules: List[SymbolicRule]
    n_rules: int
    mean_precision: float
    mean_recall: float
    mean_discriminative_power: float
    coverage: float              # fraction of cases where ≥ 1 rule fires
    top_rule_id: str


@dataclass
class RuleRefinementReport:
    """Full symbolic rule refinement v2 report."""
    disease_rule_sets: List[DiseaseRuleSet]
    rule_metrics: List[RulePerformanceMetrics]
    all_rules: List[SymbolicRule]

    # Aggregate
    n_total_rules: int
    n_pathognomonic: int
    n_strong: int
    mean_rule_precision: float
    mean_rule_recall: float
    mean_coverage: float

    # Refinement findings
    under_covered_diseases: List[str]    # coverage < 0.50
    low_precision_rules: List[str]       # precision < 0.60
    high_value_rules: List[str]          # f1 > 0.70

    recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "SYMBOLIC RULE REFINEMENT V2 REPORT",
            "=" * 70,
            f"  Total rules              : {self.n_total_rules}",
            f"  Pathognomonic rules      : {self.n_pathognomonic}",
            f"  Strong rules             : {self.n_strong}",
            f"  Mean rule precision      : {self.mean_rule_precision:.3f}",
            f"  Mean rule recall         : {self.mean_rule_recall:.3f}",
            f"  Mean coverage            : {self.mean_coverage:.1%}",
            "",
            "  ── Disease Rule Sets ─────────────────────────────────────────",
        ]
        for drs in self.disease_rule_sets:
            lines.append(
                f"    {drs.disease:<32s}  "
                f"n_rules={drs.n_rules}  "
                f"cov={drs.coverage:.1%}  "
                f"P={drs.mean_precision:.3f}  "
                f"R={drs.mean_recall:.3f}"
            )
        if self.under_covered_diseases:
            lines += [
                "",
                "  ── Under-Covered Diseases (< 50 % coverage) ─────────────────",
            ]
            for d in self.under_covered_diseases:
                lines.append(f"    ⚠  {d}")
        if self.high_value_rules:
            lines += [
                "",
                "  ── High-Value Rules (F1 > 0.70) ──────────────────────────────",
            ]
            for r in self.high_value_rules[:6]:
                lines.append(f"    ✓  {r}")
        if self.recommendations:
            lines += ["", "  ── Recommendations ──────────────────────────────────────────"]
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Rule library v2
# ──────────────────────────────────────────────────────────────────────────────

def _col(X: np.ndarray, i: int, default: float = 0.0) -> np.ndarray:
    if i < X.shape[1]:
        return X[:, i].astype(float)
    return np.full(X.shape[0], default)


def _build_rule_library() -> List[SymbolicRule]:
    """
    Returns the v2 symbolic rule set covering all 6 disease classes.
    Column indices follow the established 12-feature clinical schema:
      0=erythema, 1=scaling, 2=definite_borders, 3=itching,
      4=koebner, 5=poly_pap, 6=foll_pap, 7=oral, 8=knee_elbow,
      9=scalp, 10=family_hist, 11=melanin_inc
    """

    rules: List[SymbolicRule] = []

    # ── PSORIASIS ──────────────────────────────────────────────────────
    rules += [
        SymbolicRule(
            rule_id="PSO_001",
            target_disease="psoriasis",
            description="Scaling + knee/elbow + scalp triad (classic psoriasis distribution)",
            strength=RuleStrength.PATHOGNOMONIC,
            direction=RuleDirection.COMPOSITE,
            features_involved=["scaling", "knee_elbow", "scalp"],
            activation_fn=lambda X: np.clip(
                _col(X,1)*0.5 + _col(X,8)*0.3 + _col(X,9)*0.2, 0, 1
            ),
            clinical_reference="Christophers 2001",
        ),
        SymbolicRule(
            rule_id="PSO_002",
            target_disease="psoriasis",
            description="Koebner phenomenon + erythema (isomorphic response)",
            strength=RuleStrength.STRONG,
            direction=RuleDirection.COMPOSITE,
            features_involved=["koebner", "erythema"],
            activation_fn=lambda X: np.clip(
                _col(X,4)*0.6 + _col(X,0)*0.4, 0, 1
            ),
            clinical_reference="Weiss et al. 2002",
        ),
        SymbolicRule(
            rule_id="PSO_003",
            target_disease="psoriasis",
            description="Family history + scaling (hereditary psoriasis)",
            strength=RuleStrength.STRONG,
            direction=RuleDirection.COMPOSITE,
            features_involved=["family_history", "scaling"],
            activation_fn=lambda X: np.clip(
                _col(X,10)*0.5 + _col(X,1)*0.5, 0, 1
            ),
        ),
    ]

    # ── LICHEN PLANUS ──────────────────────────────────────────────────
    rules += [
        SymbolicRule(
            rule_id="LP_001",
            target_disease="lichen_planus",
            description="Polygonal papules + oral mucosal involvement (Wickham striae pattern)",
            strength=RuleStrength.PATHOGNOMONIC,
            direction=RuleDirection.COMPOSITE,
            features_involved=["polygonal_papules", "oral_mucosal"],
            activation_fn=lambda X: np.clip(
                _col(X,5)*0.6 + _col(X,7)*0.4, 0, 1
            ),
            clinical_reference="Boyd & Neldner 1991",
        ),
        SymbolicRule(
            rule_id="LP_002",
            target_disease="lichen_planus",
            description="Koebner + polygonal papules (LP isomorphic response)",
            strength=RuleStrength.STRONG,
            direction=RuleDirection.COMPOSITE,
            features_involved=["koebner", "polygonal_papules"],
            activation_fn=lambda X: np.clip(
                _col(X,4)*0.5 + _col(X,5)*0.5, 0, 1
            ),
        ),
        SymbolicRule(
            rule_id="LP_003",
            target_disease="lichen_planus",
            description="Oral involvement without scalp involvement (LP vs psoriasis)",
            strength=RuleStrength.MODERATE,
            direction=RuleDirection.COMPOSITE,
            features_involved=["oral_mucosal", "scalp"],
            activation_fn=lambda X: np.clip(
                _col(X,7) - _col(X,9)*0.5, 0, 1
            ),
        ),
    ]

    # ── SEBORRHEIC DERMATITIS ──────────────────────────────────────────
    rules += [
        SymbolicRule(
            rule_id="SEB_001",
            target_disease="seborrheic_dermatitis",
            description="Scalp + scaling + melanin incontinence (seborrhoeic distribution)",
            strength=RuleStrength.PATHOGNOMONIC,
            direction=RuleDirection.COMPOSITE,
            features_involved=["scalp", "scaling", "melanin_incontinence"],
            activation_fn=lambda X: np.clip(
                _col(X,9)*0.4 + _col(X,1)*0.35 + _col(X,11)*0.25, 0, 1
            ),
            clinical_reference="Borda & Wikramanayake 2015",
        ),
        SymbolicRule(
            rule_id="SEB_002",
            target_disease="seborrheic_dermatitis",
            description="Greasy scaling + no koebner (seborrhoeic without psoriasis isomorphism)",
            strength=RuleStrength.STRONG,
            direction=RuleDirection.COMPOSITE,
            features_involved=["scaling", "koebner"],
            activation_fn=lambda X: np.clip(
                _col(X,1)*0.7 - _col(X,4)*0.3, 0, 1
            ),
        ),
    ]

    # ── CHRONIC DERMATITIS ─────────────────────────────────────────────
    rules += [
        SymbolicRule(
            rule_id="CHR_001",
            target_disease="chronic_dermatitis",
            description="Persistent itching + erythema (lichenification pattern)",
            strength=RuleStrength.STRONG,
            direction=RuleDirection.COMPOSITE,
            features_involved=["itching", "erythema"],
            activation_fn=lambda X: np.clip(
                _col(X,3)*0.6 + _col(X,0)*0.4, 0, 1
            ),
        ),
        SymbolicRule(
            rule_id="CHR_002",
            target_disease="chronic_dermatitis",
            description="Itching + melanin incontinence without polygonal papules",
            strength=RuleStrength.MODERATE,
            direction=RuleDirection.COMPOSITE,
            features_involved=["itching", "melanin_incontinence", "polygonal_papules"],
            activation_fn=lambda X: np.clip(
                _col(X,3)*0.5 + _col(X,11)*0.3 - _col(X,5)*0.2, 0, 1
            ),
        ),
        SymbolicRule(
            rule_id="CHR_003",
            target_disease="chronic_dermatitis",
            description="No family history + itching (sporadic atopic pattern)",
            strength=RuleStrength.MODERATE,
            direction=RuleDirection.COMPOSITE,
            features_involved=["family_history", "itching"],
            activation_fn=lambda X: np.clip(
                (1.0 - _col(X,10))*0.4 + _col(X,3)*0.6, 0, 1
            ),
        ),
    ]

    # ── PITYRIASIS ROSEA ───────────────────────────────────────────────
    rules += [
        SymbolicRule(
            rule_id="PR_001",
            target_disease="pityriasis_rosea",
            description="Definite borders + erythema (herald patch pattern)",
            strength=RuleStrength.STRONG,
            direction=RuleDirection.COMPOSITE,
            features_involved=["definite_borders", "erythema"],
            activation_fn=lambda X: np.clip(
                _col(X,2)*0.6 + _col(X,0)*0.4, 0, 1
            ),
            clinical_reference="Drago et al. 2009",
        ),
        SymbolicRule(
            rule_id="PR_002",
            target_disease="pityriasis_rosea",
            description="No koebner, no oral, no follicular papules (negative pattern)",
            strength=RuleStrength.MODERATE,
            direction=RuleDirection.NEGATIVE,
            features_involved=["koebner", "oral_mucosal", "follicular_papules"],
            activation_fn=lambda X: np.clip(
                1.0 - (_col(X,4) + _col(X,7) + _col(X,6)) / 3.0, 0, 1
            ),
        ),
    ]

    # ── PITYRIASIS RUBRA PILARIS ───────────────────────────────────────
    rules += [
        SymbolicRule(
            rule_id="PRP_001",
            target_disease="pityriasis_rubra_pilaris",
            description="Follicular papules + scaling + knee/elbow (PRP hallmark triad)",
            strength=RuleStrength.PATHOGNOMONIC,
            direction=RuleDirection.COMPOSITE,
            features_involved=["follicular_papules", "scaling", "knee_elbow"],
            activation_fn=lambda X: np.clip(
                _col(X,6)*0.5 + _col(X,1)*0.3 + _col(X,8)*0.2, 0, 1
            ),
            clinical_reference="Griffiths 1980",
        ),
        SymbolicRule(
            rule_id="PRP_002",
            target_disease="pityriasis_rubra_pilaris",
            description="Follicular papules + scalp involvement (PRP orange-peel distribution)",
            strength=RuleStrength.STRONG,
            direction=RuleDirection.COMPOSITE,
            features_involved=["follicular_papules", "scalp"],
            activation_fn=lambda X: np.clip(
                _col(X,6)*0.6 + _col(X,9)*0.4, 0, 1
            ),
        ),
        SymbolicRule(
            rule_id="PRP_003",
            target_disease="pityriasis_rubra_pilaris",
            description="PRP: no oral + no polygonal papules (distinguishes from LP)",
            strength=RuleStrength.MODERATE,
            direction=RuleDirection.NEGATIVE,
            features_involved=["oral_mucosal", "polygonal_papules"],
            activation_fn=lambda X: np.clip(
                1.0 - (_col(X,7) + _col(X,5)) / 2.0, 0, 1
            ),
        ),
    ]

    return rules


# ──────────────────────────────────────────────────────────────────────────────
# Refiner
# ──────────────────────────────────────────────────────────────────────────────

class SymbolicRuleRefinerV2:
    """
    Builds, evaluates, and refines the v2 symbolic rule set.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names.
    activation_threshold : float
        Minimum rule activation score to consider a rule "fired".
    """

    def __init__(
        self,
        class_labels: List[str],
        activation_threshold: float = 0.40,
    ):
        self.class_labels        = class_labels
        self.activation_threshold = activation_threshold

    # ------------------------------------------------------------------
    def build_and_evaluate(
        self,
        X_clinical: np.ndarray,
        y_true: np.ndarray,
    ) -> RuleRefinementReport:
        """Build the v2 rule library and evaluate each rule empirically."""
        rules    = _build_rule_library()
        metrics  = [self._evaluate_rule(r, X_clinical, y_true) for r in rules]
        drs_list = self._build_disease_rule_sets(rules, metrics)

        n_total  = len(rules)
        n_patho  = sum(1 for r in rules if r.strength == RuleStrength.PATHOGNOMONIC)
        n_strong = sum(1 for r in rules if r.strength == RuleStrength.STRONG)
        mean_p   = statistics.mean(m.precision for m in metrics) if metrics else 0.0
        mean_r   = statistics.mean(m.recall    for m in metrics) if metrics else 0.0
        mean_cov = statistics.mean(d.coverage  for d in drs_list) if drs_list else 0.0

        under_cov = [d.disease for d in drs_list if d.coverage < 0.50]
        low_prec  = [m.rule_id for m in metrics if m.precision < 0.60]
        high_val  = [m.rule_id for m in metrics if m.f1 > 0.70]

        recs = self._generate_recommendations(
            rules, metrics, drs_list, under_cov, low_prec
        )

        return RuleRefinementReport(
            disease_rule_sets=drs_list,
            rule_metrics=metrics,
            all_rules=rules,
            n_total_rules=n_total,
            n_pathognomonic=n_patho,
            n_strong=n_strong,
            mean_rule_precision=mean_p,
            mean_rule_recall=mean_r,
            mean_coverage=mean_cov,
            under_covered_diseases=under_cov,
            low_precision_rules=low_prec,
            high_value_rules=high_val,
            recommendations=recs,
        )

    def compute_rule_activations(
        self,
        X_clinical: np.ndarray,
    ) -> np.ndarray:
        """
        Compute combined rule activation matrix.

        Returns
        -------
        ndarray of shape (n, n_rules) — activation score per case per rule.
        """
        rules = _build_rule_library()
        if X_clinical.shape[0] == 0:
            return np.zeros((0, len(rules)))
        return np.column_stack([r.activate(X_clinical) for r in rules])

    # ------------------------------------------------------------------
    def _evaluate_rule(
        self,
        rule: SymbolicRule,
        X: np.ndarray,
        y: np.ndarray,
    ) -> RulePerformanceMetrics:
        target_idx = next(
            (i for i, lbl in enumerate(self.class_labels)
             if lbl == rule.target_disease),
            -1
        )
        activations = rule.activate(X)
        fired       = activations >= self.activation_threshold
        y_true_bin  = (y == target_idx).astype(int)
        y_pred_bin  = fired.astype(int)

        tp = int(np.sum(y_pred_bin & y_true_bin))
        fp = int(np.sum(y_pred_bin & ~y_true_bin.astype(bool)))
        fn = int(np.sum(~y_pred_bin.astype(bool) & y_true_bin))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if np.std(activations) > 1e-9:
            disc = abs(float(np.corrcoef(activations, y_true_bin)[0, 1]))
        else:
            disc = 0.0

        return RulePerformanceMetrics(
            rule_id=rule.rule_id,
            target_disease=rule.target_disease,
            n_activated=int(fired.sum()),
            n_true_positive=tp,
            n_false_positive=fp,
            precision=prec,
            recall=rec,
            f1=f1,
            discriminative_power=float(np.clip(disc, 0, 1)),
        )

    def _build_disease_rule_sets(
        self,
        rules: List[SymbolicRule],
        metrics: List[RulePerformanceMetrics],
    ) -> List[DiseaseRuleSet]:
        from collections import defaultdict
        disease_rules: Dict[str, List[SymbolicRule]] = defaultdict(list)
        disease_metrics: Dict[str, List[RulePerformanceMetrics]] = defaultdict(list)
        for rule, metric in zip(rules, metrics):
            disease_rules[rule.target_disease].append(rule)
            disease_metrics[rule.target_disease].append(metric)

        drs_list: List[DiseaseRuleSet] = []
        for disease in self.class_labels:
            d_rules   = disease_rules.get(disease, [])
            d_metrics = disease_metrics.get(disease, [])
            if not d_rules:
                continue
            mean_p   = statistics.mean(m.precision for m in d_metrics) if d_metrics else 0.0
            mean_r   = statistics.mean(m.recall    for m in d_metrics) if d_metrics else 0.0
            mean_dp  = statistics.mean(m.discriminative_power for m in d_metrics) if d_metrics else 0.0
            top_rule = max(d_metrics, key=lambda m: m.f1).rule_id if d_metrics else ""
            # Coverage = fraction of target cases where ≥ 1 rule fires
            # (Proxy: max recall across rules)
            cov = max((m.recall for m in d_metrics), default=0.0)
            drs_list.append(DiseaseRuleSet(
                disease=disease,
                rules=d_rules,
                n_rules=len(d_rules),
                mean_precision=mean_p,
                mean_recall=mean_r,
                mean_discriminative_power=mean_dp,
                coverage=cov,
                top_rule_id=top_rule,
            ))
        return drs_list

    @staticmethod
    def _generate_recommendations(
        rules: List[SymbolicRule],
        metrics: List[RulePerformanceMetrics],
        drs_list: List[DiseaseRuleSet],
        under_cov: List[str],
        low_prec: List[str],
    ) -> List[str]:
        recs: List[str] = []

        if under_cov:
            names = ", ".join(under_cov)
            recs.append(
                f"Under-covered diseases [{names}] need additional rules — "
                "add at least 2 more clinically grounded rules per disease."
            )
        if low_prec:
            n = len(low_prec)
            recs.append(
                f"{n} rule(s) have precision < 0.60 — review and tighten "
                "activation thresholds or add exclusion conditions."
            )

        # Best rule
        best_m = max(metrics, key=lambda m: m.f1, default=None)
        if best_m and best_m.f1 > 0.70:
            recs.append(
                f"Rule '{best_m.rule_id}' achieves F1={best_m.f1:.3f} — "
                "use its activation pattern as a template for new rules."
            )

        # Pathognomonic coverage check
        patho_rules = [r for r in rules if r.strength == RuleStrength.PATHOGNOMONIC]
        patho_diseases = {r.target_disease for r in patho_rules}
        missing_patho = [d.disease for d in drs_list
                         if d.disease not in patho_diseases]
        if missing_patho:
            recs.append(
                f"Diseases [{', '.join(missing_patho)}] lack a pathognomonic rule — "
                "identify at least one near-definitive signal per disease."
            )

        if not recs:
            recs.append("Rule set v2 is well-configured — maintain current profile.")
        return recs[:5]
