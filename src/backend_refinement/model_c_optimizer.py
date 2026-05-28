"""
model_c_optimizer.py
=====================
Model C optimisation engine for the CASDRE clinical inference pipeline.

Identifies the strongest symbolic + discriminative hybrid configuration by
systematically evaluating multiple classification families with:

  - repeated stratified cross-validation
  - class-imbalance handling
  - calibrated probability outputs
  - disease-wise performance breakdown
  - symbolic-lift quantification (Model C vs. Model B delta)

Target performance range: Model C 88–91 % on the 12-clinical + symbolic
feature space.

Supported engines (all imported with graceful fallback):
  XGBoost / CatBoost / LightGBM / Calibrated Random Forest
"""

from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, confusion_matrix,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional engines — imported with fallback
try:
    from xgboost import XGBClassifier as _XGB
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier as _CB
    _HAS_CB = True
except ImportError:
    _HAS_CB = False

try:
    from lightgbm import LGBMClassifier as _LGB
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_LABELS = [
    "psoriasis",
    "seborrheic_dermatitis",
    "lichen_planus",
    "pityriasis_rosea",
    "chronic_dermatitis",
    "pityriasis_rubra_pilaris",
]

_N_SPLITS   = 5
_N_REPEATS  = 3
_RANDOM_STATE = 42

# Target range
_TARGET_LOW  = 0.88
_TARGET_HIGH = 0.91


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DiseasePerformance:
    disease: str
    n_cases: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ModelCConfiguration:
    """One evaluated Model C configuration."""
    name: str
    engine: str                   # "xgboost" / "catboost" / "lightgbm" / "random_forest"
    feature_set: str              # "clinical_only" / "clinical_symbolic" / "full_34"
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_balanced_accuracy: float
    cv_f1_macro: float
    symbolic_lift_pp: float       # delta over Model B (clinical-only baseline)
    disease_performance: List[DiseasePerformance]
    target_met: bool              # cv_accuracy_mean in [0.88, 0.91]
    hyperparams: Dict[str, Any]


@dataclass
class ModelBBaseline:
    """Clinical-only baseline for lift computation."""
    engine: str
    cv_accuracy_mean: float
    cv_accuracy_std: float


@dataclass
class ModelCOptimizationReport:
    """Full Model C optimisation report."""
    model_b_baseline: ModelBBaseline
    configurations: List[ModelCConfiguration]
    best_configuration: ModelCConfiguration
    best_accuracy: float
    best_symbolic_lift_pp: float
    target_achieved: bool
    configurations_meeting_target: List[str]

    # Recommendations
    recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "MODEL C OPTIMISATION REPORT",
            "=" * 70,
            f"  Model B baseline (clinical-only) : "
            f"{self.model_b_baseline.cv_accuracy_mean:.1%}",
            f"  Best Model C accuracy            : {self.best_accuracy:.1%}",
            f"  Best symbolic lift               : {self.best_symbolic_lift_pp:+.2f} pp",
            f"  Target range (88–91 %)           : "
            f"{'MET' if self.target_achieved else 'NOT YET MET'}",
            "",
            "  ── Configuration Ranking ─────────────────────────────────────",
        ]
        ranked = sorted(self.configurations,
                        key=lambda c: c.cv_accuracy_mean, reverse=True)
        for i, cfg in enumerate(ranked[:8], 1):
            marker = " ← BEST" if cfg.name == self.best_configuration.name else ""
            lines.append(
                f"    {i:2d}. {cfg.name:<40s}  "
                f"{cfg.cv_accuracy_mean:.1%} ±{cfg.cv_accuracy_std:.3f}"
                f"  lift={cfg.symbolic_lift_pp:+.1f}pp{marker}"
            )
        lines += [
            "",
            "  ── Best Configuration — Disease Breakdown ────────────────────",
        ]
        for dp in sorted(self.best_configuration.disease_performance,
                         key=lambda d: d.recall):
            lines.append(
                f"    {dp.disease:<32s}  "
                f"P={dp.precision:.3f}  R={dp.recall:.3f}  "
                f"F1={dp.f1:.3f}  n={dp.n_cases}"
            )
        if self.recommendations:
            lines += ["", "  ── Recommendations ──────────────────────────────────────────"]
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_b_baseline": {
                "engine": self.model_b_baseline.engine,
                "cv_accuracy_mean": self.model_b_baseline.cv_accuracy_mean,
            },
            "best_configuration": {
                "name": self.best_configuration.name,
                "engine": self.best_configuration.engine,
                "feature_set": self.best_configuration.feature_set,
                "cv_accuracy_mean": self.best_configuration.cv_accuracy_mean,
                "cv_accuracy_std": self.best_configuration.cv_accuracy_std,
                "symbolic_lift_pp": self.best_configuration.symbolic_lift_pp,
                "target_met": self.best_configuration.target_met,
                "hyperparams": self.best_configuration.hyperparams,
            },
            "target_achieved": self.target_achieved,
            "configurations_meeting_target": self.configurations_meeting_target,
            "all_configurations": [
                {
                    "name": c.name,
                    "cv_accuracy_mean": c.cv_accuracy_mean,
                    "symbolic_lift_pp": c.symbolic_lift_pp,
                }
                for c in sorted(self.configurations,
                                 key=lambda c: c.cv_accuracy_mean, reverse=True)
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def _build_symbolic_features(X_clinical: np.ndarray) -> np.ndarray:
    """
    Derive symbolic enrichment features from clinical inputs.
    Produces 28 derived features capturing ratio, interaction, and
    threshold-crossing signals clinically relevant to dermatological
    differential diagnosis.
    """
    n = X_clinical.shape[0]
    # Safe column access with zero padding for missing columns
    def col(i: int) -> np.ndarray:
        if i < X_clinical.shape[1]:
            return X_clinical[:, i].astype(float)
        return np.zeros(n)

    erythema      = col(0)
    scaling       = col(1)
    borders       = col(2)
    itching       = col(3)
    koebner       = col(4)
    poly_pap      = col(5)
    foll_pap      = col(6)
    oral          = col(7)
    knee_elbow    = col(8)
    scalp         = col(9)
    family_hist   = col(10)
    melanin_inc   = col(11)

    feats = np.column_stack([
        # Psoriasis indicators
        scaling * erythema,
        knee_elbow * scaling,
        scalp * erythema,
        family_hist * erythema,
        koebner * erythema,
        # Lichen planus indicators
        poly_pap * oral,
        oral * itching,
        poly_pap * foll_pap,
        koebner * poly_pap,
        # Seborrheic dermatitis indicators
        scalp * scaling,
        melanin_inc * erythema,
        scalp * melanin_inc,
        # Pityriasis rosea indicators
        borders * erythema,
        borders * itching,
        erythema / (scaling + 1.0),
        # Chronic dermatitis indicators
        itching * erythema,
        itching / (borders + 1.0),
        melanin_inc * itching,
        # PRP indicators
        foll_pap * scaling,
        knee_elbow * foll_pap,
        scalp * foll_pap,
        # Summary / interaction signals
        erythema + scaling + itching,
        poly_pap + foll_pap + oral,
        koebner + family_hist,
        np.clip(scaling - erythema, 0, None),
        np.clip(erythema - itching, 0, None),
        (erythema * scaling * itching),
        (poly_pap + foll_pap) * oral,
    ])
    return feats.astype(np.float32)


def build_model_c_features(
    X_clinical: np.ndarray,
    X_full_34: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Build Model C feature sets for evaluation.

    Returns
    -------
    dict with keys:
        "clinical_only"     : 12 clinical features (Model B baseline)
        "clinical_symbolic" : 12 clinical + 28 symbolic = 40 features
        "full_34"           : full 34-feature set (if X_full_34 provided)
        "full_34_symbolic"  : 34 + 28 = 62 features (if X_full_34 provided)
    """
    symbolic = _build_symbolic_features(X_clinical)
    result = {
        "clinical_only":     X_clinical,
        "clinical_symbolic": np.hstack([X_clinical, symbolic]),
    }
    if X_full_34 is not None:
        sym_full = _build_symbolic_features(X_full_34[:, :12])
        result["full_34"]         = X_full_34
        result["full_34_symbolic"] = np.hstack([X_full_34, sym_full])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Engine factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_engines(n_classes: int) -> List[Tuple[str, str, Any]]:
    """Return list of (name, engine_tag, estimator) tuples."""
    engines = []

    # ── Random Forest variants ─────────────────────────────────────────
    for n_est, max_d, tag in [
        (300, None, "rf_300_deep"),
        (500, 12,   "rf_500_d12"),
        (500, None, "rf_500_deep"),
    ]:
        engines.append((tag, "random_forest",
            CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_d,
                    class_weight="balanced", random_state=_RANDOM_STATE,
                    n_jobs=-1,
                ),
                method="isotonic", cv=3,
            )
        ))

    # ── XGBoost variants ───────────────────────────────────────────────
    if _HAS_XGB:
        for lr, n_est, max_d, subsample, tag in [
            (0.05, 300, 6, 0.8, "xgb_lr005_300_d6"),
            (0.10, 200, 5, 0.9, "xgb_lr010_200_d5"),
            (0.05, 500, 8, 0.8, "xgb_lr005_500_d8"),
            (0.08, 300, 6, 0.85, "xgb_lr008_300_d6"),
        ]:
            engines.append((tag, "xgboost",
                _XGB(
                    n_estimators=n_est, max_depth=max_d,
                    learning_rate=lr, subsample=subsample,
                    colsample_bytree=0.8, use_label_encoder=False,
                    eval_metric="mlogloss", random_state=_RANDOM_STATE,
                    verbosity=0, n_jobs=-1,
                )
            ))

    # ── CatBoost variants ──────────────────────────────────────────────
    if _HAS_CB:
        for lr, n_iter, depth, tag in [
            (0.05, 400, 6, "cb_lr005_400_d6"),
            (0.10, 300, 5, "cb_lr010_300_d5"),
            (0.05, 600, 7, "cb_lr005_600_d7"),
        ]:
            engines.append((tag, "catboost",
                _CB(
                    iterations=n_iter, depth=depth, learning_rate=lr,
                    auto_class_weights="Balanced",
                    random_seed=_RANDOM_STATE, verbose=0,
                )
            ))

    # ── LightGBM variants ─────────────────────────────────────────────
    if _HAS_LGB:
        for lr, n_est, max_d, tag in [
            (0.05, 400, 6, "lgb_lr005_400_d6"),
            (0.08, 300, 5, "lgb_lr008_300_d5"),
            (0.05, 600, 8, "lgb_lr005_600_d8"),
        ]:
            engines.append((tag, "lightgbm",
                _LGB(
                    n_estimators=n_est, max_depth=max_d,
                    learning_rate=lr, class_weight="balanced",
                    random_state=_RANDOM_STATE, verbosity=-1, n_jobs=-1,
                )
            ))

    return engines


# ──────────────────────────────────────────────────────────────────────────────
# CV evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_config(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    class_labels: List[str],
) -> Tuple[float, float, float, float, List[DiseasePerformance]]:
    """
    Run repeated stratified CV and return
    (mean_acc, std_acc, balanced_acc, macro_f1, disease_performance).
    """
    # cross_val_predict requires a non-repeating CV (each sample exactly once)
    cv_predict = StratifiedKFold(
        n_splits=_N_SPLITS,
        shuffle=True,
        random_state=_RANDOM_STATE,
    )
    y_pred = cross_val_predict(estimator, X, y, cv=cv_predict, n_jobs=1)

    acc_scores = []
    bal_scores = []
    f1_scores  = []
    cv2 = RepeatedStratifiedKFold(
        n_splits=_N_SPLITS, n_repeats=_N_REPEATS,
        random_state=_RANDOM_STATE
    )
    for train_idx, test_idx in cv2.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        est_clone = _clone_estimator(estimator)
        est_clone.fit(X_tr, y_tr)
        yp = est_clone.predict(X_te)
        acc_scores.append(accuracy_score(y_te, yp))
        bal_scores.append(balanced_accuracy_score(y_te, yp))
        f1_scores.append(f1_score(y_te, yp, average="macro", zero_division=0))

    mean_acc = float(np.mean(acc_scores))
    std_acc  = float(np.std(acc_scores))
    bal_acc  = float(np.mean(bal_scores))
    mac_f1   = float(np.mean(f1_scores))

    # Disease-wise breakdown from cross_val_predict
    disease_perf: List[DiseasePerformance] = []
    n_classes = len(class_labels)
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1_, sup = precision_recall_fscore_support(
        y, y_pred, labels=list(range(n_classes)),
        average=None, zero_division=0
    )
    for i, label in enumerate(class_labels):
        disease_perf.append(DiseasePerformance(
            disease=label,
            n_cases=int(np.sum(y == i)),
            precision=float(prec[i]),
            recall=float(rec[i]),
            f1=float(f1_[i]),
            support=int(sup[i]),
        ))

    return mean_acc, std_acc, bal_acc, mac_f1, disease_perf


def _clone_estimator(est: Any) -> Any:
    """Safely clone a scikit-learn-compatible estimator."""
    try:
        from sklearn.base import clone
        return clone(est)
    except Exception:
        import copy
        return copy.deepcopy(est)


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser
# ──────────────────────────────────────────────────────────────────────────────

class ModelCOptimizer:
    """
    Systematically evaluates multiple symbolic + discriminative engine
    configurations to identify the strongest Model C configuration.

    Parameters
    ----------
    class_labels : list[str]
        Ordered disease names.
    verbose : bool
        Print per-configuration progress.
    """

    def __init__(
        self,
        class_labels: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.class_labels = class_labels or CLASS_LABELS
        self.verbose      = verbose

    # ------------------------------------------------------------------
    def optimise(
        self,
        X_clinical: np.ndarray,
        y: np.ndarray,
        X_full_34: Optional[np.ndarray] = None,
    ) -> ModelCOptimizationReport:
        """
        Run full Model C optimisation.

        Parameters
        ----------
        X_clinical : ndarray (n, 12)
            Clinical feature matrix (12 features, biopsy-free).
        y : ndarray (n,)
            0-based class labels.
        X_full_34 : ndarray (n, 34), optional
            Full 34-feature matrix for Model A comparisons.
        """
        if self.verbose:
            print("Building feature sets...")
        feature_sets = build_model_c_features(X_clinical, X_full_34)

        # Model B baseline (clinical only, calibrated RF)
        if self.verbose:
            print("Evaluating Model B baseline...")
        b_baseline = self._evaluate_model_b_baseline(
            feature_sets["clinical_only"], y
        )

        # Evaluate all engine × feature_set combinations
        engines = _make_engines(len(self.class_labels))
        configs: List[ModelCConfiguration] = []
        total = len(engines) * (len(feature_sets) - 1 + (1 if X_full_34 is not None else 0))
        done = 0

        for feat_name, X_feat in feature_sets.items():
            if feat_name == "clinical_only":
                continue   # that's the Model B baseline, not Model C
            for eng_name, eng_tag, estimator in engines:
                config_name = f"{eng_name}__{feat_name}"
                done += 1
                if self.verbose:
                    print(f"  [{done}] {config_name} ...", end=" ", flush=True)
                try:
                    mean_acc, std_acc, bal_acc, mac_f1, dis_perf = _evaluate_config(
                        estimator, X_feat, y, self.class_labels
                    )
                    lift_pp = (mean_acc - b_baseline.cv_accuracy_mean) * 100.0
                    target_met = _TARGET_LOW <= mean_acc <= _TARGET_HIGH

                    hyperparams = {}
                    try:
                        hyperparams = estimator.get_params()
                    except Exception:
                        pass

                    cfg = ModelCConfiguration(
                        name=config_name,
                        engine=eng_tag,
                        feature_set=feat_name,
                        cv_accuracy_mean=mean_acc,
                        cv_accuracy_std=std_acc,
                        cv_balanced_accuracy=bal_acc,
                        cv_f1_macro=mac_f1,
                        symbolic_lift_pp=lift_pp,
                        disease_performance=dis_perf,
                        target_met=target_met,
                        hyperparams=hyperparams,
                    )
                    configs.append(cfg)
                    if self.verbose:
                        print(f"{mean_acc:.3f} (lift={lift_pp:+.1f}pp)")
                except Exception as exc:
                    if self.verbose:
                        print(f"FAILED: {exc}")

        if not configs:
            raise RuntimeError("No configurations evaluated successfully.")

        best = max(configs, key=lambda c: c.cv_accuracy_mean)
        target_configs = [c.name for c in configs if c.target_met]
        recs = self._generate_recommendations(configs, b_baseline, best)

        return ModelCOptimizationReport(
            model_b_baseline=b_baseline,
            configurations=configs,
            best_configuration=best,
            best_accuracy=best.cv_accuracy_mean,
            best_symbolic_lift_pp=best.symbolic_lift_pp,
            target_achieved=best.target_met,
            configurations_meeting_target=target_configs,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    def _evaluate_model_b_baseline(
        self,
        X_clin: np.ndarray,
        y: np.ndarray,
    ) -> ModelBBaseline:
        estimator = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=_RANDOM_STATE, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        )
        mean_acc, std_acc, _, _, _ = _evaluate_config(
            estimator, X_clin, y, self.class_labels
        )
        return ModelBBaseline(
            engine="calibrated_random_forest",
            cv_accuracy_mean=mean_acc,
            cv_accuracy_std=std_acc,
        )

    @staticmethod
    def _generate_recommendations(
        configs: List[ModelCConfiguration],
        baseline: ModelBBaseline,
        best: ModelCConfiguration,
    ) -> List[str]:
        recs: List[str] = []
        if best.target_met:
            recs.append(
                f"Configuration '{best.name}' meets the 88–91 % target at "
                f"{best.cv_accuracy_mean:.1%}."
            )
        else:
            gap = (_TARGET_LOW - best.cv_accuracy_mean) * 100
            recs.append(
                f"Best configuration is {best.cv_accuracy_mean:.1%}, "
                f"{gap:.1f} pp below the 88 % floor — deepen symbolic "
                f"feature engineering and disease-specific rule weighting."
            )

        # Feature set comparison
        feat_best: Dict[str, float] = {}
        for c in configs:
            feat_best[c.feature_set] = max(
                feat_best.get(c.feature_set, 0.0), c.cv_accuracy_mean
            )
        best_fs = max(feat_best, key=lambda k: feat_best[k])
        recs.append(
            f"Feature set '{best_fs}' achieves highest accuracy "
            f"({feat_best[best_fs]:.1%}) — prioritise this in deployment."
        )

        # Engine ranking
        eng_best: Dict[str, float] = {}
        for c in configs:
            eng_best[c.engine] = max(eng_best.get(c.engine, 0.0), c.cv_accuracy_mean)
        top_eng = max(eng_best, key=lambda k: eng_best[k])
        recs.append(
            f"Engine family '{top_eng}' dominates at "
            f"{eng_best[top_eng]:.1%} — use this as the primary inference engine."
        )

        # Disease needing attention
        worst_dp = min(best.disease_performance, key=lambda d: d.recall)
        recs.append(
            f"Disease '{worst_dp.disease}' has the lowest recall "
            f"({worst_dp.recall:.3f}) — apply disease-specific symbolic "
            f"strengthening (see disease_discrimination_refinement)."
        )

        return recs[:5]
