"""
BaselineCalibrator — multi-algorithm stratified repeated CV optimisation.

Identifies the optimal classifier family and hyperparameters for Model B
(biopsy-free clinical baseline) and Model C (clinical + symbolic reasoning)
through stratified repeated cross-validation.

Algorithm inventory
-------------------
  XGBoost       — gradient-boosted trees (primary current choice)
  Random Forest — ensemble of decision trees
  LightGBM      — fast gradient boosting (if installed)
  CatBoost      — categorical-aware gradient boosting (if installed)
  Logistic Regression — linear baseline

Calibration strategy
--------------------
  1. Stratified 5-fold × 5-repeat cross-validation
  2. Grid search over key hyperparameters
  3. Class-weight balancing to address PRP imbalance (20 records)
  4. Primary metric: macro-averaged F1 (penalises poor minority-class performance)
  5. Secondary metric: accuracy
  6. Safety constraint: escalation logic is unchanged (calibration affects
     only the ML classifier, not the symbolic reasoning pipeline)

Target performance
------------------
  Model B: ≥ 86% accuracy (up from 80%)
  Model C: 88–91% accuracy (up from 81.82%)

Important
---------
This module performs classifier-level calibration only. It does NOT:
  · Modify escalation thresholds (see EscalationSensitivityAnalyzer)
  · Modify pipeline rules (see rule_discrimination_refinement)
  · Alter symbolic feature construction (see symbolic_signal_enrichment)

The output provides recommended hyperparameter configurations to replace
the current default (n_estimators=200, max_depth=6, learning_rate=0.05).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Hyperparameter grids ──────────────────────────────────────────────────────

XGBOOST_GRID = {
    "n_estimators":  [100, 200, 300, 500],
    "max_depth":     [4, 6, 8],
    "learning_rate": [0.02, 0.05, 0.10, 0.20],
    "subsample":     [0.8, 1.0],
    "min_child_weight": [1, 3],
}

RF_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth":    [None, 8, 12, 16],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"],
}

LGBM_GRID = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [4, 6, 8],
    "learning_rate": [0.02, 0.05, 0.10],
    "num_leaves":    [15, 31, 63],
    "class_weight":  [None, "balanced"],
}

CATBOOST_GRID = {
    "iterations":   [100, 200, 300],
    "depth":        [4, 6, 8],
    "learning_rate": [0.02, 0.05, 0.10],
    "auto_class_weights": ["None", "Balanced"],
}

LR_GRID = {
    "C":           [0.01, 0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AlgorithmTrialResult:
    """
    Result of a single algorithm + hyperparameter configuration trial.

    Attributes
    ----------
    algorithm:
        Algorithm name.
    hyperparameters:
        Hyperparameter configuration evaluated.
    cv_mean_accuracy:
        Mean accuracy across all CV folds.
    cv_std_accuracy:
        Standard deviation of accuracy across folds.
    cv_mean_macro_f1:
        Mean macro F1 across all CV folds.
    cv_std_macro_f1:
        Standard deviation of macro F1 across folds.
    per_fold_accuracy:
        Accuracy for each individual fold.
    training_time_seconds:
        Total time for all CV folds.
    """

    algorithm:             str
    hyperparameters:       dict[str, Any]
    cv_mean_accuracy:      float             = 0.0
    cv_std_accuracy:       float             = 0.0
    cv_mean_macro_f1:      float             = 0.0
    cv_std_macro_f1:       float             = 0.0
    per_fold_accuracy:     list[float]       = field(default_factory=list)
    training_time_seconds: float             = 0.0

    def summary_line(self) -> str:
        return (
            f"{self.algorithm:15s} acc={self.cv_mean_accuracy:.4f}±{self.cv_std_accuracy:.4f} "
            f"f1={self.cv_mean_macro_f1:.4f}±{self.cv_std_macro_f1:.4f} "
            f"params={self.hyperparameters}"
        )


@dataclass
class CalibrationResult:
    """
    Complete calibration output for a model (B or C).

    Attributes
    ----------
    model_label:
        "model_b" or "model_c".
    all_trials:
        All evaluated algorithm × hyperparameter configurations.
    best_trial:
        Configuration with highest CV macro F1.
    top_5_trials:
        Top 5 configurations by macro F1.
    best_algorithm:
        Algorithm name of the best trial.
    best_hyperparameters:
        Hyperparameters of the best trial.
    best_cv_accuracy:
        Best mean CV accuracy.
    best_cv_macro_f1:
        Best mean CV macro F1.
    algorithms_tested:
        List of algorithms included in the search.
    n_configurations_tested:
        Total number of (algorithm, params) configurations evaluated.
    calibration_time_seconds:
        Total wall-clock time for the full calibration sweep.
    """

    model_label:               str
    all_trials:                list[AlgorithmTrialResult] = field(default_factory=list)
    best_trial:                AlgorithmTrialResult | None = None
    top_5_trials:              list[AlgorithmTrialResult] = field(default_factory=list)
    best_algorithm:            str   = ""
    best_hyperparameters:      dict[str, Any] = field(default_factory=dict)
    best_cv_accuracy:          float = 0.0
    best_cv_macro_f1:          float = 0.0
    algorithms_tested:         list[str] = field(default_factory=list)
    n_configurations_tested:   int   = 0
    calibration_time_seconds:  float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"CALIBRATION RESULT — {self.model_label.upper()}",
            "=" * 72,
            f"  Configurations tested : {self.n_configurations_tested}",
            f"  Algorithms            : {', '.join(self.algorithms_tested)}",
            f"  Calibration time      : {self.calibration_time_seconds:.1f}s",
            "-" * 72,
            f"  BEST: {self.best_algorithm}",
            f"    CV accuracy  : {self.best_cv_accuracy:.4f}",
            f"    CV macro F1  : {self.best_cv_macro_f1:.4f}",
            f"    Params       : {self.best_hyperparameters}",
            "-" * 72,
            "  TOP 5 CONFIGURATIONS:",
        ]
        for i, t in enumerate(self.top_5_trials, 1):
            lines.append(f"  {i}. {t.summary_line()}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Calibrator ────────────────────────────────────────────────────────────────

class BaselineCalibrator:
    """
    Multi-algorithm stratified repeated CV optimiser.

    Parameters
    ----------
    n_splits:
        Number of CV folds. Default 5.
    n_repeats:
        Number of CV repetitions. Default 5.
    seed:
        Master random seed.
    algorithms:
        List of algorithm names to include.
        Available: "xgboost", "random_forest", "lightgbm", "catboost",
        "logistic_regression".
    verbose:
        Print progress.
    fast_mode:
        If True, use a reduced parameter grid for faster sweeps.
    """

    def __init__(
        self,
        n_splits:   int = 5,
        n_repeats:  int = 5,
        seed:       int = 42,
        algorithms: list[str] | None = None,
        verbose:    bool = True,
        fast_mode:  bool = False,
    ) -> None:
        self.n_splits  = n_splits
        self.n_repeats = n_repeats
        self.seed      = seed
        self.algorithms = algorithms or [
            "xgboost", "random_forest", "logistic_regression"
        ]
        self.verbose   = verbose
        self.fast_mode = fast_mode

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate_model_b(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> CalibrationResult:
        """
        Calibrate Model B (12 clinical features).

        Parameters
        ----------
        X_train:
            Training matrix (n_samples × 12).
        y_train:
            0-based integer class labels.
        """
        self._log("Calibrating Model B (12 clinical features)...")
        return self._run_calibration(X_train, y_train, "model_b")

    def calibrate_model_c(
        self,
        X_train_combined: np.ndarray,
        y_train:          np.ndarray,
    ) -> CalibrationResult:
        """
        Calibrate Model C (12 clinical + symbolic signals).

        Parameters
        ----------
        X_train_combined:
            Combined feature matrix (n_samples × (12 + n_symbolic)).
        y_train:
            0-based integer class labels.
        """
        self._log(
            f"Calibrating Model C ({X_train_combined.shape[1]} combined features)..."
        )
        return self._run_calibration(X_train_combined, y_train, "model_c")

    def build_combined_matrix(
        self,
        X_clinical:       np.ndarray,
        symbolic_vectors: list[SymbolicFeatureVector],
    ) -> np.ndarray:
        """Build the combined feature matrix for Model C calibration."""
        if not symbolic_vectors:
            return X_clinical
        sym_keys = list(symbolic_vectors[0].to_dict().keys())
        X_sym    = np.array([
            [float(v.to_dict()[k]) for k in sym_keys]
            for v in symbolic_vectors
        ])
        return np.hstack([X_clinical, X_sym])

    # ── Internal calibration loop ─────────────────────────────────────────────

    def _run_calibration(
        self,
        X:     np.ndarray,
        y:     np.ndarray,
        label: str,
    ) -> CalibrationResult:
        t0     = time.monotonic()
        trials: list[AlgorithmTrialResult] = []
        algos_tested: list[str] = []

        for algo in self.algorithms:
            if algo not in algos_tested:
                algos_tested.append(algo)
            grid   = self._get_grid(algo)
            combos = list(self._expand_grid(grid))
            self._log(f"  {algo}: {len(combos)} configurations...")

            for params in combos:
                trial = self._evaluate_config(algo, params, X, y)
                trials.append(trial)

        # Sort by macro F1
        trials.sort(key=lambda t: t.cv_mean_macro_f1, reverse=True)
        best   = trials[0] if trials else None
        top5   = trials[:5]

        elapsed = time.monotonic() - t0
        return CalibrationResult(
            model_label=label,
            all_trials=trials,
            best_trial=best,
            top_5_trials=top5,
            best_algorithm=best.algorithm if best else "",
            best_hyperparameters=best.hyperparameters if best else {},
            best_cv_accuracy=best.cv_mean_accuracy if best else 0.0,
            best_cv_macro_f1=best.cv_mean_macro_f1 if best else 0.0,
            algorithms_tested=algos_tested,
            n_configurations_tested=len(trials),
            calibration_time_seconds=elapsed,
        )

    def _evaluate_config(
        self,
        algo:   str,
        params: dict[str, Any],
        X:      np.ndarray,
        y:      np.ndarray,
    ) -> AlgorithmTrialResult:
        """Run stratified repeated CV for a single configuration."""
        t0            = time.monotonic()
        fold_accs:    list[float] = []
        fold_f1s:     list[float] = []

        for repeat in range(self.n_repeats):
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed + repeat * 100,
            )
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                clf = self._build_clf(algo, params)
                try:
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_val)
                    fold_accs.append(float(accuracy_score(y_val, y_pred)))
                    fold_f1s.append(float(f1_score(
                        y_val, y_pred, average="macro", zero_division=0
                    )))
                except Exception:
                    fold_accs.append(0.0)
                    fold_f1s.append(0.0)

        elapsed = time.monotonic() - t0
        return AlgorithmTrialResult(
            algorithm=algo,
            hyperparameters=params,
            cv_mean_accuracy=float(np.mean(fold_accs)),
            cv_std_accuracy=float(np.std(fold_accs)),
            cv_mean_macro_f1=float(np.mean(fold_f1s)),
            cv_std_macro_f1=float(np.std(fold_f1s)),
            per_fold_accuracy=fold_accs,
            training_time_seconds=elapsed,
        )

    # ── Classifier factory ────────────────────────────────────────────────────

    def _build_clf(self, algo: str, params: dict[str, Any]) -> Any:
        if algo == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=params.get("n_estimators", 200),
                    max_depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.05),
                    subsample=params.get("subsample", 1.0),
                    min_child_weight=params.get("min_child_weight", 1),
                    random_state=self.seed,
                    eval_metric="mlogloss",
                    verbosity=0,
                    use_label_encoder=False,
                )
            except ImportError:
                return self._build_clf("random_forest", params)

        if algo == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                class_weight=params.get("class_weight", None),
                random_state=self.seed,
                n_jobs=-1,
            )

        if algo == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                cw = params.get("class_weight", None)
                return LGBMClassifier(
                    n_estimators=params.get("n_estimators", 200),
                    max_depth=params.get("max_depth", -1),
                    learning_rate=params.get("learning_rate", 0.05),
                    num_leaves=params.get("num_leaves", 31),
                    class_weight=cw,
                    random_state=self.seed,
                    n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                return self._build_clf("random_forest", params)

        if algo == "catboost":
            try:
                from catboost import CatBoostClassifier
                acw = params.get("auto_class_weights", "None")
                return CatBoostClassifier(
                    iterations=params.get("iterations", 200),
                    depth=params.get("depth", 6),
                    learning_rate=params.get("learning_rate", 0.05),
                    auto_class_weights=acw if acw != "None" else None,
                    random_seed=self.seed,
                    verbose=False,
                )
            except ImportError:
                return self._build_clf("random_forest", params)

        # logistic_regression (default)
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", None),
            max_iter=2000,
            random_state=self.seed,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
        )

    # ── Grid helpers ──────────────────────────────────────────────────────────

    def _get_grid(self, algo: str) -> dict[str, list[Any]]:
        full_grids = {
            "xgboost":             XGBOOST_GRID,
            "random_forest":       RF_GRID,
            "lightgbm":            LGBM_GRID,
            "catboost":            CATBOOST_GRID,
            "logistic_regression": LR_GRID,
        }
        grid = full_grids.get(algo, {})
        if self.fast_mode:
            grid = {k: v[:2] for k, v in grid.items()}
        return grid

    def _expand_grid(
        self,
        grid: dict[str, list[Any]],
    ):
        """Yield all combinations from a parameter grid dict."""
        if not grid:
            yield {}
            return
        keys   = list(grid.keys())
        values = list(grid.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[BaselineCalibrator] {msg}")
