"""
BaselineModelA — full biopsy reference classifier (Model A).

Model A uses all 34 dataset features (12 clinical + 22 histopathological)
to establish the diagnostic upper bound achievable when complete biopsy
information is available.

This is the benchmark ceiling — the performance target that cannot be
exceeded by any biopsy-free system, including Model C.

Classifier family
-----------------
  Primary:   XGBoost gradient boosting (optimised for tabular clinical data)
  Secondary: Random Forest (ensemble reference)
  Tertiary:  Logistic Regression (linear clinical baseline)

All classifiers are trained with stratified splits and reproducible seeds.
XGBoost is the primary classifier for direct comparison across A/B/C.

Performance measured
---------------------
  · Overall accuracy
  · Macro-averaged F1 (primary — penalises poor minority-class performance)
  · Per-disease precision / recall / F1
  · Confusion matrix (6 × 6)
  · Feature importances (for clinical interpretation)

Usage
-----
  from src.evaluation_pipeline.baseline_model_a import BaselineModelA, ModelAConfig

  model  = BaselineModelA(ModelAConfig(model_type="xgboost"))
  result = model.fit_and_evaluate(
      X_train, y_train, X_test, y_test,
      feature_names=ALL_FEATURE_NAMES,
      class_labels=CANONICAL_DISEASES,
  )
  print(result.macro_f1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.dataset_integration.feature_partitioning import (
    ALL_FEATURE_NAMES,
    FeaturePartition,
)
from src.dataset_integration.dataset_loader import CANONICAL_DISEASES


# ── Model A configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelAConfig:
    """
    Configuration for BaselineModelA.

    Attributes
    ----------
    model_type:
        Classifier family: "xgboost" | "random_forest" | "logistic_regression".
    n_estimators:
        Tree count for XGBoost and RandomForest.
    max_depth:
        Maximum tree depth.
    learning_rate:
        XGBoost learning rate.
    seed:
        Random seed for reproducibility.
    n_jobs:
        Parallelism for RandomForest and LogisticRegression. -1 = all cores.
    """

    model_type:    str   = "xgboost"
    n_estimators:  int   = 200
    max_depth:     int   = 6
    learning_rate: float = 0.05
    seed:          int   = 42
    n_jobs:        int   = -1


# ── Model A result ────────────────────────────────────────────────────────────

@dataclass
class ModelAResult:
    """
    Evaluation result from a single BaselineModelA training and test run.

    Attributes
    ----------
    model_type:
        Classifier family used.
    partition:
        Always "all_features" (34 features).
    feature_count:
        Always 34.
    accuracy:
        Overall classification accuracy on the test set.
    macro_f1:
        Macro-averaged F1 across all 6 disease classes.
    per_class_precision:
        Per-disease precision scores.
    per_class_recall:
        Per-disease recall scores.
    per_class_f1:
        Per-disease F1 scores.
    confusion_matrix:
        6×6 confusion matrix (rows=true, cols=predicted).
    class_labels:
        Ordered disease label list corresponding to matrix rows/columns.
    feature_importances:
        Per-feature importance score (XGBoost/RF) or coefficient magnitude (LR).
    n_train:
        Number of training records.
    n_test:
        Number of test records.
    """

    model_type:           str
    partition:            str = "all_features"
    feature_count:        int = 34
    accuracy:             float = 0.0
    macro_f1:             float = 0.0
    per_class_precision:  dict[str, float] = field(default_factory=dict)
    per_class_recall:     dict[str, float] = field(default_factory=dict)
    per_class_f1:         dict[str, float] = field(default_factory=dict)
    confusion_matrix:     list[list[int]] = field(default_factory=list)
    class_labels:         list[str] = field(default_factory=list)
    feature_importances:  dict[str, float] = field(default_factory=dict)
    n_train:              int = 0
    n_test:               int = 0

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top-n features by importance, descending."""
        return sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]

    def disease_accuracy(self, disease: str) -> float:
        """Return per-disease recall (true positive rate) for a disease."""
        return self.per_class_recall.get(disease, 0.0)

    def summary_line(self) -> str:
        return (
            f"Model A [{self.model_type}] "
            f"acc={self.accuracy:.4f} "
            f"macro_f1={self.macro_f1:.4f} "
            f"features={self.feature_count} "
            f"(train={self.n_train}, test={self.n_test})"
        )


# ── Classifier ────────────────────────────────────────────────────────────────

class BaselineModelA:
    """
    Full-biopsy reference classifier using all 34 dermatological features.

    This is the upper-bound benchmark: the best achievable diagnostic
    performance when complete histopathological biopsy information is
    available.

    The classifier is fitted once and can be evaluated on any held-out set.
    """

    def __init__(self, config: ModelAConfig | None = None) -> None:
        self.config = config or ModelAConfig()
        self._clf   = self._build_classifier()
        self._feature_names: list[str] = []
        self._class_labels:  list[str] = []
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
        class_labels: list[str] | None = None,
    ) -> "BaselineModelA":
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X_train:
            Training feature matrix (n_samples × n_features).
        y_train:
            Integer class labels (1–6) for training samples.
        feature_names:
            Ordered feature names corresponding to X_train columns.
        class_labels:
            Ordered canonical disease names (1–6 order).
        """
        self._feature_names = list(feature_names or ALL_FEATURE_NAMES)
        self._class_labels  = list(class_labels or list(CANONICAL_DISEASES))
        self._clf.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted integer class labels."""
        if not self._fitted:
            raise RuntimeError("BaselineModelA: call fit() before predict().")
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class probability estimates."""
        if not self._fitted:
            raise RuntimeError("BaselineModelA: call fit() before predict().")
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X)
        return np.zeros((len(X), 6))

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_labels: list[str] | None = None,
    ) -> ModelAResult:
        """
        Evaluate on a held-out test set and return ModelAResult.

        Parameters
        ----------
        X_test:
            Test feature matrix.
        y_test:
            True integer class labels.
        class_labels:
            Ordered canonical disease names. Uses fitted labels if None.
        """
        labels = class_labels or self._class_labels
        y_pred = self.predict(X_test)

        acc      = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm       = confusion_matrix(y_test, y_pred).tolist()

        # Per-class metrics — map integer labels back to disease names
        unique_classes = sorted(set(y_test) | set(y_pred))
        per_prec: dict[str, float] = {}
        per_rec:  dict[str, float] = {}
        per_f1:   dict[str, float] = {}

        report = classification_report(
            y_test, y_pred,
            labels=unique_classes,
            zero_division=0,
            output_dict=True,
        )
        for cls_int in unique_classes:
            # Map 0-based integer class to disease label
            disease = labels[cls_int] if 0 <= cls_int < len(labels) else str(cls_int)
            cls_str = str(cls_int)
            if cls_str in report:
                per_prec[disease] = float(report[cls_str]["precision"])
                per_rec[disease]  = float(report[cls_str]["recall"])
                per_f1[disease]   = float(report[cls_str]["f1-score"])

        importances = self._extract_importances()

        return ModelAResult(
            model_type=self.config.model_type,
            partition="all_features",
            feature_count=len(self._feature_names),
            accuracy=acc,
            macro_f1=macro_f1,
            per_class_precision=per_prec,
            per_class_recall=per_rec,
            per_class_f1=per_f1,
            confusion_matrix=cm,
            class_labels=labels,
            feature_importances=importances,
            n_train=0,       # caller sets after fit()
            n_test=len(y_test),
        )

    def fit_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        feature_names: list[str] | None = None,
        class_labels:  list[str] | None = None,
    ) -> ModelAResult:
        """Convenience method: fit then evaluate in one call."""
        self.fit(X_train, y_train, feature_names, class_labels)
        result = self.evaluate(X_test, y_test, class_labels)
        result.n_train = len(y_train)
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_classifier(self) -> Any:
        """Construct the underlying classifier from config."""
        cfg = self.config
        if cfg.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    learning_rate=cfg.learning_rate,
                    random_state=cfg.seed,
                    eval_metric="mlogloss",
                    verbosity=0,
                    use_label_encoder=False,
                )
            except ImportError:
                # Fall back to Random Forest if XGBoost not installed
                return RandomForestClassifier(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    random_state=cfg.seed,
                    n_jobs=cfg.n_jobs,
                )
        if cfg.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                random_state=cfg.seed,
                n_jobs=cfg.n_jobs,
            )
        # Logistic regression
        return LogisticRegression(
            max_iter=1000,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            solver="lbfgs",
            multi_class="multinomial",
        )

    def _extract_importances(self) -> dict[str, float]:
        """Extract feature importances / coefficient magnitudes."""
        if not self._feature_names:
            return {}

        if hasattr(self._clf, "feature_importances_"):
            vals = self._clf.feature_importances_
            return {
                self._feature_names[i]: float(vals[i])
                for i in range(min(len(vals), len(self._feature_names)))
            }

        if hasattr(self._clf, "coef_"):
            # Logistic regression: mean absolute coefficient across classes
            coef = np.abs(self._clf.coef_).mean(axis=0)
            return {
                self._feature_names[i]: float(coef[i])
                for i in range(min(len(coef), len(self._feature_names)))
            }

        return {}
