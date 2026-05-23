"""
BaselineModelB — biopsy-free baseline classifier (Model B).

Model B uses only the 12 clinical features assessable without biopsy.
It establishes the diagnostic difficulty of the biopsy-free clinical setting
— the performance gap between Model A and Model B quantifies the information
lost when histopathological biopsy features are unavailable.

This gap is the motivating clinical challenge that Model C (symbolic reasoning
augmentation) attempts to partially compensate for.

Classifier family
-----------------
Identical to Model A — same XGBoost/RF/LR classifiers, same hyperparameters.
The ONLY difference is the feature space: 12 features instead of 34.

Diagnostic degradation metrics
-------------------------------
The primary output of comparing Model B to Model A:
  · Accuracy gap = Model_A.accuracy − Model_B.accuracy
  · F1 gap       = Model_A.macro_f1 − Model_B.macro_f1
  · Per-disease recall drop (which diseases suffer most without biopsy?)
  · Confusion zone activation (which disease pairs are confused more?)

Usage
-----
  from src.evaluation_pipeline.baseline_model_b import BaselineModelB, ModelBConfig

  model  = BaselineModelB(ModelBConfig(model_type="xgboost"))
  result = model.fit_and_evaluate(
      X_train, y_train, X_test, y_test,
      feature_names=list(CLINICAL_FEATURE_NAMES),
      class_labels=list(CANONICAL_DISEASES),
  )
  print(result.macro_f1)
"""

from __future__ import annotations

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

from src.dataset_integration.feature_partitioning import CLINICAL_FEATURE_NAMES
from src.dataset_integration.dataset_loader import CANONICAL_DISEASES


# ── Model B configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelBConfig:
    """
    Configuration for BaselineModelB.

    Intentionally mirrors ModelAConfig so comparisons are apples-to-apples.
    The only structural difference is the feature space (12 vs 34).
    """

    model_type:    str   = "xgboost"
    n_estimators:  int   = 200
    max_depth:     int   = 6
    learning_rate: float = 0.05
    seed:          int   = 42
    n_jobs:        int   = -1


# ── Model B result ────────────────────────────────────────────────────────────

@dataclass
class ModelBResult:
    """
    Evaluation result from BaselineModelB.

    Structurally identical to ModelAResult but documents the biopsy-free
    constraint in its partition and feature_count fields.
    """

    model_type:           str
    partition:            str = "clinical_features"
    feature_count:        int = 12
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

    def accuracy_gap_vs_model_a(self, model_a_accuracy: float) -> float:
        """Diagnostic degradation: Model A accuracy minus this model's accuracy."""
        return model_a_accuracy - self.accuracy

    def f1_gap_vs_model_a(self, model_a_macro_f1: float) -> float:
        """F1 degradation: Model A macro F1 minus this model's macro F1."""
        return model_a_macro_f1 - self.macro_f1

    def most_degraded_disease(
        self,
        model_a_recalls: dict[str, float],
    ) -> tuple[str, float]:
        """
        Return the disease with the largest recall drop compared to Model A.

        Parameters
        ----------
        model_a_recalls:
            Per-disease recall from the Model A result.
        """
        max_drop  = -1.0
        worst_dis = ""
        for dis, rec_a in model_a_recalls.items():
            rec_b = self.per_class_recall.get(dis, 0.0)
            drop  = rec_a - rec_b
            if drop > max_drop:
                max_drop  = drop
                worst_dis = dis
        return worst_dis, max(0.0, max_drop)

    def summary_line(self) -> str:
        return (
            f"Model B [{self.model_type}] "
            f"acc={self.accuracy:.4f} "
            f"macro_f1={self.macro_f1:.4f} "
            f"features={self.feature_count} "
            f"(train={self.n_train}, test={self.n_test})"
        )


# ── Classifier ────────────────────────────────────────────────────────────────

class BaselineModelB:
    """
    Biopsy-free baseline classifier using only the 12 clinical features.

    This model represents the diagnostic challenge: without histopathological
    biopsy data, classification becomes harder — particularly for disease
    pairs that share overlapping clinical presentations (confusion zones).

    The performance gap between Model B and Model A establishes the
    clinical problem that Model C's symbolic reasoning attempts to address.
    """

    def __init__(self, config: ModelBConfig | None = None) -> None:
        self.config = config or ModelBConfig()
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
        class_labels:  list[str] | None = None,
    ) -> "BaselineModelB":
        """
        Fit on training data using only clinical features.

        Parameters
        ----------
        X_train:
            Training matrix with exactly 12 clinical features.
        y_train:
            Integer class labels (1–6).
        feature_names:
            Clinical feature names (should be CLINICAL_FEATURE_NAMES).
        class_labels:
            Ordered canonical disease names.
        """
        self._feature_names = list(feature_names or list(CLINICAL_FEATURE_NAMES))
        self._class_labels  = list(class_labels or list(CANONICAL_DISEASES))
        self._clf.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BaselineModelB: call fit() before predict().")
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BaselineModelB: call fit() before predict().")
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X)
        return np.zeros((len(X), 6))

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_labels: list[str] | None = None,
    ) -> ModelBResult:
        """Evaluate on held-out test set."""
        labels   = class_labels or self._class_labels
        y_pred   = self.predict(X_test)
        acc      = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm       = confusion_matrix(y_test, y_pred).tolist()

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
            disease = labels[cls_int] if 0 <= cls_int < len(labels) else str(cls_int)
            cls_str = str(cls_int)
            if cls_str in report:
                per_prec[disease] = float(report[cls_str]["precision"])
                per_rec[disease]  = float(report[cls_str]["recall"])
                per_f1[disease]   = float(report[cls_str]["f1-score"])

        importances = self._extract_importances()

        return ModelBResult(
            model_type=self.config.model_type,
            partition="clinical_features",
            feature_count=len(self._feature_names),
            accuracy=acc,
            macro_f1=macro_f1,
            per_class_precision=per_prec,
            per_class_recall=per_rec,
            per_class_f1=per_f1,
            confusion_matrix=cm,
            class_labels=labels,
            feature_importances=importances,
            n_train=0,
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
    ) -> ModelBResult:
        """Convenience: fit then evaluate."""
        self.fit(X_train, y_train, feature_names, class_labels)
        result = self.evaluate(X_test, y_test, class_labels)
        result.n_train = len(y_train)
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_classifier(self) -> Any:
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
        return LogisticRegression(
            max_iter=1000,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            solver="lbfgs",
            multi_class="multinomial",
        )

    def _extract_importances(self) -> dict[str, float]:
        if not self._feature_names:
            return {}
        if hasattr(self._clf, "feature_importances_"):
            vals = self._clf.feature_importances_
            return {
                self._feature_names[i]: float(vals[i])
                for i in range(min(len(vals), len(self._feature_names)))
            }
        if hasattr(self._clf, "coef_"):
            import numpy as np_inner
            coef = np_inner.abs(self._clf.coef_).mean(axis=0)
            return {
                self._feature_names[i]: float(coef[i])
                for i in range(min(len(coef), len(self._feature_names)))
            }
        return {}
