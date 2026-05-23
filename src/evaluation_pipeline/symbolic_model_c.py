"""
SymbolicModelC — symbolic reasoning augmentation classifier (Model C).

Model C is the PRIMARY CONTRIBUTION model. It combines:
  · 12 clinical features (same as Model B — no biopsy)
  · 24 symbolic reasoning signals (from SymbolicFeatureAdapter)

The symbolic reasoning signals are genuine inference outputs from the
multi-stage diagnostic reasoning pipeline executed on each patient's
clinical data — NOT hand-crafted numerical transformations.

Why this is not simply feature engineering
------------------------------------------
The SymbolicFeatureAdapter runs the full reasoning stack for each patient:
  Stage 0  — clinical grading (fuzzy conversion)
  Stage 1  — evidence activation (rule firing)
  Stage 2  — contradiction analysis (bilateral conflict detection)
  Stage 3  — certainty propagation (hypothesis weighting)
  Stage 4  — differential competition (hypothesis ranking)
  Stage 5  — evidence sufficiency (support assessment)
  Stage 6  — instability monitoring (oscillation detection)
  Stage 7  — FSM transition + safety gate + escalation decision

The signals extracted — certainty, contradiction_load, ambiguity_index,
convergence_index, oscillation_count, stabilisation_stage, etc. — are
structural properties of the reasoning trajectory over the clinical input,
not simple transformations of individual feature values.

Primary hypothesis
------------------
Symbolic reasoning signals partially compensate for missing histopathological
information by:
  · Disambiguating clinically similar diseases (contradiction detection)
  · Escalating uncertain cases before incorrect classification occurs
  · Stabilising certainty in diseases with strong clinical signatures
  · Encoding reasoning trajectory dynamics (convergence, oscillation)

Usage
-----
  from src.evaluation_pipeline.symbolic_model_c import SymbolicModelC, ModelCConfig
  from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureAdapter

  adapter = SymbolicFeatureAdapter()
  vectors = adapter.adapt_batch(train_records)

  model  = SymbolicModelC()
  result = model.fit_and_evaluate(
      X_train_clinical, vectors_train, y_train,
      X_test_clinical,  vectors_test,  y_test,
      clinical_feature_names=list(CLINICAL_FEATURE_NAMES),
      class_labels=list(CANONICAL_DISEASES),
  )
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
from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Model C configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelCConfig:
    """
    Configuration for SymbolicModelC.

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
        Parallelism for RF/LR. -1 = all cores.
    """

    model_type:    str   = "xgboost"
    n_estimators:  int   = 200
    max_depth:     int   = 6
    learning_rate: float = 0.05
    seed:          int   = 42
    n_jobs:        int   = -1


# ── Model C result ────────────────────────────────────────────────────────────

@dataclass
class ModelCResult:
    """
    Evaluation result from SymbolicModelC.

    In addition to standard classification metrics, this result tracks
    the relative importance of symbolic reasoning signals vs raw clinical
    features, providing direct evidence of symbolic contribution.

    Attributes
    ----------
    model_type:
        Classifier family used.
    partition:
        Always "clinical_plus_symbolic".
    clinical_feature_count:
        Always 12.
    symbolic_feature_count:
        Number of symbolic reasoning signals used.
    total_feature_count:
        clinical_feature_count + symbolic_feature_count.
    accuracy, macro_f1, ...:
        Standard classification metrics.
    symbolic_feature_importances:
        Importances for symbolic reasoning signals only.
    clinical_feature_importances:
        Importances for clinical features only.
    symbolic_importance_fraction:
        Sum of symbolic importances / total importance.
        Quantifies the symbolic system's contribution to classification.
    n_train, n_test:
        Dataset sizes.
    reasoning_vectors_used:
        Count of patients for which pipeline execution succeeded.
    """

    model_type:                   str
    partition:                    str   = "clinical_plus_symbolic"
    clinical_feature_count:       int   = 12
    symbolic_feature_count:       int   = 0
    total_feature_count:          int   = 0
    accuracy:                     float = 0.0
    macro_f1:                     float = 0.0
    per_class_precision:          dict[str, float] = field(default_factory=dict)
    per_class_recall:             dict[str, float] = field(default_factory=dict)
    per_class_f1:                 dict[str, float] = field(default_factory=dict)
    confusion_matrix:             list[list[int]]  = field(default_factory=list)
    class_labels:                 list[str]        = field(default_factory=list)
    feature_importances:          dict[str, float] = field(default_factory=dict)
    symbolic_feature_importances: dict[str, float] = field(default_factory=dict)
    clinical_feature_importances: dict[str, float] = field(default_factory=dict)
    symbolic_importance_fraction: float = 0.0
    n_train:                      int   = 0
    n_test:                       int   = 0
    reasoning_vectors_used:       int   = 0

    def accuracy_gap_vs_model_b(self, model_b_accuracy: float) -> float:
        """Symbolic lift: this model's accuracy minus Model B's accuracy."""
        return self.accuracy - model_b_accuracy

    def recovery_rate(
        self,
        model_a_accuracy: float,
        model_b_accuracy: float,
    ) -> float:
        """
        Fraction of the biopsy accuracy gap recovered by symbolic reasoning.

        recovery_rate = (Model_C − Model_B) / (Model_A − Model_B)

        1.0 = full recovery to biopsy-level performance.
        0.0 = no improvement over biopsy-free baseline.
        """
        gap = model_a_accuracy - model_b_accuracy
        if gap < 1e-9:
            return 0.0
        return max(0.0, (self.accuracy - model_b_accuracy) / gap)

    def summary_line(self) -> str:
        return (
            f"Model C [{self.model_type}] "
            f"acc={self.accuracy:.4f} "
            f"macro_f1={self.macro_f1:.4f} "
            f"features={self.total_feature_count} "
            f"(clinical={self.clinical_feature_count}, "
            f"symbolic={self.symbolic_feature_count}) "
            f"symbolic_weight={self.symbolic_importance_fraction:.3f} "
            f"(train={self.n_train}, test={self.n_test})"
        )


# ── Symbolic Model C classifier ───────────────────────────────────────────────

class SymbolicModelC:
    """
    Symbolic reasoning augmentation classifier.

    Combines 12 clinical features with symbolic reasoning signals derived
    from executing the full symbolic pipeline on each patient's clinical data.

    The combined feature matrix has shape (n_patients, 12 + n_symbolic_signals).
    The classifier learns which combination of clinical and reasoning features
    best predicts the correct disease label.

    Parameters
    ----------
    config:
        ModelCConfig controlling classifier family and hyperparameters.
    """

    def __init__(self, config: ModelCConfig | None = None) -> None:
        self.config = config or ModelCConfig()
        self._clf   = self._build_classifier()
        self._clinical_names:  list[str] = []
        self._symbolic_names:  list[str] = []
        self._combined_names:  list[str] = []
        self._class_labels:    list[str] = []
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_clinical: np.ndarray,
        symbolic_vectors: list[SymbolicFeatureVector],
        y_train: np.ndarray,
        clinical_feature_names: list[str] | None = None,
        class_labels: list[str] | None = None,
    ) -> "SymbolicModelC":
        """
        Fit on combined clinical + symbolic feature matrix.

        Parameters
        ----------
        X_clinical:
            Clinical feature matrix (n_samples × 12).
        symbolic_vectors:
            Symbolic reasoning signals for each patient.
        y_train:
            Integer class labels (1–6).
        clinical_feature_names:
            Names of the 12 clinical features.
        class_labels:
            Ordered canonical disease names.
        """
        self._clinical_names = list(clinical_feature_names or list(CLINICAL_FEATURE_NAMES))
        self._class_labels   = list(class_labels or list(CANONICAL_DISEASES))

        X_combined, sym_names = self._build_combined_matrix(X_clinical, symbolic_vectors)
        self._symbolic_names = sym_names
        self._combined_names = self._clinical_names + sym_names

        self._clf.fit(X_combined, y_train)
        self._fitted = True
        return self

    def predict(
        self,
        X_clinical: np.ndarray,
        symbolic_vectors: list[SymbolicFeatureVector],
    ) -> np.ndarray:
        """Return predicted integer class labels."""
        if not self._fitted:
            raise RuntimeError("SymbolicModelC: call fit() before predict().")
        X_combined, _ = self._build_combined_matrix(X_clinical, symbolic_vectors)
        return self._clf.predict(X_combined)

    def predict_proba(
        self,
        X_clinical: np.ndarray,
        symbolic_vectors: list[SymbolicFeatureVector],
    ) -> np.ndarray:
        """Return per-class probability estimates."""
        if not self._fitted:
            raise RuntimeError("SymbolicModelC: call fit() before predict().")
        X_combined, _ = self._build_combined_matrix(X_clinical, symbolic_vectors)
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X_combined)
        return np.zeros((len(X_clinical), 6))

    def evaluate(
        self,
        X_clinical: np.ndarray,
        symbolic_vectors: list[SymbolicFeatureVector],
        y_test: np.ndarray,
        class_labels: list[str] | None = None,
    ) -> ModelCResult:
        """Evaluate on held-out test set."""
        labels   = class_labels or self._class_labels
        y_pred   = self.predict(X_clinical, symbolic_vectors)
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

        all_imp  = self._extract_importances()
        sym_imp  = {k: v for k, v in all_imp.items() if k in self._symbolic_names}
        clin_imp = {k: v for k, v in all_imp.items() if k in self._clinical_names}

        total_imp  = sum(all_imp.values()) or 1.0
        sym_frac   = sum(sym_imp.values()) / total_imp

        vectors_used = sum(1 for v in symbolic_vectors if v.pipeline_success)

        return ModelCResult(
            model_type=self.config.model_type,
            partition="clinical_plus_symbolic",
            clinical_feature_count=len(self._clinical_names),
            symbolic_feature_count=len(self._symbolic_names),
            total_feature_count=len(self._combined_names),
            accuracy=acc,
            macro_f1=macro_f1,
            per_class_precision=per_prec,
            per_class_recall=per_rec,
            per_class_f1=per_f1,
            confusion_matrix=cm,
            class_labels=labels,
            feature_importances=all_imp,
            symbolic_feature_importances=sym_imp,
            clinical_feature_importances=clin_imp,
            symbolic_importance_fraction=sym_frac,
            n_train=0,
            n_test=len(y_test),
            reasoning_vectors_used=vectors_used,
        )

    def fit_and_evaluate(
        self,
        X_train_clinical: np.ndarray,
        vectors_train:    list[SymbolicFeatureVector],
        y_train:          np.ndarray,
        X_test_clinical:  np.ndarray,
        vectors_test:     list[SymbolicFeatureVector],
        y_test:           np.ndarray,
        clinical_feature_names: list[str] | None = None,
        class_labels:           list[str] | None = None,
    ) -> ModelCResult:
        """Convenience: fit then evaluate."""
        self.fit(
            X_train_clinical, vectors_train, y_train,
            clinical_feature_names, class_labels,
        )
        result = self.evaluate(X_test_clinical, vectors_test, y_test, class_labels)
        result.n_train = len(y_train)
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_combined_matrix(
        self,
        X_clinical: np.ndarray,
        symbolic_vectors: list[SymbolicFeatureVector],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Concatenate clinical features with symbolic reasoning signals.

        Returns (combined_matrix, symbolic_signal_names).
        """
        if not symbolic_vectors:
            return X_clinical, []

        sym_keys   = list(symbolic_vectors[0].to_dict().keys())
        X_symbolic = np.array([
            [float(v.to_dict()[k]) for k in sym_keys]
            for v in symbolic_vectors
        ])

        X_combined = np.hstack([X_clinical, X_symbolic])
        return X_combined, sym_keys

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
        if not self._combined_names:
            return {}
        if hasattr(self._clf, "feature_importances_"):
            vals = self._clf.feature_importances_
            return {
                self._combined_names[i]: float(vals[i])
                for i in range(min(len(vals), len(self._combined_names)))
            }
        if hasattr(self._clf, "coef_"):
            coef = np.abs(self._clf.coef_).mean(axis=0)
            return {
                self._combined_names[i]: float(coef[i])
                for i in range(min(len(coef), len(self._combined_names)))
            }
        return {}
