"""
EvaluationRunner — orchestrator for the tripartite A/B/C clinical evaluation.

Coordinates the full training and evaluation pipeline across all three models:
  Model A — full biopsy reference (34 features)
  Model B — biopsy-free baseline (12 clinical features)
  Model C — symbolic reasoning augmentation (12 clinical + reasoning signals)

Execution sequence
------------------
  1. Load dataset and apply stratified split
  2. Execute symbolic reasoning pipeline on all training and test records
  3. Fit Model A on training set (all 34 features)
  4. Fit Model B on training set (12 clinical features)
  5. Fit Model C on training set (12 clinical + symbolic signals)
  6. Evaluate all three models on the same held-out test set
  7. Compute comparative metrics and return TripartiteEvaluationResult

The symbolic pipeline execution step (step 2) is the most computationally
intensive — it runs the full 9-stage reasoning pipeline on all 366 patients.

Usage
-----
  from src.evaluation_pipeline.evaluation_runner import EvaluationRunner
  from src.dataset_integration.dataset_loader import DermatologyDatasetLoader

  dataset = DermatologyDatasetLoader.load("dermatology_with_labels.csv")
  runner  = EvaluationRunner()
  result  = runner.run(dataset)
  print(result.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.dataset_integration.dataset_loader import (
    DermatologyDataset,
    DermatologyRecord,
    CANONICAL_DISEASES,
)
from src.dataset_integration.feature_partitioning import (
    CLINICAL_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
)
from src.dataset_integration.clinical_feature_mapper import ClinicalFeatureMapper
from src.dataset_integration.symbolic_feature_adapter import (
    SymbolicFeatureAdapter,
    SymbolicFeatureVector,
)
from src.dataset_integration.dataset_splitter import DatasetSplitter, DataSplit
from src.evaluation_pipeline.baseline_model_a import (
    BaselineModelA,
    ModelAConfig,
    ModelAResult,
)
from src.evaluation_pipeline.baseline_model_b import (
    BaselineModelB,
    ModelBConfig,
    ModelBResult,
)
from src.evaluation_pipeline.symbolic_model_c import (
    SymbolicModelC,
    ModelCConfig,
    ModelCResult,
)


# ── Evaluation configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for the tripartite evaluation run.

    Attributes
    ----------
    seed:
        Master random seed for all splits and classifiers.
    train_ratio:
        Fraction of dataset for training. Default 0.70.
    validation_ratio:
        Fraction for validation. Default 0.15.
    test_ratio:
        Fraction for held-out test. Default 0.15.
    model_type:
        Classifier family for A/B/C: "xgboost" | "random_forest" | "logistic_regression".
    n_estimators:
        Tree count.
    max_depth:
        Maximum tree depth.
    age_imputation_value:
        Age median used for imputing missing age values.
    suppress_pipeline_errors:
        If True, symbolic pipeline failures produce fallback zero vectors.
    verbose:
        Print progress during execution.
    """

    seed:                     int   = 42
    train_ratio:              float = 0.70
    validation_ratio:         float = 0.15
    test_ratio:               float = 0.15
    model_type:               str   = "xgboost"
    n_estimators:             int   = 200
    max_depth:                int   = 6
    learning_rate:            float = 0.05
    age_imputation_value:     float = 35.0
    suppress_pipeline_errors: bool  = True
    verbose:                  bool  = True


# ── Tripartite evaluation result ──────────────────────────────────────────────

@dataclass
class TripartiteEvaluationResult:
    """
    Complete output of one full A/B/C evaluation run.

    Attributes
    ----------
    model_a:
        Full-biopsy reference result.
    model_b:
        Biopsy-free baseline result.
    model_c:
        Symbolic reasoning augmentation result.
    data_split:
        The stratified split used for this evaluation.
    train_symbolic_vectors:
        Reasoning signals for training records.
    test_symbolic_vectors:
        Reasoning signals for test records.
    biopsy_free_accuracy_gap:
        Accuracy drop from A to B (information lost without biopsy).
    symbolic_lift:
        Accuracy gain from B to C (symbolic reasoning contribution).
    symbolic_recovery_rate:
        Fraction of the A-B gap recovered by C. Range [0, 1].
    biopsy_free_f1_gap:
        Macro F1 drop from A to B.
    symbolic_f1_lift:
        Macro F1 gain from B to C.
    execution_time_seconds:
        Total wall-clock time for the full evaluation run.
    pipeline_success_count:
        Number of patients for whom symbolic pipeline succeeded.
    pipeline_failure_count:
        Number of patients for whom symbolic pipeline produced a fallback.
    config:
        The EvaluationConfig used for this run.
    """

    model_a:                    ModelAResult
    model_b:                    ModelBResult
    model_c:                    ModelCResult
    data_split:                 DataSplit
    train_symbolic_vectors:     list[SymbolicFeatureVector] = field(default_factory=list)
    test_symbolic_vectors:      list[SymbolicFeatureVector] = field(default_factory=list)
    biopsy_free_accuracy_gap:   float = 0.0
    symbolic_lift:              float = 0.0
    symbolic_recovery_rate:     float = 0.0
    biopsy_free_f1_gap:         float = 0.0
    symbolic_f1_lift:           float = 0.0
    execution_time_seconds:     float = 0.0
    pipeline_success_count:     int   = 0
    pipeline_failure_count:     int   = 0
    config:                     EvaluationConfig = field(
        default_factory=EvaluationConfig,
    )

    @property
    def all_symbolic_vectors(self) -> list[SymbolicFeatureVector]:
        return self.train_symbolic_vectors + self.test_symbolic_vectors

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "TRIPARTITE CLINICAL EVALUATION SUMMARY",
            "=" * 72,
            f"  Dataset:  {self.data_split.n_train} train / "
            f"{self.data_split.n_test} test",
            f"  Symbolic: {self.pipeline_success_count} pipeline successes / "
            f"{self.pipeline_failure_count} failures",
            "-" * 72,
            f"  {self.model_a.summary_line()}",
            f"  {self.model_b.summary_line()}",
            f"  {self.model_c.summary_line()}",
            "-" * 72,
            f"  Biopsy-free accuracy gap   : "
            f"{self.biopsy_free_accuracy_gap:+.4f}  (A minus B)",
            f"  Symbolic accuracy lift     : "
            f"{self.symbolic_lift:+.4f}  (C minus B)",
            f"  Symbolic recovery rate     : "
            f"{self.symbolic_recovery_rate:.1%}",
            f"  Biopsy-free F1 gap         : "
            f"{self.biopsy_free_f1_gap:+.4f}",
            f"  Symbolic F1 lift           : "
            f"{self.symbolic_f1_lift:+.4f}",
            f"  Symbolic feature weight    : "
            f"{self.model_c.symbolic_importance_fraction:.1%}",
            f"  Execution time             : "
            f"{self.execution_time_seconds:.1f}s",
            "=" * 72,
        ]
        return "\n".join(lines)


# ── Evaluation runner ─────────────────────────────────────────────────────────

class EvaluationRunner:
    """
    Orchestrates the full three-model evaluation on the UCI Dermatology dataset.

    Parameters
    ----------
    config:
        EvaluationConfig. Default settings use XGBoost 200-tree / 70-15-15 split.
    symbolic_adapter:
        Pre-built SymbolicFeatureAdapter. Built lazily if None.
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        symbolic_adapter: SymbolicFeatureAdapter | None = None,
    ) -> None:
        self.config  = config or EvaluationConfig()
        self._adapter = symbolic_adapter

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, dataset: DermatologyDataset) -> TripartiteEvaluationResult:
        """
        Execute the full A/B/C evaluation on the dataset.

        Parameters
        ----------
        dataset:
            Loaded DermatologyDataset from DermatologyDatasetLoader.

        Returns
        -------
        TripartiteEvaluationResult with all three model results and
        comparative metrics.
        """
        t0  = time.monotonic()
        cfg = self.config
        self._log("Starting tripartite evaluation.")

        # Step 1: Stratified split
        self._log("Applying stratified split...")
        splitter = DatasetSplitter(seed=cfg.seed)
        split    = splitter.split(
            dataset,
            train_ratio=cfg.train_ratio,
            validation_ratio=cfg.validation_ratio,
            test_ratio=cfg.test_ratio,
        )
        self._log(
            f"  Split: train={split.n_train}, "
            f"val={split.n_validation}, test={split.n_test}"
        )

        # Step 2: Symbolic pipeline execution on train + test
        self._log("Executing symbolic reasoning pipeline on all records...")
        adapter = self._get_adapter()
        train_vecs = adapter.adapt_batch(list(split.train_records))
        test_vecs  = adapter.adapt_batch(list(split.test_records))

        n_success = sum(1 for v in train_vecs + test_vecs if v.pipeline_success)
        n_fail    = len(train_vecs) + len(test_vecs) - n_success
        self._log(f"  Symbolic execution: {n_success} successes, {n_fail} failures.")

        # Step 3: Build feature matrices
        self._log("Building feature matrices...")
        class_labels = list(CANONICAL_DISEASES)

        X_train_all   = np.array(split.train_feature_matrix(ALL_FEATURE_NAMES))
        X_test_all    = np.array(split.test_feature_matrix(ALL_FEATURE_NAMES))
        X_train_clin  = np.array(split.train_feature_matrix(CLINICAL_FEATURE_NAMES))
        X_test_clin   = np.array(split.test_feature_matrix(CLINICAL_FEATURE_NAMES))
        y_train       = np.array(split.train_labels)
        y_test        = np.array(split.test_labels)

        # Convert string labels to 0-based integer codes for XGBoost / sklearn
        # (class_labels is ordered: index 0 = psoriasis, 1 = seborrheic_dermatitis, …)
        label_to_int  = {lbl: i for i, lbl in enumerate(class_labels)}
        y_train_int   = np.array([label_to_int.get(l, 0) for l in y_train])
        y_test_int    = np.array([label_to_int.get(l, 0) for l in y_test])

        # Step 4: Fit and evaluate Model A
        self._log("Training Model A (full biopsy reference, 34 features)...")
        model_a = BaselineModelA(ModelAConfig(
            model_type=cfg.model_type,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
        ))
        result_a = model_a.fit_and_evaluate(
            X_train_all, y_train_int,
            X_test_all,  y_test_int,
            feature_names=list(ALL_FEATURE_NAMES),
            class_labels=class_labels,
        )
        self._log(f"  {result_a.summary_line()}")

        # Step 5: Fit and evaluate Model B
        self._log("Training Model B (biopsy-free baseline, 12 features)...")
        model_b = BaselineModelB(ModelBConfig(
            model_type=cfg.model_type,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
        ))
        result_b = model_b.fit_and_evaluate(
            X_train_clin, y_train_int,
            X_test_clin,  y_test_int,
            feature_names=list(CLINICAL_FEATURE_NAMES),
            class_labels=class_labels,
        )
        self._log(f"  {result_b.summary_line()}")

        # Step 6: Fit and evaluate Model C
        self._log("Training Model C (symbolic reasoning augmentation)...")
        model_c = SymbolicModelC(ModelCConfig(
            model_type=cfg.model_type,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
        ))
        result_c = model_c.fit_and_evaluate(
            X_train_clin, train_vecs, y_train_int,
            X_test_clin,  test_vecs,  y_test_int,
            clinical_feature_names=list(CLINICAL_FEATURE_NAMES),
            class_labels=class_labels,
        )
        self._log(f"  {result_c.summary_line()}")

        # Step 7: Comparative metrics
        acc_gap   = result_a.accuracy - result_b.accuracy
        sym_lift  = result_c.accuracy - result_b.accuracy
        rec_rate  = sym_lift / acc_gap if acc_gap > 1e-9 else 0.0
        f1_gap    = result_a.macro_f1 - result_b.macro_f1
        f1_lift   = result_c.macro_f1 - result_b.macro_f1

        elapsed = time.monotonic() - t0
        self._log(f"Evaluation complete in {elapsed:.1f}s.")

        return TripartiteEvaluationResult(
            model_a=result_a,
            model_b=result_b,
            model_c=result_c,
            data_split=split,
            train_symbolic_vectors=train_vecs,
            test_symbolic_vectors=test_vecs,
            biopsy_free_accuracy_gap=acc_gap,
            symbolic_lift=sym_lift,
            symbolic_recovery_rate=max(0.0, min(1.0, rec_rate)),
            biopsy_free_f1_gap=f1_gap,
            symbolic_f1_lift=f1_lift,
            execution_time_seconds=elapsed,
            pipeline_success_count=n_success,
            pipeline_failure_count=n_fail,
            config=cfg,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_adapter(self) -> SymbolicFeatureAdapter:
        """Return or lazily build the SymbolicFeatureAdapter."""
        if self._adapter is None:
            self._adapter = SymbolicFeatureAdapter(
                age_imputation_value=self.config.age_imputation_value,
                suppress_errors=self.config.suppress_pipeline_errors,
            )
        return self._adapter

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[EvaluationRunner] {msg}")
