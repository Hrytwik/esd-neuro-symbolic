"""
evaluation_pipeline — comparative clinical evaluation across three diagnostic models.

Implements the tripartite evaluation framework that forms the core
empirical contribution of the project:

  Model A — Full biopsy reference (34 features)
  Model B — Biopsy-free baseline  (12 clinical features)
  Model C — Symbolic reasoning augmentation (12 clinical + reasoning signals)

The PRIMARY success metric is not maximum accuracy. It is:
  safe biopsy reduction with interpretable escalation behaviour.

Modules
-------
baseline_model_a        Model A: XGBoost/RF/LR on all 34 features
baseline_model_b        Model B: same classifiers on 12 clinical features
symbolic_model_c        Model C: clinical features + symbolic reasoning signals
evaluation_runner       Orchestrates A/B/C training and evaluation
escalation_evaluator    Safe biopsy avoidance and escalation appropriateness
contradiction_evaluator Contradiction prevalence and certainty decay analysis
trajectory_evaluator    Trajectory stability and convergence analysis
reasoning_metrics       Aggregated symbolic reasoning quality metrics
comparative_report_generator  Publication-grade A/B/C comparison tables

Primary entry point
-------------------
  from src.evaluation_pipeline.evaluation_runner import EvaluationRunner
  from src.dataset_integration.dataset_loader import DermatologyDatasetLoader

  dataset = DermatologyDatasetLoader.load("dermatology_with_labels.csv")
  runner  = EvaluationRunner()
  result  = runner.run(dataset)

  from src.evaluation_pipeline.comparative_report_generator import ComparativeReportGenerator
  report = ComparativeReportGenerator.generate(result)
  print(ComparativeReportGenerator.to_text_table(report))
"""
