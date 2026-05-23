"""
dataset_integration — UCI Dermatology dataset ingestion and clinical partitioning.

This package connects the symbolic diagnostic reasoning system to the real
UCI Dermatology dataset (366 patients, 34 features, 6 disease classes).

The central distinction this package enforces is the clinical partitioning:
12 features that clinicians can assess WITHOUT biopsy (the biopsy-free set)
vs 22 histopathological features that REQUIRE laboratory analysis.

This partitioning is foundational: Model B (biopsy-free baseline) and
Model C (symbolic reasoning augmentation) both operate exclusively on the
12 clinical features, while Model A (full biopsy reference) uses all 34.

Modules
-------
dataset_loader          Load and normalize the CSV; produce DermatologyDataset
feature_partitioning    Explicit clinical / histopathological feature sets
clinical_feature_mapper Map dataset rows to symbolic pipeline inputs
symbolic_feature_adapter Run pipeline per patient; extract reasoning signals
dataset_validator        Schema and distribution integrity checks
dataset_splitter         Stratified train / validation / test splits

Primary entry point
-------------------
  from src.dataset_integration.dataset_loader import DermatologyDatasetLoader
  dataset = DermatologyDatasetLoader.load("dermatology_with_labels.csv")

  from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureAdapter
  adapter = SymbolicFeatureAdapter()
  vectors = adapter.adapt_batch(dataset.records)
"""
