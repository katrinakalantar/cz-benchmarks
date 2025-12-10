"""Example usage of RareCellDetectionTask.

This example demonstrates how to evaluate cell embeddings on rare cell type
detection using the Tabula Sapiens v2 datasets.

The task automatically identifies rare cell types based on frequency thresholds
and evaluates classification performance using multiple classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from czbenchmarks.datasets import load_dataset
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.tasks.single_cell import (
    RareCellDetectionTask,
    RareCellDetectionTaskInput,
)
from czbenchmarks.tasks.task import PCABaselineInput


def main():
    """Run rare cell detection evaluation on Tabula Sapiens data."""

    # =========================================================================
    # 1. Load a Tabula Sapiens dataset
    # =========================================================================
    print("Loading Tabula Sapiens v2 dataset...")

    # Load a dataset with diverse cell types (e.g., lung tissue)
    # You can use any TSV2 dataset: tsv2_lung, tsv2_liver, tsv2_spleen, etc.
    dataset: SingleCellLabeledDataset = load_dataset("tsv2_lung")

    print(f"Loaded data shape: {dataset.adata.shape}")
    print(f"Number of cell types: {len(dataset.labels.unique())}")
    print(f"\nCell type distribution:")
    print(dataset.labels.value_counts().head(10))

    # =========================================================================
    # 2. Create task input
    # =========================================================================
    print("\n" + "=" * 80)
    print("Setting up task input...")

    task_input = RareCellDetectionTaskInput(
        obs=dataset.adata.obs,
        labels=dataset.labels,
        rarity_threshold=0.05,  # Cell types with <5% frequency
        min_cells=10,  # At least 10 cells required
        n_splits=5,  # 5-fold cross-validation
    )

    # =========================================================================
    # 3. Initialize task
    # =========================================================================
    task = RareCellDetectionTask(random_seed=42)

    # =========================================================================
    # 4. Compute PCA baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("Computing PCA baseline...")

    baseline_input = PCABaselineInput(n_components=50)
    baseline_embeddings = task.compute_baseline(
        dataset.adata.X, baseline_input
    )

    print(f"PCA baseline shape: {baseline_embeddings.shape}")

    # =========================================================================
    # 5. Run task on PCA baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("Evaluating PCA baseline on rare cell detection...")

    baseline_results = task.run(
        cell_representation=baseline_embeddings,
        task_input=task_input,
    )

    print(f"\nIdentified {len(task._rare_types)} rare cell types:")
    for ct, count in task._run_task(baseline_embeddings, task_input).rare_type_counts.items():
        freq = count / len(dataset.labels) * 100
        print(f"  - {ct}: {count} cells ({freq:.2f}%)")

    print(f"\nComputed {len(baseline_results)} metrics")

    # =========================================================================
    # 6. Display baseline results
    # =========================================================================
    print("\n" + "=" * 80)
    print("PCA Baseline Results:")
    print("=" * 80)

    # Organize results by classifier
    results_by_classifier = {}
    for result in baseline_results:
        classifier = result.params.get("classifier", "unknown")
        if classifier not in results_by_classifier:
            results_by_classifier[classifier] = {}
        results_by_classifier[classifier][result.metric_type.value] = result.value

    # Display results for each classifier
    for classifier, metrics in results_by_classifier.items():
        print(f"\n{classifier.upper()} Classifier:")
        print(f"  Binary Metrics (rare vs non-rare):")
        print(f"    F1 Score:         {metrics.get('f1_calculation', 0):.3f}")
        print(f"    Recall:           {metrics.get('recall_calculation', 0):.3f}")
        print(f"    Precision:        {metrics.get('precision_calculation', 0):.3f}")
        print(f"    Balanced Acc:     {metrics.get('balanced_accuracy', 0):.3f}")
        print(f"    MCC:              {metrics.get('mcc', 0):.3f}")
        print(f"    Specificity:      {metrics.get('specificity', 0):.3f}")
        print(f"  Per-Type Metrics:")
        print(f"    Mean Recall:      {metrics.get('mean_recall_per_type', 0):.3f}")
        print(f"    Min Recall:       {metrics.get('min_recall_per_type', 0):.3f}")
        print(f"    Mean Precision:   {metrics.get('mean_precision_per_type', 0):.3f}")

    # =========================================================================
    # 7. (Optional) Simulate model embeddings
    # =========================================================================
    print("\n" + "=" * 80)
    print("Simulating improved model embeddings...")

    # In practice, you would get these from your actual model
    # Here we simulate embeddings with better rare cell separation
    np.random.seed(42)
    model_embeddings = PCA(n_components=50).fit_transform(dataset.adata.X.toarray())

    # Add some artificial improvement for rare types
    for rare_type in task._rare_types[:2]:  # Improve first 2 rare types
        mask = dataset.labels == rare_type
        model_embeddings[mask, :5] += np.random.randn(mask.sum(), 5) * 2

    print(f"Model embeddings shape: {model_embeddings.shape}")

    # =========================================================================
    # 8. Run task on model embeddings
    # =========================================================================
    print("\n" + "=" * 80)
    print("Evaluating model embeddings on rare cell detection...")

    model_results = task.run(
        cell_representation=model_embeddings,
        task_input=task_input,
    )

    # =========================================================================
    # 9. Compare results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Comparison: Model vs PCA Baseline")
    print("=" * 80)

    # Extract key metrics for comparison
    comparison_metrics = [
        "f1_calculation",
        "balanced_accuracy",
        "mcc",
        "mean_recall_per_type",
        "min_recall_per_type",
    ]

    for classifier in ["logistic", "knn", "rf"]:
        print(f"\n{classifier.upper()} Classifier:")
        print(f"{'Metric':<25} {'Baseline':>10} {'Model':>10} {'Diff':>10}")
        print("-" * 60)

        for metric_name in comparison_metrics:
            baseline_val = next(
                (
                    r.value
                    for r in baseline_results
                    if r.metric_type.value == metric_name
                    and r.params.get("classifier") == classifier
                ),
                0,
            )
            model_val = next(
                (
                    r.value
                    for r in model_results
                    if r.metric_type.value == metric_name
                    and r.params.get("classifier") == classifier
                ),
                0,
            )
            diff = model_val - baseline_val
            diff_str = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
            print(
                f"{metric_name:<25} {baseline_val:>10.3f} {model_val:>10.3f} {diff_str:>10}"
            )

    # =========================================================================
    # 10. Interpretation guide
    # =========================================================================
    print("\n" + "=" * 80)
    print("Interpretation Guide:")
    print("=" * 80)
    print("""
Binary Metrics (rare vs non-rare detection):
  - F1 Score: Overall balance between precision and recall
  - Recall: How many rare cells were correctly identified
  - Precision: Of predicted rare cells, how many were correct
  - Balanced Accuracy: Accounts for class imbalance
  - MCC: Matthews Correlation Coefficient (-1 to 1, higher is better)
  - Specificity: How well common cells are kept as non-rare

Per-Type Metrics (granular rare cell type performance):
  - Mean Recall: Average recall across individual rare cell types
  - Min Recall: Worst-case recall (detects if any type is completely missed)
  - Mean Precision: Average precision across individual rare cell types

Good performance indicators:
  - High recall (>0.7): Model captures most rare cells
  - High min_recall_per_type (>0.5): No rare types are completely missed
  - High MCC (>0.3): Strong overall predictive power
  - High specificity (>0.9): Common cells are correctly classified

Classifier comparison:
  - Logistic Regression: Fast, good baseline, assumes linear separability
  - K-Nearest Neighbors: Captures local structure, no training needed
  - Random Forest: Robust, handles non-linear relationships well
    """)


if __name__ == "__main__":
    main()
