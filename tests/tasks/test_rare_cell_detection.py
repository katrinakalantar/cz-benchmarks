"""Tests for RareCellDetectionTask."""

import numpy as np
import pandas as pd
import pytest

from czbenchmarks.metrics.types import MetricResult, MetricType
from czbenchmarks.tasks.single_cell import (
    RareCellDetectionTask,
    RareCellDetectionTaskInput,
)
from czbenchmarks.tasks.task import PCABaselineInput


@pytest.fixture
def rare_cell_data():
    """Create synthetic data with rare cell types."""
    np.random.seed(42)

    # Create data with known rare types
    n_cells = 1000
    n_features = 50

    # Cell type distribution:
    # - common1: 400 cells (40%)
    # - common2: 300 cells (30%)
    # - medium: 200 cells (20%)
    # - rare1: 60 cells (6%)
    # - rare2: 30 cells (3%)
    # - very_rare: 10 cells (1%)

    cell_types = (
        ["common1"] * 400
        + ["common2"] * 300
        + ["medium"] * 200
        + ["rare1"] * 60
        + ["rare2"] * 30
        + ["very_rare"] * 10
    )

    # Create embeddings with some signal for cell types
    embeddings = np.random.randn(n_cells, n_features)

    # Add signal: make each cell type cluster
    for i, ct in enumerate(["common1", "common2", "medium", "rare1", "rare2", "very_rare"]):
        mask = [c == ct for c in cell_types]
        embeddings[mask, :10] += i * 2  # Shift first 10 features

    labels = pd.Series(cell_types, name="cell_type")
    obs = pd.DataFrame({"cell_type": cell_types})

    return {
        "embeddings": embeddings,
        "labels": labels,
        "obs": obs,
        "n_cells": n_cells,
        "rare_types": ["rare1", "rare2"],  # With default threshold 0.05
        "all_rare_types": ["rare1", "rare2", "very_rare"],  # Below 5%
    }


def test_rare_cell_detection_basic_execution(rare_cell_data):
    """Test that the task executes without errors."""
    task = RareCellDetectionTask()

    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
    )

    results = task.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    # Check results structure
    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)
    assert len(results) > 0

    # Check that we have results for all three classifiers
    classifiers = {r.params.get("classifier") for r in results}
    assert classifiers == {"logistic", "knn", "rf"}


def test_rare_cell_detection_identifies_rare_types(rare_cell_data):
    """Test that rare cell types are correctly identified."""
    task = RareCellDetectionTask()

    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
        rarity_threshold=0.05,
        min_cells=10,
    )

    results = task.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    # Check that rare types were correctly identified
    # With 5% threshold: rare1 (6%) is NOT rare, rare2 (3%) is rare, very_rare (1%) is rare
    assert set(task._rare_types) == {"rare2", "very_rare"}


def test_rare_cell_detection_threshold_adjustment(rare_cell_data):
    """Test that adjusting rarity threshold changes identified rare types."""
    task = RareCellDetectionTask()

    # Lower threshold to include very_rare type
    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
        rarity_threshold=0.02,  # 2% threshold
        min_cells=5,  # Lower min cells
    )

    results = task.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    # Should only identify very_rare (1%)
    assert "very_rare" in task._rare_types
    assert "rare1" not in task._rare_types  # 6% > 2% threshold


def test_rare_cell_detection_metrics_structure(rare_cell_data):
    """Test that all expected metrics are computed."""
    task = RareCellDetectionTask()

    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
    )

    results = task.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    # Expected metric types
    expected_metrics = {
        MetricType.F1_CALCULATION,
        MetricType.RECALL_CALCULATION,
        MetricType.PRECISION_CALCULATION,
        MetricType.BALANCED_ACCURACY,
        MetricType.MCC,
        MetricType.SPECIFICITY,
        MetricType.MEAN_RECALL_PER_TYPE,
        MetricType.MIN_RECALL_PER_TYPE,
        MetricType.MEAN_PRECISION_PER_TYPE,
    }

    metric_types = {r.metric_type for r in results}
    assert expected_metrics.issubset(metric_types)

    # Each metric should be computed for each classifier
    for metric_type in expected_metrics:
        metric_results = [r for r in results if r.metric_type == metric_type]
        classifiers = {r.params.get("classifier") for r in metric_results}
        assert classifiers == {"logistic", "knn", "rf"}


def test_rare_cell_detection_metric_values(rare_cell_data):
    """Test that metric values are reasonable."""
    task = RareCellDetectionTask()

    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
    )

    results = task.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    # All metrics should be in valid ranges
    for result in results:
        assert 0 <= result.value <= 1, f"Metric {result.metric_type} out of range: {result.value}"

    # Specificity should be reasonably high (most cells are not rare)
    specificity_results = [r for r in results if r.metric_type == MetricType.SPECIFICITY]
    assert all(r.value > 0.5 for r in specificity_results), "Specificity too low"


def test_rare_cell_detection_baseline_execution(rare_cell_data):
    """Test baseline computation with PCA."""
    task = RareCellDetectionTask()

    # Create expression matrix (simulate raw counts with more features than embeddings)
    # Need more features than n_components for PCA
    n_cells = rare_cell_data["n_cells"]
    n_genes = 200  # More genes to allow for PCA
    expression_matrix = np.random.randint(0, 100, size=(n_cells, n_genes)).astype(float)

    baseline_input = PCABaselineInput(n_components=20)
    baseline_embeddings = task.compute_baseline(expression_matrix, baseline_input)

    assert baseline_embeddings.shape[0] == rare_cell_data["n_cells"]
    assert baseline_embeddings.shape[1] > 0  # Should have some components

    # Test that baseline can be used with the task
    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
    )

    results = task.run(
        cell_representation=baseline_embeddings,
        task_input=task_input,
    )

    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)


def test_rare_cell_detection_no_rare_types(rare_cell_data):
    """Test error handling when no rare types are found."""
    task = RareCellDetectionTask()

    # Use threshold that excludes all types
    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
        rarity_threshold=0.001,  # 0.1% - too strict
        min_cells=100,  # Too high
    )

    with pytest.raises(ValueError, match="No rare types found"):
        task.run(
            cell_representation=rare_cell_data["embeddings"],
            task_input=task_input,
        )


def test_rare_cell_detection_input_validation():
    """Test that input validation works correctly."""
    task = RareCellDetectionTask()

    # Create minimal valid data
    embeddings = np.random.randn(100, 10)
    labels = pd.Series(["type1"] * 50 + ["type2"] * 50)
    obs = pd.DataFrame({"cell_type": labels})

    # Test invalid rarity_threshold
    with pytest.raises(ValueError, match="rarity_threshold must be between 0 and 1"):
        RareCellDetectionTaskInput(
            obs=obs,
            labels=labels,
            rarity_threshold=1.5,
        )

    with pytest.raises(ValueError, match="rarity_threshold must be between 0 and 1"):
        RareCellDetectionTaskInput(
            obs=obs,
            labels=labels,
            rarity_threshold=-0.1,
        )

    # Test invalid min_cells
    with pytest.raises(ValueError, match="min_cells must be a positive integer"):
        RareCellDetectionTaskInput(
            obs=obs,
            labels=labels,
            min_cells=-5,
        )

    # Test invalid n_splits
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        RareCellDetectionTaskInput(
            obs=obs,
            labels=labels,
            n_splits=1,
        )


def test_rare_cell_detection_cv_folds(rare_cell_data):
    """Test that different numbers of CV folds work."""
    task = RareCellDetectionTask()

    # Test with 3 folds
    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
        n_splits=3,
    )

    results_3fold = task.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    assert len(results_3fold) > 0

    # Test with 10 folds
    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
        n_splits=10,
    )

    results_10fold = task.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    assert len(results_10fold) > 0

    # Both should produce the same number of results (same classifiers and metrics)
    assert len(results_3fold) == len(results_10fold)


def test_rare_cell_detection_single_rare_type(rare_cell_data):
    """Test with only one rare cell type."""
    # Create data with only one rare type
    cell_types = ["common1"] * 900 + ["rare1"] * 100
    embeddings = np.random.randn(1000, 50)
    labels = pd.Series(cell_types)
    obs = pd.DataFrame({"cell_type": cell_types})

    task = RareCellDetectionTask()
    task_input = RareCellDetectionTaskInput(
        obs=obs,
        labels=labels,
        rarity_threshold=0.15,  # Catch rare1 (10%)
    )

    results = task.run(
        cell_representation=embeddings,
        task_input=task_input,
    )

    assert isinstance(results, list)
    assert len(results) > 0


def test_rare_cell_detection_deterministic(rare_cell_data):
    """Test that results are deterministic with same random seed."""
    task1 = RareCellDetectionTask(random_seed=42)
    task2 = RareCellDetectionTask(random_seed=42)

    task_input = RareCellDetectionTaskInput(
        obs=rare_cell_data["obs"],
        labels=rare_cell_data["labels"],
    )

    results1 = task1.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    results2 = task2.run(
        cell_representation=rare_cell_data["embeddings"],
        task_input=task_input,
    )

    # Sort results by metric type and classifier for comparison
    def sort_key(r):
        return (r.metric_type.value, r.params.get("classifier", ""))

    results1_sorted = sorted(results1, key=sort_key)
    results2_sorted = sorted(results2, key=sort_key)

    # Check that values are identical
    for r1, r2 in zip(results1_sorted, results2_sorted):
        assert r1.metric_type == r2.metric_type
        assert r1.params == r2.params
        assert np.isclose(r1.value, r2.value), f"Values differ for {r1.metric_type}"
