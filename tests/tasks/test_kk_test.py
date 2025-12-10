"""
Tests for KkTestTask (Rare Cell Type Detection)
"""

import pytest
import numpy as np
import anndata as ad
import pandas as pd
from czbenchmarks.tasks.single_cell.kk_test import (
    KkTestTask,
    KkTestTaskInput,
    KkTestTaskOutput,
)
from czbenchmarks.metrics.types import MetricResult, MetricType


@pytest.fixture
def test_adata():
    """Create synthetic AnnData with rare cell types for testing."""
    np.random.seed(42)

    # Create data with common and rare cell types
    n_cells = 500
    n_features = 50

    # Generate embeddings
    X = np.random.randn(n_cells, n_features)

    # Create cell type distribution with rare types
    # Common types: 70% type_A, 20% type_B
    # Rare types: 4% type_C, 3% type_D, 3% type_E
    cell_types = (
        ['type_A'] * 350 +
        ['type_B'] * 100 +
        ['type_C'] * 20 +
        ['type_D'] * 15 +
        ['type_E'] * 15
    )
    np.random.shuffle(cell_types)

    # Create AnnData
    adata = ad.AnnData(X=X)
    adata.obs['cell_type'] = cell_types

    return adata


@pytest.fixture
def test_adata_no_rare():
    """Create synthetic AnnData with no rare cell types."""
    np.random.seed(42)

    n_cells = 200
    n_features = 30

    X = np.random.randn(n_cells, n_features)

    # All common types
    cell_types = ['type_A'] * 100 + ['type_B'] * 100

    adata = ad.AnnData(X=X)
    adata.obs['cell_type'] = cell_types

    return adata


@pytest.fixture
def test_adata_single_rare():
    """Create synthetic AnnData with a single rare cell type."""
    np.random.seed(42)

    n_cells = 300
    n_features = 40

    X = np.random.randn(n_cells, n_features)

    # One rare type
    cell_types = ['type_A'] * 270 + ['type_B_rare'] * 30

    adata = ad.AnnData(X=X)
    adata.obs['cell_type'] = cell_types

    return adata


def test_kk_test_basic_execution(test_adata):
    """Test that the task executes without errors."""
    task = KkTestTask()
    task._adata = test_adata
    task._adata = test_adata  # Store AnnData object
    task_input = KkTestTaskInput()

    # Use embeddings from AnnData
    results = task.run(test_adata.X, task_input)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, MetricResult) for r in results)


def test_kk_test_output_structure(test_adata):
    """Test that task output has expected structure."""
    task = KkTestTask()
    task._adata = test_adata
    task._adata = test_adata
    task_input = KkTestTaskInput()

    output = task._run_task(test_adata.X, task_input)

    assert isinstance(output, KkTestTaskOutput)
    assert isinstance(output.rare_types, list)
    assert isinstance(output.results, list)
    assert isinstance(output.rare_type_counts, dict)
    assert len(output.rare_types) > 0
    assert len(output.results) > 0


def test_kk_test_rare_type_identification(test_adata):
    """Test that rare types are correctly identified."""
    task = KkTestTask()
    task._adata = test_adata
    task._adata = test_adata
    task_input = KkTestTaskInput(rarity_threshold=0.05, min_cells=10)

    output = task._run_task(test_adata.X, task_input)

    # Should identify type_C, type_D, type_E as rare (all < 5% and >= 10 cells)
    assert len(output.rare_types) == 3
    assert 'type_C' in output.rare_types
    assert 'type_D' in output.rare_types
    assert 'type_E' in output.rare_types

    # Check counts
    assert output.rare_type_counts['type_C'] == 20
    assert output.rare_type_counts['type_D'] == 15
    assert output.rare_type_counts['type_E'] == 15


def test_kk_test_no_rare_types(test_adata_no_rare):
    """Test handling when no rare types exist."""
    task = KkTestTask()
    task._adata = test_adata_no_rare
    task_input = KkTestTaskInput(rarity_threshold=0.05, min_cells=10)

    task._adata = test_adata_no_rare
    output = task._run_task(test_adata_no_rare.X, task_input)

    assert len(output.rare_types) == 0
    assert len(output.results) == 0
    assert len(output.rare_type_counts) == 0

    # Metrics should handle this gracefully
    metrics = task._compute_metrics(task_input, output)
    assert metrics == []


def test_kk_test_input_validation_cell_type_key():
    """Test that input validation works for cell_type_key."""
    with pytest.raises(ValueError, match="cell_type_key must be a non-empty string"):
        KkTestTaskInput(cell_type_key="")


def test_kk_test_input_validation_rarity_threshold():
    """Test that input validation works for rarity_threshold."""
    with pytest.raises(ValueError, match="rarity_threshold must be between 0 and 1"):
        KkTestTaskInput(rarity_threshold=0)

    with pytest.raises(ValueError, match="rarity_threshold must be between 0 and 1"):
        KkTestTaskInput(rarity_threshold=1.5)


def test_kk_test_input_validation_min_cells():
    """Test that input validation works for min_cells."""
    with pytest.raises(ValueError, match="min_cells must be at least 1"):
        KkTestTaskInput(min_cells=0)


def test_kk_test_input_validation_n_splits():
    """Test that input validation works for n_splits."""
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        KkTestTaskInput(n_splits=1)


def test_kk_test_input_validation_classifiers():
    """Test that input validation works for classifiers."""
    with pytest.raises(ValueError, match="At least one classifier must be specified"):
        KkTestTaskInput(classifiers=[])

    # Pydantic will catch invalid literals in the type definition
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        KkTestTaskInput(classifiers=["logistic", "invalid_clf"])


def test_kk_test_different_classifiers(test_adata):
    """Test running with different classifier configurations."""
    task = KkTestTask()
    task._adata = test_adata

    # Test with single classifier
    task_input = KkTestTaskInput(classifiers=["logistic"])
    task._adata = test_adata
    results = task.run(test_adata.X, task_input)
    assert len(results) > 0

    # Test with subset of classifiers
    task_input = KkTestTaskInput(classifiers=["knn", "rf"])
    results = task.run(test_adata.X, task_input)
    assert len(results) > 0


def test_kk_test_different_n_splits(test_adata):
    """Test running with different n_splits values."""
    task = KkTestTask()
    task._adata = test_adata

    task_input = KkTestTaskInput(n_splits=3)
    task._adata = test_adata
    output = task._run_task(test_adata.X, task_input)

    # With 3 classifiers and 3 folds, should have 9 results
    assert len(output.results) == 9


def test_kk_test_metric_computation(test_adata):
    """Test that metrics are computed correctly."""
    task = KkTestTask()
    task._adata = test_adata
    task_input = KkTestTaskInput()

    task._adata = test_adata
    output = task._run_task(test_adata.X, task_input)
    metrics = task._compute_metrics(task_input, output)

    assert len(metrics) > 0

    # Check that we have expected metric types
    metric_types = {m.metric_type for m in metrics}
    expected_types = {
        MetricType.MEAN_FOLD_F1_SCORE,
        MetricType.MEAN_FOLD_PRECISION,
        MetricType.MEAN_FOLD_RECALL,
        MetricType.BALANCED_ACCURACY,
        MetricType.MCC,
        MetricType.SPECIFICITY,
    }

    # Should have at least some of these metrics
    assert len(metric_types & expected_types) > 0

    # Check that all metrics have valid values
    for metric in metrics:
        assert isinstance(metric.value, (int, float))
        assert not np.isnan(metric.value)


def test_kk_test_per_classifier_metrics(test_adata):
    """Test that per-classifier metrics are computed."""
    task = KkTestTask()
    task._adata = test_adata
    task_input = KkTestTaskInput(classifiers=["logistic", "knn"])

    task._adata = test_adata
    output = task._run_task(test_adata.X, task_input)
    metrics = task._compute_metrics(task_input, output)

    # Should have metrics for each classifier plus aggregated metrics
    classifier_params = [m.params.get('classifier') for m in metrics]
    assert 'logistic' in classifier_params
    assert 'knn' in classifier_params
    assert 'MEAN(all)' in classifier_params


def test_kk_test_determinism(test_adata):
    """Test that results are deterministic with same seed."""
    task1 = KkTestTask(random_seed=42)
    task1._adata = test_adata
    task2 = KkTestTask(random_seed=42)
    task2._adata = test_adata

    task_input = KkTestTaskInput()

    task1._adata = test_adata
    results1 = task1.run(test_adata.X, task_input)
    task2._adata = test_adata
    results2 = task2.run(test_adata.X, task_input)

    # Should have same number of results
    assert len(results1) == len(results2)

    # Metric values should be the same
    for r1, r2 in zip(results1, results2):
        assert r1.metric_type == r2.metric_type
        assert np.isclose(r1.value, r2.value, rtol=1e-5)


def test_kk_test_invalid_cell_type_key(test_adata):
    """Test error handling for invalid cell_type_key."""
    task = KkTestTask()
    task._adata = test_adata
    task_input = KkTestTaskInput(cell_type_key="nonexistent_column")

    with pytest.raises(ValueError, match="cell_type_key .* not found"):
        task.run(test_adata.X, task_input)


def test_kk_test_requires_anndata():
    """Test that task requires AnnData input."""
    task = KkTestTask()
    task_input = KkTestTaskInput()

    # Should fail when _adata is not set
    embeddings = np.random.randn(100, 50)

    with pytest.raises(ValueError, match="requires an AnnData object"):
        task.run(embeddings, task_input)


def test_kk_test_baseline_not_implemented(test_adata):
    """Test that baseline computation raises NotImplementedError."""
    task = KkTestTask()
    task._adata = test_adata

    with pytest.raises(NotImplementedError, match="Baseline not implemented"):
        task.compute_baseline(test_adata)


def test_kk_test_single_rare_type(test_adata_single_rare):
    """Test with a single rare cell type."""
    task = KkTestTask()
    task._adata = test_adata_single_rare
    task_input = KkTestTaskInput(rarity_threshold=0.15, min_cells=20)

    task._adata = test_adata_single_rare
    output = task._run_task(test_adata_single_rare.X, task_input)

    assert len(output.rare_types) == 1
    assert 'type_B_rare' in output.rare_types


def test_kk_test_min_cells_threshold(test_adata):
    """Test that min_cells threshold is respected."""
    task = KkTestTask()
    task._adata = test_adata

    # With high min_cells, should exclude some rare types
    task_input = KkTestTaskInput(rarity_threshold=0.05, min_cells=18)
    task._adata = test_adata
    output = task._run_task(test_adata.X, task_input)

    # Should only identify type_C (20 cells) as rare
    # type_D and type_E have only 15 cells each
    assert len(output.rare_types) == 1
    assert 'type_C' in output.rare_types


def test_kk_test_rarity_threshold_variation(test_adata):
    """Test different rarity thresholds."""
    task = KkTestTask()
    task._adata = test_adata

    # Very strict threshold
    task_input = KkTestTaskInput(rarity_threshold=0.01, min_cells=10)
    task._adata = test_adata
    output = task._run_task(test_adata.X, task_input)
    assert len(output.rare_types) == 0

    # Very loose threshold
    task_input = KkTestTaskInput(rarity_threshold=0.5, min_cells=10)
    output = task._run_task(test_adata.X, task_input)
    assert len(output.rare_types) > 0


def test_kk_test_fold_results_structure(test_adata):
    """Test that fold results have expected structure."""
    task = KkTestTask()
    task._adata = test_adata
    task_input = KkTestTaskInput()

    task._adata = test_adata
    output = task._run_task(test_adata.X, task_input)

    # Check structure of results
    for result in output.results:
        assert 'classifier' in result
        assert 'fold' in result
        assert 'f1' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'balanced_acc' in result
        assert 'mcc' in result
        assert 'specificity' in result

        # Check that values are numeric
        for key in ['f1', 'precision', 'recall', 'balanced_acc', 'mcc', 'specificity']:
            assert isinstance(result[key], (int, float))
            assert not np.isnan(result[key])


def test_kk_test_custom_cell_type_column(test_adata):
    """Test using a custom cell type column name."""
    # Add a different column with cell types
    test_adata.obs['custom_labels'] = test_adata.obs['cell_type']

    task = KkTestTask()
    task._adata = test_adata
    task_input = KkTestTaskInput(cell_type_key='custom_labels')

    task._adata = test_adata
    results = task.run(test_adata.X, task_input)
    assert len(results) > 0
