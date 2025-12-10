"""
Kk Test Task: Rare Cell Type Detection Evaluation

This task evaluates model embeddings on their ability to identify and classify
rare cell types using multiple classifiers and stratified cross-validation.
"""

import logging
from typing import Annotated, List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
from pydantic import Field, field_validator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from ...constants import RANDOM_SEED
from ..constants import N_FOLDS
from ..task import NoBaselineInput, Task, TaskInput, TaskOutput
from ...tasks.types import CellRepresentation
from ...types import ListLike
from ...metrics.types import MetricResult, MetricType


logger = logging.getLogger(__name__)


class KkTestTaskInput(TaskInput):
    """Pydantic model for KkTestTask inputs.

    This task evaluates embeddings for rare cell type detection by:
    1. Identifying rare cell types based on frequency and minimum cell count
    2. Training multiple classifiers with cross-validation
    3. Computing binary metrics (rare vs non-rare) and per-type metrics
    """

    cell_type_key: Annotated[
        str,
        Field(
            description="Name of the column in obs containing cell type labels.",
        ),
    ] = "cell_type"

    rarity_threshold: Annotated[
        float,
        Field(
            description="Maximum frequency threshold for a cell type to be considered rare (e.g., 0.05 = 5%).",
        ),
    ] = 0.05

    min_cells: Annotated[
        int,
        Field(
            description="Minimum number of cells required for a cell type to be included as rare.",
        ),
    ] = 10

    n_splits: Annotated[
        int,
        Field(
            description="Number of cross-validation folds.",
        ),
    ] = N_FOLDS

    classifiers: Annotated[
        List[Literal["logistic", "knn", "rf"]],
        Field(
            description="List of classifiers to use for evaluation. Options: 'logistic', 'knn', 'rf'.",
        ),
    ] = ["logistic", "knn", "rf"]

    @field_validator("cell_type_key")
    @classmethod
    def _validate_cell_type_key(cls, v: str) -> str:
        if not isinstance(v, str) or len(v) == 0:
            raise ValueError("cell_type_key must be a non-empty string.")
        return v

    @field_validator("rarity_threshold")
    @classmethod
    def _validate_rarity_threshold(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("rarity_threshold must be between 0 and 1 (exclusive).")
        return v

    @field_validator("min_cells")
    @classmethod
    def _validate_min_cells(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_cells must be at least 1.")
        return v

    @field_validator("n_splits")
    @classmethod
    def _validate_n_splits(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_splits must be at least 2.")
        return v

    @field_validator("classifiers")
    @classmethod
    def _validate_classifiers(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one classifier must be specified.")
        valid_classifiers = {"logistic", "knn", "rf"}
        for clf in v:
            if clf not in valid_classifiers:
                raise ValueError(f"Invalid classifier '{clf}'. Must be one of {valid_classifiers}.")
        return v


class KkTestTaskOutput(TaskOutput):
    """Output from KkTestTask containing all classification results."""

    rare_types: List[str]
    """List of cell types identified as rare"""

    results: List[Dict[str, Any]]
    """List of dictionaries containing classifier results for each fold"""

    rare_type_counts: Dict[str, int]
    """Dictionary mapping rare cell types to their counts"""


class KkTestTask(Task):
    """Task for evaluating embeddings on rare cell type detection.

    This task identifies rare cell types in a dataset and evaluates how well
    different classifiers can distinguish rare from non-rare cell types using
    the provided embeddings. It computes multiple metrics including:
    - Binary classification metrics (rare vs non-rare)
    - Per-type performance metrics
    - Classifier-specific metrics

    The task uses stratified k-fold cross-validation to ensure robust evaluation.
    """

    display_name = "kk test task"
    description = "Evaluate embeddings for rare cell type detection using multiple classifiers."

    input_model = KkTestTaskInput
    baseline_model = NoBaselineInput

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.requires_multiple_datasets = False

    def _identify_rare_types(
        self,
        cell_types: pd.Series,
        rarity_threshold: float,
        min_cells: int,
    ) -> tuple[List[str], Dict[str, int]]:
        """Identify which cell types are rare based on frequency and count.

        Args:
            cell_types: Series containing cell type labels
            rarity_threshold: Maximum frequency for rare cell types
            min_cells: Minimum number of cells required

        Returns:
            Tuple of (list of rare cell type names, dict of rare type counts)
        """
        counts = cell_types.value_counts()
        freqs = counts / len(cell_types)

        rare_mask = (freqs <= rarity_threshold) & (counts >= min_cells)
        rare_types = freqs[rare_mask].index.tolist()
        rare_counts = counts[rare_mask].to_dict()

        logger.info(f"Found {len(rare_types)} rare cell types:")
        for ct in rare_types:
            logger.info(f"  {ct}: {counts[ct]} cells ({freqs[ct]*100:.2f}%)")

        return rare_types, rare_counts

    def _compute_binary_metrics(
        self,
        y_true_rare: np.ndarray,
        y_pred_rare: np.ndarray,
    ) -> Dict[str, float]:
        """Compute binary classification metrics for rare vs non-rare.

        Args:
            y_true_rare: Boolean array indicating true rare cells
            y_pred_rare: Boolean array indicating predicted rare cells

        Returns:
            Dictionary of metric values
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        metrics = {
            'precision': precision_score(y_true_rare, y_pred_rare, zero_division=0),
            'recall': recall_score(y_true_rare, y_pred_rare, zero_division=0),
            'f1': f1_score(y_true_rare, y_pred_rare, zero_division=0),
            'balanced_acc': balanced_accuracy_score(y_true_rare, y_pred_rare),
        }

        # MCC - good for imbalanced data
        metrics['mcc'] = matthews_corrcoef(y_true_rare, y_pred_rare)

        # Specificity
        tn = ((~y_true_rare) & (~y_pred_rare)).sum()
        fp = ((~y_true_rare) & y_pred_rare).sum()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return metrics

    def _compute_per_type_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        rare_types: List[str],
    ) -> Dict[str, float]:
        """Compute recall and precision for each individual rare type.

        Args:
            y_true: Array of true labels
            y_pred: Array of predicted labels
            rare_types: List of rare cell type names

        Returns:
            Dictionary with mean and min recall/precision across rare types
        """
        recalls = []
        precisions = []

        for ct in rare_types:
            true_mask = (y_true == ct)
            pred_mask = (y_pred == ct)

            if true_mask.sum() == 0:
                continue

            tp = (true_mask & pred_mask).sum()
            fp = (~true_mask & pred_mask).sum()
            fn = (true_mask & ~pred_mask).sum()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            recalls.append(recall)
            precisions.append(precision)

        return {
            'mean_recall_per_type': np.mean(recalls) if recalls else 0.0,
            'min_recall_per_type': np.min(recalls) if recalls else 0.0,
            'mean_precision_per_type': np.mean(precisions) if precisions else 0.0,
        }

    def _eval_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        rare_types: List[str],
        clf,
    ) -> Dict[str, float]:
        """Run evaluation for one CV fold.

        Args:
            X_train: Training embeddings
            X_test: Test embeddings
            y_train: Training labels
            y_test: Test labels
            rare_types: List of rare cell type names
            clf: Classifier instance

        Returns:
            Dictionary of metrics for this fold
        """
        # Train
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Convert to binary (rare vs not)
        y_true_rare = np.isin(y_test, rare_types)
        y_pred_rare = np.isin(y_pred, rare_types)

        # Compute metrics
        metrics = {}
        metrics.update(self._compute_binary_metrics(y_true_rare, y_pred_rare))
        metrics.update(self._compute_per_type_metrics(y_test, y_pred, rare_types))

        return metrics

    def _run_cv_eval(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        rare_types: List[str],
        clf,
        clf_name: str,
        n_splits: int,
    ) -> List[Dict[str, Any]]:
        """Run stratified k-fold cross-validation.

        Args:
            embeddings: Cell embeddings
            labels: Cell type labels
            rare_types: List of rare cell type names
            clf: Classifier instance
            clf_name: Name of classifier
            n_splits: Number of CV folds

        Returns:
            List of result dictionaries, one per fold
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            logger.debug(f"Evaluating {clf_name}, fold {fold_idx}...")
            fold_metrics = self._eval_fold(X_train, X_test, y_train, y_test, rare_types, clf)
            fold_metrics['classifier'] = clf_name
            fold_metrics['fold'] = fold_idx
            fold_results.append(fold_metrics)

        return fold_results

    def _get_classifier(self, clf_name: str):
        """Get classifier instance by name.

        Args:
            clf_name: Name of classifier ('logistic', 'knn', or 'rf')

        Returns:
            Classifier instance
        """
        if clf_name == "logistic":
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_seed,
            )
        elif clf_name == "knn":
            return KNeighborsClassifier(n_neighbors=15)
        elif clf_name == "rf":
            return RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                max_depth=20,
                random_state=self.random_seed,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: KkTestTaskInput,
    ) -> KkTestTaskOutput:
        """Run rare cell type detection evaluation.

        Args:
            cell_representation: Cell embeddings as numpy array, sparse matrix, or DataFrame.
                                Must be provided alongside an AnnData object with cell type labels.
            task_input: Task input parameters

        Returns:
            KkTestTaskOutput with results

        Note:
            This task requires an AnnData object to be stored in self._adata by the user
            before calling run(). The AnnData object must contain cell type labels in .obs.
        """
        # For this task, we need the AnnData object stored externally
        # Check if we have it stored as an instance variable
        if not hasattr(self, '_adata') or self._adata is None:
            raise ValueError(
                "KkTestTask requires an AnnData object with cell type labels. "
                "Please store the AnnData object in task._adata before calling run()."
            )

        adata = self._adata

        # Validate cell type key
        if task_input.cell_type_key not in adata.obs.columns:
            raise ValueError(
                f"cell_type_key '{task_input.cell_type_key}' not found in AnnData.obs"
            )

        # Get embeddings from cell_representation
        if hasattr(cell_representation, 'toarray'):
            # Sparse matrix
            embeddings = cell_representation.toarray()
        elif isinstance(cell_representation, pd.DataFrame):
            embeddings = cell_representation.values
        else:
            # Assume numpy array
            embeddings = np.array(cell_representation)

        # Get labels from AnnData
        labels = adata.obs[task_input.cell_type_key].values

        logger.info(f"Processing {embeddings.shape[0]} cells with {embeddings.shape[1]} features")

        # Identify rare types
        labels_series = pd.Series(labels)
        rare_types, rare_counts = self._identify_rare_types(
            labels_series,
            task_input.rarity_threshold,
            task_input.min_cells,
        )

        if len(rare_types) == 0:
            logger.warning("No rare types found! Try adjusting rarity_threshold or min_cells")
            return KkTestTaskOutput(
                rare_types=[],
                results=[],
                rare_type_counts={},
            )

        # Run evaluation for each classifier
        all_results = []
        for clf_name in task_input.classifiers:
            logger.info(f"Running evaluation with {clf_name} classifier...")
            clf = self._get_classifier(clf_name)

            results = self._run_cv_eval(
                embeddings,
                labels,
                rare_types,
                clf,
                clf_name,
                task_input.n_splits,
            )
            all_results.extend(results)

        logger.info(f"Completed evaluation with {len(all_results)} total results")

        return KkTestTaskOutput(
            rare_types=rare_types,
            results=all_results,
            rare_type_counts=rare_counts,
        )

    def _compute_metrics(
        self,
        task_input: KkTestTaskInput,
        task_output: KkTestTaskOutput,
    ) -> List[MetricResult]:
        """Compute aggregated metrics from task results.

        Args:
            task_input: Task input parameters
            task_output: Task output with results

        Returns:
            List of MetricResult objects
        """
        if not task_output.results:
            logger.warning("No results to compute metrics from")
            return []

        logger.info("Computing aggregated metrics...")
        results_df = pd.DataFrame(task_output.results)
        metrics_list = []

        # Metric type mapping for cross-validation
        metric_type_map = {
            'f1': MetricType.MEAN_FOLD_F1_SCORE,
            'precision': MetricType.MEAN_FOLD_PRECISION,
            'recall': MetricType.MEAN_FOLD_RECALL,
            'balanced_acc': MetricType.BALANCED_ACCURACY,
            'mcc': MetricType.MCC,
            'specificity': MetricType.SPECIFICITY,
            'mean_recall_per_type': MetricType.MEAN_RECALL_PER_TYPE,
            'min_recall_per_type': MetricType.MIN_RECALL_PER_TYPE,
            'mean_precision_per_type': MetricType.MEAN_PRECISION_PER_TYPE,
        }

        # Compute metrics aggregated across all classifiers
        base_params = {"classifier": "MEAN(all)"}
        for metric_name, metric_type in metric_type_map.items():
            if metric_name in results_df.columns:
                metrics_list.append(
                    MetricResult(
                        metric_type=metric_type,
                        value=results_df[metric_name].mean(),
                        params=base_params,
                    )
                )

        # Compute metrics per classifier
        for clf in results_df['classifier'].unique():
            clf_df = results_df[results_df['classifier'] == clf]
            clf_params = {"classifier": clf}

            for metric_name, metric_type in metric_type_map.items():
                if metric_name in clf_df.columns:
                    metrics_list.append(
                        MetricResult(
                            metric_type=metric_type,
                            value=clf_df[metric_name].mean(),
                            params=clf_params,
                        )
                    )

        logger.info(f"Computed {len(metrics_list)} metrics")
        return metrics_list

    def compute_baseline(
        self,
        expression_data: CellRepresentation,
        baseline_input: NoBaselineInput = None,
    ):
        """Baseline not implemented for this task.

        Rare cell detection requires cell type labels which are not available
        during baseline computation.
        """
        raise NotImplementedError(
            "Baseline not implemented for rare cell type detection task"
        )
