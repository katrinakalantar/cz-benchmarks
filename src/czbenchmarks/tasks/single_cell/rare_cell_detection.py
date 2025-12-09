"""Rare cell detection task for evaluating embeddings on rare cell types.

This task evaluates how well cell embeddings capture rare cell types by:
1. Automatically identifying rare cell types based on frequency thresholds
2. Training multiple classifiers on embeddings using stratified cross-validation
3. Computing both binary (rare vs non-rare) and per-type performance metrics

The task is particularly useful for evaluating whether embeddings preserve
information about underrepresented cell populations.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.metrics.types import MetricResult
from czbenchmarks.tasks.task import PCABaselineInput, Task, TaskInput, TaskOutput


class RareCellDetectionTaskInput(TaskInput):
    """Input parameters for rare cell detection task.

    Attributes:
        obs: Cell metadata DataFrame (must contain cell type labels).
        labels: Cell type labels for all cells.
        rarity_threshold: Maximum frequency for a cell type to be considered rare (default: 0.05).
        min_cells: Minimum number of cells required for a type to be considered (default: 10).
        n_splits: Number of cross-validation folds (default: 5).
    """

    obs: pd.DataFrame = Field(..., description="Cell metadata DataFrame")
    labels: pd.Series = Field(..., description="Cell type labels")
    rarity_threshold: float = Field(
        default=0.05,
        description="Maximum frequency threshold for rare cell types",
    )
    min_cells: int = Field(
        default=10,
        description="Minimum cells required for a rare type",
    )
    n_splits: int = Field(
        default=5,
        description="Number of cross-validation folds",
    )

    @field_validator("rarity_threshold")
    @classmethod
    def _rarity_threshold_must_be_valid(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("rarity_threshold must be between 0 and 1.")
        return v

    @field_validator("min_cells")
    @classmethod
    def _min_cells_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("min_cells must be a positive integer.")
        return v

    @field_validator("n_splits")
    @classmethod
    def _n_splits_must_be_valid(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_splits must be at least 2.")
        return v


class RareCellDetectionOutput(TaskOutput):
    """Output from rare cell detection task.

    Attributes:
        rare_types: List of identified rare cell type names.
        rare_type_counts: Dictionary mapping rare type names to cell counts.
        classifier_predictions: Dictionary mapping classifier names to prediction arrays.
        classifier_probas: Dictionary mapping classifier names to probability arrays (if available).
    """

    rare_types: List[str] = Field(..., description="Identified rare cell types")
    rare_type_counts: Dict[str, int] = Field(..., description="Cell counts per rare type")
    classifier_predictions: Dict[str, np.ndarray] = Field(
        ..., description="Predictions by classifier"
    )
    classifier_probas: Dict[str, np.ndarray] = Field(
        default_factory=dict, description="Probabilities by classifier"
    )


class RareCellDetectionTask(Task):
    """Evaluate embeddings on rare cell type detection.

    This task automatically identifies rare cell types in the dataset based on
    frequency thresholds, then evaluates how well different classifiers can
    detect these rare types using the provided embeddings.

    Three classifiers are evaluated:
    - Logistic Regression (with balanced class weights)
    - K-Nearest Neighbors (k=15)
    - Random Forest (with balanced class weights)

    Metrics include both binary classification (rare vs non-rare) and
    per-type performance metrics.

    The following parameters are required and must be supplied by
    the task input class when running the task:

    - obs (pd.DataFrame): Cell metadata
    - labels (pd.Series): Cell type labels
    - rarity_threshold (float): Maximum frequency for rare types (default: 0.05)
    - min_cells (int): Minimum cells for a rare type (default: 10)
    - n_splits (int): Number of CV folds (default: 5)
    """

    display_name = "Rare Cell Detection"
    input_model = RareCellDetectionTaskInput
    baseline_model = PCABaselineInput

    def _identify_rare_types(
        self, labels: pd.Series, rarity_threshold: float, min_cells: int
    ) -> tuple[List[str], Dict[str, int]]:
        """Identify which cell types are rare based on frequency and count thresholds.

        Args:
            labels: Cell type labels
            rarity_threshold: Maximum frequency for a type to be considered rare
            min_cells: Minimum number of cells required

        Returns:
            Tuple of (rare_type_list, rare_type_counts_dict)
        """
        counts = labels.value_counts()
        freqs = counts / len(labels)

        rare_mask = (freqs <= rarity_threshold) & (counts >= min_cells)
        rare_types = freqs[rare_mask].index.tolist()
        rare_counts = {ct: int(counts[ct]) for ct in rare_types}

        return rare_types, rare_counts

    def _compute_binary_metrics(
        self, y_true_rare: np.ndarray, y_pred_rare: np.ndarray
    ) -> Dict[str, float]:
        """Compute binary classification metrics for rare vs non-rare detection.

        Args:
            y_true_rare: Boolean array indicating true rare cells
            y_pred_rare: Boolean array indicating predicted rare cells

        Returns:
            Dictionary of metric names to values
        """
        from sklearn.metrics import (
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        metrics = {
            "precision": float(precision_score(y_true_rare, y_pred_rare, zero_division=0)),
            "recall": float(recall_score(y_true_rare, y_pred_rare, zero_division=0)),
            "f1": float(f1_score(y_true_rare, y_pred_rare, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y_true_rare, y_pred_rare)),
        }

        # Matthews Correlation Coefficient
        tp = float((y_true_rare & y_pred_rare).sum())
        tn = float(((~y_true_rare) & (~y_pred_rare)).sum())
        fp = float(((~y_true_rare) & y_pred_rare).sum())
        fn = float((y_true_rare & ~y_pred_rare).sum())

        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics["mcc"] = float(mcc_num / mcc_den if mcc_den > 0 else 0)

        # Specificity (true negative rate)
        metrics["specificity"] = float(tn / (tn + fp) if (tn + fp) > 0 else 0)

        return metrics

    def _compute_per_type_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, rare_types: List[str]
    ) -> Dict[str, float]:
        """Compute per-type recall and precision for each rare cell type.

        Args:
            y_true: True cell type labels
            y_pred: Predicted cell type labels
            rare_types: List of rare cell type names

        Returns:
            Dictionary of aggregated per-type metrics
        """
        recalls = []
        precisions = []

        for ct in rare_types:
            true_mask = y_true == ct
            pred_mask = y_pred == ct

            if true_mask.sum() == 0:
                continue

            tp = float((true_mask & pred_mask).sum())
            fp = float((~true_mask & pred_mask).sum())
            fn = float((true_mask & ~pred_mask).sum())

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            recalls.append(recall)
            precisions.append(precision)

        return {
            "mean_recall_per_type": float(np.mean(recalls) if recalls else 0),
            "min_recall_per_type": float(np.min(recalls) if recalls else 0),
            "mean_precision_per_type": float(np.mean(precisions) if precisions else 0),
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
        """Evaluate one cross-validation fold.

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
        # Train classifier
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Convert to binary (rare vs not rare)
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
        n_splits: int,
    ) -> Dict[str, float]:
        """Run stratified k-fold cross-validation and return averaged metrics.

        Args:
            embeddings: Cell embeddings
            labels: Cell type labels
            rare_types: List of rare cell type names
            clf: Classifier instance
            n_splits: Number of CV folds

        Returns:
            Dictionary of averaged metrics across folds
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        fold_results = []
        for train_idx, test_idx in skf.split(embeddings, labels):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            fold_metrics = self._eval_fold(X_train, X_test, y_train, y_test, rare_types, clf)
            fold_results.append(fold_metrics)

        # Average across folds
        avg_metrics = {}
        for key in fold_results[0].keys():
            values = [f[key] for f in fold_results]
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f"{key}_std"] = float(np.std(values))

        return avg_metrics

    def _run_task(
        self, cell_representation: np.ndarray, task_input: RareCellDetectionTaskInput
    ) -> RareCellDetectionOutput:
        """Run rare cell detection evaluation.

        Args:
            cell_representation: Cell embeddings or expression data
            task_input: Validated task input parameters

        Returns:
            RareCellDetectionOutput with predictions and identified rare types
        """
        # Identify rare cell types
        rare_types, rare_counts = self._identify_rare_types(
            task_input.labels,
            task_input.rarity_threshold,
            task_input.min_cells,
        )

        if len(rare_types) == 0:
            raise ValueError(
                f"No rare types found with rarity_threshold={task_input.rarity_threshold} "
                f"and min_cells={task_input.min_cells}. Try adjusting these parameters."
            )

        # Convert labels to numpy array for sklearn
        labels_array = task_input.labels.values

        # Define classifiers
        classifiers = {
            "logistic": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_seed,
            ),
            "knn": KNeighborsClassifier(n_neighbors=15),
            "rf": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                max_depth=20,
                random_state=self.random_seed,
                n_jobs=-1,
            ),
        }

        # Store results for each classifier
        classifier_results = {}
        for name, clf in classifiers.items():
            metrics = self._run_cv_eval(
                cell_representation,
                labels_array,
                rare_types,
                clf,
                task_input.n_splits,
            )
            classifier_results[name] = metrics

        # Store results as instance variables for _compute_metrics
        self._classifier_results = classifier_results
        self._rare_types = rare_types

        # Create dummy predictions for output (actual predictions happen in CV)
        # We'll use the full dataset predictions for illustration
        classifier_predictions = {}
        classifier_probas = {}

        for name, clf in classifiers.items():
            clf.fit(cell_representation, labels_array)
            classifier_predictions[name] = clf.predict(cell_representation)
            if hasattr(clf, "predict_proba"):
                classifier_probas[name] = clf.predict_proba(cell_representation)

        return RareCellDetectionOutput(
            rare_types=rare_types,
            rare_type_counts=rare_counts,
            classifier_predictions=classifier_predictions,
            classifier_probas=classifier_probas,
        )

    def _compute_metrics(
        self,
        task_input: RareCellDetectionTaskInput,
        task_output: RareCellDetectionOutput,
    ) -> List[MetricResult]:
        """Compute evaluation metrics from task output.

        Args:
            task_input: Task input parameters
            task_output: Task output with predictions

        Returns:
            List of MetricResult objects with metrics for each classifier
        """
        from czbenchmarks.metrics.types import MetricType

        metric_results = []

        # Create metrics for each classifier
        for classifier_name, metrics in self._classifier_results.items():
            # Binary metrics (rare vs non-rare)
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.F1_CALCULATION,
                    value=metrics["f1"],
                    params={"classifier": classifier_name},
                )
            )
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.RECALL_CALCULATION,
                    value=metrics["recall"],
                    params={"classifier": classifier_name},
                )
            )
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.PRECISION_CALCULATION,
                    value=metrics["precision"],
                    params={"classifier": classifier_name},
                )
            )
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.BALANCED_ACCURACY,
                    value=metrics["balanced_acc"],
                    params={"classifier": classifier_name},
                )
            )
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.MCC,
                    value=metrics["mcc"],
                    params={"classifier": classifier_name},
                )
            )
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.SPECIFICITY,
                    value=metrics["specificity"],
                    params={"classifier": classifier_name},
                )
            )

            # Per-type metrics
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.MEAN_RECALL_PER_TYPE,
                    value=metrics["mean_recall_per_type"],
                    params={"classifier": classifier_name},
                )
            )
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.MIN_RECALL_PER_TYPE,
                    value=metrics["min_recall_per_type"],
                    params={"classifier": classifier_name},
                )
            )
            metric_results.append(
                MetricResult(
                    metric_type=MetricType.MEAN_PRECISION_PER_TYPE,
                    value=metrics["mean_precision_per_type"],
                    params={"classifier": classifier_name},
                )
            )

        return metric_results
