"""
Utility functions and configuration for the Developer Salary & Job Satisfaction project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Set global random seed for numpy, python and hash seed."""
    import os
    import random

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_regression(y_true, y_pred) -> RegressionMetrics:
    """Return common regression metrics as a dataclass."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2)


def evaluate_classification(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Return classification metrics given true labels and predicted probabilities."""
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc = None

    return ClassificationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
    )


def split_task_specific(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a dataframe into task‑specific train/test sets.

    If the target has few unique values (e.g. classification),
    stratified split is used automatically.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    stratify = y if y.nunique() <= 10 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test
