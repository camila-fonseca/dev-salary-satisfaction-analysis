"""
Classification models for job satisfaction prediction (JobSat ≥ 8).
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    RANDOM_STATE,
    evaluate_classification,
    ClassificationMetrics,
    split_task_specific,
)


def build_jobsat_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Create a ColumnTransformer for job satisfaction classification."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def train_jobsat_models(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Train Logistic Regression and Random Forest models for
    high job satisfaction prediction.
    """
    feature_cols = numeric_features + categorical_features
    X_train, X_test, y_train, y_test = split_task_specific(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    preprocessor = build_jobsat_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        # Use predicted probabilities for the positive class
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics: ClassificationMetrics = evaluate_classification(
            y_test,
            y_proba,
            threshold=0.5,
        )
        results[name] = {
            "pipeline": pipe,
            "metrics": metrics,
        }

    return results
