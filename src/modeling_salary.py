"""
Regression models for salary prediction.

This module wraps the main models used in the notebook so that they
can be trained and evaluated in a reusable way.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    RANDOM_STATE,
    evaluate_regression,
    RegressionMetrics,
    split_task_specific,
)


def build_salary_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Create a ColumnTransformer for salary regression."""
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


def train_salary_models(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Train several regression models for salary prediction.

    Returns a dictionary with fitted pipelines and evaluation metrics.
    """
    feature_cols = numeric_features + categorical_features
    X_train, X_test, y_train, y_test = split_task_specific(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    preprocessor = build_salary_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=[0.1, 1.0, 10.0]),
        "LassoCV": LassoCV(
            alphas=[0.01, 0.1, 1.0],
            random_state=RANDOM_STATE,
            max_iter=2000,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
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
        y_pred = pipe.predict(X_test)
        metrics: RegressionMetrics = evaluate_regression(y_test, y_pred)

        results[name] = {
            "pipeline": pipe,
            "metrics": metrics,
        }

    return results
