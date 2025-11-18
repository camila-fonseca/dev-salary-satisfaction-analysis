"""
Preprocessing utilities for the Developer Salary & Job Satisfaction project.

This module centralises data loading, cleaning and feature construction
so the notebook can focus on analysis and interpretation.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


def load_survey_data(data_path: str) -> pd.DataFrame:
    """Load the Stack Overflow survey data from a CSV file."""
    df = pd.read_csv(data_path)
    return df


def to_numeric_safe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce a list of columns to numeric, keeping the original dataframe."""
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def create_job_satisfaction_binary(
    df: pd.DataFrame,
    source_col: str = "JobSat",
    threshold: float = 8.0,
    new_col: str = "JobSat_bin",
) -> pd.DataFrame:
    """Create a binary column for high job satisfaction (>= threshold)."""
    df = df.copy()
    df[new_col] = np.where(df[source_col] >= threshold, 1, 0).astype("Int8")
    return df


def encode_remote_work_binary(
    df: pd.DataFrame,
    source_col: str = "RemoteWork",
    new_col: str = "RemoteWork_bin",
) -> pd.DataFrame:
    """
    Encode detailed remote work categories into a binary variable.

    1 = remote / flexible
    0 = in‑person or mostly in‑person
    Other / unexpected entries are set to NaN.
    """
    df = df.copy()

    remote_positive = {
        "Remote",
        "Your choice (very flexible, you can come in when you want or just as needed)",
        "Hybrid (some in-person, leans heavy to flexibility)",
    }
    remote_negative = {
        "In-person",
        "Hybrid (some remote, leans heavy to in-person)",
    }

    df[new_col] = np.where(
        df[source_col].isin(remote_positive),
        1,
        np.where(df[source_col].isin(remote_negative), 0, np.nan),
    ).astype("Int8")

    return df


def log_transform_column(
    df: pd.DataFrame,
    source_col: str,
    new_col: str,
    offset: float = 1e-6,
) -> pd.DataFrame:
    """Apply a log transformation to a positive numeric column."""
    df = df.copy()
    df[new_col] = np.where(
        df[source_col] > 0,
        np.log(df[source_col] + offset),
        np.nan,
    )
    return df


def select_salary_subset(
    df: pd.DataFrame,
    cols: List[str],
    target_col: str = "LogConvertedCompYearly",
) -> pd.DataFrame:
    """
    Convenience helper to select the subset of columns
    used in salary modelling and drop rows with missing target.
    """
    keep_cols = list(dict.fromkeys(cols + [target_col]))
    subset = df[keep_cols].dropna(subset=[target_col]).reset_index(drop=True)
    return subset


def select_jobsat_subset(
    df: pd.DataFrame,
    cols: List[str],
    target_col: str = "JobSat_bin",
) -> pd.DataFrame:
    """
    Convenience helper to select the subset of columns
    used in job satisfaction modelling.
    """
    keep_cols = list(dict.fromkeys(cols + [target_col]))
    subset = df[keep_cols].dropna(subset=[target_col]).reset_index(drop=True)
    return subset
