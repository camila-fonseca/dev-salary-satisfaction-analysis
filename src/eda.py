"""
Exploratory Data Analysis (EDA) utilities.

These helper functions generate the main figures used in the notebook:
distributions, bivariate relationships and group comparisons.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_salary_distribution(
    df: pd.DataFrame,
    salary_col: str = "ConvertedCompYearly",
    log_col: Optional[str] = "LogConvertedCompYearly",
) -> None:
    """Plot raw and log‑transformed salary distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df[salary_col].dropna(), bins=50, ax=axes[0])
    axes[0].set_title("Salary distribution (raw)")
    axes[0].set_xlabel(salary_col)

    if log_col in df.columns:
        sns.histplot(df[log_col].dropna(), bins=50, ax=axes[1])
        axes[1].set_title("Salary distribution (log‑transformed)")
        axes[1].set_xlabel(log_col)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_experience_vs_salary(
    df: pd.DataFrame,
    exp_col: str = "WorkExp",
    salary_col: str = "LogConvertedCompYearly",
) -> None:
    """Scatter plot of experience vs (log) salary with a smooth trend line."""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x=exp_col,
        y=salary_col,
        alpha=0.3,
        s=20,
    )
    sns.regplot(
        data=df,
        x=exp_col,
        y=salary_col,
        scatter=False,
        lowess=True,
        color="black",
    )
    plt.title("Experience vs log salary")
    plt.tight_layout()
    plt.show()


def plot_salary_by_category(
    df: pd.DataFrame,
    category_col: str,
    salary_col: str = "LogConvertedCompYearly",
    top_n: int = 15,
) -> None:
    """Boxplot of salary by a categorical column (e.g. Country, Industry)."""
    counts = df[category_col].value_counts().head(top_n).index
    subset = df[df[category_col].isin(counts)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=subset,
        x=salary_col,
        y=category_col,
        orient="h",
    )
    plt.title(f"{salary_col} by {category_col} (top {top_n})")
    plt.tight_layout()
    plt.show()


def plot_remote_vs_salary(
    df: pd.DataFrame,
    remote_col: str = "RemoteWork_bin",
    salary_col: str = "LogConvertedCompYearly",
) -> None:
    """Compare salary distributions by remote / in‑person groups."""
    plt.figure(figsize=(6, 5))
    sns.boxplot(
        data=df,
        x=remote_col,
        y=salary_col,
    )
    plt.xticks([0, 1], ["In‑person / mostly in‑person", "Remote / flexible"])
    plt.title("Log salary by remote work arrangement")
    plt.tight_layout()
    plt.show()


def plot_job_satisfaction_distribution(
    df: pd.DataFrame,
    sat_col: str = "JobSat",
) -> None:
    """Plot the distribution of the original job satisfaction scores."""
    plt.figure(figsize=(6, 5))
    sns.histplot(df[sat_col].dropna(), bins=10, discrete=True)
    plt.title("Job satisfaction distribution")
    plt.xlabel(sat_col)
    plt.tight_layout()
    plt.show()
