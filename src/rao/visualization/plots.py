"""All visualization functions. Each saves a PNG if save_path is provided."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from rao.config import FIGURES_DIR

plt.style.use("seaborn-v0_8-whitegrid")
_FIG_SIZE = (10, 6)


def _save_or_show(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Generic EDA plots
# ---------------------------------------------------------------------------

def plot_date_distribution(
    df: pd.DataFrame,
    date_col: str,
    title: str,
    save_path: Path | None = None,
) -> None:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    monthly = df.groupby(df[date_col].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    monthly.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_missing_heatmap(
    df: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 0.6), 4))
    null_pct = df.isnull().mean() * 100
    sns.barplot(x=null_pct.index, y=null_pct.values, ax=ax, color="tomato")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing %")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty or len(numeric.columns) < 2:
        return
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_distribution(
    df: pd.DataFrame,
    col: str,
    title: str,
    hue_col: str | None = None,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    if hue_col and hue_col in df.columns:
        for label, grp in df.groupby(hue_col):
            grp[col].dropna().plot(kind="hist", bins=30, alpha=0.6, label=label, ax=ax)
        ax.legend()
    else:
        df[col].dropna().plot(kind="hist", bins=30, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Domain-specific EDA plots
# ---------------------------------------------------------------------------

def plot_inventory_stockout_risk(
    inv_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    if "Stockout_Risk_Score" not in inv_df.columns or "Item_Name" not in inv_df.columns:
        return
    df = inv_df.copy()
    df["Date"] = pd.to_datetime(df.get("Date", pd.NaT), errors="coerce")
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    pivot = df.pivot_table(values="Stockout_Risk_Score", index="Item_Name", columns="Month", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)), max(4, len(pivot.index))))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, annot=False)
    ax.set_title("Inventory Stockout Risk Score (Item × Month)", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Item")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_staff_utilization(
    staff_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    if "Utilization_Rate" not in staff_df.columns or "Staff_Type" not in staff_df.columns:
        return
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    order = staff_df["Staff_Type"].value_counts().index.tolist()
    sns.boxplot(data=staff_df, x="Staff_Type", y="Utilization_Rate", hue="Staff_Type",
                order=order, ax=ax, palette="Set2", legend=False)
    ax.set_title("Staff Utilization Rate by Type", fontsize=14)
    ax.set_xlabel("Staff Type")
    ax.set_ylabel("Utilization Rate (patients per 8h shift)")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_financial_trends(
    fin_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    df = fin_df.copy()
    cat_col = "Corrected_Category" if "Corrected_Category" in df.columns else "Expense_Category"
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby(["Month", cat_col])["Amount"].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    monthly.plot(kind="area", stacked=True, ax=ax, alpha=0.7, colormap="tab10")
    ax.set_title("Monthly Expense by Category", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Amount ($)")
    plt.xticks(rotation=45)
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_patient_diagnosis_mix(
    patient_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    if "Primary_Diagnosis" not in patient_df.columns:
        return
    counts = patient_df["Primary_Diagnosis"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140,
           colors=sns.color_palette("pastel")[:len(counts)])
    ax.set_title("Patient Diagnosis Distribution", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Algorithm comparison plots
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    gwo_histories: list[list[float]],
    pso_histories: list[list[float]],
    save_path: Path | None = None,
) -> None:
    """Mean ± std convergence curve across N runs for both algorithms."""
    gwo_arr = np.array(gwo_histories)
    pso_arr = np.array(pso_histories)
    iters = np.arange(1, gwo_arr.shape[1] + 1)

    gwo_mean, gwo_std = gwo_arr.mean(axis=0), gwo_arr.std(axis=0)
    pso_mean, pso_std = pso_arr.mean(axis=0), pso_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(iters, gwo_mean, label="GWO", color="steelblue", linewidth=2)
    ax.fill_between(iters, gwo_mean - gwo_std, gwo_mean + gwo_std, alpha=0.2, color="steelblue")
    ax.plot(iters, pso_mean, label="PSO", color="tomato", linewidth=2)
    ax.fill_between(iters, pso_mean - pso_std, pso_mean + pso_std, alpha=0.2, color="tomato")
    ax.set_title("Convergence Curves (Mean ± Std, 30 Runs)", fontsize=14)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness (Cost)")
    ax.legend(fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_fitness_boxplot(
    gwo_results: list[float],
    pso_results: list[float],
    save_path: Path | None = None,
) -> None:
    """Side-by-side box plots of final fitness values."""
    data = pd.DataFrame({"GWO": gwo_results, "PSO": pso_results})
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(data=data, palette=["steelblue", "tomato"], ax=ax)
    ax.set_title("Final Fitness Distribution (30 Runs)", fontsize=14)
    ax.set_ylabel("Fitness (Total Cost + Penalty)")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_allocation_heatmap(
    allocation: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    save_path: Path | None = None,
) -> None:
    """Annotated heatmap of a resource allocation matrix."""
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.8), max(6, len(row_labels) * 0.4)))
    sns.heatmap(
        allocation,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap="Blues",
        annot=True,
        fmt=".0f",
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Batch EDA save
# ---------------------------------------------------------------------------

def save_all_eda_plots(dfs: dict[str, pd.DataFrame], figures_dir: Path | None = None) -> None:
    """Generate and save all EDA plots to figures_dir."""
    out = figures_dir or FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)

    print("[viz] Saving EDA plots...")

    # Financial
    fin = dfs.get("financial")
    if fin is not None:
        plot_date_distribution(fin, "Date", "Financial Records Over Time",
                               save_path=out / "financial_date_dist.png")
        plot_financial_trends(fin, save_path=out / "financial_trends.png")
        plot_distribution(fin, "Amount", "Expense Amount Distribution",
                          hue_col="Corrected_Category" if "Corrected_Category" in fin.columns else "Expense_Category",
                          save_path=out / "financial_amount_dist.png")

    # Inventory
    inv = dfs.get("inventory")
    if inv is not None:
        plot_inventory_stockout_risk(inv, save_path=out / "inventory_stockout_risk.png")
        plot_distribution(inv, "Unit_Cost", "Unit Cost Distribution",
                          hue_col="Item_Type", save_path=out / "inventory_unit_cost_dist.png")
        plot_correlation_matrix(inv, "Inventory Numeric Correlations",
                                save_path=out / "inventory_correlation.png")

    # Staff
    staff = dfs.get("staff")
    if staff is not None:
        plot_staff_utilization(staff, save_path=out / "staff_utilization.png")
        plot_distribution(staff, "Hours_Worked", "Hours Worked Distribution",
                          hue_col="Staff_Type", save_path=out / "staff_hours_dist.png")
        plot_distribution(staff, "Patients_Assigned", "Patients Assigned per Shift",
                          hue_col="Staff_Type", save_path=out / "staff_patients_dist.png")

    # Patient
    patient = dfs.get("patient")
    if patient is not None:
        plot_patient_diagnosis_mix(patient, save_path=out / "patient_diagnosis_mix.png")
        plot_distribution(patient, "Bed_Days", "Bed Days Distribution",
                          hue_col="Room_Type", save_path=out / "patient_bed_days_dist.png")

    print(f"[viz] Saved plots to {out}\n")
