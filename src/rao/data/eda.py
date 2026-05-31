"""EDA, feature engineering, and aggregation of optimization inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rao.config import STAFF_HOURLY_COST, OVERTIME_MULTIPLIER, HOLDING_COST_FACTOR


# ---------------------------------------------------------------------------
# Per-dataset feature engineering
# ---------------------------------------------------------------------------

def compute_staff_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cost_map = STAFF_HOURLY_COST
    df["Hourly_Rate"] = df["Staff_Type"].map(cost_map).fillna(45.0)
    df["Regular_Hours"] = df["Hours_Worked"].clip(upper=8.0)
    df["OT_Hours"] = df["Hours_Worked"].clip(lower=8.0) - 8.0
    df["Daily_Staffing_Cost"] = df["Regular_Hours"] * df["Hourly_Rate"]
    df["Overtime_Cost"] = df["OT_Hours"] * df["Hourly_Rate"] * OVERTIME_MULTIPLIER
    df["Total_Shift_Cost"] = df["Daily_Staffing_Cost"] + df["Overtime_Cost"]

    # Specialty mix per date (fraction of each type)
    if "Shift_Date" in df.columns:
        daily_counts = df.groupby(["Shift_Date", "Staff_Type"]).size().unstack(fill_value=0)
        daily_totals = daily_counts.sum(axis=1)
        for col in daily_counts.columns:
            daily_counts[f"{col}_Pct"] = (daily_counts[col] / daily_totals * 100).round(1)
        df = df.merge(daily_counts.reset_index()[["Shift_Date"] + [f"{c}_Pct" for c in daily_counts.columns if not c.endswith("_Pct")]],
                      on="Shift_Date", how="left")

    return df


def compute_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Stockout risk score: how close to stockout relative to minimum stock
    numerator = (df["Min_Required"] - df["Current_Stock"]).clip(lower=0)
    denominator = df["Min_Required"].replace(0, np.nan)
    df["Stockout_Risk_Score"] = (numerator / denominator).fillna(0).round(3)

    # Holding cost per day
    df["Holding_Cost_Per_Day"] = (df["Current_Stock"] * df["Unit_Cost"] * HOLDING_COST_FACTOR).round(2)

    # Restock quantities and costs
    df["Order_Quantity"] = (df["Max_Capacity"] - df["Current_Stock"]).clip(lower=0)
    df["Restock_Cost"] = df["Order_Quantity"] * df["Unit_Cost"]

    return df


def compute_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Acuity score (may already be present from preprocessing)
    if "Acuity_Score" not in df.columns:
        acuity_map = {"ICU": 3, "Emergency": 2, "General Ward": 1}
        df["Acuity_Score"] = df["Room_Type"].map(acuity_map).fillna(1).astype(int)

    # Staff intensity: total staff per bed-day
    total_staff = df.get("Staff_Surgeons", 0) + df.get("Staff_Nurses", 0) + df.get("Staff_Doctors", 0)
    safe_bed_days = df["Bed_Days"].replace(0, 1)
    df["Staff_Intensity"] = (total_staff / safe_bed_days).round(2)

    # Temporal features
    if "Admission_Date" in df.columns:
        df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")
        df["Admission_Month"] = df["Admission_Date"].dt.month
        df["Admission_DayOfWeek"] = df["Admission_Date"].dt.day_of_week

    return df


def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")

    # Rolling 7-row cost (integer window; data is ~daily so ~7 days)
    df["Rolling_7Day_Cost"] = (
        df["Amount"].rolling(7, min_periods=1).mean()
    )

    # Cost anomaly flag: > mean + 2*std in rolling window
    rolling_mean = df["Rolling_7Day_Cost"]
    rolling_std = df["Amount"].rolling(7, min_periods=1).std().fillna(0)
    df["Cost_Anomaly"] = df["Amount"] > (rolling_mean + 2 * rolling_std)

    # Category budget share per month
    cat_col = "Corrected_Category" if "Corrected_Category" in df.columns else "Expense_Category"
    df["YearMonth"] = df["Date"].dt.to_period("M")
    monthly_total = df.groupby("YearMonth")["Amount"].transform("sum")
    df["Category_Budget_Share"] = (df["Amount"] / monthly_total * 100).round(2)

    return df


# ---------------------------------------------------------------------------
# Aggregate optimization inputs
# ---------------------------------------------------------------------------

def build_optimization_inputs(
    staff_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    financial_df: pd.DataFrame,
) -> dict:
    """Aggregate cleaned/featured data into arrays needed by the optimization problem."""
    # Number of unique shifts (days × assignment combos); use dates as shifts
    if "Shift_Date" in staff_df.columns:
        shift_dates = sorted(staff_df["Shift_Date"].unique())
    else:
        shift_dates = list(range(30))
    n_shifts = len(shift_dates)

    staff_types = list(STAFF_HOURLY_COST.keys())
    n_staff_types = len(staff_types)

    items = inventory_df["Item_Name"].unique() if "Item_Name" in inventory_df.columns else []
    n_items = len(items)

    # Budget limit: mean daily spend from financial data
    if "Date" in financial_df.columns and "Amount" in financial_df.columns:
        daily_spend = financial_df.groupby("Date")["Amount"].sum()
        budget_limit = float(daily_spend.mean())
    else:
        budget_limit = 50_000.0

    # Per-shift min/max staff counts (shape: n_shifts x n_staff_types)
    min_staff_per_shift = np.zeros((n_shifts, n_staff_types), dtype=float)
    max_staff_per_shift = np.zeros((n_shifts, n_staff_types), dtype=float)

    for i, stype in enumerate(staff_types):
        mask = staff_df["Staff_Type"] == stype if "Staff_Type" in staff_df.columns else pd.Series([True] * len(staff_df))
        assigned = staff_df[mask].groupby("Shift_Date")["Patients_Assigned"].count() if "Shift_Date" in staff_df.columns else pd.Series([1])
        min_v = max(1, int(assigned.min())) if len(assigned) else 1
        max_v = max(min_v + 1, int(assigned.max())) if len(assigned) else 5
        min_staff_per_shift[:, i] = min_v
        max_staff_per_shift[:, i] = max_v

    # Inventory restock bounds (shape: n_items)
    if n_items > 0:
        item_stats = inventory_df.groupby("Item_Name").agg(
            min_order=("Order_Quantity", "min"),
            max_order=("Order_Quantity", "max"),
            unit_cost=("Unit_Cost", "mean"),
        )
        min_order_qty = item_stats["min_order"].clip(lower=0).values.astype(float)
        max_order_qty = item_stats["max_order"].clip(lower=1).values.astype(float)
        inventory_unit_costs = item_stats["unit_cost"].values.astype(float)
    else:
        min_order_qty = np.zeros(1)
        max_order_qty = np.ones(1) * 100
        inventory_unit_costs = np.ones(1) * 10.0

    # Patient demand per shift
    if "Admission_Date" in patient_df.columns:
        patient_df = patient_df.copy()
        patient_df["Admission_Date"] = pd.to_datetime(patient_df["Admission_Date"], errors="coerce")
        daily_admissions = patient_df.groupby(patient_df["Admission_Date"].dt.date).size()
        mean_demand = float(daily_admissions.mean()) if len(daily_admissions) else 5.0
    else:
        mean_demand = 5.0

    patient_demand = np.full(n_shifts, mean_demand)
    staff_cost_rates = np.array([STAFF_HOURLY_COST[t] for t in staff_types], dtype=float)

    return {
        "n_shifts": n_shifts,
        "n_staff_types": n_staff_types,
        "n_inventory_items": n_items if n_items > 0 else 1,
        "budget_limit": budget_limit,
        "min_staff_per_shift": min_staff_per_shift,
        "max_staff_per_shift": max_staff_per_shift,
        "min_order_qty": min_order_qty,
        "max_order_qty": max_order_qty,
        "staff_cost_rates": staff_cost_rates,
        "inventory_unit_costs": inventory_unit_costs,
        "patient_demand": patient_demand,
        "shift_dates": shift_dates,
        "staff_types": staff_types,
        "item_names": list(items),
    }


# ---------------------------------------------------------------------------
# Full EDA runner
# ---------------------------------------------------------------------------

def run_full_eda(dfs: dict[str, pd.DataFrame]) -> dict:
    """Apply all feature engineering functions and build optimization inputs."""
    print("[eda] Computing features...")

    staff = compute_staff_features(dfs["staff"])
    inventory = compute_inventory_features(dfs["inventory"])
    patient = compute_patient_features(dfs["patient"])
    financial = compute_financial_features(dfs["financial"])

    inputs = build_optimization_inputs(staff, inventory, patient, financial)

    print(f"  Shifts: {inputs['n_shifts']}, Staff types: {inputs['n_staff_types']}, "
          f"Inventory items: {inputs['n_inventory_items']}")
    print(f"  Decision variable dim: {inputs['n_shifts'] * inputs['n_staff_types'] + inputs['n_inventory_items']}")
    print(f"  Daily budget limit: ${inputs['budget_limit']:,.0f}")
    print("[eda] Done.\n")

    return {
        "staff": staff,
        "inventory": inventory,
        "patient": patient,
        "financial": financial,
        "vendor": dfs["vendor"],
        "optimization_inputs": inputs,
    }
