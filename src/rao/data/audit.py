"""Data quality audit — reads raw data without modifying it."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from rao.data.loader import load_all_raw


# ---------------------------------------------------------------------------
# Generic auditor
# ---------------------------------------------------------------------------

def audit_dataset(df: pd.DataFrame, name: str) -> dict:
    """Return basic quality metrics for any dataset."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    return {
        "name": name,
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": null_counts.to_dict(),
        "null_pct": null_pct.to_dict(),
        "numeric_stats": df.describe(include="all").to_dict(),
    }


# ---------------------------------------------------------------------------
# Dataset-specific auditors
# ---------------------------------------------------------------------------

def audit_patient(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")
    df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"], errors="coerce")

    inverted_mask = df["Discharge_Date"] < df["Admission_Date"]
    inverted_count = int(inverted_mask.sum())
    inverted_pct = round(inverted_count / len(df) * 100, 1)

    computed_bed_days = (df["Discharge_Date"] - df["Admission_Date"]).dt.days
    mismatch_mask = (computed_bed_days - df["Bed_Days"]).abs() > 0
    bed_day_mismatch_count = int(mismatch_mask.sum())

    return {
        "inverted_dates_count": inverted_count,
        "inverted_dates_pct": inverted_pct,
        "inverted_indices": df.index[inverted_mask].tolist(),
        "bed_day_mismatch_count": bed_day_mismatch_count,
        "staff_needed_unique": df["Staff_Needed"].unique().tolist(),
    }


def audit_inventory(df: pd.DataFrame) -> dict:
    stock_overflow = (df["Current_Stock"] > df["Max_Capacity"]).sum()
    min_exceeds_max = (df["Min_Required"] > df["Max_Capacity"]).sum()

    q1, q3 = df["Unit_Cost"].quantile(0.25), df["Unit_Cost"].quantile(0.75)
    iqr = q3 - q1
    unit_cost_outliers = int(((df["Unit_Cost"] < q1 - 3 * iqr) | (df["Unit_Cost"] > q3 + 3 * iqr)).sum())

    stockout_risk = (df["Avg_Usage_Per_Day"] > df["Current_Stock"]).sum()

    return {
        "stock_overflow_count": int(stock_overflow),
        "min_exceeds_max_count": int(min_exceeds_max),
        "unit_cost_outliers_count": unit_cost_outliers,
        "unit_cost_range": (float(df["Unit_Cost"].min()), float(df["Unit_Cost"].max())),
        "stockout_risk_count": int(stockout_risk),
        "item_names": df["Item_Name"].unique().tolist(),
    }


def audit_staff(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["Shift_Date"] = pd.to_datetime(df["Shift_Date"], errors="coerce")

    def _parse_time_minutes(t: str) -> int:
        """Convert '06:00 AM' / '07:00 PM' to minutes since midnight."""
        t = t.strip()
        try:
            dt = pd.to_datetime(t, format="%I:%M %p")
        except Exception:
            dt = pd.to_datetime(t)
        return dt.hour * 60 + dt.minute

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_min = df["Shift_Start_Time"].apply(_parse_time_minutes)
        end_min = df["Shift_End_Time"].apply(_parse_time_minutes)

    computed_hours = (end_min - start_min) / 60.0
    # Handle cross-midnight shifts
    computed_hours = computed_hours.where(computed_hours > 0, computed_hours + 24)

    mismatch_mask = (computed_hours - df["Hours_Worked"]).abs() > 0.5
    cross_midnight_mask = end_min < start_min

    return {
        "shift_hour_mismatch_count": int(mismatch_mask.sum()),
        "cross_midnight_shifts_count": int(cross_midnight_mask.sum()),
        "date_range": (str(df["Shift_Date"].min().date()), str(df["Shift_Date"].max().date())),
        "staff_types": df["Staff_Type"].unique().tolist(),
        "assignments": df["Current_Assignment"].unique().tolist(),
    }


def audit_financial(df: pd.DataFrame) -> dict:
    category_keywords: dict[str, set[str]] = {
        "Staffing": {"salary", "surgeon", "nurse", "technician", "doctor"},
        "Supplies": {"mask", "glove", "iv", "gown", "drip", "bandage", "syringe"},
        "Equipment": {"ventilator", "mri", "x-ray", "machine", "monitor", "table"},
    }

    def _infer_category(desc: str) -> str:
        desc_lower = str(desc).lower()
        for cat, kws in category_keywords.items():
            if any(kw in desc_lower for kw in kws):
                return cat
        return "Unknown"

    df = df.copy()
    df["Inferred_Category"] = df["Description"].apply(_infer_category)
    mismatch_count = int((df["Expense_Category"] != df["Inferred_Category"]).sum())

    q1, q3 = df["Amount"].quantile(0.25), df["Amount"].quantile(0.75)
    iqr = q3 - q1
    amount_outliers = int(((df["Amount"] < q1 - 3 * iqr) | (df["Amount"] > q3 + 3 * iqr)).sum())

    return {
        "category_mismatch_count": mismatch_count,
        "category_mismatch_pct": round(mismatch_count / len(df) * 100, 1),
        "amount_outliers_count": amount_outliers,
        "expense_categories": df["Expense_Category"].unique().tolist(),
        "descriptions_sample": df["Description"].value_counts().head(10).to_dict(),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_audit() -> dict[str, dict]:
    """Load all raw datasets, run all audits, print summary, return results."""
    raw = load_all_raw()

    results: dict[str, dict] = {}
    for name, df in raw.items():
        results[name] = audit_dataset(df, name)

    results["patient"].update(audit_patient(raw["patient"]))
    results["inventory"].update(audit_inventory(raw["inventory"]))
    results["staff"].update(audit_staff(raw["staff"]))
    results["financial"].update(audit_financial(raw["financial"]))

    _print_audit_summary(results, raw)
    return results


def _print_audit_summary(results: dict, raw: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 70)
    print("DATA QUALITY AUDIT SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Rows':>6} {'Cols':>5} {'Nulls':>7} {'Key Issues'}")
    print("-" * 70)
    for name, r in results.items():
        rows, cols = r["shape"]
        total_nulls = sum(r["null_counts"].values())
        issues = []
        if name == "patient":
            issues.append(f"inverted dates: {r.get('inverted_dates_count', 0)} ({r.get('inverted_dates_pct', 0)}%)")
        if name == "inventory":
            issues.append(f"stock overflow: {r.get('stock_overflow_count', 0)}")
            issues.append(f"min>max: {r.get('min_exceeds_max_count', 0)}")
        if name == "staff":
            issues.append(f"hour mismatches: {r.get('shift_hour_mismatch_count', 0)}")
        if name == "financial":
            issues.append(f"category mismatches: {r.get('category_mismatch_count', 0)}")
        print(f"{name:<15} {rows:>6} {cols:>5} {total_nulls:>7}   {'; '.join(issues) or 'none detected'}")
    print("=" * 70 + "\n")
