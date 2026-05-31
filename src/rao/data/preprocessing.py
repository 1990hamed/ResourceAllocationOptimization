"""Clean all five raw datasets and save to data/processed/."""
from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd

from rao.data.loader import load_all_raw, save_processed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iqr_clip(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """Clip values outside Q1 - k*IQR / Q3 + k*IQR to the boundary."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(lower=q1 - multiplier * iqr, upper=q3 + multiplier * iqr)


def _parse_time_minutes(t: str) -> int:
    """Convert '06:00 AM' / '07:00 PM' to minutes since midnight."""
    t = str(t).strip()
    try:
        dt = pd.to_datetime(t, format="%I:%M %p")
    except Exception:
        dt = pd.to_datetime(t)
    return dt.hour * 60 + dt.minute


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

def clean_patient(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")
    df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"], errors="coerce")

    # Swap inverted dates (assumed data entry error)
    inverted = df["Discharge_Date"] < df["Admission_Date"]
    n_inverted = inverted.sum()
    if n_inverted:
        df.loc[inverted, ["Admission_Date", "Discharge_Date"]] = (
            df.loc[inverted, ["Discharge_Date", "Admission_Date"]].values
        )
        print(f"  [patient] Swapped {n_inverted} inverted date pairs.")

    # Recompute Bed_Days from corrected dates
    df["Bed_Days"] = (df["Discharge_Date"] - df["Admission_Date"]).dt.days.clip(lower=0)

    # Parse Staff_Needed into integer columns
    def _parse_staff(s: str) -> tuple[int, int, int]:
        s = str(s)
        surgeons = int(m.group(1)) if (m := re.search(r"(\d+)\s*Surgeon", s, re.I)) else 0
        nurses = int(m.group(1)) if (m := re.search(r"(\d+)\s*Nurse", s, re.I)) else 0
        doctors = int(m.group(1)) if (m := re.search(r"(\d+)\s*Doctor", s, re.I)) else 0
        return surgeons, nurses, doctors

    parsed = df["Staff_Needed"].apply(_parse_staff)
    df["Staff_Surgeons"] = [x[0] for x in parsed]
    df["Staff_Nurses"] = [x[1] for x in parsed]
    df["Staff_Doctors"] = [x[2] for x in parsed]

    # Acuity score
    acuity_map = {"ICU": 3, "Emergency": 2, "General Ward": 1}
    df["Acuity_Score"] = df["Room_Type"].map(acuity_map).fillna(1).astype(int)

    return df


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Fix stock overflow
    overflow = df["Current_Stock"] > df["Max_Capacity"]
    if overflow.sum():
        print(f"  [inventory] Capping {overflow.sum()} rows where Current_Stock > Max_Capacity.")
    df["Current_Stock"] = df[["Current_Stock", "Max_Capacity"]].min(axis=1)

    # Fix Min_Required > Max_Capacity
    min_exceeds = df["Min_Required"] > df["Max_Capacity"]
    if min_exceeds.sum():
        print(f"  [inventory] Fixing {min_exceeds.sum()} rows where Min_Required > Max_Capacity.")
    df.loc[min_exceeds, "Min_Required"] = (df.loc[min_exceeds, "Max_Capacity"] * 0.8).astype(int)

    # IQR-clip unit cost
    df["Unit_Cost"] = _iqr_clip(df["Unit_Cost"])

    # Derived features
    safe_usage = df["Avg_Usage_Per_Day"].replace(0, np.nan)
    df["Days_Of_Stock"] = (df["Current_Stock"] / safe_usage).fillna(0).round(1)
    df["Restock_Urgency"] = df["Days_Of_Stock"] < df["Restock_Lead_Time"]

    return df


# ---------------------------------------------------------------------------
# Staff
# ---------------------------------------------------------------------------

def clean_staff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Shift_Date"] = pd.to_datetime(df["Shift_Date"], errors="coerce").dt.date

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_min = df["Shift_Start_Time"].apply(_parse_time_minutes)
        end_min = df["Shift_End_Time"].apply(_parse_time_minutes)

    computed_hours = (end_min - start_min) / 60.0
    # Handle cross-midnight shifts
    computed_hours = computed_hours.where(computed_hours > 0, computed_hours + 24)

    mismatch = (computed_hours - df["Hours_Worked"]).abs() > 0.5
    if mismatch.sum():
        print(f"  [staff] Correcting {mismatch.sum()} mismatched Hours_Worked values.")
    df["Hours_Worked"] = computed_hours.round(1)

    df["Overtime_Hours"] = (df["Hours_Worked"] - 8.0).clip(lower=0).round(1)

    # Utilization rate: patients per standard 8-hour shift
    safe_hours = df["Hours_Worked"].replace(0, np.nan)
    df["Utilization_Rate"] = (df["Patients_Assigned"] / (safe_hours / 8.0)).fillna(0).round(2)

    return df


# ---------------------------------------------------------------------------
# Financial
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "Staffing": {"salary", "salaries", "surgeon", "nurse", "technician", "doctor", "staff"},
    "Supplies": {"mask", "glove", "gloves", "iv", "gown", "drip", "bandage", "syringe"},
    "Equipment": {"ventilator", "mri", "x-ray", "machine", "monitor", "table", "scanner"},
}


def _infer_category(desc: str) -> str:
    desc_lower = str(desc).lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in kws):
            return cat
    return "Unknown"


def clean_financial(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Inferred_Category"] = df["Description"].apply(_infer_category)
    df["Category_Mismatch"] = df["Expense_Category"] != df["Inferred_Category"]
    df["Corrected_Category"] = df["Inferred_Category"].where(
        df["Inferred_Category"] != "Unknown", df["Expense_Category"]
    )

    df["Amount"] = _iqr_clip(df["Amount"])

    mismatch_count = df["Category_Mismatch"].sum()
    if mismatch_count:
        print(f"  [financial] Found {mismatch_count} category/description mismatches — Corrected_Category added.")

    return df


# ---------------------------------------------------------------------------
# Vendor
# ---------------------------------------------------------------------------

def clean_vendor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Column name may have spaces / parentheses — normalise
    df.columns = df.columns.str.strip()

    for col in ("Last_Order_Date", "Next_Delivery_Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    lead_col = next((c for c in df.columns if "Lead_Time" in c), None)
    if lead_col and "Last_Order_Date" in df.columns and "Next_Delivery_Date" in df.columns:
        df["Actual_Lead_Time"] = (df["Next_Delivery_Date"] - df["Last_Order_Date"]).dt.days

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def preprocess_all(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Load all raw datasets, clean them, save to data/processed/, return dict."""
    print("\n[preprocessing] Starting full pipeline...")
    raw = load_all_raw()

    cleaned: dict[str, pd.DataFrame] = {}
    cleaners = {
        "patient": lambda df: clean_patient(df, seed=seed),
        "inventory": clean_inventory,
        "staff": clean_staff,
        "financial": clean_financial,
        "vendor": clean_vendor,
    }

    for name, cleaner in cleaners.items():
        print(f"\n  Processing '{name}'...")
        cleaned[name] = cleaner(raw[name])
        save_processed(cleaned[name], name)
        print(f"  Saved data/processed/{name}_clean.csv  ({len(cleaned[name])} rows)")

    print("\n[preprocessing] Done.\n")
    return cleaned
