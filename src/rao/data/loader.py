"""Thin I/O wrappers for raw and processed datasets."""
import pandas as pd
from rao.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

_DATASET_NAMES = ("financial", "inventory", "patient", "staff", "vendor")


def load_raw(name: str) -> pd.DataFrame:
    """Load a raw CSV by dataset name (financial, inventory, patient, staff, vendor)."""
    if name not in _DATASET_NAMES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {_DATASET_NAMES}")
    path = DATA_RAW_DIR / f"{name}_data.csv"
    return pd.read_csv(path)


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all five raw datasets. Returns dict keyed by dataset name."""
    return {name: load_raw(name) for name in _DATASET_NAMES}


def load_processed(name: str) -> pd.DataFrame:
    """Load a cleaned CSV from data/processed/."""
    if name not in _DATASET_NAMES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {_DATASET_NAMES}")
    path = DATA_PROCESSED_DIR / f"{name}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}. Run preprocessing first.")
    return pd.read_csv(path)


def load_all_processed() -> dict[str, pd.DataFrame]:
    """Load all five cleaned datasets."""
    return {name: load_processed(name) for name in _DATASET_NAMES}


def save_processed(df: pd.DataFrame, name: str) -> None:
    """Write a cleaned DataFrame to data/processed/<name>_clean.csv."""
    path = DATA_PROCESSED_DIR / f"{name}_clean.csv"
    df.to_csv(path, index=False)
