"""Central configuration: paths, seeds, hyperparameters."""
from pathlib import Path

# Reproducibility
RANDOM_SEED: int = 42

# Experiment settings
N_RUNS: int = 30
MAX_ITER: int = 200
POP_SIZE: int = 30

# Paths — derived from this file's location so they work from any cwd
_ROOT = Path(__file__).resolve().parents[2]  # project root (two levels up from src/rao/)

DATA_RAW_DIR = _ROOT / "data" / "raw"
DATA_PROCESSED_DIR = _ROOT / "data" / "processed"
REPORTS_DIR = _ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create output dirs on import
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Domain constants
STAFF_HOURLY_COST: dict[str, float] = {
    "Surgeon": 150.0,
    "Nurse": 60.0,
    "Technician": 45.0,
}

ROOM_CAPACITY: dict[str, int] = {
    "ICU": 10,
    "General Ward": 30,
    "Emergency": 15,
}

MIN_STAFF_PER_SHIFT: dict[str, int] = {
    "Surgeon": 1,
    "Nurse": 2,
    "Technician": 1,
}

OVERTIME_THRESHOLD_HOURS: float = 8.0
OVERTIME_MULTIPLIER: float = 1.5
HOLDING_COST_FACTOR: float = 0.0002   # ~2% annual / 365 days
PENALTY_WEIGHT: float = 1e6
