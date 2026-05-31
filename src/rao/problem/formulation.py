"""Hospital resource allocation problem definition."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rao.config import HOLDING_COST_FACTOR


@dataclass
class ProblemConfig:
    n_shifts: int
    n_staff_types: int
    n_inventory_items: int
    budget_limit: float
    min_staff_per_shift: np.ndarray      # shape (n_shifts, n_staff_types)
    max_staff_per_shift: np.ndarray      # shape (n_shifts, n_staff_types)
    min_order_qty: np.ndarray            # shape (n_inventory_items,)
    max_order_qty: np.ndarray            # shape (n_inventory_items,)
    staff_cost_rates: np.ndarray         # shape (n_staff_types,)  $/hour
    inventory_unit_costs: np.ndarray     # shape (n_inventory_items,)
    patient_demand: np.ndarray           # shape (n_shifts,)
    hours_per_shift: float = 8.0
    holding_cost_factor: float = field(default_factory=lambda: HOLDING_COST_FACTOR)

    def __post_init__(self) -> None:
        # Ensure arrays are float64
        for attr in (
            "min_staff_per_shift", "max_staff_per_shift",
            "min_order_qty", "max_order_qty",
            "staff_cost_rates", "inventory_unit_costs", "patient_demand",
        ):
            setattr(self, attr, np.asarray(getattr(self, attr), dtype=float))


class HospitalAllocationProblem:
    """Encapsulates the optimization problem: bounds, decode, feasibility."""

    def __init__(self, config: ProblemConfig) -> None:
        self.config = config
        self._staff_dim = config.n_shifts * config.n_staff_types
        self._inv_dim = config.n_inventory_items

    @property
    def dim(self) -> int:
        """Total number of decision variables."""
        return self._staff_dim + self._inv_dim

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bound arrays of shape (dim,)."""
        cfg = self.config
        lb_staff = cfg.min_staff_per_shift.flatten()
        ub_staff = cfg.max_staff_per_shift.flatten()
        lb_inv = cfg.min_order_qty
        ub_inv = cfg.max_order_qty
        lb = np.concatenate([lb_staff, lb_inv])
        ub = np.concatenate([ub_staff, ub_inv])
        # Ensure lb < ub (safety)
        ub = np.maximum(ub, lb + 1.0)
        return lb, ub

    def decode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split decision vector into staff matrix and order quantities."""
        cfg = self.config
        staff_flat = x[: self._staff_dim]
        order_qty = x[self._staff_dim :]
        staff_matrix = staff_flat.reshape(cfg.n_shifts, cfg.n_staff_types)
        return staff_matrix, order_qty

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check all hard constraints (no penalty)."""
        cfg = self.config
        staff_matrix, order_qty = self.decode(x)

        # Staff lower bounds
        if np.any(staff_matrix < cfg.min_staff_per_shift):
            return False
        # Staff upper bounds
        if np.any(staff_matrix > cfg.max_staff_per_shift):
            return False
        # Inventory bounds
        if np.any(order_qty < cfg.min_order_qty):
            return False
        if np.any(order_qty > cfg.max_order_qty):
            return False
        # Budget (approximate: per-shift staffing cost must not exceed limit)
        daily_staff_cost = (staff_matrix * cfg.hours_per_shift) @ cfg.staff_cost_rates
        if np.any(daily_staff_cost > cfg.budget_limit):
            return False
        return True


def build_problem_from_inputs(inputs: dict) -> HospitalAllocationProblem:
    """Construct a HospitalAllocationProblem from build_optimization_inputs() output."""
    cfg = ProblemConfig(
        n_shifts=inputs["n_shifts"],
        n_staff_types=inputs["n_staff_types"],
        n_inventory_items=inputs["n_inventory_items"],
        budget_limit=inputs["budget_limit"],
        min_staff_per_shift=inputs["min_staff_per_shift"],
        max_staff_per_shift=inputs["max_staff_per_shift"],
        min_order_qty=inputs["min_order_qty"],
        max_order_qty=inputs["max_order_qty"],
        staff_cost_rates=inputs["staff_cost_rates"],
        inventory_unit_costs=inputs["inventory_unit_costs"],
        patient_demand=inputs["patient_demand"],
    )
    return HospitalAllocationProblem(cfg)
