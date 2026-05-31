"""Fitness function for the hospital resource allocation problem."""
from __future__ import annotations

import numpy as np

from rao.config import OVERTIME_THRESHOLD_HOURS, OVERTIME_MULTIPLIER, PENALTY_WEIGHT
from rao.problem.formulation import HospitalAllocationProblem


def compute_fitness(
    x: np.ndarray,
    problem: HospitalAllocationProblem,
    penalty_weight: float = PENALTY_WEIGHT,
) -> float:
    """
    Fitness = total_cost + penalty_weight * total_constraint_violation

    Lower is better.

    Cost components:
      1. Regular staffing cost per shift
      2. Overtime cost per shift
      3. Inventory restock cost
      4. Inventory holding cost

    Penalty terms:
      - Under-staffing: max(0, min_staff - allocated) per shift/type
      - Over-budget: max(0, daily_cost - budget_limit) per shift
      - Stockout: max(0, min_order - allocated) per inventory item
    """
    cfg = problem.config
    staff_matrix, order_qty = problem.decode(x)

    # --- Staffing costs ---
    regular_hours = np.minimum(staff_matrix * cfg.hours_per_shift,
                               cfg.hours_per_shift * np.ones_like(staff_matrix))
    overtime_hours = np.maximum(0.0, staff_matrix * cfg.hours_per_shift - cfg.hours_per_shift)

    # cost rate broadcast: (n_shifts, n_staff_types) * (n_staff_types,)
    staffing_cost = (regular_hours * cfg.staff_cost_rates).sum(axis=1)        # per shift
    overtime_cost = (overtime_hours * cfg.staff_cost_rates * OVERTIME_MULTIPLIER).sum(axis=1)
    daily_total = staffing_cost + overtime_cost

    total_staffing_cost = float(daily_total.sum())

    # --- Inventory costs ---
    restock_cost = float((order_qty * cfg.inventory_unit_costs).sum())
    # Approximate held stock after reorder
    held_stock = order_qty
    holding_cost = float((held_stock * cfg.inventory_unit_costs * cfg.holding_cost_factor).sum())

    total_cost = total_staffing_cost + restock_cost + holding_cost

    # --- Constraint violations ---
    violation = 0.0

    # Under-staffing penalty (per shift per type)
    under_staff = np.maximum(0.0, cfg.min_staff_per_shift - staff_matrix)
    violation += float(under_staff.sum())

    # Over-budget penalty (per shift)
    over_budget = np.maximum(0.0, daily_total - cfg.budget_limit)
    violation += float(over_budget.sum())

    # Stockout penalty (per inventory item)
    stockout = np.maximum(0.0, cfg.min_order_qty - order_qty)
    violation += float(stockout.sum())

    return total_cost + penalty_weight * violation
