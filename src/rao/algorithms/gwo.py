"""Grey Wolf Optimizer — Mirjalili et al. (2014)."""
from __future__ import annotations

import numpy as np

from rao.algorithms.base import OptimizerBase
from rao.problem.formulation import HospitalAllocationProblem


class GreyWolfOptimizer(OptimizerBase):
    """
    Grey Wolf Optimizer (GWO).

    Wolf hierarchy:
      alpha — best solution found so far
      beta  — second-best
      delta — third-best
      omega — all remaining wolves, updated each iteration

    Position update is the mean of three leader-guided movements.
    The coefficient 'a' decreases linearly from 2 → 0, balancing
    exploration (|A| > 1) and exploitation (|A| < 1).
    """

    def __init__(
        self,
        problem: HospitalAllocationProblem,
        pop_size: int = 30,
        max_iter: int = 200,
        seed: int = 42,
    ) -> None:
        super().__init__(problem, pop_size, max_iter, seed)

    def optimize(self) -> tuple[np.ndarray, float, list[float]]:
        dim = self.problem.dim
        lb, ub = self.problem.bounds

        # --- Initialise ---
        pop = self._initialize_population()
        fitness = self._evaluate_population(pop)

        # Sort and assign hierarchy
        idx = np.argsort(fitness)
        alpha_pos, alpha_fit = pop[idx[0]].copy(), fitness[idx[0]]
        beta_pos, beta_fit = pop[idx[1]].copy(), fitness[idx[1]]
        delta_pos, delta_fit = pop[idx[2]].copy(), fitness[idx[2]]

        self.history = []

        # --- Main loop ---
        for t in range(self.max_iter):
            a = 2.0 - 2.0 * t / self.max_iter  # linearly decreases 2 → 0

            for i in range(self.pop_size):
                x = pop[i]

                # Movement toward each leader
                new_x = np.zeros(dim)
                for leader_pos in (alpha_pos, beta_pos, delta_pos):
                    r1 = self.rng.random(dim)
                    r2 = self.rng.random(dim)
                    A = 2.0 * a * r1 - a          # A coefficient
                    C = 2.0 * r2                   # C coefficient
                    D = np.abs(C * leader_pos - x)  # distance to leader
                    new_x += leader_pos - A * D

                pop[i] = self._clip_to_bounds(new_x / 3.0)

            # Evaluate updated population
            fitness = self._evaluate_population(pop)

            # Update hierarchy
            for i in range(self.pop_size):
                f = fitness[i]
                if f < alpha_fit:
                    delta_pos, delta_fit = beta_pos.copy(), beta_fit
                    beta_pos, beta_fit = alpha_pos.copy(), alpha_fit
                    alpha_pos, alpha_fit = pop[i].copy(), f
                elif f < beta_fit:
                    delta_pos, delta_fit = beta_pos.copy(), beta_fit
                    beta_pos, beta_fit = pop[i].copy(), f
                elif f < delta_fit:
                    delta_pos, delta_fit = pop[i].copy(), f

            self.history.append(alpha_fit)

        self.best_x = alpha_pos.copy()
        self.best_fitness = alpha_fit
        return self.best_x, self.best_fitness, list(self.history)
