"""Particle Swarm Optimization — Kennedy & Eberhart (1995)."""
from __future__ import annotations

import numpy as np

from rao.algorithms.base import OptimizerBase
from rao.problem.formulation import HospitalAllocationProblem


class ParticleSwarmOptimizer(OptimizerBase):
    """
    Standard PSO with linear inertia weight decay.

    Parameters
    ----------
    w_max, w_min : inertia weight bounds (linearly decayed per iteration)
    c1           : cognitive coefficient (personal best attraction)
    c2           : social coefficient   (global best attraction)
    """

    def __init__(
        self,
        problem: HospitalAllocationProblem,
        pop_size: int = 30,
        max_iter: int = 200,
        seed: int = 42,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
    ) -> None:
        super().__init__(problem, pop_size, max_iter, seed)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def optimize(self) -> tuple[np.ndarray, float, list[float]]:
        lb, ub = self.problem.bounds
        v_max = 0.2 * (ub - lb)

        # --- Initialise positions and velocities ---
        pos = self._initialize_population()
        vel = np.zeros_like(pos)

        fitness = self._evaluate_population(pos)

        pbest_pos = pos.copy()
        pbest_fit = fitness.copy()

        g_idx = int(np.argmin(fitness))
        gbest_pos = pos[g_idx].copy()
        gbest_fit = fitness[g_idx]

        self.history = []

        # --- Main loop ---
        for t in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * t / self.max_iter

            r1 = self.rng.random(pos.shape)
            r2 = self.rng.random(pos.shape)

            vel = (
                w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            vel = np.clip(vel, -v_max, v_max)
            pos = self._clip_to_bounds(pos + vel)

            fitness = self._evaluate_population(pos)

            # Update personal bests
            improved = fitness < pbest_fit
            pbest_pos[improved] = pos[improved].copy()
            pbest_fit[improved] = fitness[improved]

            # Update global best
            g_idx = int(np.argmin(pbest_fit))
            if pbest_fit[g_idx] < gbest_fit:
                gbest_pos = pbest_pos[g_idx].copy()
                gbest_fit = pbest_fit[g_idx]

            self.history.append(gbest_fit)

        self.best_x = gbest_pos.copy()
        self.best_fitness = gbest_fit
        return self.best_x, self.best_fitness, list(self.history)
