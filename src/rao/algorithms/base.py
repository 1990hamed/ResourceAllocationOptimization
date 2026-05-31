"""Abstract base class for all optimizers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rao.problem.formulation import HospitalAllocationProblem
from rao.problem.objective import compute_fitness


class OptimizerBase(ABC):
    def __init__(
        self,
        problem: HospitalAllocationProblem,
        pop_size: int = 30,
        max_iter: int = 200,
        seed: int = 42,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.history: list[float] = []
        self.best_x: np.ndarray | None = None
        self.best_fitness: float = float("inf")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _initialize_population(self) -> np.ndarray:
        """Uniform random positions within bounds, shape (pop_size, dim)."""
        lb, ub = self.problem.bounds
        return self.rng.uniform(lb, ub, size=(self.pop_size, self.problem.dim))

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        lb, ub = self.problem.bounds
        return np.clip(x, lb, ub)

    def _evaluate_population(self, pop: np.ndarray) -> np.ndarray:
        """Return fitness array of shape (pop_size,)."""
        return np.array([compute_fitness(pop[i], self.problem) for i in range(len(pop))])

    def reset(self, seed: int) -> None:
        """Reset internal state for a new independent run."""
        self.rng = np.random.default_rng(seed)
        self.history = []
        self.best_x = None
        self.best_fitness = float("inf")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def optimize(self) -> tuple[np.ndarray, float, list[float]]:
        """
        Returns:
            best_x      — solution vector of shape (dim,)
            best_fitness — scalar fitness value
            history     — list of best fitness per iteration (length = max_iter)
        """
