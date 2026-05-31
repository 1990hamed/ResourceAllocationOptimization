"""30-run experiment harness for GWO vs PSO comparison."""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rao.algorithms.base import OptimizerBase
from rao.algorithms.gwo import GreyWolfOptimizer
from rao.algorithms.pso import ParticleSwarmOptimizer
from rao.config import MAX_ITER, N_RUNS, POP_SIZE, RANDOM_SEED, REPORTS_DIR
from rao.problem.formulation import HospitalAllocationProblem


@dataclass
class RunResult:
    algorithm: str
    run_id: int
    seed: int
    best_fitness: float
    best_x: np.ndarray
    history: list[float]
    runtime_seconds: float


def run_single(
    algorithm_cls: type[OptimizerBase],
    problem: HospitalAllocationProblem,
    run_id: int,
    seed: int,
    pop_size: int = POP_SIZE,
    max_iter: int = MAX_ITER,
) -> RunResult:
    """Execute one independent run of a given optimizer."""
    optimizer = algorithm_cls(problem, pop_size=pop_size, max_iter=max_iter, seed=seed)
    t0 = time.perf_counter()
    best_x, best_fitness, history = optimizer.optimize()
    runtime = time.perf_counter() - t0
    return RunResult(
        algorithm=algorithm_cls.__name__,
        run_id=run_id,
        seed=seed,
        best_fitness=best_fitness,
        best_x=best_x,
        history=history,
        runtime_seconds=runtime,
    )


def run_experiment(
    problem: HospitalAllocationProblem,
    n_runs: int = N_RUNS,
    pop_size: int = POP_SIZE,
    max_iter: int = MAX_ITER,
    base_seed: int = RANDOM_SEED,
) -> tuple[list[RunResult], list[RunResult]]:
    """
    Run GWO and PSO each n_runs times with seeds = [base_seed + i for i in range(n_runs)].
    Returns (gwo_results, pso_results).
    """
    seeds = [base_seed + i for i in range(n_runs)]
    gwo_results: list[RunResult] = []
    pso_results: list[RunResult] = []

    print(f"\n{'='*65}")
    print(f"EXPERIMENT: {n_runs} runs × 2 algorithms  |  pop={pop_size}  iter={max_iter}")
    print(f"{'='*65}")
    print(f"{'Run':>4} | {'GWO Fitness':>16} {'GWO Time':>10} | {'PSO Fitness':>16} {'PSO Time':>10}")
    print("-" * 65)

    for i, seed in enumerate(seeds):
        gwo_r = run_single(GreyWolfOptimizer, problem, i, seed, pop_size, max_iter)
        pso_r = run_single(ParticleSwarmOptimizer, problem, i, seed, pop_size, max_iter)
        gwo_results.append(gwo_r)
        pso_results.append(pso_r)
        print(
            f"{i+1:>4} | {gwo_r.best_fitness:>16.2f} {gwo_r.runtime_seconds:>9.2f}s"
            f" | {pso_r.best_fitness:>16.2f} {pso_r.runtime_seconds:>9.2f}s"
        )

    print("=" * 65)
    gwo_mean = np.mean([r.best_fitness for r in gwo_results])
    pso_mean = np.mean([r.best_fitness for r in pso_results])
    print(f"{'MEAN':>4} | {gwo_mean:>16.2f}            | {pso_mean:>16.2f}")
    print("=" * 65 + "\n")

    return gwo_results, pso_results


def results_to_dataframe(results: list[RunResult]) -> pd.DataFrame:
    """Flatten a list of RunResult into a tidy DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "algorithm": r.algorithm,
            "run_id": r.run_id,
            "seed": r.seed,
            "best_fitness": r.best_fitness,
            "runtime_seconds": r.runtime_seconds,
        })
    return pd.DataFrame(rows)


def save_results(
    gwo_results: list[RunResult],
    pso_results: list[RunResult],
    output_dir: Path | None = None,
) -> None:
    """Save results as pickle (preserves arrays) and a CSV summary."""
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Pickle for full data (including best_x and history arrays)
    with open(out / "experiment_results.pkl", "wb") as f:
        pickle.dump({"gwo": gwo_results, "pso": pso_results}, f)

    # CSV summary
    df = pd.concat([results_to_dataframe(gwo_results), results_to_dataframe(pso_results)])
    df.to_csv(out / "experiment_summary.csv", index=False)
    print(f"[runner] Results saved to {out}")


def load_results(output_dir: Path | None = None) -> tuple[list[RunResult], list[RunResult]]:
    """Reload saved results from pickle."""
    out = output_dir or REPORTS_DIR
    with open(out / "experiment_results.pkl", "rb") as f:
        data = pickle.load(f)
    return data["gwo"], data["pso"]
