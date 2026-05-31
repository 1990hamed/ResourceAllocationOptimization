"""Statistical comparison of GWO vs PSO experiment results."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rao.experiments.runner import RunResult


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def wilcoxon_test(
    gwo_fitnesses: list[float],
    pso_fitnesses: list[float],
) -> dict:
    """Paired Wilcoxon signed-rank test."""
    stat, p = stats.wilcoxon(gwo_fitnesses, pso_fitnesses, alternative="less")
    return {
        "test": "Wilcoxon signed-rank (one-sided: GWO < PSO)",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def mann_whitney_test(
    gwo_fitnesses: list[float],
    pso_fitnesses: list[float],
) -> dict:
    """Mann-Whitney U test (unpaired, robustness check)."""
    stat, p = stats.mannwhitneyu(gwo_fitnesses, pso_fitnesses, alternative="less")
    return {
        "test": "Mann-Whitney U (one-sided: GWO < PSO)",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def rank_biserial_correlation(
    gwo_fitnesses: list[float],
    pso_fitnesses: list[float],
) -> float:
    """Effect size r = 1 − 2U / (n1 * n2).  Range [−1, 1]; positive → GWO better."""
    n1, n2 = len(gwo_fitnesses), len(pso_fitnesses)
    mw = stats.mannwhitneyu(gwo_fitnesses, pso_fitnesses, alternative="less")
    r = 1.0 - 2.0 * mw.statistic / (n1 * n2)
    return round(float(r), 4)


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_comparison(
    gwo_histories: list[list[float]],
    pso_histories: list[list[float]],
) -> pd.DataFrame:
    """Per-iteration statistics across all runs."""
    gwo_arr = np.array(gwo_histories)
    pso_arr = np.array(pso_histories)
    n_iters = gwo_arr.shape[1]
    return pd.DataFrame({
        "iteration": np.arange(1, n_iters + 1),
        "gwo_mean": gwo_arr.mean(axis=0),
        "gwo_std": gwo_arr.std(axis=0),
        "gwo_min": gwo_arr.min(axis=0),
        "pso_mean": pso_arr.mean(axis=0),
        "pso_std": pso_arr.std(axis=0),
        "pso_min": pso_arr.min(axis=0),
    })


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(
    gwo_results: list[RunResult],
    pso_results: list[RunResult],
) -> pd.DataFrame:
    """Produce the full statistical comparison table."""
    gwo_fit = [r.best_fitness for r in gwo_results]
    pso_fit = [r.best_fitness for r in pso_results]
    gwo_rt = [r.runtime_seconds for r in gwo_results]
    pso_rt = [r.runtime_seconds for r in pso_results]

    wil = wilcoxon_test(gwo_fit, pso_fit)
    mwu = mann_whitney_test(gwo_fit, pso_fit)
    effect = rank_biserial_correlation(gwo_fit, pso_fit)

    rows = [
        ("Mean Fitness", f"{np.mean(gwo_fit):.4f}", f"{np.mean(pso_fit):.4f}"),
        ("Std Fitness", f"{np.std(gwo_fit):.4f}", f"{np.std(pso_fit):.4f}"),
        ("Min Fitness", f"{np.min(gwo_fit):.4f}", f"{np.min(pso_fit):.4f}"),
        ("Median Fitness", f"{np.median(gwo_fit):.4f}", f"{np.median(pso_fit):.4f}"),
        ("Mean Runtime (s)", f"{np.mean(gwo_rt):.3f}", f"{np.mean(pso_rt):.3f}"),
        ("Wilcoxon p-value", f"{wil['p_value']:.4e}", "—"),
        ("Wilcoxon significant", str(wil["significant"]), "—"),
        ("Mann-Whitney p-value", f"{mwu['p_value']:.4e}", "—"),
        ("Mann-Whitney significant", str(mwu["significant"]), "—"),
        ("Effect size r (rank-biserial)", f"{effect:.4f}", "—"),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "GWO", "PSO"])
    return df


def print_summary(gwo_results: list[RunResult], pso_results: list[RunResult]) -> None:
    df = summary_table(gwo_results, pso_results)
    print("\n" + "=" * 55)
    print("STATISTICAL COMPARISON: GWO vs PSO")
    print("=" * 55)
    print(df.to_string(index=False))
    print("=" * 55 + "\n")
