# Show experiment results summary

Load saved experiment results and print a comparison summary of GWO vs PSO.

$ARGUMENTS

Run the following to load and print the summary table and convergence comparison:

```python
from rao.experiments.runner import load_results
from rao.experiments.stats import print_summary, summary_table, convergence_comparison

gwo_results, pso_results = load_results()
print_summary(gwo_results, pso_results)

s_df = summary_table(gwo_results, pso_results)
print(s_df.to_string())

gwo_histories = [r.history for r in gwo_results]
pso_histories = [r.history for r in pso_results]
c_df = convergence_comparison(gwo_histories, pso_histories)
print(c_df.to_string())
```

Use `uv run python -c "..."` to execute it.

If results files don't exist yet, tell the user to run `/pipeline optimize` first.
List the figures that exist under `reports/figures/` as well.
