# Run pipeline step

Run a single pipeline step or the full pipeline end-to-end.

Usage:
- `/pipeline audit` — data quality checks
- `/pipeline preprocess` — clean and transform raw data
- `/pipeline eda` — exploratory analysis + feature engineering + EDA plots
- `/pipeline optimize` — run GWO vs PSO experiment, save results, generate plots
- `/pipeline report` — build Jupyter report notebook from saved results
- `/pipeline all` — run every step in sequence

$ARGUMENTS

Run the requested pipeline step using:
```
uv run main.py --step <step>
```
or for `all`:
```
uv run main.py --all
```

Show the command output and summarize what was produced (files written, figures saved, key metrics printed).
