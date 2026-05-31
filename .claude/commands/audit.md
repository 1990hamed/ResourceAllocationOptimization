# Run data quality audit

Run the data audit step and surface any issues found.

$ARGUMENTS

Execute:
```
uv run main.py --step audit
```

After running, highlight:
- Any missing values detected
- Any columns with unexpected types or out-of-range values
- Whether raw data files in `data/raw/` are all present and readable
- A pass/fail summary

If no raw data is found, tell the user to place datasets in `data/raw/` before proceeding.
