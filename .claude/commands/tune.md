# Tune hyperparameters

Inspect and update hyperparameters in `src/rao/config.py`.

$ARGUMENTS

If arguments are provided, parse them as key=value pairs (e.g. `POP_SIZE=50 MAX_ITER=300 N_RUNS=20`)
and update the corresponding constants in `src/rao/config.py`.

If no arguments are provided, read `src/rao/config.py` and print the current values of:
- `RANDOM_SEED`
- `N_RUNS`
- `MAX_ITER`
- `POP_SIZE`
- `PENALTY_WEIGHT`
- `OVERTIME_MULTIPLIER`

After any change, remind the user to re-run `/pipeline optimize` for results to reflect the new settings.
