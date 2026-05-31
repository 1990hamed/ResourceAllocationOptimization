# Open the report notebook

Build the report notebook (if not already built) and print the command to open it.

$ARGUMENTS

1. Check if a `.ipynb` file exists under `reports/`. If it does, print its path and the open command.
2. If no notebook exists, run:
   ```
   uv run main.py --step report
   ```
   then print the path and open command.

The open command is:
```
uv run jupyter notebook <path>
```

Remind the user that the notebook is read-only output — to regenerate it with fresh data, run `/pipeline optimize` first.
