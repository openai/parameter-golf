# AGENTS.md

## Surprise Alerts

- `train_gpt.py` writes the full source code into `logs/<RUN_ID>.txt` before the runtime metrics. Any parser that scans those logs must anchor on concrete metric lines and ignore template strings in the source dump, or it will mis-parse placeholders like `{q_val_loss:.8f}` as real values.
