# Junkyard Rat Shroud Mini

This is a reduced-size copy lane for visualizing architecture-function flow with Shroud.

## Run

```bash
bash experiments/Junkyard_Rat_Shroud_Mini/run.sh
```

Outputs:

- Fresh run directory: `results/shroud_junkyard_mini_<timestamp>/`
- Stable latest trace: `results/shroud_junkyard_mini_latest.trace.jsonl`
- Stable latest points: `results/shroud_junkyard_mini_latest.trace.points.json`
- Architecture graph: `results/shroud_junkyard_mini_latest.architecture_flow.json`

Open viewer:

```bash
python3 -m http.server 8787
```

Then open:

`http://127.0.0.1:8787/experiments/Shroud/visualizer/shroud_viewer.html`

Preset `JUNKYARD MINI (latest)` is wired as default.
