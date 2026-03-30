# Shroud Visualizer

This folder provides a copy-safe visualization lane for Bandit activity traces.

## Files

- `build_shroud_points.py`: converts Shroud JSONL trace events into compact point-cloud JSON.
- `shroud_viewer.html`: Three.js particle viewer with a dark theme, complementary gradients, flow projection, and a loop-depth interaction map overlay.
- Viewer modes include `Flow`, `Loop Map`, `Morph` (step-growth metamorphosis on Y), and `Overlay`.

## Quick start

1. Run the copied Shroud build:
   ```bash
   bash experiments/Shroud/run.sh
   ```
2. Convert trace manually (if needed):
   ```bash
   python3 experiments/Shroud/visualizer/build_shroud_points.py \
     --input logs/shroud_trace_*.jsonl \
     --output logs/shroud_trace.points.json
   ```
3. Open the viewer over a local server:
   ```bash
   cd experiments/Shroud/visualizer
   python3 -m http.server 8787
   ```
4. Visit `http://localhost:8787/shroud_viewer.html` and load `*.points.json` or raw `*.jsonl`.

The `*.points.json` payload now includes compact crawler math fields (`stage`, `loop`, `block`, `head`, `kv_head`,
`qk_align`, `rms`, `std`, `amax`, `q_rms`, `k_rms`, `v_rms`, `step`, `micro_step`, `energy`) which drive animated
particle motion in the viewer. Existing files without these fields still work via label/geometry fallback.

Head-edge interactions are rendered as moving packet flow on top of Q↔KV links. For richer interaction dynamics,
fresh traces can include attention transfer fields (`transfer`, `attn_entropy`, `attn_lag`, `recent_mass`,
`attn_peak`, `out_rms`, `token_count`) from `head_event` instrumentation.

## DGX Spark micro-run

For a small architecture-preserving loop/head interaction run on Spark:

```bash
bash Nitrust/scripts/spark_shroud_loopviz_smoke.sh
```

Outputs are written under `results/shroud_loopviz_smoke_<timestamp>/`:

- `*.trace.jsonl` raw trace (activation + head interaction + compression events)
- `*.trace.points.json` viewer-ready points/edges payload
