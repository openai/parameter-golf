# Research Hub

Local scientific dashboard for the Parameter Golf repo.

## Files

- `index.html` - single-page dashboard shell.
- `styles.css` - layout and visual system.
- `app.js` - client-side filtering, sorting, and details panel logic.
- `scripts/build_index.py` - read-only crawler/indexer that scans `experiments`, `results`, and `logs`.
- `data/hub_index.json` - generated data consumed by the dashboard.

## Rebuild

From the repository root:

```bash
python3 experiments/GreenRod_X_1/lab_protocol_20260327/research_hub_20260327_131801/scripts/build_index.py
```

You can also target a custom output file:

```bash
python3 experiments/GreenRod_X_1/lab_protocol_20260327/research_hub_20260327_131801/scripts/build_index.py \
  --out experiments/GreenRod_X_1/lab_protocol_20260327/research_hub_20260327_131801/data/hub_index.json
```

## View

Serve the hub folder locally so `fetch("./data/hub_index.json")` works:

```bash
cd experiments/GreenRod_X_1/lab_protocol_20260327/research_hub_20260327_131801
python3 -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## Notes

- The indexer only reads files; it does not modify source experiments.
- Status classification is heuristic: errors are marked from tracebacks/failures, warn captures proxy or promotion notes, and ok means metric-bearing non-error records.
- Large logs are summarized into short snippets, not fully mirrored into the page.
