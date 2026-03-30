# Research Hub // Darklab 02

Dark, analysis-first Research Hub focused on extracting actionable signal fast.

## What this version adds

- Front-page **Personal SOTAs by Category** (cards, compact table, and chart).
- Top **Hypothesis** block (current thesis, support, contradiction, next test).
- **Ablation Insights** cards + table with deltas and verdicts.
- **Chart suite** (status distribution, top ablation deltas, timeline) via local Apache ECharts.
- Detail-pane **Log Writeup** that highlights critical metric numbers and risk keywords.

## Files

- `index.html` - darklab dashboard shell.
- `styles.css` - dark theme visual system and responsive layout.
- `app.js` - full UI wiring (filters, records, SOTAs, hypothesis, ablations, charts, writeups).
- `scripts/build_index.py` - enhanced indexer producing `hypothesis`, `personal_sotas`, `ablations`, and `charts`.
- `data/hub_index.json` - generated index payload.
- `vendor/echarts.min.js` - vendored Apache ECharts runtime (no CDN needed).
- `scripts/playwright_smoke.js` - Playwright smoke flow used for screenshots.
- `artifacts/playwright/*.png` - screenshot artifacts.

## Rebuild data

From repo root:

```bash
python3 experiments/GreenRod_X_1/lab_protocol_20260327/research_hub_20260327_darklab_02/scripts/build_index.py
```

## Run locally

```bash
cd experiments/GreenRod_X_1/lab_protocol_20260327/research_hub_20260327_darklab_02
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000/`.

## Playwright validation used

```bash
cd experiments/GreenRod_X_1/lab_protocol_20260327/research_hub_20260327_darklab_02
npx playwright install chromium
node scripts/playwright_smoke.js
```

## Notes

- Static and read-only; this UI does not modify source runs/logs.
- Classification and summaries are heuristic because source logs are heterogeneous.
