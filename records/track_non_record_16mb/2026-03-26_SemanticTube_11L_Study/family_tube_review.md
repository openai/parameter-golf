# Semantic Tube Family Study

- Family: `semantic_tube`
- Source JSONL: `family_tube.jsonl`
- Log snapshots: `logs/`
- Runs: `11`
- Current best: `seq2048_tube_5e-4` at `1.2177 val_bpb`

## Fixed Baseline

11L/512d fixed backbone with EMA, XSA4, SmearGate, NTK-RoPE, and FA3 strict. The core family sweep uses `seq1024`; the extension uses matched controls at `seq1536` and `seq2048`.

## Results

| ID | Variation | val_bpb | Steps | Time (s) | Artifact Est (MB) | Drift | Curvature | Isotropy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | control_metrics | 1.2595 | 1692 | 600.0 | 27.53 | 0.7361 | 20.5762 | 0.0135 |
| 2 | tube_5e-5 | 1.2559 | 1676 | 600.1 | 27.53 | 0.7684 | 3.6776 | 0.0137 |
| 3 | tube_1e-4 | 1.2558 | 1680 | 600.2 | 27.53 | 0.7762 | 2.3371 | 0.0130 |
| 4 | tube_2e-4 | 1.2553 | 1671 | 599.9 | 27.53 | 0.7770 | 1.5189 | 0.0143 |
| 5 | tube_5e-4 | 1.2549 | 1668 | 600.0 | 27.53 | 0.7947 | 0.7371 | 0.0148 |
| 6 | tube_1e-3 | 1.2555 | 1673 | 600.2 | 27.53 | 0.7843 | 0.4293 | 0.0133 |
| 7 | tube_2e-3 | 1.2560 | 1679 | 600.2 | 27.53 | 0.7999 | 0.2321 | 0.0148 |
| 8 | seq1536_control | 1.2316 | 2189 | 600.2 | 27.53 | 0.7362 | 37.7799 | 0.0107 |
| 9 | seq1536_tube_5e-4 | 1.2293 | 2166 | 600.1 | 27.53 | 0.7947 | 0.7371 | 0.0148 |
| 10 | seq2048_control | 1.2207 | 2406 | 600.0 | 27.53 | 0.8514 | 33.2481 | 0.0076 |
| 11 | seq2048_tube_5e-4 | 1.2177 | 2380 | 600.1 | 27.53 | 0.8800 | 0.6183 | 0.0094 |

## Notes

- This report is generated from the study JSONL and the corresponding logs.
- `Artifact Est (MB)` is a proxy field from the `SKIP_QUANT=1` study harness, not a final compressed artifact size.
