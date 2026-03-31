# TTT Sweep Record (2026-03-31, seed 444)

Run context:
- Base checkpoint: `/workspace/parameter-golf-lab/final_model.pt`
- Sweep command: `MODEL_PATH=/workspace/parameter-golf-lab/final_model.pt bash experiments/Rascal_Stripper/ttt_sweep.sh`
- Evaluator baseline (from sweep): `final_sliding_window_exact val_bpb=1.11055027`

## Results

| Arm | Config | TTT BPB | Delta vs baseline | Verdict |
|---|---|---:|---:|---|
| A_conservative | `lr=1e-4, epochs=1, freeze_blocks=2, chunk=65536` | 1.11134960 | +0.00079933 | WORSE |
| B_balanced | `lr=1e-4, epochs=2, freeze_blocks=2, chunk=32768` | 1.11149799 | +0.00094772 | WORSE |
| C_aggressive | `lr=5e-4, epochs=3, freeze_blocks=2, chunk=32768` | 1.11163602 | +0.00108575 | WORSE |

## Decision

TTT is a regression on this checkpoint and should be treated as **bust** for this line.

- Best arm (`A_conservative`) is still worse than baseline by `+0.00079933` BPB.
- For this run family, prefer no TTT post-processing.

## Notes

- Baseline reference run excerpt (same checkpoint family):
  - `final_int6_roundtrip_exact val_bpb: 1.14464324`
  - `final_sliding_window_exact val_bpb: 1.11052831`
- Sweep logs are produced under `experiments/Rascal_Stripper/ttt_sweep_logs/`.
