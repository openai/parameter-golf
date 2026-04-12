# MLX Experiment Notes

These are local Apple Silicon experiment notes from the Codex exploration branch.
They are not leaderboard-quality results, because they used capped validation via
`VAL_MAX_BATCHES` for faster iteration.

## Useful Local Knob

- `VAL_MAX_BATCHES`
  - Local-only helper for faster smoke tests.
  - `0` means full validation.
  - Small values like `16` or `64` are useful for quick comparisons on a laptop.

## Best Local Direction So Far

The strongest local result we found was:

```bash
RUN_ID=mlx_mlp1_layers11_ab300 \
ITERATIONS=300 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
VAL_MAX_BATCHES=64 \
MLP_MULT=1 \
NUM_LAYERS=11 \
.venv/bin/python train_gpt_mlx.py
```

Observed result:

- `serialized_model_int8_zlib: 10418730 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 2.30746143`

## Key Comparisons

Best comparison set so far used: `TRAIN_BATCH_TOKENS=8192`, `VAL_BATCH_SIZE=524288`, `VAL_MAX_BATCHES=64`.

| Config | Compressed Bytes | Roundtrip `val_bpb` | Notes |
|---|---:|---:|---|
| Baseline (`MLP_MULT=2`, `NUM_LAYERS=9`, `ITERATIONS=200`) | 11258346 | 2.44989649 | Strong quality baseline |
| Compact MLP (`MLP_MULT=1`, `NUM_LAYERS=9`, `ITERATIONS=200`) | 8303583 | 2.47262920 | Much smaller, but worse |
| Deeper compact (`MLP_MULT=1`, `NUM_LAYERS=11`, `ITERATIONS=200`) | 9984744 | 2.44481887 | Beat baseline while staying smaller |
| Deeper compact (`MLP_MULT=1`, `NUM_LAYERS=12`, `ITERATIONS=200`) | 10736033 | 2.46008228 | Worse than 11 layers |
| Deeper compact (`MLP_MULT=1`, `NUM_LAYERS=11`, `NUM_KV_HEADS=2`, `ITERATIONS=200`) | 10759634 | 2.45219887 | KV-head reduction hurt |
| Deeper compact (`MLP_MULT=1`, `NUM_LAYERS=11`, `ITERATIONS=300`) | 10418730 | 2.30746143 | Current best local result |

## Dead Ends We Tested

- Embedding factorization (`EMBED_RANK`)
  - Helped in tiny smoke tests.
  - Lost to baseline in longer runs.

- Post-training magnitude pruning
  - Barely changed compressed size.
  - Slightly hurt `val_bpb`.

## Recommended Next Step

Use `MLP_MULT=1` and `NUM_LAYERS=11` as the local working baseline and scale training from there.
