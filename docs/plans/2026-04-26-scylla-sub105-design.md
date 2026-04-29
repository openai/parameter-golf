# Scylla Sub-1.05 Design

## Objective

Reach a defensible sub-`1.05` `val_bpb` submission by reproducing the measured
Scylla frontier before adding any new ideas. Local March records top out at
`1.12278022`; the SP8192 frontier in PR #1797 reaches `1.06157`; the only
measured sub-`1.05` lane found in the current repo/PR landscape is PR #1813 at
`0.94166052` over three seeds.

## Decision

Use the PR #1813 Scylla lane as the primary path. Keep the current SP1024/SP8192
scripts intact and create an isolated Scylla reproduction lane with explicit
provenance, asset checks, artifact checks, and launcher scripts. Do not mutate
`train_gpt_kl.py` until the Scylla lane has reproduced the reference behavior.

## Reference Configuration

The target record is:

- Record path: `records/track_10min_16mb/2026-04-25_Scylla_QK525_DepthRecurrence_Experiment`
- Score: `0.94166052` 3-seed mean, std `0.00066536`
- Seeds: `1337`, `42`, `2025`
- Worst artifact: `15,868,157` bytes, leaving only `131,843` bytes under the
  decimal `16,000,000` byte cap.
- Architecture: 11 physical layers, 512 dim, 8 query heads, 4 KV heads, Scylla
  vocab size `998`, tied embeddings, train seq len `2048`, XSA on all layers.
- Core knobs: `QK_GAIN_INIT=5.25`, `NUM_LOOPS=2`, `LOOP_START=3`,
  `LOOP_END=5`, `ENABLE_LOOPING_AT=0.35`, `BIGRAM_VOCAB_SIZE=2816`,
  `BIGRAM_DIM=40`, `USE_GPTQ=1`, `GPTQ_RESERVE_MS=9000`, `TTT_ENABLED=0`.
- Compression: full GPTQ int6, `torch.save` quant payload, `lzma.compress`
  preset 6.

## Why This Lane

Compression-only work cannot close the gap from `1.1228` to sub-`1.05`. The
useful compression findings are mostly negative: byte shuffle makes real
artifacts larger, FP16 last-layer escape hatches exceed the cap, and INT4 hurts
quality too much. PR #1813 crosses the target by changing the tokenizer/data
regime and architecture schedule while still fitting under the size cap.

## Architecture

Create a separate lane with four responsibilities:

1. **Provenance capture**: copy the PR #1813 `train_gpt.py`, logs, and
   `submission.json` into `frontier_sources/scylla_pr1813/` for local review.
2. **Asset validation**: add a script that verifies the Scylla tokenizer and
   dataset assets exist before any paid GPU launch.
3. **Run launch**: add a shell launcher that runs the exact reference config
   for one or all canonical seeds.
4. **Artifact validation**: add a checker that computes code bytes + model bytes
   against `16,000,000` and fails below a configurable safety margin.

## Data Flow

The launcher passes explicit env vars into the copied Scylla `train_gpt.py`.
Training reads only training shards during optimization and GPTQ calibration.
Validation remains disabled during training via `VAL_LOSS_EVERY=0`; scoring
runs only after the wallclock stop. The artifact checker runs after training and
validates the generated compressed model plus script size.

## Compliance Guardrails

- Keep `TTT_ENABLED=0` for the first reproduction.
- Reject cache/PPM/SLOT/ETLB-style additions in the Scylla preflight.
- Keep Scylla recurrence + GPTQ allowed only for the exact proven loop schedule.
- Use decimal bytes, not MiB.
- Treat the `131,843` byte PR #1813 margin as fragile; any code or serializer
  growth must be offset by measured artifact savings.

## Testing Strategy

Local tests should run without GPU and without the Scylla dataset:

- Python compile checks for new scripts.
- Asset-check tests using temporary fake files.
- Artifact-check tests using temporary fake model/code files.
- Launcher dry-run or env-render check that confirms exact PR #1813 defaults.

GPU validation is staged:

1. Run one seed exactly, no ablations.
2. Compare steps, artifact bytes, and final BPB to the PR #1813 logs.
3. Run all three seeds only after the one-seed reproduction is within expected
   tolerance.
4. Only then test narrow ablations: `BIGRAM_DIM=36/40/44`,
   `QK_GAIN_INIT=5.0/5.25`, loop `3-5` vs `4-5`, and LZMA preset `6/9`.

## Rejected Approaches

- **Compression-only local cleanup**: useful for hygiene, but not enough BPB.
- **SP8192 as primary**: lower compliance risk, but the measured frontier is
  around `1.06157`, still above target.
- **PPM/cache lane**: potentially strong but compliance-risky; keep it as a
  separate research lane, not the primary submission path.
