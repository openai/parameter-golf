# MetaStack Runbook

## Submission Phases

| Phase | Delta from v0 | Purpose |
|-------|--------------|---------|
| v0 | None (exact #60 clone) | Control. Reproduce merged SOTA 1.1748 BPB. |
| A | Int6-range export + selective fp16 (tok_emb) | Measure artifact size improvement. Training unchanged. |
| B | MLP 3x (MLP_MULT=3) | Measure step time + artifact size with wider MLP. |
| C | Late QAT (QAT_START_FRAC=0.75, QAT_LR_DROP=0.5) | Measure quant gap closure. Training changes. |

## Exact Diffs Between Phases

**v0 → A**: Export path only. `QUANT_BITS`, `COMPRESSOR`, `LOWBIT_KEEP_FLOAT_NAME_PATTERNS` env vars added. `quantize_state_dict_lowbit` replaces `quantize_state_dict_int8`. Dynamic artifact naming.

**A → B**: One hyperparameter: `MLP_MULT` default 2 → 3 (line 71).

**B → C**: QAT added to `CastedLinear.forward` behind `_qat` flag. `QAT_START_FRAC` and `QAT_LR_DROP` control activation timing. Default: off (QAT_ENABLED=0).

## H100 Experiment Matrix

Run each phase once on 8xH100 to build the evidence ladder. Record: step time, total steps, final val_bpb (standard + sliding window), artifact size.

### Tuned Env Vars (apply to all runs)

Based on merged SOTA + frontier PRs (2026-03-20). Key correction: **batch 524K, not 786K** (PR #236 showed 524K wins by -0.017 BPB due to more gradient updates in fixed time).

Note: our env var names are `ADAMW_WEIGHT_DECAY` and `MUON_WEIGHT_DECAY` (not `ADAM_WD`/`MUON_WD`).

```bash
TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=524288
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
GRAD_CLIP_NORM=0.3
ADAMW_WEIGHT_DECAY=0.04
MUON_WEIGHT_DECAY=0.04
```

### Run 1: Phase B (MLP 3x + tuned hyperparams + int6)

Skip v0/A — the #60 SOTA is already proven at 1.1748. Go straight to Phase B.

```bash
RUN_ID=metastack_phaseB \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
GRAD_CLIP_NORM=0.3 \
ADAMW_WEIGHT_DECAY=0.04 \
MUON_WEIGHT_DECAY=0.04 \
QUANT_BITS=6 \
COMPRESSOR=zlib \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_MetaStack/train_gpt.py
```

**Key questions**: step time with MLP 3x + seq2048? Artifact size with int6? BPB?
Expected: ~1.15-1.16 BPB, ~65ms/step, artifact TBD.

### Run 2: Phase D (EMA) — after Phase B, before QAT

EMA replaces SWA as the weight averaging technique (PR #287 frontier at 1.1271 uses EMA=0.997). Only run after EMA is scaffolded in the code.

```bash
RUN_ID=metastack_phaseD \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
GRAD_CLIP_NORM=0.3 \
ADAMW_WEIGHT_DECAY=0.04 \
MUON_WEIGHT_DECAY=0.04 \
QUANT_BITS=6 \
COMPRESSOR=zlib \
EVAL_STRIDE=64 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_MetaStack/train_gpt.py
```

Expected: ~0.005 BPB improvement over Phase B from smoother weight distribution + lower quant penalty.

### Run 3: Phase C (Late QAT) — deferred, only if EMA underdelivers

QAT and SWA/EMA may work at cross purposes. Only test QAT if EMA doesn't close the quant gap sufficiently.

```bash
RUN_ID=metastack_phaseC \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
GRAD_CLIP_NORM=0.3 \
ADAMW_WEIGHT_DECAY=0.04 \
MUON_WEIGHT_DECAY=0.04 \
QUANT_BITS=6 \
COMPRESSOR=zlib \
EVAL_STRIDE=64 \
QAT_ENABLED=1 \
QAT_START_FRAC=0.75 \
QAT_LR_DROP=0.5 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_MetaStack/train_gpt.py
```

Expected: similar BPB to Phase B but smaller quant gap. Skip if EMA already closes the gap.

## Metrics to Record

For each run, extract from the log:

| Metric | Log pattern |
|--------|------------|
| Step time | `step_avg:XXms` (from last training log line) |
| Total steps | `step:X/20000` (from stopping_early or last step line) |
| Pre-quant val_loss | `step:X/X val_loss:X.XXXX` (last validation before export) |
| Pre-quant val_bpb | `step:X/X ... val_bpb:X.XXXX` |
| Post-quant val_loss | `final_intX_zlib_roundtrip val_loss:X.XXXX` |
| Post-quant val_bpb | `final_intX_zlib_roundtrip ... val_bpb:X.XXXX` |
| Sliding window val_bpb | `sliding_window ... val_bpb:X.XXXX` (if EVAL_STRIDE>0) |
| Artifact size | `Serialized model intX+zlib: XXXX bytes` |
| Code size | `Code size: XXXX bytes` |
| Total submission size | `Total submission size intX+zlib: XXXX bytes` |
| Peak memory | `peak memory allocated: XXXX MiB` |

## Quick 1xH100 Timing Probe

If you want step time without a full run:

```bash
RUN_ID=timing_probe \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=50 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-19_MetaStack/train_gpt.py
```

Multiply 1xH100 step time by ~0.85 to estimate 8xH100 step time (DDP overhead is small).

## What's Next After Phase B

If Phase B succeeds (fits in 16MB, step time < 80ms):

1. **EMA** (Phase D, decay=0.997) — ~10 lines, ~0.005 BPB. PR #287 frontier.
2. **SmearGate + BigramHash(10240)** — ~100 lines, ~0.005 BPB. In all top-3 merged.
3. **Orthogonal init** (gain=1.0, output 1/sqrt(2L)) — ~15 lines, ~0.001 BPB.
4. **FA3 with SDPA fallback** — ~15 lines, +600 steps.
5. **Nibble separation** in serialization — ~20 lines, 0.3-1.0 MB savings.
6. **Late QAT** (Phase C) — only if EMA doesn't close quant gap.

Each is an independent delta. Add one at a time, measure after each.

See `docs/research-synthesis.md` for the full prioritized findings from the research sprint.
