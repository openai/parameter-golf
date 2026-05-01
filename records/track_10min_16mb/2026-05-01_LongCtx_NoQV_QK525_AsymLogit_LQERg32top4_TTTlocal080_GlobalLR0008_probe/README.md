# LongCtx No-QV QK5.25 + AsymLogit + LQER g32/top4 + TTT-local 0.80 + GlobalTTT LR 0.0008

Probe candidate based on PR #2060:

- Base recipe: `2026-05-01_LongCtx_NoQV_QK525_AsymLogit_LQERg32top4_TTTlocal080_1.0579`
- Only intended change vs #2060: `GLOBAL_TTT_LR=0.0008` instead of the train_gpt.py default `0.001`
- `PHASED_TTT_PREFIX_DOCS` stays at `3000`
- `TTT_LOCAL_LR_MULT` stays at `0.80`
- Dataset, tokenizer, training, quantizer, compressor, and TTT mask are unchanged

## Why

The earlier `prefix2750` experiment on the #1953-style stack spent more eval time but did not improve BPB, suggesting larger phased-TTT prefixes can saturate or over-adapt if the full-model global TTT step is too strong.

This probe first tests a smaller global full-parameter TTT step at the same #2060 prefix length. If it does not hurt seed 42, it becomes a safer base for later `prefix3500 + lower GLOBAL_TTT_LR` testing.

## Decision Rule

Run seed 42 first and compare to #2060 seed 42:

- #2060 seed 42: `val_bpb=1.05781454`, TTT eval `397.125s`
- Continue only if this candidate is at least neutral or better, ideally `<=1.0578`
- Stop if it worsens by more than about `0.0003 BPB`

## Reproduce

Download the prebuilt CaseOps data once:

```bash
python3 download_caseops_data.py --local-dir /workspace/caseops_data
```

For a quick path/layout smoke test, download only one shard each:

```bash
python3 download_caseops_data.py --local-dir /tmp/caseops_smoke --train-shards 1 --val-shards 1
```

Run seed 42:

```bash
SEED=42 \
CASEOPS_ROOT=/workspace/caseops_data \
RUN_ID=global0008_seed42 \
./run_current_candidate.sh
```

The script launches:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

This is not a record claim yet. It is a controlled single-knob probe to decide whether lower global TTT LR should be combined with larger phased prefixes.
