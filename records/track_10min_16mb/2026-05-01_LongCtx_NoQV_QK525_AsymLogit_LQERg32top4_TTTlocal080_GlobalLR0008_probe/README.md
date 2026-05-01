# LongCtx No-QV QK5.25 + AsymLogit + LQER g32/top4 + TTT-local 0.80 + Prefix3500 GlobalTTT LR 0.0008

Non-record follow-up on PR #2060:

- Base recipe: `2026-05-01_LongCtx_NoQV_QK525_AsymLogit_LQERg32top4_TTTlocal080_1.0579`
- Change vs #2060: `PHASED_TTT_PREFIX_DOCS=3000 -> 3500`
- Change vs #2060: `GLOBAL_TTT_LR=0.001 -> 0.0008`
- The five #2060 tuning knobs are retained: `MATRIX_LR=0.028`, `LQER_RANK=2`, `LQER_ASYM_GROUP=32`, `LQER_TOP_K=4`, `TTT_LOCAL_LR_MULT=0.80`
- Dataset, tokenizer, architecture, quantizer, compressor, and TTT mask are unchanged.

## Result

Single seed-42 run on 8x H100:

| Metric | Value |
|---|---:|
| Pre-quant BPB | 1.06172454 |
| Quantized BPB | 1.06984890 |
| Quantized phased-TTT BPB | 1.05807364 |
| Train wallclock | 596.082 s |
| TTT eval time | 498.781 s |
| Total submission size | 15,977,802 B |

This is not a record claim. It is a small TTT follow-up experiment. Relative to
#2060 seed 42 (`1.05781454`), this setting is worse by `+0.00025910 BPB`, but
it is still useful as a documented negative/near-neutral result for the
longer-prefix + lower-global-LR direction.

## Why

The motivation was to test whether a larger phased-TTT prefix could benefit from
a lower full-parameter global TTT step. The run showed that the TTT gain relative
to the quantized model increased, but the trained/quantized base for this seed
started worse, so the final BPB did not beat #2060 seed 42.

## Reproduce

Download CaseOps data:

```bash
python3 download_caseops_data.py --local-dir /workspace/caseops_data
```

Run seed 42:

```bash
SEED=42 \
CASEOPS_ROOT=/workspace/caseops_data \
RUN_ID=prefix3500_global0008_seed42 \
./run_current_candidate.sh
```

The script launches:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Log

- `train_seed42_prefix3500_global0008.log` contains the complete seed-42 run.
