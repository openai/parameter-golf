# Non-record: 13L INT4 Attention — Failed Experiment

**val_bpb: 1.1640** (worse than best) | **15.14 MB** | 8×H100 SXM

## Hypothesis

By quantizing attention to INT4 (from INT6), we save enough space to add a 13th layer, gaining extra model capacity.

## Result

| Metric | v9_13l (this) | v7_ve seed 2 (prev best) |
|--------|---------------|--------------------------|
| Layers | **13** | 12 |
| Attn quant | **INT4** | INT6 |
| step_avg | **172.8ms** | 148ms |
| steps (10min) | **3485** | 4058 |
| post_quant bpb | 1.1696 | 1.1624 |
| ttt_bpb | **1.1640** | **1.1574** |
| artifact | 15,137,448 | 16,408,223 |

**Conclusion: Failed.** 172.8ms/step means only 3485 steps — 573 fewer than 12L VE. The extra layer does not compensate for both: (1) slower steps from INT4 recompilation overhead, and (2) INT4 attention quality degradation. net result is 0.0066 BPB worse.

## Lesson

Consistent with the community finding: 12L+ at seq2048 is worse than 11L because slower steps cancel extra capacity. INT4 attention adds overhead without sufficient quality benefit at this model size.

## Config

```bash
RUN_ID=v9_13l_int4attn_seed1 SEED=1 \
NUM_LAYERS=13 MLP_QUANT_BITS=4 ATTN_QUANT_BITS=4 XSA_LAST_N=4 \
EMA_ENABLED=1 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_FRAC=0.65 TTT_ENABLED=1 \
VALUE_EMBED_LAYERS=2 VALUE_EMBED_DIM=128
```
