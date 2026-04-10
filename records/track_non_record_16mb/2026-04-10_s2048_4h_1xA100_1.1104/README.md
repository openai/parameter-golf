# 11L s2048 4h on 1xA100 — 1.1104 BPB (non-record)

**Author:** Huanyi Xie (`xiehuanyi`)
**Date:** 2026-04-10
**Track:** `non_record_16mb`
**Result:** **val_bpb = 1.11044406** (int6 GPTQ + LZMA + sliding window eval stride=64)

## TL;DR

Drop-in longer-context variant of the existing `1.1147 ValCalib_GPTQ_XSA_BigramHash3072`-style stack: **11 layers, 3x MLP, LeakyReLU(0.5)^2, XSA-all, BigramHash(2048), Partial RoPE(16/64), LN Scale, SmearGate, U-Net skip, Muon+AdamW(WD=0.04), EMA(0.997), SWA, Late QAT@0.15, Int6 GPTQ with self-gen autoregressive calibration, LZMA preset=9, sliding window eval s64.**

The only changes vs. a classic ~1.13 BPB s1024 stack are:
1. `TRAIN_SEQ_LEN=2048` (longer training and eval context)
2. 4 hours of training on 1x A100 (~240 A100-min ≈ 76-80 equivalent H100-min, close to the official 80 H100-min compute budget)

## Why non-record

This submission does **not** qualify for `track_10min_16mb` because it was trained on **1x A100 for 4 hours**, not on **8x H100 for 10 min**. A100 ≈ H100 / 3.17 raw BF16 throughput (excluding FA3). The total compute is roughly comparable (~76–80 H100-min-equivalent vs. 80 allowed) but the submission was never verified on actual 8xH100 hardware, so it belongs in the non-record track.

FA3 is unavailable on Ampere; the attention forward uses PyTorch SDP (flash backend) as a drop-in via a small wrapper.

## Numbers (seed 1337)

| Metric | Value |
|---|---|
| **Int6 Sliding Window BPB** | **1.11044406** |
| Int6 Roundtrip BPB | 1.13437381 |
| Pre-quant val_bpb (post-EMA) | 1.1323 |
| Training steps | 14065 / 20000 |
| Step avg | 1023.86 ms |
| Peak memory | 16.3 GiB |
| Model params | 26,993,756 |
| Artifact bytes (int6+lzma) | 15,941,100 |
| **Total (code + artifact)** | **16,040,603** (under 16 MiB = 16,777,216) |

## Ablation context

This result was the top performer in a 24-experiment ablation run on 1x A100 with identical infrastructure. Summary of the biggest levers:

| Change | BPB (lower = better) | Notes |
|---|---|---|
| seq_len=512, full stack, 2h (exp07 Round 2) | 1.1484 | old baseline |
| seq_len=1024, full stack, 2h (exp13 Round 2) | 1.1317 | **+context alone = -0.017** |
| seq_len=1024, full stack, 4h (exp30 Round 3) | 1.1177 | **+time alone = -0.014** |
| **seq_len=2048, full stack, 4h (exp34, this)** | **1.1104** | **+context+time = -0.021 vs exp13** |

Trick ablations that did NOT help noticeably once training time was sufficient:
- Gated Attention (neutral)
- Value Residual (neutral)
- V-Norm on V projection (neutral)
- BigramHash 3072×112 vs 2048×128 (noise)
- warmdown=4000 vs 3500 (marginal)
- 13 layers (step overhead > depth gain at this budget)
- Attention Residuals (Kimi) — strongly negative (-0.036)
- Differential Attention — negative (-0.014)
- Layer Tying (22L/11B, 11L/6B) — strongly negative
- MTP 2 heads — slight negative

The key finding is that **at this model size and compute budget, simply extending context length and training longer dominates all micro-architectural tricks.**

## Reproduction

```bash
# On 1x A100 80GB:
RUN_ID=v3_exp34_s2048 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=14400 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
WARMDOWN_ITERS=3500 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=11 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
LATE_QAT_THRESHOLD=0.15 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Files

- `train_gpt.py` — modified training script (based on `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` with A100/FA2/SDP fallback, deferred EMA start for short runs, optional layer tying / V-norm / diff attn / AttnRes / MTP toggles, none of which were active for this run)
- `final_model.int6.ptz` — 15.94 MB int6-quantized model (LZMA preset=9)
- `train_seed1337.log` — full training log
- `submission.json` — structured metadata
- `requirements.txt` — pip deps

## Caveats

- **Single seed (1337) only.** A proper 3-seed mean (42, 314, 999) has NOT been run yet. This makes the reported BPB noisier than the main-leaderboard records. Seeds 42 and 999 are planned.
- EMA is deferred to start at 20% of wallclock to avoid random-init contamination on shorter runs (discovered during Round 2 experiments).
- The attention backend falls back to PyTorch SDP (flash backend) because FA3 is Hopper-only; FA2 is not installed in the current venv.
