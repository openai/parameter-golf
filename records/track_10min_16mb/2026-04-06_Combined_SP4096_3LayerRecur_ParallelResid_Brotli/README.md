# Record: Combined 3-Layer Depth Recurrence + Parallel Residuals + Polar Express + Brotli — val_bpb 1.1067 (3-seed mean)

## Results

| Seed | Sliding BPB | Artifact |
|------|------------|---------|
| 1337 | 1.1080 | 13,866,319 |
| 42 | 1.1055 | 13,871,505 |
| 2025 | 1.1067 | 13,861,924 |
| **Mean** | **1.1067** | **13,866,583** |

Beats merged SOTA (PR #1019, 1.1147) by 0.008 BPB. All artifacts under 16MB (13.87MB — 2.13MB headroom). No TTT, no SLOT, no n-gram cache. Clean neural record.

## How This Was Built

I'm a documentary filmmaker, not an ML engineer. I built this submission using Claude Opus 4.6 (Anthropic) as a co-author throughout the entire process — from understanding the codebase to identifying which techniques to combine to merging the code.

My process: I systematically analyzed all 50+ open PRs in the competition, identified that PR #1344 (3-layer recurrence + Polar Express) and PR #1392 (parallel residuals + Brotli) each had techniques the other didn't, and that nobody had tested the full combination. Claude helped me read and merge the two codebases. The strategic decisions — what to combine, what to leave out, when to pivot — were mine. The code comprehension and implementation were AI-assisted.

This is my first ML submission of any kind. Everything I know about transformers, quantization, and depth recurrence I learned in the past two weeks by reading PRs in this repo.

## Novel Contribution

First submission combining 3-layer depth recurrence (from @omrigotlieb's #1344) with parallel residuals (from @dexhunter's #1392). Neither PR tested this combination. Additionally identifies 2.13MB of unused artifact headroom as a future optimization opportunity.

## Source PRs

- **PR #1392** (2026-04-05, score 1.1020 BPB): SP4096, 2-layer depth recurrence, parallel residuals, Brotli compression, QK-Gain 5.0
- **PR #1344** (2026-04-03, score 1.0923 BPB): SP4096, 3-layer depth recurrence (layers 3,4,5), Polar Express Newton-Schulz (4 steps), MuonEq-R, WD=0.105

## What was combined

| Technique | Source | Detail |
|-----------|--------|--------|
| SP4096 vocabulary | Both | `VOCAB_SIZE=4096` |
| 3-layer depth recurrence | PR #1344 | `RECUR_LAYERS=3,4,5` (layers replayed after max layer) |
| Parallel residuals | PR #1392 | `PARALLEL_START_LAYER=7` (attn+mlp computed in parallel from layer 7) |
| Brotli compression | PR #1392 | `COMPRESSOR=brotli` with byte-shuffle preprocessing |
| Polar Express Newton-Schulz | PR #1344 | 5 optimized coefficient pairs, 4 steps per iteration |
| MuonEq-R | PR #1344 | Row-norm equalization before NS (`MUON_EQ_R=1`) |
| Weight decay 0.105 | PR #1344 | `MUON_WD=0.105`, `EMBED_WD=0.105` |
| QK-Gain 5.0 | PR #1392 | `QK_GAIN_INIT=5.0` |
| Full Hessian GPTQ int6 | Both | `GPTQ_ENABLED=1` with calibration data |
| No TTT | Clean | All TTT code removed for compliance |

## Architecture

- Base: #1344 codebase (cleaner, non-banked weights, standard CastedLinear layers)
- 11 layers, 512 model_dim, 8 heads, 4 KV heads
- U-Net skip connections with skip gates
- Value embeddings on layers 9,10
- XSA on all 11 layers
- RoPE dims=16, ln_scale=True
- EMA decay=0.997, warmup=20 steps

## Key hyperparameters changed from base #1344

```
QK_GAIN_INIT:   4.0 -> 5.0    (from #1392)
MUON_WD:        0.090 -> 0.105 (from #1344's best WD finding)
EMBED_WD:       0.090 -> 0.105
RECUR_LAYERS:   '' -> '3,4,5'  (3-layer recurrence)
PARALLEL_START_LAYER: N/A -> 7 (parallel residuals from #1392)
MUON_EQ_R:      N/A -> 1       (from #1392)
```

## What was NOT included

- TTT (test-time training) -- removed entirely for compliance
- Weight banking (qo_bank, kv_bank) from #1392 -- kept #1344's standard per-layer weights
- Bigram/trigram hash embeddings from #1392
- SmearGate from #1392
- QAT, MTP heads, n-gram scoring from #1392
- SWA/LAWA averaging from #1392

## Reproduction

```bash
VOCAB_SIZE=1024 QK_GAIN_INIT=5.0 RECUR_LAYERS="3,4,5" \
PARALLEL_START_LAYER=7 MUON_WD=0.105 MUON_EQ_R=1 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

The techniques in this submission belong to the people who invented them. I combined their work.

- **@omrigotlieb** (PR #1344) — 3-layer depth recurrence, Polar Express Newton-Schulz, MuonEq-R, WD=0.105 tuning. The architecture backbone.
- **@dexhunter** (PR #1392) — Parallel residuals, Brotli + byte-shuffle compression, QK-Gain 5.0, MLP 4x. The compression and architecture innovations.
- **@abaybektursun** (PR #1019) — Merged SOTA base stack that everything builds on. XSA-all, GPTQ, BigramHash, EMA.
- All upstream contributors credited in #1344 and #1392 — this work stands on many shoulders.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
