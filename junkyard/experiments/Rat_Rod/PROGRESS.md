# Rat Rod Green — Base Model Optimization Campaign

## Goal
Push base model BPB from 1.11 → 1.08 on 8xH100, 600s wallclock, no GPTQ.

## Results Table

| Run | Base BPB (sliding) | Post-EMA BPB | N-gram Legal BPB | Steps | ms/step | Config Changes |
|-----|-------------------|--------------|-------------------|-------|---------|----------------|
| Old SOTA (green_1) | 1.1195 | 1.1384 | 0.3200 (ILLEGAL oracle) | 6823 | 87.95 | XSA=4, Bigram=1536, RoPE=24, SWA=100, GPTQ outside wallclock |
| **Rat Rod Green v1** | **1.1129** | **1.1364** | **0.4489** | **6882** | **87.20** | Parallel Muon, XSA=11, Bigram=2048, RoPE=16, SWA=50, no GPTQ, no complementary |
| Rat Rod Green v2 | 1.1132 | 1.1367 | 0.4490 | 6875 | 87.28 | v1 + TRIGRAM=1, LATE_QAT_THRESHOLD=0 — **WASH** |
| Rat Rod Green v3 | PENDING | — | — | — | — | v1 base + MTP_NUM_HEADS=2 (vanilla MTP) |
| Rat Rod Green v4 "Synapse" | ~1.14+ | 1.1702 | — | 5872 | 102.20 | HS-MTP + CPU Bridge — **DEAD** (15ms overhead) |
| Rat Rod Green v5 "Synapse v2" | 1.1296 | 1.1529 | 0.4534 | 6819 | 88.01 | GPU-native hash bridge — **DEAD** (worse on both metrics) |
| Rat Rod Green v6 | PENDING | — | — | — | — | v1 + WARMDOWN_ITERS=2000 (not yet run at 600s) |
| Rat Rod Green v7 | 1.1169 | 1.1405 | 0.4500 | 6873 | 87.31 | v1 + WD=2000 + COMPLEMENT_ALPHA=0.5 — **WORSE** (+0.004 sliding vs v1) |

## v1 Full Log Metrics (2026-03-27)
```
model_params:26993756
train_batch_tokens:786432 train_seq_len:2048
max_wallclock_seconds:600.000
XSA:last_11 active_layers:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

step:0/20000   val_bpb:4.1049
step:4000      val_bpb:1.2114   train_time:348324ms
step:6882      val_bpb:1.1374   train_time:600115ms (wallclock cap)

post_ema        val_bpb:1.1364
sliding_window  val_bpb:1.1129   stride:64
ngram9          val_bpb:0.4489

peak memory: 22860 MiB
late_qat kicked in at step 6362 (BUG — no GPTQ to benefit, injected noise for 520 steps)
```

## v1 run.sh Config
```bash
SEED=1337
MAX_WALLCLOCK_SECONDS=600
COMPLEMENT_ALPHA=0
XSA_LAST_N=11
BIGRAM_VOCAB_SIZE=2048
ROPE_DIMS=16
SWA_EVERY=50
MTP_NUM_HEADS=0
NGRAM_EVAL_ORDER=9
NGRAM_EVAL_MIN_ORDER=2
NGRAM_EVAL_ADAPTIVE=1
NGRAM_EVAL_ALPHA=0.30
NGRAM_EVAL_ALPHA_MIN=0.05
NGRAM_EVAL_ALPHA_MAX=0.60
NGRAM_EVAL_ENTROPY_CENTER=3.0
NGRAM_EVAL_ENTROPY_SCALE=2.0
NGRAM_EVAL_MIN_COUNT=2
NGRAM_EVAL_BUCKETS=8388608
NGRAM_EVAL_MAX_SECONDS=0
CUBRIC_CADENCE=0
NGRAM_ENTROPY_SHIFT=1
NGRAM_ORDER_MULTS="0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0"
```

## v2 Changes from v1
- `TRIGRAM=1` — trigram hash reuses bigram embedding table, zero extra params
- `LATE_QAT_THRESHOLD=0` — kills quantization noise that was hurting last 520 steps

## Architecture (PR#609 Parallel Muon base)
- 11 layers (5 encoder + 6 decoder with skip connections)
- 512d, 8 heads, 4 KV heads (GQA)
- MLP mult 3.0 (1536 MLP dim), leaky_relu_sq activation (slope 0.5)
- Parameter banks: weights stored as contiguous 3D tensors for batched Muon optimizer
- Parallel Muon: reduce-scatter → local NS5 → all-gather (overlapped with Adam)
- BigramHash 2048 (128 dim → 512 projection)
- XSA (cross-sequence attention) on all 11 layers
- Partial RoPE (16 dims of 64 head_dim)
- SmearGate on input embeddings
- Value Embeddings on layers 9,10 (128 dim)
- Tied embeddings, logit softcap 30
- EMA decay 0.997, SWA every 50 steps (starts when LR scale < 0.2)
- torch.compile(fullgraph=True)

## What We Removed from PR#609
- Full GPTQ pipeline (~660 lines) — eats wallclock, not needed for base model testing
- INT8 quantization helpers (dead code)
- Complementary training (COMPLEMENT_ALPHA=0) — intentionally weakens base model

## "Synapse" — HS-MTP + CPU N-gram Bridge (v4)

A symbiotic system connecting the training loop and the n-gram eval pipeline.

**Architecture:**
```
GPU training loop                      CPU background thread
─────────────────                      ────────────────────
tokens (x) ──────────────────────────→ XOR hash → count tables
                                              │
model forward:                                │
  main CE loss (unchanged)                    │
  + HS-MTP loss ←── weight tensor ←───────────┘
    4 heads (512→2048 hash vocab)        (per-token: high where
    predict bigram hash patterns          n-grams are weak)
    at depths 1-4
```

**How it works:**
1. CPU thread builds n-gram frequency tables from training tokens (same XOR hash as BigramHash/n-gram eval)
2. Each step, CPU computes per-token "n-gram confidence" — how well n-grams will predict each token
3. Confidence is inverted to weight: easy tokens (n-gram confident) → 0.3x, hard tokens → 3.0x
4. HS-MTP heads are trained to predict n-gram hash patterns, weighted by difficulty
5. Model learns richer representations exactly where n-grams can't help

**Why it's novel:**
- No GPU cost for weighting (CPU integer ops, async queue, never blocks GPU)
- Training signal directly mirrors eval behavior (same hash space)
- Model and n-gram eval become symbiotic — model focuses on what n-grams can't do

**Config:** `HSMTP_NUM_HEADS=4 HSMTP_LOSS_WEIGHT=0.3`
**Dir:** `experiments/Rat_Rod/green_v4_hsmtp/`

## Tested Levers
1. ~~LATE_QAT_THRESHOLD=0~~ (v2) — **WASH**, EMA/SWA smoothed the noise anyway
2. ~~TRIGRAM=1~~ (v2) — **WASH**, shared table pulled in conflicting directions
3. ~~WARMDOWN_ITERS (2000/3500/5000)~~ (200s sweep, 2026-03-27) — **WINNER: 2000** (1.3504 vs 1.3775 vs 1.4111 cap BPB)
4. ~~SWA_EVERY 50 vs 100~~ (200s sweep, 2026-03-27) — **EDGE: 100** (1.3773 vs 1.3778 cap BPB)
5. ~~Synapse v1 (CPU bridge)~~ (v4) — **DEAD**, 15ms overhead, worse per-step
6. ~~Synapse v2 (GPU-native)~~ (v5) — **DEAD**, worse on both base AND ngram. Concept disproven.
7. ~~VALUE_RESIDUAL=1~~ — **WORSE** (+0.0012 sliding at 200s)
8. ~~Siphon (ensemble loss training)~~ — **DEAD**, sliding +0.151, ngram +0.017. Model can't learn under ensemble objective.
9. ~~COMPLEMENT_ALPHA=0.5~~ (v7, 600s) — **WORSE**, sliding +0.004, ngram +0.001 vs v1. Bigram weighting doesn't help our n-gram system.

## Untested Levers
1. GATED_ATTENTION=1 — learned per-head attention gating
2. MTP_NUM_HEADS=2 — vanilla MTP (moderate overhead expected)
3. ROPE_DIMS 16 vs 24 — old SOTA used 24, needs clean modern A/B in current harness

## A/B Tests (200s wallclock, directional signal only)

| Test | val_bpb @cap | Post-EMA | Sliding BPB | N-gram BPB | Steps | Variable |
|------|-------------|----------|-------------|------------|-------|----------|
| ROPE_DIMS=16 (control) | 1.2053 | 1.2079 | 1.1847 | 0.4702 | 2292 | baseline |
| ROPE_DIMS=24 (test) | PENDING | — | — | — | — | old SOTA value |
| WARMDOWN_ITERS=2000 | 1.3504 | 1.3979 | skipped | skipped | 1036 | 200s/seed1337 |
| WARMDOWN_ITERS=3500 | 1.3775 | 1.4344 | skipped | skipped | 1034 | 200s/seed1337 |
| WARMDOWN_ITERS=5000 | 1.4111 | 1.4764 | skipped | skipped | 1032 | 200s/seed1337 |
| SWA_EVERY=50 | 1.3778 | 1.4354 | skipped | skipped | 1032 | 200s/seed1337 |
| SWA_EVERY=100 | 1.3773 | 1.4335 | skipped | skipped | 1037 | 200s/seed1337 |
| Siphon OFF (WD=2000) | 1.1970 | 1.1996 | 1.1760 | 0.4674 | 2299 | control |
| Siphon ON α=0.50 (WD=2000) | 1.3459 | 1.3538 | 1.3269 | 0.4841 | 2284 | **DEAD** +0.151 sliding |

## Checkpoints
- `checkpoints/final_model_ratrod_green_v1_1.1129.pt` — saved on pod
- `experiments/Rat_Rod/green_v1_1.1129/` — frozen v1 experiment copy on pod

## Legality Status
- No oracle alpha (legal)
- No GPTQ (no wallclock concern)
- No forward-looking anything
- N-gram eval uses entropy-adaptive alpha with per-order shift + fixed mults (B-WING system)
- All training inside 600s wallclock
