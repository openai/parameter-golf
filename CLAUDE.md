# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Context

**OpenAI Parameter Golf** (March 18 – April 30, 2026): Train the best language model that fits in **16MB decimal** (code + compressed weights), trainable in **≤10 minutes on 8×H100s**, evaluated by bits-per-byte (bpb) on the FineWeb validation set — lower is better.

- Current leaderboard SOTA: **1.2244 bpb** (Naive Baseline, 9L×512D)
- Artifact = `len(train_gpt.py bytes)` + `len(compressed_weights bytes)` — everything must be in one file
- `train_gpt.py` hard limit: **1500 lines** (enforced by docstring at line 1)
- Submission PR goes to `/records/track_10min_16mb/` with README, `submission.json`, train log, and `train_gpt.py`
- SOTA claims require p < 0.01 that the run beats previous SOTA by ≥ 0.005 nats

## Training Commands

### Local single-GPU (WSL):
```bash
# Quick smoke test (fast, no val)
RUN_ID=smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  /usr/bin/python3 -u train_gpt.py

# Standard local test (300 steps, all features)
bash launch_test_960d.sh   # or launch_sota_512d.sh

# Monitor running experiments
monitor.bat   # (Windows) — calls monitor.sh via WSL
bash monitor.sh  # (WSL directly)
```

### Multi-GPU H100 submission:
```bash
bash launch_h100_submission.sh   # 8×H100, 10L×960D, 590s wallclock
# Or manually:
torchrun --standalone --nproc_per_node=8 -m train_gpt
```

### Dataset download:
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 2
# Full: --train-shards 10 (80 shards for H100 runs)
```

All hyperparameters are set via environment variables — see `class Hyperparameters` in `train_gpt.py` for the full list with defaults.

## Architecture (`GPT` class in `train_gpt.py`)

The model is a pure-attention transformer (no recurrence, no Mamba). Key components:

**Forward pass order:**
1. `tok_emb` → token embedding (tied to lm_head when `TIE_EMBEDDINGS=1`)
2. `BigramHash` → additive (prev_token, curr_token) hash embedding, projected to model_dim
3. `SmearGate` → per-dim learned gate blending current and previous token embeddings
4. Input `rms_norm`
5. Stack of `AttnBlock` layers (each: RMSNorm → GQA attention with RoPE + QK-norm → residual → optional SwiGLU FFN)
6. Final `RMSNorm` → logits with `logit_softcap` (tanh-based)
7. Loss via `cggr_loss` (hard-token selection after warmup)

**Key design choices proven to work:**
- All-attention wins over hybrid attention+Mamba, pure Mamba, or DeltaNet variants
- GQA with 8 Q-heads / 4 KV-heads
- `CastedLinear`: weights stored fp32, cast to bf16 at matmul time
- `RESET_ON_BOS=1`: resets attention mask at BOS tokens (document boundaries)
- `TIE_EMBEDDINGS=1`: ties token embedding and lm_head (saves params, needs `TIED_EMBED_LR`)
- `ATTN_FFN_EXPAND=3.0` with SwiGLU beats no-FFN for equivalent artifact size

**Known-bad approaches (don't revisit):** BitLinear (negative result), pure Mamba, DeltaNet.

## Optimizer Split

Three separate optimizers with different LR schedules:
- **`optimizer_tok`** (Adam): `tok_emb.weight` at `TIED_EMBED_LR`, `bigram_hash.emb.weight` at `EMBED_LR`
- **`optimizer_muon`** (Muon): all 2D matrix params in blocks + `bigram_hash.proj.weight` at `MATRIX_LR`
- **`optimizer_scalar`** (Adam): 1D params in blocks + `smear_gate.gate` at `SCALAR_LR`
- **`optimizer_head`** (Adam, only if `TIE_EMBEDDINGS=0`): `lm_head.weight` at `HEAD_LR`

**Critical**: Any new learnable module must be explicitly assigned to an optimizer group — it won't be picked up automatically.

## Quantization + Serialization Pipeline

Export uses mixed precision quantization then zstd-22 compression:
- **int8** (per-row scale): `tok_emb`, `bigram_hash.emb`, `qkv`, `out`/`out_proj`, `lm_head` — configured via `MIXED_INT8_NAME_PATTERNS`
- **int4** (per-row scale, packed 2/byte): all other 2D matrices (MLP weights)
- **fp16 passthrough**: tensors with ≤ 65536 elements (`INT8_KEEP_FLOAT_MAX_NUMEL`)
- **QAT** (`QAT_START_FRACTION`): fake-quantizes int4 weights during training with STE; attention layers have `skip_qat=True`
- **SWA** (`SWA_START_FRACTION`): averages weights over the last fraction of training; swapped in before export

Artifact budget: ~0.126–0.141 bytes/param after zstd-22. Larger models compress better per-param.

## Logging

Each run produces two log files in `logs/`:
- `logs/{RUN_ID}.txt` — stdout: step metrics, val_bpb, final roundtrip results
- `logs/{RUN_ID}_meta.txt` — nvidia-smi output + full source code + all the same metrics

Key log lines to grep: `^step:`, `^val_bpb:`, `^qat_active`, `^swa:`, `^final_mixed_zstd_roundtrip_exact`, `^Serialized model mixed`, `^Total submission size mixed`

## `attention_playground.py`

Optional experimental features, activated by environment variables:
- `INIT_CKPT`: load checkpoint with `strict=False` (for continuation training)
- `ATTN_TIED_LAYERS`: layer tying / unrolled depth recurrence
- `MEMORY_TOKENS`: learned memory K/V slots appended to each attention layer
- `EMA_DECAY`: EMA shadow weights that swap in at export instead of SWA

All are no-ops unless the corresponding env var is set.

## H100 Submission Config (Current Best)

`launch_h100_submission.sh` — 10L×960D, SwiGLU×3, BigramHash(10240×128), SmearGate, CGGR, SWA, QAT:
- 113M params, estimated ~14–16MB artifact (verify before submitting)
- `TRAIN_BATCH_TOKENS=786432` = 8 GPUs × 1 grad_accum × 2048 seq_len × 48 seqs/GPU
- `MAX_WALLCLOCK_SECONDS=590` (10-second buffer before the 10-minute hard cap)
- Wallclock countdown starts after warmup and step-0 val eval; torch.compile time not counted

## Known Bugs (All Fixed)

1. **QAT double-quantization**: `_QAT_ACTIVE` check must include `self.training` — fixed in `CastedLinear.forward`
2. **Logfile double-write**: stdout goes to `{RUN_ID}.txt`; meta log uses `{RUN_ID}_meta.txt`
3. **Optimizer coverage**: `bigram_hash.emb`, `bigram_hash.proj`, `smear_gate.gate` must be explicitly added to optimizer groups
4. **QAT on int8 layers**: attention `qkv` and `out` have `skip_qat=True` — they're int8 at export, not int4
5. **BigramHash zero-gradient deadlock**: `emb.weight` initialized with `N(0, 0.01²)` (not zeros); `proj.weight` stays at zeros so initial contribution is zero but gradients flow immediately
