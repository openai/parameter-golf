First SSM-based entry in either track. Trained on 4×H200 SXM for one hour rather than 8×H100 for ten minutes — non-record on both hardware and time. Eval is the standard root-harness full-window val + int8/brotli quant; no TTT, no sliding-window, no GPTQ.

**val_bpb 1.30040 / 12.07 MB / seed=1337**

## Architecture

**Topology.** 7 distinct transformer-style blocks, each shared across 3 sequential applications via depth recurrence (`NUM_UNIQUE_LAYERS=7 NUM_LOOPS=3`). Effective compute depth is 21; total stored body parameters are equivalent to 7 layers. No U-Net skip; the loop is plain weight reuse. `MODEL_DIM=512`, tied input/output embeddings, sp1024 vocab.

**Block contents.** Each block is `(attn || SSM) + MLP`, not the standard `attn → MLP` chain:

1. RMSNorm → run attention and the SSM in parallel on the same normalized input → sum their outputs (with independent per-channel learned scales `attn_scale`, `s4d_scale`) into the residual stream.
2. RMSNorm → SwiGLU MLP (`MLP_MULT=8`, hidden = 8·dim) → add to the residual stream with its own per-channel scale (`mlp_scale`).
3. A learned 2-vector `resid_mix` per block interpolates the incoming residual between the live stream `x` and the original post-embedding `x0` before normalization. Cheap, lets each block decide how much it cares about deep context vs the original embedding.

**Attention branch.** Standard causal multi-head attention with grouped-query (`NUM_HEADS=8`, `NUM_KV_HEADS=4`) and RoPE positional encoding.

**SSM branch — kill-Mamba-2.** Standard Mamba-2 has an `in_proj` that produces `x` plus three input-dependent quantities `dt`, `B`, `C` ("selectivity" — the ability to modulate the recurrence per-token), runs a depthwise causal `conv1d` (kernel=4) on `x`, then an SSD chunkwise selective scan (`d_state=64`, `expand=2`, `chunk_size=64`, `headdim=64`, 16 SSD heads), then `out_proj`. **kill-Mamba-2 replaces `dt`, `B`, `C` with learned per-head/per-state constants** (`_B_const`, `_C_const`) and a per-head `dt_bias`, making the recurrence linear time-invariant (LTI) instead of input-dependent. Same conv1d, same gating, same `A_log`, same `D_skip`, same in/out projections — only the dynamics become LTI. The intuition: at sub-records training scale, the input-dependent projections are under-trained and add noise; the LTI variant keeps the structural advantages of Mamba-2 (conv1d local recall, gated SSD scan) without the gradient surface area of selectivity.

**Why parallel attention || SSM.** The two mixers have complementary recall: attention does exact content-addressable lookup over the context, conv1d-equipped SSM does structured local recall. Running them on the same normalized input and summing outputs is consistently better at this scale than alternating attention-only and SSM-only blocks (cross-class hybrid finding from earlier experiments).

**No BigramHash.** BigramHash recall (`BIGRAM_VOCAB_SIZE=0` here) helps S4D-Lin family blocks but interacts negatively with the conv1d already present in Mamba-2 — it ends up adding noise to the recall niche conv1d already occupies.

**Quantization (`TERNARY_BODY=1`).** Body weights of the attention `qkv`/`out` projections, Mamba-2 `in_proj`/`out_proj`, and SwiGLU MLP gates are constrained to `{−γ, 0, +γ}` ternary via BitNet-b1.58 absmean straight-through estimation: at every forward pass each weight matrix is quantized to ternary (with a per-row scale γ = mean absolute value) before the matmul; gradients pass through unchanged. Roughly 1.58 bits per parameter of effective resolution. At quant export, ternary weights are stored as 2 bits per parameter packed (4 vals per byte) in a custom format that bypasses int8 entirely — lossless ternary→ternary round-trip. 1D and small (≤65,536-element) tensors — the SSM dynamics buffers `A_log`, `B_proj`, `C_proj`, `dt_bias`, `D_skip`, `conv1d` weights, all RMSNorm scales, all per-channel scales — stay fp32 throughout via `CONTROL_TENSOR_NAME_PATTERNS`.

**EMA-of-weights (`EMA_BETA=0.999`).** A shadow copy of the model weights is updated each step as `shadow = β·shadow + (1−β)·model`, then swapped into the model immediately before final eval. β=0.999 gives an effective averaging window of ~1000 steps — about the last 23% of this run's 4,380 steps. EMA is reliable when `(1−β)·steps ≫ 1`; for shorter runs (<3,000 steps) β=0.99 is the right choice.

**Optimizer.** Muon (Newton-Schulz orthogonalization, `MUON_BACKEND_STEPS=15`) for 2D weight matrices; AdamW for low-dim parameters and embeddings. `MATRIX_LR=0.045` is the modded-nanogpt baseline value. The optimizer split is by tensor pattern, same as the records-track convention.

**Compression.** int8 quantization of fp32 buffers + 2-bit packed ternary for body weights → brotli q=11 → final artifact. brotli/zlib ratio on this bytestream is ~0.985; brotli is the standard records-track choice.

## Configuration summary

- Track: `non-record`, 16 MB cap
- `NUM_UNIQUE_LAYERS=7 NUM_LOOPS=3` (effective depth 21)
- `MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`, tied embeddings, sp1024
- `PARALLEL_LAYER_POSITIONS=0,1,2,3,4,5,6 PARALLEL_SSM_TYPE=mamba2_kill MAMBA2_KILL_SELECTIVITY=1`
- `BIGRAM_VOCAB_SIZE=0` (off)
- `TERNARY_BODY=1` (BitNet-b1.58, exported 2-bit packed)
- `EMA_BETA=0.999` (shadow swapped at last step)
- Schedule: `WARMDOWN_ITERS=1800 LR_WARMUP_STEPS=30 MATRIX_LR=0.045 TIED_EMBED_INIT_STD=0.05 MUON_BACKEND_STEPS=15 TRAIN_BATCH_TOKENS=524288`

## Command

Run from the **repo root** (so `./data/datasets/fineweb10B_sp1024` and `./data/tokenizers/fineweb_1024_bpe.model` resolve — same convention as the other records-folder submissions). The path to `train_gpt.py` is given relative to repo root:

```bash
SEED=1337 \
NUM_UNIQUE_LAYERS=7 NUM_LOOPS=3 \
PARALLEL_LAYER_POSITIONS=0,1,2,3,4,5,6 \
PARALLEL_SSM_TYPE=mamba2_kill MAMBA2_KILL_SELECTIVITY=1 \
BIGRAM_VOCAB_SIZE=0 \
TERNARY_BODY=1 \
EMA_BETA=0.999 \
TRAIN_BATCH_TOKENS=524288 ITERATIONS=20000 \
WARMDOWN_ITERS=1800 LR_WARMUP_STEPS=30 MATRIX_LR=0.045 \
TIED_EMBED_INIT_STD=0.05 MUON_BACKEND_STEPS=15 \
MAX_WALLCLOCK_SECONDS=3600 VAL_TOKENS=0 \
torchrun --standalone --nproc_per_node=4 \
  records/track_non_record_16mb/2026-04-30_KillMamba2_TriParallel_n7_Ternary_EMA_4xH200_1hr/train_gpt.py
```

`MAX_WALLCLOCK_SECONDS=3600` is the binding cap; `ITERATIONS=20000` is just an upper bound. The full env.sh in this folder lists every variable; `CONTROL_TENSOR_NAME_PATTERNS` is load-bearing for SSMs.

## Key metrics

- Pre-quant val_bpb: `1.2983`
- Post-quant val_bpb: `1.30040229`
- Quant tax: `0.0021`
- Wallclock: `3,600s` (cap fired)
- Step time: `821.94 ms`
- Steps: `4,380`
- Tokens trained: `4,380 × 524,288 ≈ 2.30B`
- Artifact (int8 + ternary-packed + brotli): `12,074,422 bytes`
- Code size: `104,676 bytes`
- Model parameters: `61,657,752`
- Hardware: 4×H200 SXM (141GB HBM3e per GPU), `--nproc 4`, grad_accum=2 (peak GPU memory at run: 114,125 MiB allocated)

## Comparison

- vs `track_10min_16mb` records baseline `2026-03-31_ParallelResiduals_MiniDepthRecurrence` (1.1063): +0.194 BPB
- vs `track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L...` (1.1239, 8×H100, 2.15h, 8192 BPE): +0.177 BPB
- vs `track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (1.2074, 8×H100, 4h): +0.093 BPB

## Files

- `train_gpt.py` — code snapshot
- `env.sh` — full env (every variable used by the run)
- `train_seed1337.log` — training log (partial: pod was stopped before the full `run.log` synced from `/workspace`; lines below were captured via the monitor stream during training and cover the headline numbers — pre/post-quant val_bpb, artifact bytes, step times, peak memory, EMA shadow swap)
- `result.json`, `submission.json` — leaderboard metadata
- `requirements.txt` — `brotli` and `sentencepiece` are required at quant-export
