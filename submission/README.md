# 12L SwiGLU-3.75x + NorMuon + Full GPTQ + Batched PerDoc Q/K/V/Proj-LoRA 30-Pass

**val_bpb: TBD** (awaiting 8xH100 SXM run — RunPod $25 credits available)
**PR:** openai/parameter-golf#844
**Branch:** `submission/preyam2002` on `preyam2002/parameter-golf`

## Run Command

```bash
# On 8xH100 SXM pod (spot recommended ~$8/run)
git clone https://github.com/preyam2002/parameter-golf.git
cd parameter-golf && git checkout submission/preyam2002
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

RUN_ID=submission_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 submission/train_gpt.py
```

BPB prints at end in `final_int8_zlib_roundtrip` lines. Seeds: 42, 1337, 7.

## Architecture

- 12 layers, dim=512, 8 heads, 4 KV heads (GQA), seq=1024
- **SwiGLU MLP** 3.75x (hidden=1280): Gated linear unit with SiLU activation
- **XSA (Exclusive Self Attention)**: All 12 layers (arXiv:2603.09078)
- **CANON-AC + DeltaGate**: Last 10 layers — depthwise causal conv1d (kernel=3) pre-attention and pre-MLP, sigmoid gate init -4.0
- **Value Residual (ResFormer)**: Layer-0 V mixed into subsequent layers (arXiv:2410.17897)
- **Gated Attention**: Per-head sigmoid gate after SDPA (arXiv:2505.06708)
- **Partial RoPE**: First 16 of 64 head dims rotated
- **LN Scale Factor**: Block contributions scaled by 1/sqrt(layer_idx+1)
- **Stochastic Depth**: DropPath 0→0.1 across layers (training only)
- **SmearGate + BigramHash(16384, dim=128) + TrigramHash(8192, dim=48)**
- **U-Net skip connections**: 6 encoder, 6 decoder layers
- Orthogonal init, tied embeddings, logit softcap 30.0

## Training

- **NorMuon** (arXiv:2510.05491): Per-neuron normalization in Muon, second-moment EMA beta2=0.95
- **Polar Express Muon** (arXiv:2505.16932): matrix_lr=0.025, WD=0.04, momentum warmup 0.92→0.99, 5-step Newton-Schulz
- AdamW for embeddings (lr=0.035) and scalars (lr=0.025), WD=0.04
- LR warmup 200 steps, cosine warmdown 2100 iterations
- Batch=786,432 tokens, seq_len=2048, grad_clip=0.3, Z-loss 1e-4
- EMA 0.997 (fp32 CPU, every 5 steps) + exponential SWA (alpha=0.85)
- **Soft-Round QAT**: Sigmoid soft-round, temperature anneal 1→16, threshold=0.40

## Post-Training Pipeline

1. **GradQuant**: 8-seq probe, rank by gradient L2 norm — top 10%→int8, middle 65%→int6, bottom 25%→int5
2. **Magnitude pruning**: 5% smallest weights zeroed
3. **Full GPTQ**: Hessian-aware Cholesky error compensation, actorder, 256-seq calibration, 7+6 percentile search, progressive damping (0/0.01/0.1/1.0)
4. **zstd-22 compression** (~15.7 MB artifact)

## Eval Pipeline (sessions 14-15)

**No global TTT** — all eval time allocated to per-document LoRA (PR #596/#611 style).

1. **Baseline** (1-temp fast inference, ~2s)
2. **Batched Per-Doc Q/K/V/Proj LoRA TTT** (~580s, 30 all-batched passes)
3. **Sliding window final** (stride=48, shuffled, 15s reserve, 25-temp)
4. **min(NLL) per token** across all temperatures and passes

### Per-Doc LoRA Details

**Targets:** Q, K, V, attn output proj (`.attn.proj`) + LM-head (rank-16)
**Ranks:** Q/K/V/Proj = rank-8 (rank_overrides per-module)
**Per-module LR scaling:** Q 0.5x · K 0.3x · V 1.5x · Proj 0.2x · LM-head 2x · bias 3x
**alpha = rank** (scale=1.0, matching PR #611 — no extra scaling)
**Optimizer:** Adam (eps=1e-10), no gradient clipping, cosine LR decays to 0 (no floor), no label smoothing

**25-temperature diversity:** T from 0.46 to 1.50, offsets ±0.52, take min(NLL) per token
**Progressive PonderTTT:** skip easiest docs by fraction — 25% (early passes) → 40% → 50% → 0% (last 3 passes)
**baddbmm fused LoRA forward:** eliminates intermediate allocation
**Chunk diversity:** passes cycle through 256/512/1024/2048/4096 token chunks
**Front-loaded time:** pass-0 gets 3.0x budget, passes 1-7 get 1.5x, rest 1.0x
**LR cycling:** 5 LR candidates on pass-0, 3 candidates on passes 1-7, per-batch per-epoch

### 30-Pass Schedule (abbreviated)
- Passes 0-7: Phase 1 (proven config, lr=0.010, 8ep, chunk=256/512, 5/3 LR candidates)
- Passes 8-13: Phase 2 (near-proven variants, rank-8, diverse chunks)
- Passes 14-23: Phase 3 (rank-8/4 diversity, various chunks)
- Passes 24-29: Phase 4 (late proven, rank-1 FLoRA style)

## Key Innovations vs PR #611

1. **Attn output proj as 4th LoRA target** (Proj, 0.2x LR) — learns document-specific attention head mixing
2. **25-temperature diversity** (T=0.46 to 1.50) vs PR #611's flat T=0.98 rescaling
3. **Progressive PonderTTT** — allocates more passes to hard documents, 0% skip on final passes
4. **30 all-batched passes** with chunk diversity vs PR #611's simpler schedule

## PR #611 Differences We've Matched

- Adam (not AdamW) ✓
- alpha=rank (scale=1.0, no extra scaling) ✓
- No gradient clipping in TTT ✓
- Cosine LR decays to 0, no floor ✓
- No label smoothing ✓

## Ablation Knobs

```bash
TTT_ENABLED=0              # Global TTT disabled (default — all time to per-doc LoRA)
SKIP_MULTIPASS=1           # Skip multipass streaming eval (default)
SKIP_ONLINE_LORA=1         # Skip online LoRA streaming (default)
PERDOC_MAX_PASSES=N        # Limit per-doc LoRA passes
TTT_MAX_SECONDS=N          # Override eval time budget
TTT_DISABLE_XSA=1          # Disable XSA during TTT
```
