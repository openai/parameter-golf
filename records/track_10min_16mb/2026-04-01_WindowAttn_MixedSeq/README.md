# Record: Fused MLP (Triton+CUTLASS EVT) + MLP 3.5× + Mixed int5/int6 + Brotli

**val_bpb: 1.1125** (3-seed mean) | **1.8784 nats** | **~14.52 MB** | 8×H100 SXM, 600s | No TTT

Continuation of our merged PR 1019 (current SOTA, 1.1138 BPB). Fused MLP kernels recover throughput; mechanistic analysis identified MLP as the capacity bottleneck, leading to MLP 3.5× enabled by Hessian-based mixed int5/int6 quantization.

Our merged PR 1019 (current SOTA): **1.88059 nats** (1.1138 BPB). Delta: **−0.00215 nats** (−0.0013 BPB).
Prior leaderboard SOTA (our PR 549): **1.89002 nats** (1.1194 BPB). Delta: **−0.01158 nats** (−0.0069 BPB). Welch's t = −17.63, df ≈ 3.24, p < 0.01.

## Results (3-seed)

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 314 | 6,844 | 87.7 | 1.1253 | **1.1123** | 1.87802 | 14,519,698 |
| 999 | 6,846 | 87.7 | 1.1256 | **1.1124** | 1.87821 | 14,517,302 |
| 1337 | 6,828 | 87.7 | 1.1261 | **1.1129** | 1.87910 | 14,525,480 |
| **Mean** | **6,839** | **87.7** | | **1.1125** | **1.8784** | |

## Changes vs PR 1019

### 1. Fused Triton TMA Forward MLP Kernel

Fuses `F.linear(x, up_w) -> LeakyReLU(0.5) -> square` into a single Triton TMA kernel. The raw matmul output never hits HBM — activation computed in-register before first store. Backward uses explicit cuBLAS matmuls to preserve torch.compile's cross-layer fusion.

Builds on our kernel profiling in PR 670 (abaybektursun).

### 2. CUTLASS EVT Backward MLP Fusion

Fuses `(go @ down_w) * act_grad` into a single CUTLASS 3.x kernel via Epilogue Visitor Tree. The multiply happens in the GEMM epilogue while tiles are still in registers. Uses `KernelTmaWarpSpecializedPingpong` on sm90a with 128x128 tiles.

| Variant | dpre time | vs Unfused |
|---|---|---|
| cuBLAS unfused | 1.199 ms | baseline |
| Triton precomp | 1.130 ms | -0.069 ms |
| **CUTLASS Pingpong** | **1.095 ms** | **-0.104 ms** |

CUTLASS EVT is a hard dependency — no silent fallback.

### 3. Pre-Computed Activation Gradient

Store `act_grad = where(pre > 0, 2*pre, 0.5*pre)` in forward instead of `pre`. Zero extra memory cost. Derive `post = 0.5 * act_grad * c0` via algebraic identity. Eliminates `where()` branching from both forward and backward, and enables the CUTLASS EVT to use a trivial 3-node epilogue tree (`multiplies(AccFetch, AuxLoad)`) with no conditional logic.

### 4. Brotli-11 Compression (replaces LZMA-9)

-581 KB (-5.9%) vs LZMA-9. Independently discovered; also used in PR 1089 (mikeapedia).

### 5. Memmap Multi-Shard Data Pipeline + GPU Prefetch

Coprime-stride sampling, daemon thread, CUDA stream prefetch. Credit: DeepReinforce (PR 726).

## Negative Results

- **Turbo-Muon (AOL + Polar Express NS4):** +0.0018 BPB worse on 8xH100 AND artifact over 16MB. Early convergence advantage at step 500 doesn't hold at 7000+ steps. Reverted to standard NS5.
- **2:4 Structured Sparsity:** +0.672 BPB. Dead.

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA / 4 KV heads) | Baseline |
| MLP | 3x (1536), LeakyReLU(0.5)^2 | PR 493 (parinzee) |
| MLP Forward | **Fused Triton TMA kernel** | **This work** (profiling: our PR 670) |
| MLP Backward | **CUTLASS EVT Pingpong + pre-computed act_grad** | **This work** |
| Attention | XSA on all 11 layers | PR 478 (gowtham0992) |
| BigramHash | 3072 x 112 | Our PR 1019 (concept: PR 162 (raahilshah)) |
| RoPE | Partial (16/64 dims) | PR 315 (jfprincz) |
| LN Scale | 1/sqrt(layer+1) | PR 315 (jfprincz) |
| VE128 | Layers 9-10 | PR 374 (unnir) |
| SmearGate | Position-mixing gate | PR 65 (aquariouseworkman) |
| U-Net skips | Encoder-decoder connections | PR 289 |
| Weight avg | EMA(0.997) + SWA(every 50) | PR 401 (newjordan) |
| Quantization | Full Hessian GPTQ int6 (AR self-gen calibration) | Our PR 1019 (GPTQ: PR 535 (raahilshah)) |
| Compression | **Brotli quality=11** | **This work** (independently: PR 1089 (mikeapedia)) |
| Data Pipeline | **Memmap multi-shard + GPU prefetch** | PR 726 (DeepReinforce) |
| Warmdown | 4000 iterations | PR 364 (shikhar1729) |
| Optimizer | Parallel Muon (NS5) | Our PR 399 |
| Late QAT | STE at LR scale < 0.15 | PR 286 (chris-buckley) |
| Selective pruning | +/-1 by reconstruction error | PR 609 (saml212) |
| Flash Attention 3 | Hopper kernels | PR 122 (mtybadger) |

**Calibration legality:** AR self-generated (64 seqs x 2048 tokens, temp=0.8). No val data, no train data accessed during quantization. Same method as our PR 1019.

## Setup & Reproduction

```bash
# 1. Python dependencies
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece brotli

# 2. CUTLASS headers (header-only, no build needed for CUTLASS itself)
cd /opt && git clone --depth 1 --branch v3.7.0 https://github.com/NVIDIA/cutlass

# 3. Build CUTLASS EVT extension
cd cutlass_evt_fusion
CUTLASS_PATH=/opt/cutlass python3 setup.py build_ext --inplace
cd ..

# 4. Set library paths (auto-detect from Python packages)
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$(python3 -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0] + '/lib')"):${LD_LIBRARY_PATH:-}

# 5. Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# 6. Train (3 seeds)
for SEED in 314 42 999; do
  BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${SEED}.log
done
```
