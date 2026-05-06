# Record: AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112

**val_bpb: 1.1147** (3-seed mean, std 0.0004) | **~15.91 MB** | 8×H100 SXM, 600s | No TTT

**This submission uses only AR (autoregressive) self-generated calibration data.** After training, the model autoregressively generates its own calibration tokens (64 seqs × 2048 tokens, temp=0.8). No val data and no train data are accessed during quantization.

**Improvement over current SOTA ([PR #549](https://github.com/openai/parameter-golf/pull/549), 1.1194 BPB):** −0.0078 nats (−0.0046 BPB)

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 314 | 6,927 | 86.6 | 1.1354 | **1.1151** | 15,863,278 |
| 42 | 6,922 | 86.7 | 1.1349 | **1.1144** | 15,984,850 |
| 999 | 6,917 | 86.8 | 1.1353 | **1.1148** | 15,876,310 |
| **Mean** | | | | **1.1147** | |

Current SOTA (PR #549, exact 3-seed mean): **1.11937967 BPB** (**1.89002068 nats**). This run's exact 3-seed mean is **1.11473509 BPB** (**1.88217853 nats**). Delta: **−0.00784215 nats** (**−0.00464458 BPB**).

Using the exact per-seed scores from the PR #549 logs (`1.11922988`, `1.12002032`, `1.11888882`) and this run (`1.11508120`, `1.11437394`, `1.11475014`), Welch's t-test gives **t = -11.83**, **df ≈ 3.31**.

---

## Main Changes

The comparison baseline is [PR #549](https://github.com/openai/parameter-golf/pull/549), the current legal leaderboard entry at **1.1194 BPB**. The implementation lineage is closer to [PR #609](https://github.com/openai/parameter-golf/pull/609): this run keeps the XSA-all + Full GPTQ + selective-pruning stack, but uses AR self-generated GPTQ calibration (no external data), bumps BigramHash to **3072 × 112**, and uses `lzma preset=9`.

### 1. AR Self-Generated Full Hessian GPTQ

PR #549 used GPTQ-lite (diagonal Hessian approximation). We use Full Hessian GPTQ with Cholesky error compensation and column reordering — a strictly better quantizer.

The calibration problem: prior Full Hessian GPTQ implementations (PRs #535, #569, #593, #609) calibrated on training data, ruled illegal after the 600s window. We solve this by having the model generate its own calibration data. After training completes, the model autoregressively generates 64 sequences of 2048 tokens (temperature=0.8, fixed seed). Hessians H = X^T X are collected from these self-generated sequences. No val data, no train data accessed during quantization.

### 2. BigramHash 3072 × dim=112 (up from 1536)

Lineage: [PR #549](https://github.com/openai/parameter-golf/pull/549) (1536) → [PR #609](https://github.com/openai/parameter-golf/pull/609) (2048) → this run (**3072 × dim=112**). Fits under 16MB; going wider increased artifact pressure past the break-even point.

### 3. XSA on all 11 layers (up from last 4)

PR #549 applied XSA to the last 4 layers. Extending to all 11 layers forces cross-position information mixing from layer 0 at zero parameter cost. Source: [PR #478](https://github.com/openai/parameter-golf/pull/478) by @gowtham0992.

### Dropped: TTT

PR #549 used Legal Score-First TTT for −0.0025 BPB. On this stack, TTT is neutral or negative (25 failed attempts across two stacks — see our [PR #756](https://github.com/openai/parameter-golf/pull/756)). The Full Hessian GPTQ improvement more than compensates for dropping TTT.

---

## Architecture

| Component | Setting | First introduced by |
|-----------|---------|---------------------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 3× (1536) with LeakyReLU(0.5)² | [#493](https://github.com/openai/parameter-golf/pull/493) @parinzee |
| Attention | XSA on all 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) @gowtham0992 |
| BigramHash | **3072 × dim=112** | **This work** (concept: [#162](https://github.com/openai/parameter-golf/pull/162) @raahilshah) |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| LN Scale | 1/√(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| VE128 | Layers 9-10 | [#374](https://github.com/openai/parameter-golf/pull/374) @unnir |
| SmearGate | Position-mixing gate | [#65](https://github.com/openai/parameter-golf/pull/65) @aquariouseworkman |
| U-Net skips | Encoder-decoder connections | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Weight avg | EMA(0.997) + Tight SWA(every 50) | [#401](https://github.com/openai/parameter-golf/pull/401) @newjordan |
| Quantization | **Full Hessian GPTQ int6 (AR self-gen calibration)** | **This work** (GPTQ: [#535](https://github.com/openai/parameter-golf/pull/535) @raahilshah) |
| Compression | LZMA preset=9 | [#160](https://github.com/openai/parameter-golf/pull/160) @ChaseWNorton |
| Warmdown | 4000 iterations | [#364](https://github.com/openai/parameter-golf/pull/364) @shikhar1729 |
| Optimizer | **Parallel Muon + Parameter Banking** | **[#399](https://github.com/openai/parameter-golf/pull/399) @abaybektursun** |
| Late QAT | STE at LR scale < 0.15 | [#286](https://github.com/openai/parameter-golf/pull/286) @chris-buckley |
| Selective pruning | ±1 values by reconstruction error | [#609](https://github.com/openai/parameter-golf/pull/609) @saml212 |
| Flash Attention 3 | Hopper warp-specialized kernels | [#122](https://github.com/openai/parameter-golf/pull/122) @mtybadger |

## Requirements

**Flash Attention 3 (Hopper) is required.** The script imports `flash_attn_interface` directly and was run with PyTorch 2.9.1+cu128.

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard
python3 -c "from flash_attn_interface import flash_attn_func; import sentencepiece, zstandard; print('deps OK')"
```

## Run Command

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

```
PR #549 (Legal SOTA, 1.1194) — our Parallel Muon base with LeakyReLU² + legal TTT
    └── This work adds:
        ├── AR self-gen GPTQ calibration (no external data during quantization)
        ├── BigramHash 3072 × 112 (wider setting that still fits under 16MB)
        ├── XSA-all (from #478/@gowtham0992, applied via #609/@saml212)
        ├── Selective ±1 pruning (from #609/@saml212)
        ├── warmdown=4000, LZMA=9 (from #364/@shikhar1729, #160/@ChaseWNorton)
        └── Guided by PR #670 negative results (30+ failed experiments)
```
