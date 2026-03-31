# 10L E2E TTT-Linear + FlowRefiner (Non-Record)

**val_bpb: 1.1335 ± 0.0010** (4-seed mean ± std, int6 sliding window, stride=64) | **~15.1 MB** artifact | 2×A100 PCIe 40GB

## Track

Non-record submission (`track_non_record_16mb`). Trained on 2×A100 PCIe 40GB for ~2.2 hours (7,185 steps). Artifact fits within the 16,000,000-byte cap. **4-seed reproducibility established** (seeds 42, 99, 1337, 2025).

**Contribution**: Demonstrates end-to-end TTT-Linear refinement combined with 1-step flow matching, compressed into a 10-layer architecture that fits under the 16 MB artifact limit. The lightweight FlowRefiner is inspired in part by FLOWR's use of learned flow-matching vector fields with efficient Euler-style updates, but adapted here into a tiny hidden-state refiner rather than a pocket-conditioned 3D ligand generator. Includes a three-variant comparison (11L over-budget, 10L legal, 11L+int5 legal) as supplementary data.

## Results

### Multi-Seed Reproducibility (4 seeds, 2×A100 PCIe 40GB)

| Seed | SLURM Job | Int6 Roundtrip BPB | Int6 Sliding Window BPB | Artifact Size |
|------|-----------|-------------------|------------------------|---------------|
| 42 | 55383562 | 1.15790913 | 1.13472408 | 15,094,152 |
| 99 | 55392385 | 1.15743221 | 1.13387877 | 15,198,948 |
| 1337 | 55392383 | 1.15614893 | 1.13269366 | 15,070,964 |
| 2025 | 55392384 | 1.15630189 | 1.13283672 | 15,117,416 |
| **Mean** | — | **1.15694804** | **1.13353331** | — |
| **Std** | — | **0.00085911** | **0.00095351** | — |

### Primary (seed=42, this artifact)

| Metric | Value |
|--------|-------|
| Pre-quant val BPB (step 7185) | 1.1492 |
| Post-EMA val BPB | 1.1483 |
| Int6 roundtrip BPB | 1.15790913 |
| **Int6 sliding window BPB (stride=64)** | **1.13472408** |
| Quantization gap | +0.0096 BPB |
| Sliding window gain | −0.0232 BPB |
| Model (int6+lzma) | 15,094,152 bytes |
| Code (`train_gpt.py`) | 104,955 bytes |
| **Total artifact** | **15,199,107 bytes** |
| Headroom | 800,893 bytes |
| Base model params | 25,749,398 |
| Refiner params | 1,182,530 (TTT: 1,083,905 + Flow: 98,625) |
| Step avg | 1,116 ms |
| Peak memory | 23,546 MiB |
| Training steps | 7,185 |
| Warmdown steps | 4,311 (60%) |

## Architecture

- **10 layers**, 512D, 8 attention heads, 4 KV heads (GQA)
- 3× MLP expansion (1536D hidden) with LeakyReLU(0.5)² (per PR #549)
- Tied embeddings (1024-token SentencePiece vocab)
- BigramHash(1536), XSA (last 4 layers), Partial RoPE (16D)
- LN Scale, Value Embeddings (128D, layers 8–9)
- EMA(0.997) + SWA (starts step 6350)
- Late QAT (starts step 6539, scale 0.15)
- Mixed int6/int8 per-row quantization + lzma compression (preset=6)
- SmearGate + OrthoInit, U-Net skip connections

### Refiners (applied after final LayerNorm, before lm_head)

**E2E TTT-Linear** (Sun et al., 2024): Per-head inner-loop SGD on learned projections during both training and inference. Learned projection matrices θ_K, θ_V, θ_Q ∈ ℝ^{512×512}. Per-head W ∈ ℝ^{num_heads × head_dim × head_dim} + bias. 8 heads, 16-token mini-batches, lr=1.0. Sigmoid-gated additive output (init=−5.0). 1,083,905 parameters.

**1-Step FlowRefiner**: Single-step flow matching in latent space. Down-project 512D → 64D, velocity network 64D → 256D (GELU) → 64D, up-project 64D → 512D. 1-step Euler integration: z_refined = z + v_net(z). Sigmoid-gated residual (init=−5.0). 98,625 parameters. The formulation is inspired by the FLOWR paper's use of learned vector fields and Euler-style transport updates for efficient refinement, but our version is much smaller and operates on transformer hidden states rather than molecular coordinates or pocket-conditioned ligand graphs.

## Training Config

```bash
SEED=42
ITERATIONS=7185
WARMDOWN_ITERS=4311
WARMUP_STEPS=20
TRAIN_BATCH_TOKENS=786432
TRAIN_SEQ_LEN=2048
NUM_LAYERS=10
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3
BIGRAM_VOCAB_SIZE=1536
BIGRAM_DIM=128
MATRIX_LR=0.025
SCALAR_LR=0.025
TIED_EMBED_LR=0.035
MUON_WD=0.04
ADAM_WD=0.04
EVAL_STRIDE=64
E2E_TTT_ENABLED=1
E2E_TTT_NUM_HEADS=8
E2E_TTT_MINI_BATCH=16
E2E_TTT_BASE_LR=1.0
FLOW_ENABLED=1
FLOW_LATENT_DIM=64
FLOW_HIDDEN_DIM=256
FLOW_INIT_SCALE=0.01
VE_ENABLED=1
VE_DIM=128
VE_LAYERS="8,9"
```

## Run Command

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

SEED=42 ITERATIONS=7185 WARMDOWN_ITERS=4311 \
  NUM_LAYERS=10 BIGRAM_VOCAB_SIZE=1536 \
  E2E_TTT_ENABLED=1 FLOW_ENABLED=1 \
  VE_LAYERS="8,9" \
  MATRIX_LR=0.025 SCALAR_LR=0.025 MUON_WD=0.04 ADAM_WD=0.04 \
  EVAL_STRIDE=64 MAX_WALLCLOCK_SECONDS=0 \
  torchrun --standalone --nproc_per_node=2 train_gpt.py
```

## Training Trajectory

| Step | val_bpb | Event |
|------|---------|-------|
| 500 | 1.4063 | |
| 1000 | 1.3275 | |
| 2000 | 1.2674 | |
| 3000 | 1.2417 | |
| 4000 | 1.2220 | |
| 5000 | 1.2007 | |
| 6000 | 1.1790 | |
| 6350 | — | SWA start |
| 6539 | — | Late QAT enabled |
| 7000 | 1.1524 | |
| 7185 | 1.1492 | Final |

## Supplementary: Three-Variant Size–Quality Comparison

We explored three strategies to fit the combined E2E-TTT + FlowRefiner architecture under the 16 MB artifact cap. All variants use the same `train_gpt.py` (Var A and B are identical code; only runtime environment variables differ). Var C adds int5 quantization for MLP weights.

| Variant | Layers | Quant | val_bpb (sw) | Model Size | Code Size | Total | Status |
|---------|--------|-------|--------------|------------|-----------|-------|--------|
| A: 11L + 60% warmdown | 11 | int6/int8 | **1.12356295** | 16,576,172 | 104,955 | 16,681,127 | Over budget |
| **B: 10L (this submission)** | **10** | **int6/int8** | **1.13353 ± 0.00095 (4-seed)** | **~15.1 MB** | **104,955** | **~15.2 MB** | **Legal** |
| C: 11L + int5 MLP | 11 | int5/int6/int8 | 1.15074174 | 14,196,568 | 106,694 | 14,303,262 | Legal |

**Key observations:**
- Variant A achieves the best BPB (1.1236) but exceeds the 16 MB cap by 681 KB. No compression algorithm tested (lzma-6, lzma-9e, zstd-16/19/22, zlib-9) could close the gap.
- Variant B (this submission) trades one transformer layer for budget compliance, losing ~0.011 BPB vs Variant A.
- Variant C saves the most space via int5 MLP quantization but incurs a large quantization gap (0.0361 vs Var A's 0.0094), making it the worst performer despite identical pre-quant training (Var A and C have the same 28.1M base model params and near-identical pre-quant BPB of 1.1386 vs 1.1391).

Variant A and C training logs are included in `supplementary/` for reproducibility.

## Prior 11L Ablations on the Same Refiner Pair

To understand whether FlowRefiner helps on its own or mainly in combination with E2E TTT, we also ran an earlier 11-layer ablation study on the same refiner formulation. These are **not** the main numbers for this 10-layer legal submission; they are prior supporting experiments from `experiments_pr549/`.

| Prior 11L run | Sliding BPB | Δ vs 11L baseline | Roundtrip BPB | Post-EMA BPB |
|---------------|-------------|-------------------|---------------|--------------|
| Baseline | 1.12440473 | — | 1.14795319 | 1.1424 |
| + E2E-TTT only | 1.12414225 | -0.00026 | 1.14782916 | 1.1423 |
| + Flow only | 1.12531495 | +0.00091 | 1.14870788 | 1.1426 |
| + Both (Combined) | 1.12344104 | -0.00096 | 1.14698496 | 1.1412 |

### Synergy Discussion

In that earlier 11-layer study, FlowRefiner alone regressed after quantization, while the combined E2E-TTT + Flow model was the best of the four. Using the isolated deltas, the additive expectation would have been worse than baseline:

- Expected additive BPB = $1.12440473 + (1.12414225 - 1.12440473) + (1.12531495 - 1.12440473) = 1.12505247$
- Actual combined BPB = $1.12344104$
- Gap vs additive expectation = $1.12344104 - 1.12505247 = -0.00161143$

So in that prior 11-layer setting, the combined model outperformed the additive expectation by about **0.00161 BPB**, which is consistent with a positive interaction between E2E TTT and FlowRefiner. We treat this as evidence that FlowRefiner is most useful when paired with TTT, but we do **not** claim that the same four-way ablation has been rerun separately for the present 10-layer legal submission.

For provenance, the earlier ablation logs are:

- `experiments_pr549/exp_baseline/logs/train_55374937.txt`
- `experiments_pr549/exp_e2e_ttt/logs/train_55374938.txt`
- `experiments_pr549/exp_flow/logs/train_55374939.txt`
- `experiments_pr549/exp_combined/logs/train_55374940.txt`

## Variant Details

### Variant A: 11 Layers + 60% Warmdown (Over Budget)

- 11 layers, 28,110,750 base params + 1,182,530 refiner params
- Same training steps (7185) and warmdown ratio (60%)
- Post-EMA val_bpb: 1.1377 | Int6 roundtrip: 1.14705342 | Sliding window: **1.12356295**
- Artifact: 16,681,127 bytes (over by 681,127)
- SLURM job: 55383561 on node g022

### Variant C: 11 Layers + Int5 MLP Quantization (Legal)

- 11 layers, 28,110,750 base params + 1,182,530 refiner params
- MLP weights quantized to int5 (clip_range=15); attention and other weights remain int6/int8
- Post-EMA val_bpb: 1.1382 | Int6 roundtrip: 1.17431049 | Sliding window: **1.15074174**
- Artifact: 14,303,262 bytes (1.70 MB headroom)
- The int5 quantization gap (+0.0361) is 3.8× larger than int6 (+0.0094), overwhelming the extra layer's benefit
- SLURM job: 55383563 on node g022

## Provenance

| Item | Value |
|------|-------|
| SLURM Job ID (seed=42) | 55383562 |
| SLURM Job ID (seed=99) | 55392385 |
| SLURM Job ID (seed=1337) | 55392383 |
| SLURM Job ID (seed=2025) | 55392384 |
| Node | g022 |
| GPU | 2×NVIDIA A100-PCIE-40GB |
| Run directory | `experiments_16mb/varB_10L/runs/seed{42,99,1337,2025}_<jobid>/` |
| Training log | `train.log` (seed=42, this directory); additional seeds in `supplementary/` |
| Training scripts | `run_seed{42,99,1337,2025}.sh` (in `supplementary/`) |
| Runtime | ~3 hours per seed |
| Exit code | 0 (all seeds) |
| Prior 11L ablation study | `experiments_pr549/exp_{baseline,e2e_ttt,flow,combined}/logs/` |

## Limitations

1. **Non-record hardware**: 2×A100 PCIe 40GB, not the required 8×H100 SXM for record-track submissions.
2. **Non-record training time**: ~2.2 hours, exceeding the 10-minute record-track constraint.
3. **No post-training TTT**: The sliding window result (1.1335 mean) does not include score-first TTT adaptation, which could improve results further.
4. **10L vs 11L tradeoff**: The 10-layer variant sacrifices ~0.011 BPB relative to the 11-layer Variant A to fit within the artifact size budget.

## Credits

This submission builds on work from many contributors to the parameter-golf competition:

- **Base architecture** — PR #549 (abaybektursun): 11L 512D GQA, BigramHash, 3×MLP LeakyReLU(0.5)², SmearGate, XSA, U-Net skips, VE128, Partial RoPE, LN Scale, OrthoInit, Late QAT, SWA, EMA
- **Muon optimizer** — Baseline (`modded-nanogpt`); Newton-Schulz orthogonal preconditioning
- **BigramHash embeddings** — PR #65 (aquariouseworkman)
- **SmearGate** — PR #65 (aquariouseworkman)
- **XSA (Exclusive Self Attention)** — PR #187 (Idan3011), GQA variant in PR #265 (unnir)
- **U-Net skip connections** — PR #65 (aquariouseworkman), PR #69 (TevBenji)
- **Mixed int6/int8 quantization** — PR #76 (unixmadtoonslab / Will DePue)
- **SWA** — PR #69 (TevBenji)
- **Late QAT** — PR #315 (jfprincz), working implementation in PR #374 (unnir)
- **Sliding window evaluation** — PR #50 (mattqlf / Matthew Li)
- **Value Embeddings (VE128)** — PR #374 (unnir)
- **Partial RoPE** — PR #315 (jfprincz), PR #374 (unnir)
- **LN Scale** — PR #315 (jfprincz), PR #374 (unnir)
- **Legal TTT framework** — PR #77 (samacqua)
- **Flow-inspired refinement framing** — informed by FLOWR (Cremer et al., arXiv:2504.10564), adapted here into a tiny hidden-state vector-field refiner rather than a pocket-conditioned ligand generator
- **LeakyReLU² / ReLU², GQA** — Baseline (`modded-nanogpt`), PR #549 variant uses slope=0.5

**TTT-Linear reference**: Sun, Y., et al. "Learning to (Learn at Test Time): RNNs with Expressive Hidden States." arXiv:2407.04620 (2024).

**Flow Matching reference**: Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR 2023.

**FLOWR inspiration**: Cremer, J., Irwin, R., Tibo, A., Janet, J. P., Olsson, S., Clevert, D.-A. "FLOWR: Flow Matching for Structure-Aware De Novo, Interaction- and Fragment-Based Ligand Generation." arXiv:2504.10564 (2025).

Built on the [parameter-golf](https://github.com/openai/parameter-golf) starter code by Beren Millidge & Keller Jordan.
