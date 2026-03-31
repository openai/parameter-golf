# Non-Record: 11L NativeFlowMatcher + Legal Score-First TTT

**val_bpb: 1.11991** (sliding window, stride=64, int6/int5 quantized, legal TTT)
**val_bpb: 1.12312** (sliding window, stride=64, int6/int5 quantized, no TTT)

Single-seed (seed=42) non-record submission exploring NativeFlowMatcher (NFM) — a 393K-parameter OT-CFM (Optimal Transport Conditional Flow Matching) velocity network that applies gated hidden-state correction, jointly trained with the AR objective. Combined with legal score-first TTT for additional compression.

> **Note:** This is a single-seed exploratory submission. Three-seed reproducibility runs were attempted but failed due to an SDPA backend incompatibility on the evaluation cluster (A100 PCIe). Statistical significance is therefore not claimed. The purpose of this submission is to document the NFM architectural idea and its interaction with legal TTT, not to claim a new record.

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 11 |
| Model dim | 512 |
| Attention heads | 8 (4 KV heads, GQA) |
| MLP expansion | 3× (1536 hidden) |
| Vocab size | 1024 (SentencePiece BPE) |
| Sequence length | 2048 (training), 1024 (eval window) |
| Total params | 27,530,952 |
| NFM params | 393,729 |
| Tied embeddings | Yes |
| BigramHash | vocab=4096, dim=128 |
| Positional encoding | Partial RoPE (16 dims, base 10000) |
| XSA | All 11 layers (XSA_LAST_N=11) |
| Value residual | Yes |
| Gated attention | Yes |
| Activation | LeakyReLU(0.5)² |
| Logit softcap | 30.0 |
| LN scale | 1 |
| QK gain init | 1.5 |
| EMA | decay=0.997, applied at end |

### NativeFlowMatcher (NFM)

NFM is a conditional flow matching module inserted after the final LayerNorm in the transformer stack, before the language model head. It learns a velocity field over hidden states using OT-CFM:

- **Time embedding:** Sinusoidal positional encoding of scalar t → projected to 256-dim via Linear(512,256) + GELU
- **Velocity network:** Linear(512,256) + time-conditioning (additive) + GELU → Linear(256,512, no bias)
- **Gate:** Scalar parameter initialized at −5.0 (sigmoid ≈ 0.007), learned during training
- **Training loss:** MSE between predicted velocity v(x_t, t) and OT target velocity (x − z), where x_t = (1−t)·z + t·x with z ~ N(0,I). Weighted by `NATIVE_FLOW_LOSS_WEIGHT=0.1` added to AR cross-entropy loss.
- **Inference:** Single Euler step at t=1 on clean input x, gated: `x_out = x + sigmoid(gate) · v(x, t=1)`

The NFM velocity network has zero-initialized output weights and a near-zero initial gate, ensuring the correction starts negligible and grows only as the velocity field learns useful structure.

## Results

### Primary (This Submission)

| Evaluation | val_loss | val_bpb |
|------------|----------|---------|
| Roundtrip (dequantized) | 1.93630746 | 1.14679034 |
| Sliding window (stride=64), no TTT | 1.89632895 | 1.12311579 |
| **Sliding window (stride=64), legal TTT** | **1.89091021** | **1.11990650** |

Legal TTT improvement: **−0.00321 BPB** (from 1.12312 → 1.11991)

### Supplementary Comparison

These are reference results from related configurations, included for context. All use the same base architecture (PR #940 stack) and evaluation protocol.

| Config | Steps | Params | No-TTT Sliding BPB | Legal TTT Sliding BPB |
|--------|-------|--------|---------------------|----------------------|
| Base (no refiners) 20k | 20,000 | ~27.1M | 1.10050 | 1.09292 |
| FlowRefiner 20k | 20,000 | ~27.2M | 1.10002 | 1.09279 |
| **NFM 7k (this submission)** | **7,000** | **27.5M** | **1.12312** | **1.11991** |
| E2E TTT + FlowRefiner 7k | 7,000 | 28.3M | — | ~1.124† |

† E2E TTT + FlowRefiner legal TTT eval was incomplete — SLURM job timed out at chunk 1271/1893. The BPB at truncation was 1.12408; final value is unknown.

> **Important context:** The 20k-step results (base, flow) use a longer training schedule (20,000 steps vs 7,000). Direct BPB comparison between 7k and 20k is not meaningful for architecture evaluation. The NFM contribution should be assessed relative to the base architecture at matched step count, but no 7k-step base-only run exists in this evaluation set. The training-time val_bpb at step 7000 was 1.1380 (pre-quantization, non-sliding, no TTT).

## Quantization

| Property | Value |
|----------|-------|
| Base scheme | Per-row int8 (2D weights) / per-tensor int8 (1D) |
| MLP layers 0–4, 7–10 | int6 (GPTQ-lite) |
| MLP layers 5–6 | int5 (auto-downgrade fallback to fit 16MB) |
| Compression | zstd level 16 |
| Quantized model | 15,630,744 bytes |
| Code (`train_gpt.py`) | 115,032 bytes |
| **Total artifact** | **15,745,776 bytes** (headroom: 254,224 bytes) |

The auto-downgrade mechanism progressively applies int5 quantization to middle MLP layers (starting from layer 5 outward) until the compressed artifact fits within the 16MB budget.

## Training

| Property | Value |
|----------|-------|
| Hardware | 1× A100 PCIe 40GB |
| Steps | 7,000 |
| Wallclock | 13,879 seconds (3.86 hours) |
| Step average | 1,982.77 ms |
| Training tokens | ~5.51B (7000 × 786432) |
| Sequence length | 2048 |
| Optimizer | Muon (matrix) + Adam (scalars/embeddings) |
| Matrix LR | 0.025 |
| Scalar LR | 0.025 |
| Muon weight decay | 0.04 |
| Adam weight decay | 0.04 |
| Gradient clip | 0.3 |
| Warmup | 20 steps |
| Warmdown | 2,800 steps |
| Seed | 42 |
| SLURM job | 55342820 |
| Peak GPU memory | 25,832 MiB |

### Training Trajectory

| Step | val_bpb |
|------|---------|
| 0 | 4.1055 |
| 500 | 1.3813 |
| 1000 | 1.3058 |
| 2000 | 1.2499 |
| 3000 | 1.2283 |
| 4000 | 1.2199 |
| 5000 | 1.1975 |
| 6000 | 1.1707 |
| 6500 | 1.1527 |
| 7000 | 1.1380 |

## Legal TTT Configuration

Score-first test-time training that complies with the rule that training may only occur on tokens that have already been scored (no future information leakage).

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD with momentum=0.9 |
| Learning rate | 0.002 |
| Epochs per chunk | 10 |
| Chunk size | 32,768 tokens |
| Frozen blocks | 2 (first 2 transformer layers frozen) |
| Gradient clip | 1.0 |
| Total chunks | 1,893 |
| TTT eval time | ~7,190 seconds (~2.0 hours) |
| SLURM job | 55375245 |

## Provenance

All artifacts trace back to verifiable SLURM jobs:

1. **Training:** SLURM job 55342820 → `runs/nflow_55342820/models/final_model_pr940_nflow_55342820.pt`
2. **Eval (no TTT):** SLURM job 55375246 → sliding BPB = 1.12312, artifact = 15,745,776 bytes
3. **Eval (legal TTT):** SLURM job 55375245 → sliding BPB = 1.11991, artifact = 15,745,776 bytes
4. **Submitted model:** `final_model.int6.ptz` is the quantized+compressed artifact from eval job 55375245

The training SLURM script and both evaluation SLURM scripts are included in `supplementary/` for full reproducibility.

> **Code size note:** The submitted `train_gpt.py` (115,032 bytes) reflects the version used at evaluation time. The training log (`train.log`) reports a code size of 104,738 bytes, reflecting the version at training time. The code evolved between training and evaluation but the model checkpoint is unchanged. Supplementary SLURM scripts reference the working filename `train_gpt_pr940.py`; rename to `train_gpt.py` for reproduction.

## Limitations

1. **Single seed:** Only seed=42 results are available. Three-seed runs (seeds 42, 1337, 2025) were attempted on a PR #549 codebase port but failed due to `RuntimeError('Invalid backend')` in `scaled_dot_product_attention` under `torch.compile` on A100 PCIe hardware. Statistical significance is not established.

2. **No matched baseline:** There is no 7k-step base-only (no NFM) run with the exact same architecture and evaluation protocol. The training-time val_bpb of 1.1380 cannot be directly compared to the sliding-window eval BPB of 1.12312 due to different evaluation methodologies.

3. **Non-competitive BPB:** The best result (1.11991) is above the current leaderboard SOTA. This submission documents the NFM idea rather than competing for a record.

4. **Incomplete supplementary eval:** The E2E TTT + FlowRefiner legal TTT evaluation was truncated at chunk 1271/1893 due to SLURM time limits. Its partial BPB (~1.124) is included as supplementary data only.

## Reproduction

```bash
# Training (single GPU, ~4 hours)
# See supplementary/slurm_pr940_nflow_7k.sh for full env vars
export NATIVE_FLOW_ENABLED=1
export NATIVE_FLOW_HIDDEN_DIM=256
export NATIVE_FLOW_INIT_SCALE=0.01
export NATIVE_FLOW_LOSS_WEIGHT=0.1
export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
export ITERATIONS=7000 SEED=42
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Evaluation with legal TTT (~2 hours)
# See supplementary/slurm_eval_nflow7k_legal_ttt.sh for full env vars
export EVAL_ONLY=/path/to/final_model.pt
export TTT_ENABLED=1 LEGAL_TTT=1
export TTT_LR=0.002 TTT_EPOCHS=10 TTT_FREEZE_BLOCKS=2
export TTT_CHUNK_TOKENS=32768 TTT_OPTIMIZER=sgd TTT_MOMENTUM=0.9
export EVAL_STRIDE=64
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Credits

This submission builds on the work of many contributors to the parameter golf contest:

| Component | Source | Author(s) |
|-----------|--------|-----------|
| Base AR architecture | PR #549 | @abaybektursun |
| Muon optimizer | Baseline | Contest organizers |
| BigramHash, SmearGate | PR #65 | @aquariouserworkman |
| XSA (Cross-Sequence Attention) | PR #187, #265 | @Idan3011, @unnir |
| U-Net skip connections | PR #65, #69 | @aquariouserworkman |
| SWA (Stochastic Weight Averaging) | PR #69 | @aquariouserworkman |
| Mixed int6/int8 quantization | PR #76 | Contest community |
| Sliding window evaluation | PR #50 | @mattqlf |
| Legal score-first TTT | PR #77 | @samacqua |
| VE, Partial RoPE, LN Scale | PR #315, #374 | @jfprincz, @unnir |
| LeakyReLU² activation | Baseline / PR #549 | @abaybektursun |
| EMA | PR #65 | @aquariouserworkman |
| Gated attention, value residual | PR #940 | Contest community |
| NativeFlowMatcher (this work) | PR #940 experiments | @mcclec07 |

## File Manifest

```
README.md                   — This file
submission.json             — Structured metadata
train_gpt.py               — Training/eval script (2,601 lines)
train.log                   — Training log (SLURM 55342820, 2,479 lines)
final_model.int6.ptz        — Quantized model artifact (15,630,744 bytes)
supplementary/
  eval_nflow7k_legal_ttt.log        — Legal TTT evaluation log (SLURM 55375245)
  eval_nflow7k_nottt.log            — No-TTT evaluation log (SLURM 55375246)
  eval_e2ettt_flow7k_legal_ttt.log  — E2E TTT+Flow eval (SLURM 55375247, INCOMPLETE)
  slurm_pr940_nflow_7k.sh           — Training SLURM script
  slurm_eval_nflow7k_legal_ttt.sh   — Legal TTT eval SLURM script
  slurm_eval_nflow7k_nottt.sh       — No-TTT eval SLURM script
```
