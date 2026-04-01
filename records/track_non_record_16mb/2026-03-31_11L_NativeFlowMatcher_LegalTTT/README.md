# Non-Record: 11L NativeFlowMatcher + Legal Score-First TTT

**val_bpb: 1.11991** (seed=42, sliding window, stride=64, int6/int5 quantized, legal TTT)
**3-seed mean sliding BPB (no TTT): 1.12252** ± 0.00151 | **3-seed mean legal TTT: 1.11928** ± 0.00146

Non-record submission exploring **NativeFlowMatcher (NFM)** — a 393K-parameter OT-CFM (Optimal Transport Conditional Flow Matching) velocity network that applies gated hidden-state correction, jointly trained with the AR objective. Combined with legal score-first TTT for additional compression.

> **Update (2026-04-01):** All ablation studies complete. Three-seed legal TTT evals finished. NFM does not improve BPB vs matched base — see Ablation Studies section for full 2×2 matrix, loss weight sweep, and hidden dim sweep.

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

### Three-Seed Reproducibility (Training-Time Eval)

All three seeds trained identically: 7,000 steps, 1×A100 PCIe 40GB, same architecture and optimizer config.

| Seed | SLURM Job | Training val_bpb | Roundtrip BPB | Sliding (no TTT) BPB | Legal TTT BPB | Artifact Bytes |
|------|-----------|-----------------|---------------|----------------------|---------------|----------------|
| 42 | 55342820 | 1.1380 | 1.14679034 | **1.12311579** | **1.11990650** | 15,745,776 |
| 1337 | 55398556 | 1.1385 | 1.14729126 | **1.12366996** | **1.12032079** | 15,736,933 |
| 2025 | 55398557 | 1.1359 | 1.14444585 | **1.12077485** | **1.11761299** | 15,745,950 |
| **Mean** | | **1.1375** | **1.14617582** | **1.12252020** | **1.11928009** | — |
| **Std** | | **0.0014** | **0.00157** | **0.00151** | **0.00146** | — |

> Legal TTT evaluation complete for all seeds.

### Primary (Seed=42, This Submission)

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
| E2E TTT + FlowRefiner 7k | 7,000 | 28.3M | — | 1.12418 |

> **Update:** The E2E TTT + FlowRefiner legal TTT eval (SLURM 55398555) completed with val_bpb=1.12418. Previous submission had partial data (truncated at chunk 1271/1893).

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

### Training Trajectory (All Seeds)

| Step | Seed 42 | Seed 1337 | Seed 2025 |
|------|---------|-----------|-----------|
| 0 | 4.1055 | 4.1175 | 4.1065 |
| 500 | 1.3813 | 1.3849 | 1.3859 |
| 1000 | 1.3058 | 1.3100 | 1.3070 |
| 2000 | 1.2499 | 1.2497 | 1.2490 |
| 3000 | 1.2283 | 1.2285 | 1.2269 |
| 4000 | 1.2199 | 1.2205 | 1.2190 |
| 5000 | 1.1975 | 1.1983 | 1.1958 |
| 6000 | 1.1707 | 1.1710 | 1.1686 |
| 6500 | 1.1527 | 1.1532 | 1.1508 |
| 7000 | 1.1380 | 1.1385 | 1.1359 |

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

### Seed 42 (Primary)
1. **Training:** SLURM job 55342820 → `runs/nflow_55342820/models/final_model_pr940_nflow_55342820.pt`
2. **Eval (no TTT):** SLURM job 55375246 → sliding BPB = 1.12312, artifact = 15,745,776 bytes
3. **Eval (legal TTT):** SLURM job 55375245 → sliding BPB = 1.11991, artifact = 15,745,776 bytes
4. **Submitted model:** `final_model.int6.ptz` is the quantized+compressed artifact from eval job 55375245

### Seed 1337 (Reproducibility)
1. **Training:** SLURM job 55398556 → `runs/nflow_s1337_55398556/models/final_model_pr940_nflow_s1337_55398556.pt`
2. **Training-time sliding BPB (no TTT):** 1.12367, artifact = 15,736,933 bytes
3. **Eval (legal TTT):** SLURM job 55411651 → sliding BPB = 1.12032
4. **Eval (no TTT):** SLURM job 55411652 → sliding BPB = 1.12367

### Seed 2025 (Reproducibility)
1. **Training:** SLURM job 55398557 → `runs/nflow_s2025_55398557/models/final_model_pr940_nflow_s2025_55398557.pt`
2. **Training-time sliding BPB (no TTT):** 1.12077, artifact = 15,745,950 bytes
3. **Eval (legal TTT):** SLURM job 55411653 → sliding BPB = 1.11761
4. **Eval (no TTT):** SLURM job 55411654 → sliding BPB = 1.12077

### Supplementary
5. **E2E TTT + FlowRefiner eval (complete):** SLURM job 55398555 → legal TTT BPB = 1.12418

The training SLURM scripts and evaluation SLURM scripts for all seeds are included in `supplementary/` for full reproducibility.

> **Code size note:** The submitted `train_gpt.py` (115,032 bytes) reflects the version used at evaluation time. The training log (`train.log`) reports a code size of 104,738 bytes, reflecting the version at training time. The code evolved between training and evaluation but the model checkpoint is unchanged. Supplementary SLURM scripts reference the working filename `train_gpt_pr940.py`; rename to `train_gpt.py` for reproduction.

## Ablation Studies

Ablation studies isolating the NFM contribution and exploring hyperparameter sensitivity. All runs use seed=42, 7,000 steps, identical architecture and optimizer settings.

### Three-Seed Reproducibility

Training completed for all three seeds. Sliding window (no TTT) results from training-time eval:

| Seed | SLURM Job | Training val_bpb | Sliding BPB (no TTT) |
|------|-----------|-----------------|---------------------|
| 42 | 55342820 | 1.1380 | 1.12312 |
| 1337 | 55398556 | 1.1385 | 1.12367 |
| 2025 | 55398557 | 1.1359 | 1.12077 |
| **Mean ± Std** | | **1.1375 ± 0.0014** | **1.12252 ± 0.00151** |

Legal TTT evaluation jobs submitted: 55411651 (s1337, **complete: 1.12032**), 55411653 (s2025, **complete: 1.11761**).

### 2×2 Matrix: NFM × TTT

Isolates the NFM and legal-TTT contributions independently. All runs use seed=42.

| Configuration | Params | No TTT (BPB) | Legal TTT (BPB) | Δ (TTT effect) |
|---------------|--------|--------------|-----------------|------------------|
| Base (no NFM) | 27,137,223 | 1.12106 | 1.11861 | −0.00245 |
| NFM (hd=256, lw=0.1) | 27,530,952 | 1.12312 | 1.11991 | −0.00321 |
| **Δ (NFM effect)** | **+393,729** | **+0.00206** | **+0.00130** | — |

**NFM hurts by +0.00206 BPB (no TTT) or +0.00130 BPB (with TTT).** The extra 393K parameters do not improve compression. Base ablation: SLURM 55398693 (train), 55398694 (eval no-TTT), 55398695 (eval TTT).

### Loss Weight Sweep (hidden_dim=256)

Explores the balance between NFM auxiliary loss and AR cross-entropy loss.

| loss_weight | No TTT (BPB) | Δ vs base |
|-------------|--------------|----------|
| 0.01 | 1.12344 | +0.00238 |
| 0.05 | 1.12294 | +0.00188 |
| **0.10 (default)** | **1.12312** | **+0.00206** |
| 0.20 | 1.12368 | +0.00262 |

Best loss weight is 0.05, but still +0.00188 BPB worse than base (1.12106).

### Hidden Dim Sweep (loss_weight=0.1)

Explores the capacity of the NFM velocity network.

| hidden_dim | NFM Params | Total Params | No TTT (BPB) | Δ vs base |
|------------|------------|--------------|--------------|----------|
| 128 | ~197K | 27,334,088 | 1.12228 | +0.00122 |
| **256 (default)** | **393,729** | **27,530,952** | **1.12312** | **+0.00206** |
| 512 | ~787K | 27,924,680 | 1.12219 | +0.00113 |

Best hidden dim is 512, but still +0.00113 BPB worse than base (1.12106). Increasing NFM capacity does not help.

> **Conclusion:** NFM consistently hurts across all configurations tested. The auxiliary parameters are better allocated to the main AR model.

## Limitations & Conclusions

1. **NFM does not improve val_bpb.** Across all configurations tested (3 loss weights × 3 hidden dims), NFM consistently hurts by +0.001 to +0.003 BPB vs the matched base. The auxiliary parameters are better spent on the main AR model.

2. **Three-seed reproducibility achieved:** No-TTT mean = 1.12252 ± 0.00151, legal TTT mean = 1.11928 ± 0.00146.

3. **Non-competitive BPB:** The best result (1.11991) is above the current leaderboard SOTA. This submission documents the NFM negative result and ablation methodology.

4. **TTT interaction:** NFM shows slightly larger TTT gains (−0.00321) than base (−0.00245), but the absolute score with TTT is still worse than base+TTT (1.11991 vs 1.11861).

5. **E2E TTT + FlowRefiner eval completed:** SLURM job 55398555 completed with legal TTT BPB = 1.12418.

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
train.log                   — Training log for seed=42 (SLURM 55342820)
final_model.int6.ptz        — Quantized model artifact (15,630,744 bytes)
supplementary/
  eval_nflow7k_legal_ttt.log                — Legal TTT eval log, seed=42 (SLURM 55375245)
  eval_nflow7k_nottt.log                    — No-TTT eval log, seed=42 (SLURM 55375246)
  eval_e2ettt_flow7k_legal_ttt.log          — E2E TTT+Flow eval, INCOMPLETE (SLURM 55375247)
  eval_e2ettt_flow7k_legal_ttt_complete.log — E2E TTT+Flow eval, COMPLETE (SLURM 55398555)
  slurm_pr940_nflow_7k.sh                   — Training SLURM script (seed=42)
  slurm_eval_nflow7k_legal_ttt.sh           — Legal TTT eval SLURM script (seed=42)
  slurm_eval_nflow7k_nottt.sh               — No-TTT eval SLURM script (seed=42)
  seed_runs/
    slurm_nflow_train_s1337.sh              — Training script (seed=1337)
    slurm_nflow_train_s2025.sh              — Training script (seed=2025)
    slurm_eval_s1337_legal_ttt.sh           — Legal TTT eval (seed=1337)
    slurm_eval_s1337_nottt.sh               — No-TTT eval (seed=1337)
    slurm_eval_s2025_legal_ttt.sh           — Legal TTT eval (seed=2025)
    slurm_eval_s2025_nottt.sh               — No-TTT eval (seed=2025)
    train_s1337.log                         — Training log (SLURM 55398556)
    train_s2025.log                         — Training log (SLURM 55398557)
```
