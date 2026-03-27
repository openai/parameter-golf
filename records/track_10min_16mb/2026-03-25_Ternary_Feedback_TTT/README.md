# KoopCaps-HRM Ternary Reasoner

**Author**: Aki Gogikar (OneNewAI)

Submission for the 10-minute / 16MB track of OpenAI's Parameter Golf Challenge.

## Core Thesis

**Ternary weights buy parameters. Koopman dynamics buy reasoning.**

Standard transformers do a single forward pass. The Ternary Reasoner iterates:
encode once, then run multiple decoder passes where each pass is corrected by a
compressed semantic sketch from the previous pass via Hadamard-gated adapters.

**KoopCaps-HRM** upgrades this with three first-principles innovations:
1. **Koopman capsule dynamics**: predictive state evolution (not just blending)
2. **Adaptive halting**: entropy-based early stopping per position
3. **Cross-window capsule carry**: persistent structured memory during eval

## What Makes This Different

While most submissions optimize storage (better quantization), we optimize **computation**: getting more reasoning per byte by making the model do structured iterative refinement with predictive dynamics.

### Ternary Weights → 87M Parameters in 12MB
We go to the extreme: **ternary weights {-1, 0, +1}** packed as base-3 (5 trits/byte). This gives 3-4x more parameters than int6 submissions at the same budget. The noisier per-parameter signal is compensated by structured reasoning:

### KoopCaps — Koopman Capsule Dynamics (640 new params = 1.3KB)
The capsule update across correction passes is a discrete-time dynamical system. We add diagonal + low-rank stable linear dynamics:
```
c_pred = D ⊙ c + U(V^T c)       # predict next-pass state
c_new  = α ⊙ c_observed + (1-α) ⊙ c_pred  # blend prediction with observation
```
- **D initialized at 0.9** (critical damping, ρ(D)=0.9 < 1 guaranteed stable)
- **UV^T small** (spectral perturbation << 0.1, stability at init)
- **α at sigmoid(0)=0.5** (maximum-entropy prior)
- Auxiliary consistency loss (λ=0.005) trains the dynamics to be predictive

This is **Anderson acceleration** applied to the correction loop: the model learns to predict where the fixed point is, reaching it in fewer passes.

### Adaptive Halting (0 new params)
At eval time, capsule convergence norm decides when to stop iterating:
- δ = ‖c_k - c_{k-1}‖₂ / ‖c_k‖₂
- Halt when δ < 0.05 (standard numerical convergence criterion)
- Easy tokens get 1 pass, hard tokens get 2-3

### Cross-Window Capsule Carry (0 new params)
During sliding eval, capsule state persists across windows with exponential decay (0.8):
```
carry = 0.8 · carry + 0.2 · this_window_capsules
```
This gives structured long-range memory at zero parameter cost.

## Architecture (Default: 12L/768d)

- **Ternary U-Net trunk**: 12-layer encoder-decoder with skip connections, ~87M ternary params
- **KoopCaps**: 16 capsules × 64 dims with Koopman dynamics (rank-4 D+UV^T)
- **Iterative correction**: 1 feedback pass during training, 2-3 adaptive at eval
- **Hadamard-gated feedback**: multiplicative + additive backward semantic correction
- **XSA**: Exclusive Self-Attention on last 4 layers
- **LeakyReLU(0.5)²**: proven -0.003 BPB over ReLU²
- **Partial RoPE**: 16/96 dims rotated, rest attend without position
- **VRL**: first-layer values blended into deep-layer attention (layers 10+)
- **LN Scale Damping**: 1/√(layer+1) for training stability
- **BigramHash**: 4096-bucket bigram hashing for local context
- **EMA** (decay=0.997): weight averaging for smoother quantization
- **GPTQ-lite**: per-row clip percentile search before ternary packing

## Eval Stack

- **Sliding window** (stride=64) with temperature scaling
- **Cross-window capsule carry** (decay=0.8)
- **Adaptive halting** (capsule delta < 0.05)
- **N-gram cache** (order=5, entropy-adaptive mixing)
- **Legal score-first TTT** (3 epochs, feedback scope, SGD with momentum)

## Training Configuration

- **Muon optimizer**: lr=0.025, momentum=0.95, WD=0.04, 5 Newton-Schulz steps
- **Adam for Koopman**: lr=0.025 (scalar routing), stability-clamped diagonal
- **Batch**: 786K tokens/step, seq_len=2048
- **Warmdown**: 50% of wallclock time
- **Gradient clipping**: 0.3
- **Capsule consistency loss**: λ=0.005 × CE loss
- **8xH100 SXM**: 10 minutes training, 10 minutes eval

## Run

```bash
bash setup.sh
conda activate golf
bash run_cuda_feedback.sh
```

### Key env knobs

| Variable | Default | Description |
|----------|---------|-------------|
| `KOOPMAN_ENABLED` | 1 | Enable Koopman capsule dynamics |
| `KOOPMAN_RANK` | 4 | Low-rank coupling dimension |
| `KOOPMAN_DIAG_INIT` | 0.9 | Diagonal initial value (stability) |
| `KOOPMAN_CONSISTENCY_WEIGHT` | 0.005 | Auxiliary loss weight |
| `ADAPTIVE_HALT_ENABLED` | 1 | Enable eval-time adaptive halting |
| `ADAPTIVE_HALT_THRESHOLD` | 0.05 | Relative capsule delta threshold |
| `MAX_EVAL_PASSES` | 3 | Maximum correction passes in eval |
| `CAPSULE_CARRY_ENABLED` | 1 | Cross-window capsule persistence |
| `CAPSULE_CARRY_DECAY` | 0.8 | Exponential decay for carry |
| `FEEDBACK_PASSES` | 1 | Training correction passes |
| `EVAL_FEEDBACK_PASSES` | 2 | Eval correction passes |

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design rationale.

### Ablation variants

```bash
# KoopCaps off (baseline comparison)
KOOPMAN_ENABLED=0 ADAPTIVE_HALT_ENABLED=0 CAPSULE_CARRY_ENABLED=0 bash run_cuda_feedback.sh

# Full stack minus feedback (isolate feedback contribution)
FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 bash run_cuda_feedback.sh

# Quick smoke test (1 GPU, 60s)
ITERATIONS=200 MAX_WALLCLOCK_SECONDS=60 SLIDING_EVAL=0 TEMP_SCALING=0 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 NPROC_PER_NODE=1 bash run_cuda_feedback.sh
```
