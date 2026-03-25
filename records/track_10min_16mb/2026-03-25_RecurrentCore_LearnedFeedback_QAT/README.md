# Recurrent Core from Current Best Record

Adds a shared recurrent core with quantization-aware training and error-feedback correction on top of the current best 10-min / 16 MB Parameter Golf record.

## Preserved from the current best record

| Component | Detail |
|-----------|--------|
| Activation | LeakyReLU(0.5)^2 |
| BigramHash | 1536 |
| XSA | Last 4 unique layers |
| Partial RoPE | 16 / 64 dims |
| LayerNorm scaling | 1 / sqrt(layer + 1) |
| VE128 | Configurable unique-layer indices |
| Weight averaging | EMA(0.997) + tight SWA(every 50) |
| Export path | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |
| TTT | Legal score-first, SGD+momentum, 32K chunks |

## Borrowed conceptually from PR #363

PR #363 demonstrated that a 4-block × 3-cycle looped architecture suffered ~900× quantization error amplification (2.0711 → 2.4402 BPB post-quant). Its "noisy QAT" experiment showed the gap could be largely removed.

From PR #363 we take:

- The dynamical-systems framing: with shared quantized weights W_q = W + ε, perturbation grows as ‖J‖^k · ‖εh_0‖.
- The delta-sigma / error-feedback idea: approximate the quantization residual and inject a compensation term.
- The conclusion that full-rollout QAT is necessary but may not be sufficient alone.

## What was changed to make recurrence quantization-stable

### Architecture: stem / recurrent core / tail

Instead of converting the full 11-layer stack, the model partitions into:

- **Stem** (default 3 unique layers) — early processing, collects U-Net skips
- **Recurrent core** (default 2 shared layers × K passes) — middle depth via weight reuse
- **Tail** (default 3 unique layers) — late refinement, consumes skips

Banks store weights for `num_unique = stem + core + tail` layers. Core bank entries are reused across K recurrence passes.

### Full-rollout QAT with STE

Core bank weights are fake-quantized (symmetric int6, per-row, STE) on every forward pass during training. Loss is computed only after the final recurrence pass. Stem and tail weights are not fake-quantized by default.

### Error feedback

Quantization residual is approximated as a low-rank branch:

    e_k = U (V^T h_k),     U, V in R^{d × r},  r in {1, 2, 4}

Three correction variants:

| Script | Correction | Parameters added |
|--------|-----------|-----------------|
| `train_bestbase_recurrent_qat.py` | None (QAT only) | 0 |
| `train_bestbase_recurrent_feedback_fixed.py` | Identity or shared diagonal | Very small |
| `train_bestbase_recurrent_feedback_learned.py` | Learned diagonal/low-rank, per-pass option, optional affine junction | Small |

### Stabilizers

- Optional hidden-state clipping (value or norm)
- Optional learnable per-pass residual scaling
- Optional Jacobian spectral-norm proxy penalty

### Recurrence-safe TTT

Five regimes controlling which parameters adapt at test time:

| Regime | Adapts |
|--------|--------|
| `tail_only` | Tail blocks only (safest) |
| `tail_plus_stem` | Stem + tail, core frozen |
| `all_unique_layers` | All blocks at full LR |
| `all_layers` | Alias for all_unique |
| `all_layers_with_recurrent_lr_scale` | Core at reduced LR (e.g. 0.1×) |

## File structure

```
records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT/
├── model_recurrent_bestbase.py    # RecurrentGPT model
├── quant.py                       # Fake quantization (STE) + export
├── feedback.py                    # Error feedback modules
├── stability.py                   # Diagnostics, clipping, Jacobian proxy
├── ttt_recurrent.py               # Recurrence-aware TTT
├── train_utils_recurrent.py       # Hyperparameters, Muon, data, eval, export
├── train_bestbase_recurrent_qat.py            # Script 1: QAT only
├── train_bestbase_recurrent_feedback_fixed.py # Script 2: fixed feedback
├── train_bestbase_recurrent_feedback_learned.py # Script 3: learned feedback
├── smoke_test.sh                  # 1-GPU correctness check
├── submission.json
└── README.md
```

## Quick start

```bash
# Script 3 (learned feedback) — the main experimental target
NUM_STEM_LAYERS=3 NUM_CORE_LAYERS=2 NUM_TAIL_LAYERS=3 NUM_PASSES=3 \
CORE_QUANT_BITS=6 CORE_QUANT_ENABLED=1 \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=6,7 \
EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  train_bestbase_recurrent_feedback_learned.py \
  --feedback-mode diagonal --feedback-rank 2 --ttt-regime tail_only
```

## Experimental plan

| Experiment | Script | Key question |
|-----------|--------|-------------|
| A | qat | Does QAT alone fix recurrence? |
| B | fixed | Does a tiny correction path help further? |
| C | learned | Does learned feedback beat fixed at same budget? |
| D | learned + TTT | Which TTT regime is safest for shared weights? |
| E | learned + stabilizers | Do clipping / scaling / Jacobian penalty help? |
