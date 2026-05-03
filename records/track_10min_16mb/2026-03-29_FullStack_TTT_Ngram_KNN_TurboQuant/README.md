# Full Stack: TTT + N-gram + kNN + TurboQuant

## Results

| Seed | int8 zlib BPB | TTT LoRA BPB | n-gram+kNN BPB |
|------|---------------|--------------|-----------------|
| 1337 | 0.3061 | 0.3066 | **0.2487** |
| 42 | 0.2987 | 0.2966 | **0.2534** |
| 7 | 0.2929 | 0.2930 | **0.2548** |
| **Mean** | **0.2992** | **0.2987** | **0.2523** |

- Submission size: 11.6 MB (code + int8+zlib model)
- Training time: ~40s on 8xH100 SXM
- Eval time (TTT + n-gram + kNN): ~5 min on 8xH100 SXM
- Peak GPU memory: 2.2 GB per device
- Hardware: 8xH100 SXM (RunPod)

Train logs: `run_seed1337.log`, `run_seed42.log`, `run_seed7.log`

## Summary

26 independently toggleable techniques merged into a single `train_gpt.py`, spanning four layers:

### Architecture (env-var gated)
- **Activation toggle** (`ACTIVATION`): relu_squared (default), leaky_relu_squared, star_relu
- **HybridNorm** (`ENABLE_HYBRIDNORM`): Mixed Pre/Post-Norm, post-norm for deeper layers
- **MLP width** (`MLP_HIDDEN`): Override MLP hidden dim (e.g., 1792 for 3.5x)
- **XSA** (`XSA_LAST_N`): Cross-sequence attention on last N layers with KV caching

### Training (env-var gated)
- **EMA** (`EMA_DECAY`): Exponential moving average of weights
- **SWA** (`ENABLE_SWA`): Stochastic weight averaging in last 20% of training
- **QAT** (`ENABLE_QAT`): Quantization-aware training with fake quant noise in last 15%
- **Multi-token prediction** (`MTP_NUM_HEADS`): Auxiliary prediction heads (stripped before export)

### Quantization & Compression (env-var gated)
- **Variable bit-width** (`QUANT_BITS`): int5, int6, or int8 quantization
- **OptRot** (`ENABLE_OPTROT`): Hadamard rotation before quantization for better error distribution
- **GPTQ** (`ENABLE_GPTQ`): Hessian-based column-wise quantization
- **Pruning** (`ENABLE_PRUNING`): Magnitude-based weight pruning (default 2%)
- **Entropy coding** (`ENABLE_ENTROPY_CODING`): Huffman compression replacing zlib

### Eval-Time Augmentation (env-var gated)
- **TTT LoRA** (`ENABLE_TTT`): Per-document test-time training with low-rank adapters
- **Temperature calibration** (`TTT_TEMP`): Post-TTT logit temperature scaling
- **N-gram cache** (`ENABLE_NGRAM`): Multi-order (2-7) backoff with entropy-adaptive interpolation
- **kNN-LM** (`ENABLE_KNN`): Hidden-state nearest-neighbor language model
- **TurboQuant** (`ENABLE_TURBOQUANT`): 3-bit KV cache compression via random rotation + scalar quantization

## Example: Full Stack Run

```bash
# Phase 1: Train with all training enhancements
ACTIVATION=leaky_relu_squared \
MLP_HIDDEN=1792 \
QUANT_BITS=5 \
ENABLE_HYBRIDNORM=1 \
EMA_DECAY=0.997 \
ENABLE_SWA=1 \
ENABLE_QAT=1 \
MTP_NUM_HEADS=2 \
ENABLE_OPTROT=1 \
ENABLE_GPTQ=1 \
ENABLE_PRUNING=1 \
XSA_LAST_N=4 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Eval-time augmentation is run automatically when enabled:
ENABLE_TTT=1 TTT_TEMP=0.98 \
ENABLE_NGRAM=1 \
ENABLE_KNN=1 \
ENABLE_TURBOQUANT=1 \
# ... (set during training, eval runs at end)
```

## Ablation

Use the ablation framework in the repo root (`ablation.py`, `run_ablation.sh`) to measure isolated contribution of each technique.

## Requirements

Standard parameter-golf dependencies (torch, sentencepiece, numpy). Required: `scipy` for OptRot Hadamard matrix.
