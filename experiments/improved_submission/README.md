# Improved Parameter Golf Submission

Building on the #1 SOTA (bigbag, 1.0810 BPB) with multiple improvement directions.

## Improvements Over SOTA

### 1. Mixed-Precision Per-Layer GPTQ Quantization

Instead of uniform int6 for all weight matrices, allocates bits based on Hessian trace importance:
- **int8** for the top 15% most critical matrices (highest Hessian trace)
- **int6** for the middle 60%
- **int5** for the bottom 25% (least critical)

Each tier has tuned `clip_sigmas` (25.0 / 12.85 / 8.0 respectively).

Enable: `MIXED_PRECISION_QUANT=1` (default)

### 2. LoRA-Based Test-Time Training

Injects low-rank adapters (rank 8) into Q and V projections of the last 4 layers during eval.
Far fewer trainable parameters -> faster backward pass -> more TTT epochs (6 vs 3) in the
same 600s eval budget.

Enable: `TTT_LORA_ENABLED=1 TTT_LORA_RANK=8 TTT_LORA_LAYERS=4 TTT_LORA_EPOCHS=6`

### 3. SwiGLU MLP Option

SwiGLU is used in most modern LLMs and is more expressive per parameter than LeakyReLU^2.
Uses gated activation: `SiLU(gate) * up` instead of `LeakyReLU(x, 0.5)^2`.

Enable: `USE_SWIGLU=1 SWIGLU_MLP_MULT=2.67`

### 4. Tuned Hyperparameters

- QK-Gain: 5.25 -> 5.5 (monotonic trend suggests higher is better)
- Weight decay: 0.095 -> 0.10
- EMA decay: 0.9965 -> 0.997

### 5. Better Compression Options

Added zstd support alongside brotli and lzma. Set via `COMPRESSOR=zstd`.

### 6. Configurable Deeper Recurrence

Env vars `LOOP_START`, `LOOP_END`, `NUM_LOOPS` allow testing 4-layer recurrence (L2-5)
and other configurations.

## Experiment Scripts

| Script | Purpose |
|--------|---------|
| `sweep_hp.sh` | HP sweep: QK-Gain, WD, EMA, LR, quant fractions, compressor |
| `run_deeper_recurrence.sh` | Test 4-layer recurrence configurations |
| `run_lora_ttt.sh` | Compare standard TTT vs LoRA TTT |
| `run_swiglu.sh` | Compare SwiGLU vs LeakyReLU^2 |
| `run_large_vocab.sh` | SP16384 with aggressive embedding quantization |
| `generate_sp16384.sh` | Generate SP16384 tokenizer and dataset |
| `run_final.sh` | 3-seed final evaluation with best config |

## Recommended Execution Order

1. `sweep_hp.sh` -- find optimal QK-Gain, WD, EMA, LR, quant fractions
2. `run_deeper_recurrence.sh` -- test deeper recurrence
3. `run_swiglu.sh` -- test SwiGLU
4. `run_lora_ttt.sh` -- test LoRA TTT
5. Update `run_final.sh` defaults based on sweep results
6. `run_final.sh` -- 3-seed evaluation for submission

## Requirements

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir \
    "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
```

## Quick Start (1xH100)

```bash
# Download data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
    python3 data/cached_challenge_fineweb.py --variant sp8192

# Run single seed with all improvements
NPROC=1 SEED=42 TTT_ENABLED=1 TTT_LORA_ENABLED=1 MIXED_PRECISION_QUANT=1 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py
```
