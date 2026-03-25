# BigramHash(4096) + Cosine EMA + Earlier QAT + LZMA-9

**val_bpb: 1.4775** (1xH100, 940 steps) | **~7.9 MB** | Non-record submission

> **Note:** This is a non-record submission tested on 1xH100 only (940 steps in 10 min). On 8xH100 (~7500 steps), performance would be significantly better. We're submitting to share our approach and get on the board.

## Results (1xH100 80GB SXM, PyTorch 2.6.0+cu126)

| Metric | Value |
|--------|-------|
| Steps (1xH100, 10 min) | 940 |
| Step avg | 638.6ms |
| Pre-quant val BPB | 1.3628 |
| Post-EMA val BPB | 1.3671 |
| Int6 roundtrip BPB | 2.2125 |
| Sliding window BPB (stride=64) | 1.9946 |
| **Legal TTT BPB** | **1.4775** |
| Artifact size | 7,896,113 bytes (49% of 16MB) |

## Changes from PR #549 (SOTA 1.1194)

### 1. BigramHash vocabulary: 2048 -> 4096
Doubles the bigram hash embedding table for richer local context. Earlier submissions showed BigramHash scaling from 1536->2048->10240 consistently helps. With our 7.9MB artifact, there's ample room.

### 2. Cosine EMA schedule: fixed 0.997 -> cosine 0.99->0.999
Instead of fixed exponential moving average decay, we use a cosine schedule that starts at 0.99 (captures more training signal early) and ramps to 0.999 (more averaging late). This trades some quantization robustness for potentially better weight averaging.

### 3. Earlier late QAT: threshold 0.15 -> 0.10
Activates quantization-aware training earlier in the warmdown phase, giving the model more steps to adapt to int6 quantization.

### 4. LZMA compression: preset 6 -> 9
Maximum LZMA compression for the quantized weights. Slower compression but smaller artifact, freeing bytes for future model expansion.

## Observations

- **Quantization gap is large** (1.37 -> 2.21 BPB). The cosine EMA schedule produces weights that don't quantize as cleanly as the fixed 0.997 decay. TTT recovers much of this (2.21 -> 1.48), but the gap suggests the cosine EMA trade-off may not be worth it without further tuning.
- **Artifact is very small** (7.9MB vs 16MB limit). This suggests we could significantly expand the model (e.g., model_dim=640 or 12+ layers) while still fitting the constraint.
- **On 8xH100**, this script would train ~7500 steps (vs 940 on 1xH100). The SOTA achieves 1.12 pre-TTT at 7180 steps, so our variant would likely land in a similar range, with TTT bringing it under 1.12.

## Development Process

This submission was developed using [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI's evolutionary code optimization framework) as part of an experiment to apply LLM-driven evolutionary search to the Parameter Golf challenge. The evolution loop used GPT-5.4 and Gemini 3 Pro as mutation operators via the OSV API, with 1xH100 proxy evaluation on RunPod.

## Training Architecture

Built on PR #549 stack (PR #414 + Parameter Banking + Parallel Muon + Legal TTT):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| **BigramHash** | **4096** (was 2048) |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | **Cosine EMA (0.99->0.999)** + SWA(every 50) |
| Quantization | GPTQ-lite int6 + **LZMA preset 9** |
| **Late QAT** | **threshold 0.10** (was 0.15) |
| Optimizer | Parameter Banking + Parallel Muon |

## Run Command

```bash
TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are embedded in the script defaults. No environment variable overrides needed.

## Credits

- **Base model + all proven techniques**: PR #549 by @abaybektursun, PR #414 by @signalrush, PR #461 by @Christopher-Lee-McClendon, PR #399 by @abaybektursun
- **LeakyReLU^2**: PR #493 by @parinzee, PR #518 by @sofiabod
- **ShinkaEvolve framework**: Sakana AI (@SakanaAI)
