# Asymmetric 1/10 Encoder-Decoder Split: 1.1492 Pre-Quant BPB on 8xH100

## TL;DR

One-line change to the hourglass architecture: `self.num_encoder_layers = 1` instead of `num_layers // 2`. This shifts nearly all layers to the decoder side, giving **monotonic improvement across every configuration tested**. Applied to the SOTA stack (PR #549) on 8xH100 SXM, reached **1.1492 pre-quant BPB at step 5666/9000** before the run was cut short by FA2 speed limitations and a pod crash during eval.

## The Finding

The hourglass/U-Net architecture splits layers into encoder and decoder blocks with skip connections. Every submission uses the default 50/50 split (`num_encoder_layers = num_layers // 2`). We swept the split ratio and found the decoder benefits far more from additional layers:

### Baseline Code Sweep (RTX 5090, 11 layers, 300 steps)

| Encoder/Decoder Split | int8_bpb | vs Default (5/6) |
|----------------------|----------|-------------------|
| 5/6 (default) | 1.5455 | — |
| 3/8 | 1.5421 | -0.0034 |
| 2/9 | 1.5369 | -0.0086 |
| **1/10** | **1.5298** | **-0.0157** |

Monotonic. No sign of plateau. The pattern is clear: the decoder does the heavy lifting.

### SOTA Code Validation (RTX 5090, PR #549 stack, 300 steps)

| Split | val_bpb (pre-quant) | vs Default |
|-------|---------------------|------------|
| 5/6 (default) | 1.8070 | — |
| 1/10 | 1.8034 | -0.0036 |

Improvement transfers to the fully-optimized SOTA code.

### 8xH100 SXM Run (SOTA + Asymmetric Split)

Full competition settings, all SOTA techniques enabled (EMA, SWA, QAT, Muon, XSA, BigramHash, etc.):

| Metric | Value |
|--------|-------|
| **Pre-quant val_bpb** | **1.1492** |
| Steps completed | 5666 / 9000 |
| Step avg | 105.9 ms |
| Wall clock | 600s (10 min cap hit) |
| Peak memory | 25.9 GB / 80 GB per GPU |

**Why only 5666 steps:** Flash Attention 3 (`flash_attn_interface`) is not available as a pip package. We used FA2 as fallback, which runs at 105.9ms/step vs FA3's 83.3ms/step. This cost us ~3300 training steps within the 10-minute window. The pod then crashed during the TTT + quantization eval phase, so the final int8 score was never obtained.

**Estimated full-run performance:** SOTA reaches 1.1194 at 9000 steps with the default 5/6 split. With our 1/10 split and FA3 enabling all 9000 steps, we estimate **~1.115-1.119 BPB**, which would place in the **top 3** on the leaderboard.

## Why This Was Missed

Every submission copies `num_encoder_layers = num_layers // 2` from the baseline. The symmetric split is a convention borrowed from U-Net in image segmentation, where encoder and decoder have symmetric roles. In language modeling, the decoder's autoregressive generation is the harder task — it makes sense that it benefits more from capacity.

## The Change

```python
# Before (every existing submission):
self.num_encoder_layers = num_layers // 2

# After (our finding):
self.num_encoder_layers = 1
```

That's it. One line. Zero extra parameters. Zero extra compute.

## Background Experiments

See PR #1073 for 27 systematic experiments on M4 MacBook covering deep supervision, LR tuning, batch scaling, architecture, and convergence techniques. Key findings from that work:

- Deep supervision helps at small batch (-0.05 BPB) but disappears at large batch
- LR 0.08 beats default 0.04 by -0.025 at 300 steps
- Gradient clipping gives -0.019
- EMA/SWA hurt at 300 steps but help at 9000 (consistent with leaderboard submissions)

## Request for GPU Credits

We ran out of credits ($16 spent) and H100 availability before completing a clean run. With FA3 built from source, we believe this one-line change would produce a top-3 record. Requesting credits to:

1. Build FA3 from source on H100
2. Run the asymmetric split at full 9000 steps
3. Test combined with LR tuning from our M4 experiments

## Reproduce

```bash
# Clone and setup
git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
pip install sentencepiece huggingface-hub datasets tiktoken flash-attn

# Apply one-line change to any train_gpt.py:
# self.num_encoder_layers = num_layers // 2  →  self.num_encoder_layers = 1

# Run on 8xH100
python data/cached_challenge_fineweb.py --variant sp1024
NUM_LAYERS=11 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
