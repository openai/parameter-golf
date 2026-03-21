# Depth Recurrence + Mixed-Precision Quantization for 16MB Parameter Golf

**Non-record submission** | val_bpb: 2.4402 (3-seed mean, post-quant) | Pre-quant: 2.0711 | Artifact: ~1.5 MB | We still ran 3x runs for verification though cause I'm not giving you guys only 1 sample cmon now.
Also tldr: this is basically a test to see if we can do a different architecture than all the leaderboard runs are doing right now. It kinda worked and we figured out why some people trying this earlier got errors.

## Who I Am

I'm a high school student with no formal ML background. I used Claude (sorry guys I already had the subscription q-q) to help me understand all of this and debug implementation. This submission represents about 12 hours of intensive work, starting from zero knowledge of transformer training. (also itsmeaura on discord in the #parameter-golf-discussions channel if anyone wants to laugh at a flaw I made with this)

## Core Idea

Instead of training 9-11 independent transformer layers like every other submission, I share weights across a small set of unique blocks and cycle through them multiple times. This gives more effective depth for fewer stored parameters, leaving headroom for wider layers or better compression.

4 unique transformer blocks × 3 cycles = 12 effective layers of depth, stored in the parameter budget of 4.

This approach is inspired by Relaxed Recursive Transformers (arXiv:2410.20672), Huginn (arXiv:2502.05171), MobileLLM's block-wise sharing (arXiv:2402.14905), and Samsung's Tiny Recursive Models.

## Key Finding: Quantization Error Amplifies Through Recurrence

We (Claude & I) also managed to figure out why depth recurrence has failed for other competitors (PR #212 got catastrophic 4.34 bpb, PR #213 got 1.60 bpb).

Recurrence amplifies quantization error by approximately 900× over 3 cycles. When the same slightly-wrong quantized weights are applied 3 times in sequence, errors compound multiplicatively through the residual stream. Both int6 and int8 suffer equally in relative amplification (~896×), but int6 starts with 4× more absolute error per weight, making it 4× worse after cycling.

This means:
- Int6 quantization (used by all top submissions) is incompatible with depth recurrence unless the error is managed
- Int8 for shared/recycled weights + Int6 for single-use tensors is the correct mixed-precision strategy for recurrent architectures

In summary, I believe this explains why PR #212's Huginn approach catastrophically failed. As they probably used standard quantization without accounting for error amplification. (like I did on my first 8xH100 run)

This interaction between recurrence and quantization has not been documented in the competition or (to my knowledge) in the published literature on recursive transformers.

## Architecture

```
Input → Embedding (tied, fp16) → BigramHash (4096 buckets, 128d)
     → [Block 0 → Block 1 → Block 2 → Block 3] × 3 cycles (12 effective layers)
     → XSA (last 4 virtual layers) → Output Head
```

Each block contains:
- Multi-head attention (8 heads, 4 KV heads, GQA)
- 3× MLP (hidden dim = 3 × model_dim)
- RMSNorm, RoPE, residual connections

Additional components:
- BigramHash: Hashes consecutive token pairs into 4096 buckets with 128-dim embeddings. Adds bigram context for ~590K extra parameters. Contributed -0.20 bpb in our experiments, the single most impactful addition.
- XSA (Exclusive Self Attention): Zero-parameter technique from PR #287. Removes self-value bias via orthogonal projection on the last 4 virtual layers. ~0.005 bpb improvement.
- LoRA adapters (rank 4): Per-virtual-layer adaptation allowing each cycle through a shared block to specialize slightly. 129K extra parameters total.

## Compression Strategy

Standard pipeline: int8 quantization for shared block weights + zstd-22 compression. We deliberately use int8 (not int6) for recycled weights to minimize the amplified quantization error through recurrence cycles.

The artifact is significantly under the 16MB cap, reflecting a tradeoff: recurrence saves parameters but requires higher precision, so the parameter savings are partially offset by the larger per-parameter storage.

## Research Process

Can not forget to acknowledge all the people who have done work in the PRs allowing me to jump way ahead and not have to spend as much time debugging. Thanks guys. (shoutout techniques from PRs #76, #77, #208, #213, #236, #287, #288, #297 specifically!)

Key papers that influenced the design:
- Relaxed Recursive Transformers (Google DeepMind, ICLR 2025) — LoRA adapters for layer specialization in recursive models
- MobileLLM (Meta, ICML 2024) — deep-and-thin beats wide-and-shallow at small scale
- Mixture-of-Recursions (NeurIPS 2025) — adaptive depth per token with weight sharing
- MiniCPM (OpenBMB) — WSD learning rate schedule, 192× data-to-model ratio at small scale
- Simplified Transformer Blocks (ETH Zurich, ICLR 2024) — removing components without quality loss

## Experimental Results

### Technique Ablations (A4500, ~170 steps, no torch.compile)

| Config | Params | val_bpb | vs Control | Notes |
|--------|--------|---------|------------|-------|
| Baseline (9L unique, 512d) | 17.1M | 2.2409 | — | Reference |
| Recurrent 3×3, NoLoRA | 6.0M | 2.2894 | -0.049 gap | 65% fewer params |
| Recurrent 3×3, LoRA=4 | 6.1M | 2.3168 | +0.076 | LoRA hurts at low step count |
| + BigramHash | 6.4M | 2.1373 | -0.205 | Huge win |
| + SmearGate | 6.1M | 2.4167 | +0.075 | Hurts with recurrence |
| + SmearGate + BigramHash | 6.4M | 2.1735 | -0.169 | SmearGate drags BigramHash down |
| **Best: Rec 3×3 + BigramHash** | **6.4M** | **2.0981** | **-0.244** | **Best overall** |

### Quantization Error Amplification (measured)

Simulated with a 512×512 weight matrix passed through 3 recurrence cycles:

| Quantization | Error per weight (1 cycle) | After 3 cycles | Amplification factor |
|-------------|---------------------------|----------------|---------------------|
| Int8 | 0.133 | 119.6 | 896× |
| Int6 | 0.545 | 488.8 | 896× |

Observed bpb gaps on 8×H100 SXM (seed 1337):

| Quantization | Pre-quant bpb | Post-quant bpb | Gap |
|-------------|---------------|----------------|-----|
| Int6 (all tensors) | 2.0723 | 3.2168 | **1.144** |
| Int8 (shared blocks) | 2.0730 | 2.3889 | **0.316** |

### 8×H100 SXM Runs (3-seed validation)

| Seed | Steps | Pre-quant bpb | Post-quant bpb (int8) | Quant gap |
|------|-------|---------------|----------------------|-----------|
| 1337 | 2908 | 2.0730 | 2.3889 | 0.316 |
| 42 | 2967 | 2.0650 | 2.3876 | 0.323 |
| 7 | 2963 | 2.0753 | 2.5440 | 0.469 |
| **Mean** | **2946** | **2.0711** | **2.4402** | **0.369** |
| **Std** | | **0.0054** | | |

- 22.8M parameters, 4 unique blocks × 3 cycles = 12 effective depth
- 768d model, 3× MLP, BigramHash 4096×128, XSA on last 4 layers, LoRA rank 4
- ~195ms/step on 8×H100 SXM with torch.compile, ~2950 steps in 600s
- Late STE QAT activated at 85% wallclock (~step 2615)
- Artifact: ~1.5 MB with int8+zstd-22

Note: We initially ran with int6 quantization (matching competition standard) and got a catastrophic 1.14 bpb gap (2.07 → 3.22). Switching shared block weights to int8 reduced the gap to ~0.37 bpb. The remaining gap is from the ~900× error amplification through recurrence. This is the fundamental tradeoff: recurrence saves parameters but requires higher-precision quantization. Seed 7's larger gap may indicate sensitivity to weight initialization in the recurrence pathway, I'll leave that for someone else though.

## Acknowledgements

- Thanks for the compute credits OpenAI! Maybe this is cool enough for the larger grant??? *wink wink nudge nudge* Hey I'll even take another round of the 25$ I'm not picky, I just cant afford to fund too many 8xH100 runs on my waitress salary lmao. Hoping this quantization find helps out the ~~competition~~ other wonderful people in this competition! (aka PLEASSSEE GIVE ME MORE CREDITS)
- PRs #76, #77, #208, #213, #236, #287, #288, #297 again for letting me not have to debug as much.
- Runpod for the A4500 since my 3070 can only handle so much before we needed more vram.
- Claude (Anthropic) for research assistance, code review, and helping me understand the ML concepts involved. (Listen I can't realistically justify the 200$/mo subscription for gpt sorry guys)
- The authors of Relaxed Recursive Transformers, Huginn, MobileLLM, and BitNet whose published work made this approach possible

## Files

| File | Description |
|------|-------------|
| `train_gpt.py` | Self-contained training script with recurrence, BigramHash, XSA, LoRA, mixed-precision quantization |
| `train.log` | Training log from 8×H100 SXM run |
| `submission.json` | Competition metadata |
| `README.md` | This file |
| `requirements.txt` | External dependencies (zstandard) |
