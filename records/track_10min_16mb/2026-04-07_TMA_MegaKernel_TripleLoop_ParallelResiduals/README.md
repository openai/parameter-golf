# Record: TMA Megakernel + Triple Loop + Parallel Residuals + SP8192 — val_bpb 1.08480

**val_bpb: 1.08480** (5-seed mean, std=0.0007)

| Seed | Steps | **Sliding BPB** | Artifact |
|-|-|-|-|
| 1234 | 4624 | **1.08359** | 15,750,778 |
| 1    | 4613 | **1.08512** | 15,750,312 |
| 42   | 4612 | **1.08530** | 15,751,243 |
| 1337 | 4614 | **1.08448** | 15,747,957 |
| 2025 | 4614 | **1.08553** | 15,752,675 |
| **Mean** | **4615** | **1.08480** | **15,750,593** |

## Summary

This submission combines frontier training techniques with a custom Triton TMA megakernel that fuses the MLP forward pass on Hopper GPUs. The fused kernel avoids materializing the 2048-wide MLP intermediate activation (~384MB), yielding **+10.5% training throughput** and **+127 extra training steps** in the 600s budget. Together with triple depth recurrence and GPT-J parallel residuals, this achieves 1.08480 BPB — a 0.030 improvement over the current merged SOTA.

## Changes

This script builds on [#1394](https://github.com/openai/parameter-golf/pull/1394) (clarkkev's SP8192 + depth recurrence + SDClip baseline). The main additions are:

### Megakernel: Triton TMA Fused MLP Forward

The MLP forward pass — `proj(leaky_relu(fc(x), 0.5)^2)` — normally materializes a `(batch*seq, 2048)` intermediate activation in HBM. Our Triton kernel fuses `fc → leaky_relu → square` into a single kernel using Hopper TMA (Tensor Memory Accelerator) descriptors for async global-to-shared memory transfers.

Key implementation details:
- **TMA descriptors** via `TensorDescriptor.from_tensor()` for hardware-managed address generation
- **Persistent kernel** — one program per SM, loops over output tiles via `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)`
- **Interleaved writes** — splits output into two halves for better memory throughput
- **Block sizes 128×256×64** optimized for H100 SXM
- **Dual output** — writes both `leaky_relu(h)^2` (for `proj`) and `2*leaky_relu(h)*leaky_relu'(h)` (for backward) in the same kernel pass
- Falls back to unfused `torch.compile` path during evaluation

Measured throughput improvement: **+10.5%** across all training phases (pre-loop and post-loop), translating to ~127 extra training steps in 600s.

### Triple Depth Recurrence (NUM_LOOPS=3)

Layers 4-5 are executed 4 times total (1 base + 3 loops), creating 17 virtual layers from 11 physical. The encoder path is `[0,1,2,3,4,5,4,5]` and decoder is `[4,5,4,5,6,7,8,9,10]`. Loop activation at training fraction 0.35 (vs 0.50 in #1394). This follows the approach from [#1420](https://github.com/openai/parameter-golf/pull/1420).

### Parallel Residuals (GPT-J Style, Layers 7-10)

Attention and MLP both read from the same pre-residual input, with outputs summed in parallel. This follows [Wang & Komatsuzaki, 2021](https://github.com/kingoflolz/mesh-transformer-jax) as applied in [#1420](https://github.com/openai/parameter-golf/pull/1420). Provides a small throughput boost from the faster forward pass.

### Inherited from #1394

- SP8192 vocabulary with tied embeddings
- GPTQ-quantized embeddings (int8, k=20) and weight matrices (int6, k=13.5)
- SDClip: `clip = k * std(row)` for principled quantization clipping
- Row-normalized Muon (MuonEq-R) from [#1217](https://github.com/openai/parameter-golf/pull/1217)
- ShuffledSequenceLoader
- EMA (decay=0.997)
- Brotli-11 + byte shuffle compression
- Skip connections with learnable gates

## Legality Audit

Audited against all four conditions from [#1017](https://github.com/openai/parameter-golf/issues/1017):

1. **Causal dependence**: The model scores position `t` using only tokens `x_1..x_{t-1}`. The sliding window evaluation uses prior context positions but never future tokens. No eval-time adaptation or state modification.
2. **Full normalized distribution**: Cross-entropy loss is computed over the full vocabulary (8192 tokens). Softmax normalization is implicit in `F.cross_entropy`. No auxiliary distributions or blending.
3. **Score-before-update**: No test-time training or state updates during evaluation. The model is frozen after GPTQ quantization.
4. **Single left-to-right pass**: Sliding window evaluates each token exactly once (context positions provide context but are not re-scored).

No TTT, no SLOT, no n-gram caching, no multi-pass scoring.

## Requirements

- 8xH100 80GB SXM
- PyTorch with Triton 3.5+ (for TMA descriptor support)
- Flash Attention 3 (Hopper)
- SentencePiece, Brotli

```bash
# Download SP8192 data from clarkkev's HuggingFace
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

# Run training
SEED=1234 \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 train_gpt_mega.py
```

## Credits

| Component | Origin | Author |
|-----------|--------|--------|
| **This PR** | | |
| Triton TMA fused MLP forward kernel | This work, inspired by [#1420](https://github.com/openai/parameter-golf/pull/1420) | @andrewbaggio1 |
| Triple depth recurrence (NUM_LOOPS=3) | [#1420](https://github.com/openai/parameter-golf/pull/1420) | @abaybektursun |
| Earlier loop activation (0.35) | [#1420](https://github.com/openai/parameter-golf/pull/1420) | @abaybektursun |
| Parallel residuals (GPT-J, layers 7+) | [#1420](https://github.com/openai/parameter-golf/pull/1420), [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) | @abaybektursun, @kingoflolz |
| **Architecture** | | |
| SP8192 vocabulary | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| SDClip quantization (c = k*std) | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| GPTQ on embeddings (int8) | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| MuonEq-R (row-normalized Muon) | [#1217](https://github.com/openai/parameter-golf/pull/1217) | @bigbag |
| Depth recurrence (loop layers 4-5) | [#1204](https://github.com/openai/parameter-golf/pull/1204) | @msisovic |
| U-Net sigmoid-gated skip connections | [#289](https://github.com/openai/parameter-golf/pull/289), [#1089](https://github.com/openai/parameter-golf/pull/1089) | @integrate-your-mind, @mikeapedia |
| XSA on all layers | [#265](https://github.com/openai/parameter-golf/pull/265), [#478](https://github.com/openai/parameter-golf/pull/478) | @unnir, @gowtham0992 |
| Partial RoPE (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) | @jfprincz |
| LeakyReLU(0.5)^2 activation | [#185](https://github.com/openai/parameter-golf/pull/185) | @dttdrv |
| **Optimizer** | | |
| Muon (Newton-Schulz) | Baseline + [#399](https://github.com/openai/parameter-golf/pull/399) | @abaybektursun |
| EMA (decay=0.997) | [#315](https://github.com/openai/parameter-golf/pull/315) | @jfprincz |
| **Quantization & Compression** | | |
| Full Hessian GPTQ (actorder + Cholesky) | [#535](https://github.com/openai/parameter-golf/pull/535), [#1060](https://github.com/openai/parameter-golf/pull/1060) | @raahilshah, @dexhunter |
| Brotli-11 + byte shuffle | [#1089](https://github.com/openai/parameter-golf/pull/1089) | @mikeapedia |
| **Evaluation** | | |
| Sliding window (stride=64) | [#122](https://github.com/openai/parameter-golf/pull/122) | @mtybadger |
| Flash Attention 3 (Hopper) | [#122](https://github.com/openai/parameter-golf/pull/122) | @mtybadger |
| **Data** | | |
| ShuffledSequenceLoader | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| SP8192 tokenizer + datasets (HuggingFace) | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
