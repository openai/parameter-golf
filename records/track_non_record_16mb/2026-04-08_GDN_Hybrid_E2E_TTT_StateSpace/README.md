# Non-record: GDN Hybrid — Gated DeltaNet as E2E TTT / State-Space Model — val_bpb 1.14502

**val_bpb: 1.14502** (seed 1234, 8xH100, 600s training)

| Seed | Steps | Sliding BPB | Artifact |
|-|-|-|-|
| 1234 | 3673 | 1.14502 | 13,828,304 |

## Summary

This submission replaces 8 of 10 transformer attention layers with **Gated DeltaNet (GDN)** — a linear attention mechanism based on the gated delta rule (Yang et al., ICLR 2025). GDN is mathematically equivalent to **E2E TTT-Linear with MSE loss**: each head maintains a state matrix S that is updated via one step of gradient descent on a reconstruction objective at every token. This update is baked into the forward pass and trained end-to-end, making it simultaneously a state-space model and a test-time training mechanism.

This submission targets two items from the OpenAI bounty list:
- **State-space models** — GDN is a linear RNN with gated recurrent state
- **E2E TTT** — the delta rule update S_t = α·S_{t-1}·(I - β·k_t·k_t^T) + β·v_t·k_t^T is exactly one step of SGD on L = 0.5·‖S·k - v‖², learned end-to-end during pre-training

## Architecture

- **10 layers total:** 8 GDN layers (positions 0-3, 5-7, 9) + 2 softmax attention layers (positions 4, 8)
- dim=512, 8 heads, head_dim=64, MLP 3x (LeakyReLU(0.5)²)
- GDN config: expand_v=1.0, use_short_conv=True (causal conv1d, kernel=4), mode='chunk' (chunk_size=64)
- Attention layers use RoPE (16-dim partial), QK gain=5.0, GQA 8Q/4KV
- SP8192 vocab, tied embeddings, SDClip GPTQ (int6 matrices, int8 embeddings, k=15.0)
- EMA (decay=0.997), brotli-11 compression
- 37.4M parameters, 13.83 MB artifact

## Why GDN = E2E TTT

The GDN state update per head:
```
S_t = α_t · S_{t-1} · (I - β_t · k_t · k_t^T) + β_t · v_t · k_t^T
```

This is equivalent to TTT-Linear (Sun et al. 2024) with:
- Self-supervised loss: L = 0.5 · ‖S·k - v‖²
- Gradient step: ∇_S L = k^T · (S·k - v)
- Update: S_new = S - β · ∇_S L = S · (I - β·k·k^T) + β·v·k^T
- Plus a decay gate α for memory clearing

The outer training loop backpropagates through these inner updates end-to-end, teaching the model how to adapt efficiently. At eval time, the same mechanism runs naturally — no separate TTT phase needed.

## Results and Analysis

**1.14502 BPB is not competitive with softmax attention** at this training budget. The key bottleneck is throughput: GDN achieves 4.91M tok/s on 8xH100 vs 6.93M tok/s for our softmax attention baseline, yielding 3673 steps vs 4624 steps in 600s. The 20% training deficit is not compensated by GDN's per-step learning advantage at 37M parameters.

However, GDN shows promise:
- **Training is stable** — no NaN, smooth convergence from 9.0 to 2.93 train loss
- **GPTQ quantization works** — only +0.022 BPB quant gap (comparable to softmax attention)
- **Artifact is small** — 13.83 MB, leaving 2.17 MB headroom for larger models
- PR #1370 achieved 1.003 BPB with GDN at 7000 steps (unlimited compute), suggesting the architecture is capable if given more training time

The path to competitive GDN results requires either faster Triton kernels (the FLA chunk_gated_delta_rule kernel doesn't benefit from torch.compile) or longer training budgets.

## Requirements

```bash
pip install flash-linear-attention==0.4.2 brotli sentencepiece

rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

SEED=1234 VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

| Component | Origin | Author |
|-----------|--------|--------|
| Gated DeltaNet (FLA v0.4.2) | [arXiv 2412.06464](https://arxiv.org/abs/2412.06464), [FLA library](https://github.com/fla-org/fla) | Yang et al. (NVIDIA), @sustcsonglin |
| GDN in parameter-golf | [#1370](https://github.com/openai/parameter-golf/pull/1370) (PureGDN, 1.003 BPB unlimited) | @Christopher-Lee-McClendon |
| SP8192 + SDClip + GPTQ embeddings | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| TMA fused MLP kernel | [#1450](https://github.com/openai/parameter-golf/pull/1450) | @andrewbaggio1 |
| E2E TTT-Linear equivalence | [arXiv 2407.04620](https://arxiv.org/abs/2407.04620) | Sun et al. (Stanford) |
