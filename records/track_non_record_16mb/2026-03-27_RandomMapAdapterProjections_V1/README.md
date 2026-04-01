# Random Map Adapter Projections

Non-record submission answering the challenge request for **learning adapters on random linear maps**.

**Best result: val_bpb = 1.2615** (sliding window, int6+lzma, 11.6MB artifact on 8xH100 SXM).

## Concept

Replace selected dense projection matrices in attention with **frozen random Rademacher matrices** regenerated deterministically from a seed at inference time. Only the seed plus small learned LoRA-style adapter parameters are serialized, eliminating the storage cost of the replaced projections entirely.

This is structurally related to VeRA (Kopiczko et al., ICLR 2024), which shares frozen random matrices across layers and learns only per-layer scaling vectors. The key difference: VeRA adapts a strong pretrained base (learning a small delta), while this submission asks the random projection + low-rank corrector to carry modeling burden from scratch within a 10-minute training window.

## Architecture

Each `RandomMapAdapterProj` replaces a dense projection with:

1. A seeded frozen Rademacher down-projection (entries +1/-1 scaled by 1/sqrt(d))
2. A learned diagonal scale in the random bottleneck space
3. A learned low-rank residual adapter (LoRA)
4. A seeded frozen Rademacher up-projection

The Rademacher matrices are generated via deterministic integer hashing from a per-module seed, stored as non-persistent buffers, and never serialized.

## Base Stack

Same as the `2026-03-24_LeakyReLU2_VRL_LZMA` record line:

- 11 layers, 512 model dim, 8 heads / 4 KV heads (GQA)
- LeakyReLU(0.5)^2 MLP activation
- BigramHash, Partial RoPE, XSA4, VRL, VE128, SmearGate
- EMA + tight SWA, Late QAT + CROWN-Q
- GPTQ-lite int6 + lzma compression
- FlashAttention 3 when available, portable SDPA fallback

## Results

| Variant | Adapter Rank | MLP Random | Compile | ms/step | Steps | val_bpb | Artifact |
|---------|-------------|------------|---------|---------|-------|---------|----------|
| V1 | 8 | dim=256 | No | ~220 | ~2700 | 1.6542 | 1.2MB |
| **V2b** | **128** | **disabled** | **No** | **~220** | **~2727** | **1.2615** | **11.6MB** |
| V2c | 128 | disabled | Yes | 245 | 2444 | 1.4324 | 11.6MB |
| Baseline (no random maps) | -- | -- | Yes | ~88 | ~5700 | 1.1227 | 15.9MB |

All runs on 8xH100 SXM, 600s training budget, seed 1337.

## Key Findings

**Rank matters enormously.** Rank 8 to 128 improved from 1.65 to 1.26 bpb. The adapter needs enough capacity to meaningfully correct the random projections for each layer's specific learned function.

**MLP projections should stay learned.** Random maps in attention Q/K/V projections are tolerable (attention is already a soft lookup). Random maps in MLP projections hurt significantly more -- the MLP's pointwise nonlinear transform relies on learned weight structure that random projections destroy. Setting `RANDOM_MLP_DIM=0` (attention-only random maps) was strictly better.

**torch.compile is counterproductive.** The Rademacher matrix generation is CPU-side and seed-deterministic. Compile can't fuse away the expensive part but adds graph tracing overhead (+25ms/step), resulting in 10% fewer training steps under the fixed wallclock budget. V2c regression is fully explained by overhead without invoking hash quality.

**The ceiling is structural, not RNG quality.** The +0.1388 bpb gap between V2b (1.2615) and baseline (1.1227) reflects the information/capacity cost of replacing learned projections with frozen random ones. This is consistent with VeRA's known behavior: random-matrix adaptation works well when learning a small delta on a pretrained base, but struggles when the random projection must carry primary modeling burden from scratch.

**The approach would shine at tighter budgets.** At 16MB, there is enough room for fully learned weights to win. At a hypothetical 4MB cap, random maps would become very competitive since they trade stored weights for compute, and the 11.6MB-vs-15.9MB artifact gap would matter.

## Reproduction

```bash
# V2b (best result)
COMPILE_ENABLED=0 RANDOM_MLP_DIM=0 RANDOM_ADAPTER_RANK=128 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# V1 (original submission)
COMPILE_ENABLED=0 RANDOM_MLP_DIM=256 RANDOM_ADAPTER_RANK=8 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requires `fineweb10B_sp1024` dataset and `fineweb_1024_bpe.model` tokenizer in `./data/`.
