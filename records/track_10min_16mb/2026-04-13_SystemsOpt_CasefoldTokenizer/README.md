# Record: Custom Casefold Tokenizer + Parallel Residuals + Systems Optimization

**val_bpb = 1.0639** (3-seed mean, std 0.0006) | **3.0705 nats** | **~15.98 MB** | 8xH100 SXM, 600s | Legal TTT

This submission applies systems-level performance optimizations to PR #1578's casefold tokenizer + PR #1529's parallel residual architecture. The ML is unchanged; faster per-step throughput yields extra training steps in the same 600s budget.

> **Submission series:** This PR is one of three related submissions applying the same systems optimizations to different base stacks:
>
> 1. On PR #1493 (current merged SOTA)
> 2. On PR #1529 (pending review)
> 3. On PR #1578 (pending review) -- **this PR**
>
> The optimizations are identical across all three -- fused Muon kernel, batched EMA, and loader prealloc. We submit against multiple bases so that a ready-to-merge option exists regardless of how the pending PRs are resolved. Judges should feel free to evaluate whichever base(s) they consider valid and disregard the rest.

**Note on record criteria:** Per the official contest rules: *"For submissions that improve speed through systems optimization without changing the ML, this requirement [0.005 nats] is waived."* The three changes are purely systems-level. That said, this submission also clears the 0.005-nat threshold outright (0.0083 nats vs PR #1578).

## 3-Seed Results

| Seed | Steps | ms/step | Post-EMA BPB | **TTT BPB** | Artifact |
|------|-------|---------|-------------|-------------|----------|
| 1337 | 4,716 | 124.5 | 1.0709 | **1.0646** | 15,985,530 |
| 2024 | 4,731 | 124.1 | 1.0697 | **1.0634** | 15,980,244 |
| 42   | 4,726 | 124.3 | 1.0701 | **1.0639** | 15,982,918 |
| **Mean** | **4,724** | **124.3** | **1.0702** | **1.0639** | **15,982,897** |
| **Std** | | | | **0.0006** | |

PR #1578 original (same seeds): **1.0668 BPB mean**. Delta: **-0.0029 BPB** / **-0.0083 nats**.

## Systems Optimizations (3 changes, training-step only)

1. **Fused Muon transform** -- Single `@torch.compile` function combining momentum update, Nesterov extrapolation, row normalization, and Newton-Schulz orthogonalization. Eliminates kernel launch overhead between sequential operations.

2. **EMA foreach** -- Replaces per-tensor EMA loop with `torch._foreach_mul_` / `torch._foreach_add_` for batched parameter averaging.

3. **Numpy prealloc loader** -- Pre-allocates a reusable numpy buffer for data loading instead of allocating a new `np.array` per sequence.

No eval, serialization, or model architecture changes.

## What Changed vs PR #1578 (Only Systems Optimization)

The **only difference** from PR #1578 is the three systems optimizations listed above. Architecture, optimizer logic, hyperparameters, tokenizer, dataset, TTT, and quantization are all identical. The casefold v2 vocabulary and retokenized dataset are unchanged from PR #1578.

## Architecture (from PR #1529)

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: loops layers 3-5 (activated at frac=0.35). Dual-lane parallel residuals from physical layer 8. Fused Triton TMA MLP kernel + CUTLASS EVT backward.

## Tokenizer (from PR #1578)

Casefold v2 vocabulary: SP8192 retrained on NFKC + lowercased text. 374 freed case-duplicate slots refilled with BPP-optimized subwords. ~10.4% better compression. Byte counting verified correct on 15.4M FineWeb docs (0 mismatches). See `CASEFOLD_TOKENIZER.md` and `verify_bytes.py`.

## Training

Muon optimizer (sharded reduce-scatter + all-gather, Newton-Schulz 5 steps), AdamW for embeddings/scalars. ~4,724 steps in 587s. Warmdown frac=0.72, Muon momentum=0.97, EMA decay=0.997. GPTQ reserve 13s.

## TTT (Test-Time Training)

Score-first chunk-based SGD: 32K-token chunks, 3 epochs per chunk, cosine LR decay (lr=0.005, momentum=0.9). Hash embedding (16384-dim bigram hash, zero-initialized, learned during TTT). Gradient clipping at 1.0.

## Compliance

- **Condition 1 (Causality):** Sliding-window eval is strictly causal.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab.
- **Condition 3 (Score before update):** Each chunk scored under `torch.no_grad()` before SGD.
- **Condition 4 (Single pass):** Each token scored exactly once.
- No SLOT, no pre-quant TTT, no ETLB, no n-gram cache.

## Reproducibility

```bash
pip install brotli sentencepiece flash_attn_3 huggingface_hub

# CUTLASS EVT build:
git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
cd /opt/cutlass && git checkout 08185b9c3e90510ee2b656662ed0d53b06d28157
cd /workspace && pip install --no-build-isolation ./cutlass_evt_fusion

# Casefold data (from HuggingFace):
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Mikeapedia/fineweb10B-sp8192-casefold-v2', repo_type='dataset', local_dir='data/datasets/fineweb10B_sp8192_casefold_v2', allow_patterns='*.bin')"

# Training (per seed):
for SEED in 1337 2024 42; do
    SEED=$SEED TTT_ENABLED=1 HASH_EMBED_ENABLED=1 TTT_LR=0.005 \
    MUON_MOMENTUM=0.97 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 \
    EMA_DECAY=0.997 WARMDOWN_FRAC=0.72 \
    DATASETS_DIR=./data/datasets/fineweb10B_sp8192_casefold_v2 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_casefold_refined_v2.model \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Attribution

- **PR #1578** (@mikeapedia): Casefold v2 vocabulary and retokenized dataset
- **PR #1529** (@msisovic): Dual-lane parallel residual architecture, Triton fused MLP, CUTLASS EVT
- **PR #1394** (@clarkkev): SP8192 tokenizer, GPTQ SDClip
- **PR #1413** (@dexhunter): Legal TTT framework
