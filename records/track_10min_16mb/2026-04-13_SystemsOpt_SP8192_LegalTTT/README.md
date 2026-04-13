# Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT + Systems Optimization

**val_bpb = 1.0801** (3-seed mean, std 0.0001) | **2.7899 nats** | **~15.99 MB** | 8xH100 SXM, 600s | Legal TTT

This submission applies systems-level performance optimizations to the PR #1493 SOTA stack. The ML is unchanged; faster per-step throughput yields extra training steps in the same 600s budget.

> **Submission series:** This PR is one of three related submissions applying the same systems optimizations to different base stacks:
>
> 1. On PR #1493 (current merged SOTA) -- **this PR**
> 2. On PR #1529 (pending review)
> 3. On PR #1578 (pending review)
>
> The optimizations are identical across all three -- fused Muon kernel, batched EMA, and loader prealloc. We submit against multiple bases so that a ready-to-merge option exists regardless of how the pending PRs are resolved. Judges should feel free to evaluate whichever base(s) they consider valid and disregard the rest.

**Note on record criteria:** This submission improves speed through systems optimization without changing the ML. Per the official contest rules: *"For submissions that improve speed through systems optimization without changing the ML, this requirement [0.005 nats] is waived."* The changes (fused Muon kernel, batched EMA, superchunk eval, rank-0 serialize) are purely systems-level and do not alter model architecture, optimizer logic, loss function, or any hyperparameter.

## 3-Seed Results

| Seed | Steps | ms/step | Post-EMA BPB | Sliding BPB | **TTT BPB** | Artifact |
|------|-------|---------|-------------|-------------|-------------|----------|
| 0    | 4,607 | 127.4 | 1.0866 | 1.0815 | **1.0799** | 15,993,737 |
| 3141 | 4,622 | 127.0 | 1.0868 | 1.0817 | **1.0801** | 15,995,437 |
| 42   | 4,619 | 127.1 | 1.0869 | 1.0815 | **1.0802** | 15,993,201 |
| **Mean** | **4,616** | **127.2** | **1.0868** | **1.0816** | **1.0801** | **15,994,125** |
| **Std** | | | | | **0.0001** | |

Current merged SOTA (PR #1493): **1.0810 BPB**. Delta: **-0.0009 BPB**.

## Systems Optimizations

1. **Fused Muon transform** -- Single `@torch.compile` function combining momentum update, Nesterov extrapolation, row normalization, and Newton-Schulz orthogonalization. (+0.43% step time on 2xH100 benchmark)

2. **EMA foreach** -- Replaces per-tensor EMA loop with `torch._foreach_mul_` / `torch._foreach_add_`. (+0.08% step time)

3. **Muon prealloc + foreach apply** -- Pre-allocated flat update buffer reused across steps; `torch._foreach_mul_`/`_foreach_add_` for weight updates. (+0.07% step time)

4. **Superchunk eval** -- Contiguous copy + `torch.as_strided` overlapping views for sliding window eval, replacing per-window data loading. (+2.65% eval time)

5. **Rank-0 serialize** -- Only rank 0 performs GPTQ serialization; other ranks skip. Saves redundant work on 7 of 8 GPUs.

6. **Eval batch 128** -- Increased sliding window eval batch from 32 to 128 sequences.

No model architecture or hyperparameter changes.

## Architecture (from PR #1493)

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: loops layers 3-5 (activated at frac=0.35), 17 virtual layers from 11 physical. Parallel residuals from layer 7: GPT-J style, attention and MLP read from same input. Skip gates (sigmoid-gated U-Net connections).

## Training

Muon optimizer (flat-buffer all-reduce, Newton-Schulz 5 steps), AdamW for embeddings/scalars. ~4,616 steps in 588s. Warmdown frac=0.72, EMA decay=0.9965, WD=0.095. GPTQ reserve 12s.

## Quantization

Full-Hessian GPTQ with SDClip: int6 for attention/MLP matrices (k=12.85), int8 for token embeddings (k=20.0). Byte-shuffle + Brotli-11 compression.

## TTT (Test-Time Training)

Score-first chunk-based SGD: 32K-token chunks, 3 epochs per chunk, cosine LR decay (lr=0.005, momentum=0.9). Gradient clipping at 1.0.

## Compliance

- **Condition 1 (Causality):** Sliding-window eval is strictly causal.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab.
- **Condition 3 (Score before update):** Each chunk scored under `torch.no_grad()` before SGD.
- **Condition 4 (Single pass):** Each token scored exactly once.
- No SLOT, no pre-quant TTT, no ETLB, no n-gram cache.

## Reproducibility

```bash
pip install brotli sentencepiece flash_attn_3 huggingface_hub

# Data:
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Training (per seed):
for SEED in 0 3141 42; do
    SEED=$SEED TTT_ENABLED=1 TTT_LR=0.005 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Attribution

- **PR #1493** (@bigbag): Full SOTA stack
- **PR #1394** (@clarkkev): SP8192 tokenizer, GPTQ SDClip, depth recurrence base
- **PR #1413** (@dexhunter): Legal TTT framework
- **PR #1412** (@Robby955), **PR #1204** (@msisovic): Parallel residuals
- **PR #1445** (@X-Abhishek-X): Hyperparameter tuning
