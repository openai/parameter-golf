# Record: Improved Parallel Residuals + Systems Optimization

**val_bpb = 1.0752** (3-seed mean, std 0.0006) | **2.7773 nats** | **~15.98 MB** | 8xH100 SXM, 600s | Legal TTT

This submission applies three systems-level performance optimizations to PR #1529's dual-lane parallel residual architecture. The ML is unchanged; faster per-step throughput yields ~20 extra training steps in the same 600s budget.

> **Submission series:** This PR is one of three related submissions applying the same systems optimizations to different base stacks:
>
> 1. On PR #1493 (current merged SOTA)
> 2. On PR #1529 (pending review) -- **this PR**
> 3. On PR #1578 (pending review)
>
> The optimizations are identical across all three -- fused Muon kernel, batched EMA, and loader prealloc. We submit against multiple bases so that a ready-to-merge option exists regardless of how the pending PRs are resolved. Judges should feel free to evaluate whichever base(s) they consider valid and disregard the rest.

**Note on record criteria:** This submission improves speed through systems optimization without changing the ML. Per the official contest rules: *"For submissions that improve speed through systems optimization without changing the ML, this requirement [0.005 nats] is waived."* The three changes (fused Muon kernel, batched EMA, loader prealloc) are purely systems-level and do not alter model architecture, optimizer logic, loss function, or any hyperparameter.

## 3-Seed Results

| Seed | Steps | ms/step | Post-EMA BPB | Sliding BPB | **TTT BPB** | Artifact |
|------|-------|---------|-------------|-------------|-------------|----------|
| 1337 | 4,745 | 123.8 | 1.0823 | 1.0756 | **1.0745** | 15,983,819 |
| 2024 | 4,724 | 124.3 | 1.0833 | 1.0769 | **1.0755** | 15,982,374 |
| 42   | 4,744 | 123.8 | 1.0832 | 1.0773 | **1.0755** | 15,979,637 |
| **Mean** | **4,738** | **123.9** | **1.0829** | **1.0766** | **1.0752** | **15,981,943** |
| **Std** | | | | | **0.0006** | |

PR #1529 original (same seeds): **1.0753 BPB mean**. Delta: **-0.0001 BPB** (from extra training steps).

## Systems Optimizations (3 changes, training-step only)

1. **Fused Muon transform** -- Single `@torch.compile` function combining momentum update, Nesterov extrapolation, row normalization, and Newton-Schulz orthogonalization. Eliminates kernel launch overhead between sequential operations. (+0.43% step time on 2xH100 benchmark)

2. **EMA foreach** -- Replaces per-tensor EMA loop with `torch._foreach_mul_` / `torch._foreach_add_` for batched parameter averaging. (+0.08% step time)

3. **Numpy prealloc loader** -- Pre-allocates a reusable numpy buffer for data loading instead of allocating a new `np.array` per sequence. (+0.11% step time)

No eval, serialization, or model architecture changes. The three optimizations together save ~0.5% step time, translating to ~20 extra steps over 600s.

## Architecture (from PR #1529)

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: loops layers 3-5 (activated at frac=0.35). Dual-lane parallel residuals from physical layer 8: attention and MLP write to both lanes with learned post-lambdas and residual-lambdas. Final output: mean of two lanes. Skip connections: lane0 only.

Fused Triton TMA MLP kernel + CUTLASS EVT backward for throughput.

## Training

Muon optimizer (sharded reduce-scatter + all-gather, Newton-Schulz 5 steps), AdamW for embeddings/scalars. ~4,738 steps in 587s. Warmdown frac=0.667, Muon momentum=0.97, EMA decay=0.9965. GPTQ reserve 13s.

## Quantization

Full-Hessian GPTQ with SDClip: int6 for attention/MLP matrices (k=12.85), int8 for token embeddings (k=20.0). Byte-shuffle + Brotli-11 compression.

## TTT (Test-Time Training)

Score-first chunk-based SGD: 32K-token chunks, 3 epochs per chunk, cosine LR decay (lr=0.01, momentum=0.9). Hash embedding (16384-dim bigram hash, zero-initialized, learned during TTT). Gradient clipping at 1.0.

## Compliance

- **Condition 1 (Causality):** Sliding-window eval is strictly causal.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab.
- **Condition 3 (Score before update):** Each chunk scored under `torch.no_grad()` before SGD.
- **Condition 4 (Single pass):** Each token scored exactly once.
- No SLOT, no pre-quant TTT, no ETLB, no n-gram cache.

## Reproducibility

```bash
pip install brotli sentencepiece flash_attn_3 huggingface_hub
# CUTLASS EVT build (required for full throughput):
git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
cd /opt/cutlass && git checkout 08185b9c3e90510ee2b656662ed0d53b06d28157
cd /workspace && pip install --no-build-isolation ./cutlass_evt_fusion

# Data:
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Training (per seed):
for SEED in 1337 2024 42; do
    SEED=$SEED TTT_ENABLED=1 HASH_EMBED_ENABLED=1 TTT_LR=0.01 \
    MUON_MOMENTUM=0.97 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Attribution

- **PR #1529** (@msisovic): Dual-lane parallel residual architecture, Triton fused MLP, CUTLASS EVT
- **PR #1394** (@clarkkev): SP8192 tokenizer, GPTQ SDClip, depth recurrence base
- **PR #1413** (@dexhunter): Legal TTT framework
- **PR #1445** (@X-Abhishek-X): Hyperparameter tuning
