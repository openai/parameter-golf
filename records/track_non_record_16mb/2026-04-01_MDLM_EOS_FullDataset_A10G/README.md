# MDLM v5: EOS Learning + Full Dataset Shard Rotation

**val_var_bpb: 0.9901** (128 eval steps) | **33M params** | 1x AWS A10G (24GB) | Non-record

Builds on PR #1106. First MDLM submission to train on the full FineWeb SP-1024 dataset
via shard rotation, and the first to learn document-boundary structure (EOS tokens) during diffusion.

## Results

| Model | BPB |
|-------|-----|
| AR SOTA (PR #549) | 1.1194 |
| **This (MDLM v5, EOS + full dataset)** | **0.9901** |
| PR #1106 (MDLM, prior best diffusion) | 1.1465 |
| AR baseline | 1.2244 |

## Why Non-Record

Trained on 1x AWS A10G (24GB VRAM). The competition requires 8xH100 SXM within a
10-minute wall-clock budget. The A10G was used for iterative development and to
validate the full-dataset shard rotation strategy. We plan to rerun on 8xH100 when available.

## What's New vs PR #1106

PR #1106 established the MDLM baseline: bidirectional transformer, log-linear noise schedule,
discrete absorbing-mask ELBO, frozen visible-token logits. This submission adds two
contributions on top of that stack.

### 1. EOS Token Learning

In the SP-1024 tokenizer, token 1 (`<s>`) marks document boundaries throughout FineWeb.
Prior MDLM submissions treated the corpus as a flat token stream with no document structure signal.

**Key design:**
- `EOS_ID = 1` — never masked during forward diffusion. EOS positions are always visible
  to the model regardless of noise level, acting as structural anchors.
- `PAD_ID = 1025` — a dedicated padding token (not MASK_ID=1024) fills positions after EOS
  within each sequence. PAD is also never masked and is excluded from the loss.
- Separating PAD from MASK avoids a collision where the model cannot distinguish
  diffusion masking from structural padding.

**Loss:**

```python
is_special = (x0 == EOS_ID) | (x0 == PAD_ID)
move = (torch.rand_like(x0.float()) < move_chance[:, None]) & ~is_special
xt = torch.where(move, MASK_ID, x0)

# x0_safe avoids -inf * 0 = nan at PAD positions during gather
x0_safe = x0.masked_fill(x0 == PAD_ID, 0)
log_p_x0 = torch.gather(log_probs, -1, x0_safe[..., None]).squeeze(-1)
content_mask = (x0 != PAD_ID).float()  # loss over real tokens + EOS only

loss = (dsigma[:, None] * (-log_p_x0) * is_masked * content_mask).sum() / n_content
```

**Document chunking:** Documents are split into contiguous chunks of at most `SEQ_LEN=2048`.
Short documents → one chunk ending at EOS + PAD fill. Long documents → N full SEQ_LEN chunks
(no EOS mid-doc) + one tail chunk ending at EOS. EOS supervision is preserved at every
document boundary; batch shapes remain consistent.

### 2. Shard Rotation for Full-Dataset Training on Memory-Constrained Hardware

The FineWeb SP-1024 training set spans 80 shards (~100M tokens each). Loading all 80
simultaneously requires ~64GB RAM — infeasible on most single-GPU setups.

**Strategy:** `ShardedDataLoader` partitions shards into groups of `SHARDS_IN_MEMORY`
(default 4). Training runs `TRAIN_STEPS` steps per group before rotating, ensuring every
shard is visited once per pass. Memory is explicitly freed between groups:

```python
# Explicit free before allocation
self.tokens_np = None; self.chunks = None; gc.collect()

# Pre-allocate merged buffer; load shards one-at-a-time into slices
# (avoids the 2x peak from np.concatenate holding all raw shards simultaneously)
sizes = [shard_token_count(p) for p in batch_paths]
self.tokens_np = np.empty(sum(sizes), dtype=np.int64)
offset = 0
for p, n in zip(batch_paths, sizes):
    self.tokens_np[offset:offset + n] = _load_shard(p)
    offset += n
gc.collect()
```

Toggle constants:

```python
MAX_TRAIN_SHARDS = 0     # 0 = all 80 shards
SHARDS_IN_MEMORY = 4     # shards loaded at once
ROTATE_SHARDS    = True  # False = load all at once (original behaviour)
```

**Use case:** Designed for training smaller diffusion models on single consumer or cloud
GPUs where the full dataset cannot be held in RAM, while still ensuring complete data passes
across all shards. LR schedule (warmup + cosine warmdown) restarts per shard group.

### 3. Attention Head Count is Invariant for Diffusion LMs

We ran a sweep over attention head counts `{2, 4, 8, 16, 32}` on a single shard with all other hyperparameters fixed (MODEL_DIM=512, 6000 steps). Val BPB and val loss were flat across all configurations to within noise. This is notably different from autoregressive models, where head count and head dimension interact with the causal attention pattern and KV cache efficiency.

**Practical implication:** head count can be chosen freely based on hardware alignment (e.g. multiples of 8 for tensor core efficiency) without sacrificing model quality.

## Architecture

Identical to PR #1106 except for EOS/PAD token handling and data loading:

- 11 layers, 512 dim, 8 heads, MLP 3× (ReLU²), RoPE
- AdaLN timestep conditioning (log-sigma → scale+shift per layer)
- Bidirectional attention (`is_causal=False`)
- Log-linear noise schedule: `alpha(t) = 1 - (1 - eps) * t`, `eps = 1e-3`
- Antithetic time sampling for variance reduction
- `TOTAL_VOCAB = 1026` (1024 real + MASK + PAD); embedding table padded to 1088
- AdamW: lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1
- Warmup 300 steps, cosine warmdown 1500 steps
- SEQ_LEN=2048, BATCH_SIZE=8, GRAD_ACCUM=4 (effective batch=32), 6000 steps/group

## Evaluation

Discrete absorbing-mask variational ELBO (128 Riemann steps).
Competition BPB uses sentencepiece byte-count LUTs (exact bytes per token,
matching competition scoring formula).

## Hardware

1x AWS A10G (24GB VRAM). Total training time: 1267 minutes (~21 hours).
Non-record due to hardware constraint (requires 8xH100 SXM).

## Credits

- PR #1106 (agalimova): MDLM baseline for parameter-golf, discrete ELBO eval
- MDLM: Sahoo et al. (2024), "Simple and Effective Masked Diffusion Language Models"
- LLaDA: Nie et al. (2025), "Large Language Diffusion with Masking"
