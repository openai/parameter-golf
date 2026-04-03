# Spectral Flood Walk LM — V1 Staged Design

## Summary

`v1` is no longer a single monolithic jump from `v0` to "fill all 640GB." The revised plan is a staged program that isolates the biggest unknowns first:

1. `V1a`: small transformer + static semantic memory only
2. `V1b`: add same-GPU episodic memory growth
3. `V1c`: add routed multi-GPU episodic memory only if `V1b` earns it

This changes the question from:

> Can one giant hybrid architecture saturate the entire machine immediately?

to:

> Which memory mechanism actually buys `val_bpb`, under a real 16MB artifact budget, before we spend 8xH100 time on routing and giant HBM pools?

## Why The Old V1 Draft Needed Revision

The earlier draft was directionally interesting but overcommitted on a few numbers:

- The artifact budget for an `8`-layer `d=512` transformer was too optimistic by about an order of magnitude.
- The bucket occupancy math for the routed episodic pool did not line up with the stated bucket count and write volume.
- The Zobrist routing section stated semantic locality as a property, when it is still a hypothesis.
- The document mixed three questions together:
  - does semantic memory help?
  - does episodic memory help?
  - does cross-GPU routing help?

Those should be tested in that order, not assumed together.

## Rules We Optimize For

- `16MB` artifact
- `600s` train on `8x H100 SXM`
- `600s` eval on `8x H100 SXM`
- full normalized distribution over the whole vocab
- score-before-update
- single left-to-right pass

References:

- [README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/README.md#L6)
- [README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/README.md#L184)

## Core V1 Thesis

The broad design thesis still stands:

1. The learned artifact should remain small.
2. Runtime memory should do more work than current leaderboard models allow themselves to do.
3. The first memory object worth betting on is not "reconstructed hidden state."
4. Static semantic memory and dynamic episodic memory should be treated as separate levers.

What changes is the implementation discipline:

- no claim that we should immediately fill all available HBM
- no claim that every semantic-memory byte is justified before the baseline wins
- no claim that routing is free or inherently helpful

## Stage Breakdown

### V1a — Semantic Memory Only

`V1a` replaces the recurrent `v0` controller with a small transformer and tests only one new mechanism:

> does static semantic memory improve next-token prediction under a realistic artifact budget?

Architecture:

- small causal transformer controller
- selected FFN blocks replaced by product-key memory layers
- no episodic pool
- no online append
- no cross-GPU routing

This is now implemented in [spectral_flood_walk_v1.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v1.py).

### V1b — Local Episodic Memory

If `V1a` helps, then `V1b` adds:

- append-only episodic memory on the same GPU
- score-after-write legality preserved
- fixed local bucket scan
- no cross-GPU routing yet

This stage answers:

> does growing same-stream memory help before we pay any distributed complexity?

### V1c — Routed Episodic Memory

Only if `V1b` wins do we add:

- GPU routing
- bucket routing
- NVLink point-to-point retrieval

This stage answers:

> does routing across GPUs buy enough to justify the added system complexity?

## V1a Architecture

The current `V1a` implementation is intentionally narrower than the original grand design.

Controller:

- causal transformer
- learned absolute position embeddings
- tied embedding / LM head
- standard self-attention blocks

Semantic memory:

- product-key FFN replacement in selected layers
- exact lookup via top-`k` on two factored sub-tables
- compressed codes expanded through a learned projection
- no pre-expanded HBM-filling table yet

This keeps `V1a` focused on whether semantic lookup helps at all. If it does, we can separately test whether fully expanded semantic tables are worth the runtime memory.

## Corrected Artifact Math

The previous draft budgeted `5` standard transformer layers at `2.5MB` total. That is not realistic for a conventional `d=512` transformer block.

For rough planning, the right scale is:

```text
attention weights per layer  ~= 4 * d^2
ffn weights per layer        ~= 2 * d * (ff_mult * d)
```

At `d=512`, `ff_mult=4`:

```text
attention ~= 1.05M params
ffn       ~= 2.10M params
block     ~= 3.15M params
```

Even at `int8`, that is already about `3.15MB` per standard block before metadata, code bytes, or semantic tables.

So the original budget table was not a runnable spec.

## Sizing Worksheet

The active sizing calculator now lives in:

- [spectral_flood_walk_v1.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v1.py) via `estimate_v1a_sizes(...)`
- [tools/spectral_flood_walk_v1_sizing.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/tools/spectral_flood_walk_v1_sizing.py)

Current worksheet outputs:

- compact model byte estimate
- expanded semantic-memory byte estimate
- semantic layer count
- product-key entry count per layer
- compact code bytes per semantic layer

Example:

```bash
python3 tools/spectral_flood_walk_v1_sizing.py \
  --vocab-size 1024 \
  --embed-dim 256 \
  --num-layers 6 \
  --semantic-layers 2,4 \
  --pk-num-subkeys 64 \
  --pk-key-dim 16 \
  --pk-code-dim 64 \
  --json
```

This is the source of truth for `V1a` sizing until we have a byte-exact exported artifact shape for the new path.

## Immediate Experiment Ladder

These are the experiments worth running on the pod, in order.

### 1. Baseline Transformer

Purpose:

- establish the small-transformer floor
- measure actual train/eval speed
- verify artifact size and `val_bpb`

Command shape:

```bash
python3 spectral_flood_walk_v1.py \
  --use-semantic-memory false \
  --semantic-layers '' \
  --embed-dim 256 \
  --num-layers 6 \
  --num-heads 8 \
  --seq-len 128
```

### 2. One Semantic Layer

Purpose:

- test whether semantic lookup helps at all
- keep the added memory small enough that training remains easy to read

Command shape:

```bash
python3 spectral_flood_walk_v1.py \
  --semantic-layers 2 \
  --pk-num-subkeys 64 \
  --pk-code-dim 64 \
  --pk-topk-sub 4 \
  --pk-topk-final 8
```

### 3. Two Semantic Layers

Purpose:

- test whether semantic memory compounds or just adds latency

Command shape:

```bash
python3 spectral_flood_walk_v1.py \
  --semantic-layers 2,4 \
  --pk-num-subkeys 64 \
  --pk-code-dim 64
```

### 4. Entry-Count Sweep

Purpose:

- determine whether more product-key entries help enough to justify artifact and VRAM

Sweep:

- `pk_num_subkeys`: `32`, `64`, `96`, `128`

Since the effective table is `num_subkeys^2`, this is the fastest way to explore capacity without changing the controller.

### 5. Code-Size Sweep

Purpose:

- determine whether semantic values are undercompressed or overcompressed

Sweep:

- `pk_code_dim`: `32`, `64`, `96`, `128`

### 6. 1024 vs 8192 Tokenizer Budget Check

Purpose:

- determine whether `8192` BPE is actually affordable once the controller and semantic tables are sized honestly

This should start as a worksheet exercise, not a full training run.

## Go / No-Go Gates

Move from `V1a` to `V1b` only if:

1. The best semantic-memory run beats the no-semantic baseline on `val_bpb`.
2. The exported artifact is still plausibly inside `16MB`.
3. The train/eval throughput remains sane enough for a 600-second run.

Move from `V1b` to `V1c` only if:

1. same-GPU episodic memory helps
2. local bucket scans are cheap enough
3. the write representation is actually predictive

## What We Are Explicitly Not Doing Yet

- TTL or eviction policy design
- TTT / LoRA adaptation
- bucket routing on the assumption that it must help
- giant fully expanded semantic tables just to light up VRAM

Those are later-stage optimizations. First we need one memory mechanism that clearly earns its keep.

## Open Questions

1. Is semantic memory useful enough to justify the artifact tradeoff at all?
2. Does `8192` BPE remain viable once the controller and memory are budgeted honestly?
3. If semantic memory wins, should the next increment be:
   - larger static tables
   - same-GPU episodic memory
   - or a different semantic value representation?
