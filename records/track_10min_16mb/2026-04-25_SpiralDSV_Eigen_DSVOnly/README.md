# 2026-04-25 — SpiralDSV + Eigen: DSV-Only Architecture

> **BPB formula audit (2026-04-25):** `build_token_byte_arrays()` in
> [`_semantic_layer.py`](_semantic_layer.py) was updated to exactly match the
> official competition formula. Key fixes:
> 1. `is_boundary_token` initialised to **all True** (not zeros) — matches
>    `is_boundary_token_np = np.ones(...)` in every official `train_gpt.py`.
> 2. Byte tokens (`sp.is_byte()`) handled explicitly: `base_bytes=1`, remain
>    boundary, `continue` — matches the official `if sp.is_byte: ... continue` branch.
> 3. Normal word-piece tokens clear `is_boundary_token[tok_id] = False`.
> 4. Byte-count formula uses the exact official pattern:
>    `tok_bytes = base_bytes[tgt] + (has_leading_space[tgt] & ~is_boundary_token[prev])`

**Track:** `10min_16mb`  
**Date:** 2026-04-25  
**Author:** Ashley Klimpel (`viasky657`)

---

## Summary

Complete replacement of the 2026-04-07 NMF + DSV pipeline with a **DSV-only** system.

All NMF phases (0–5, 8–9) are removed. The freed 16 MB budget is reallocated to the
DSV tables, increasing hypervector dimensionality from 1,024 bits to **65,536 bits**
(a 64× increase). The XOR-bundle confidence resolution scales as `O(sqrt(n_bits))`,
giving an **8× improvement in signal-to-noise ratio**.

---

## Architecture

### Before (2026-04-07 pipeline)

```
tokens (500M)
    │
    ├─ Phase 2: freq tabulation → (TABLE_SIZE, vocab) freq table
    ├─ Phase 4: XOR orbit regularisation
    ├─ Phase 5: NMF 1-iter → embed (TABLE_SIZE, 16) fp16   [16 MB]
    │                       W_out (16, 1024) fp16
    │
    └─ Phase 6: DirectionalSemanticVec scatter-XOR
                → sem_fwd (1024, 16) uint64  [128 KB]
                → sem_bwd (1024, 16) uint64  [128 KB]

Eval:
    bucket filled + fp match → embed @ W_out  (NMF, ~random)
    collision               → sem_fwd XOR codebook  (DSV)
    miss                    → sem_fwd lag blend      (DSV)
```

### After (this submission — DSV-only)

```
tokens (500M)
    │
    └─ Phase 6 only: EigenTrainer.build_bilateral_from_tokens()
                     with GoldenAxisShift per-lag codebook rotation
                     + PMI centering
                     → sem_fwd (1024, 1024) uint64  [8 MB]
                     → sem_bwd (1024, 1024) uint64  [8 MB]
                     [Total: 16 MB — full budget used by DSV]

Eval (all positions → single DSV path):
    ALL positions → SpiralDSVLanguageModel.vote_scores_all_vocab()
                    prev-only: sem_fwd_pm1[prev] @ codebook_pm1.T
                    → (batch, 1024) float32 scores
                    → p_correct = scores[range(B), tgt_toks]
                    → BPB = Σ(-log2 p) / Σ(utf8_bytes)
```

---

## What Was Removed

| Component | Removed? | Reason |
|---|---|---|
| `precompute_g_states()` | ✅ Yes | Only needed for NMF bucket hashing |
| `tabulate_bucket_frequencies()` | ✅ Yes | Only needed for NMF freq table |
| `xor_orbit_regularise()` | ✅ Yes | Only needed for NMF freq smoothing |
| `nmf_kl_fit()` | ✅ Yes | 1-iter NMF ≈ random, marginal contribution |
| `build_frozen_prior()` | ✅ Yes | Only needed for NMF regularisation |
| `embed` array (TABLE_SIZE × EMBED_DIM) | ✅ Yes | 16 MB freed for DSV |
| `W_out` array (EMBED_DIM × VOCAB_SIZE) | ✅ Yes | Freed for DSV |
| `fingerprint_packed` array | ✅ Yes | Only needed for NMF collision detection |
| Hadamard codebook (vocab × 16 uint64) | ✅ Yes | Replaced by SpiralDSV internal codebook |
| `_transition_codebook.py` | ✅ Yes | Only used by suffix grammar + old codebook |
| Phases 0, 1, 2, 3, 4, 5, 8, 9 | ✅ Yes | All NMF phases |
| `_bidi_hdc_engine.py` | ✅ Yes | BiDirHDC engine not needed for DSV-only |
| `_bidi_train.py` | ✅ Yes | BiDirHDC training not needed |
| `_safety_oxytocin.py` | ✅ Yes | ARC-AGI-3 steering not needed for LM |

## What Was Kept

| Component | Kept? | Reason |
|---|---|---|
| `_spiral_dsv_lm.py` | ✅ Yes | Core DSV + GoldenAxisShift |
| `_eigen_convergence.py` | ✅ Yes | `EigenTrainer.build_bilateral_from_tokens()` |
| `_gpu.py` | ✅ Yes | GPU matmul acceleration |
| Phase 6 (DSV build) | ✅ Yes | Primary prediction mechanism |
| `.hgz` artifact format | ✅ Yes | Modified to HGZ3 (sem_fwd + sem_bwd only) |
| BPB formula | ✅ Yes | Unchanged |
| Distributed token loading | ✅ Yes | All 8 ranks load tokens for DSV scan |

---

## Budget Reallocation

| Component | Old size | New size |
|---|---|---|
| `embed` (TABLE_SIZE × EMBED_DIM × 2B) | 16 MB | **0 MB** — removed |
| `W_out` (EMBED_DIM × VOCAB_SIZE × 2B) | ~32 KB | **0 MB** — removed |
| `sem_fwd` (vocab × n_words × 8B) | 128 KB | **8 MB** — expanded |
| `sem_bwd` (vocab × n_words × 8B) | 128 KB | **8 MB** — expanded |
| **Total** | ~16.3 MB | **≤ 16 MB** ✅ |

With `n_words=1024`: each token gets a **65,536-bit** hypervector (vs 1,024 bits before).

---

## n_words Scaling

| n_words | n_bits | sem_fwd+bwd | CB_composite (ctx=4) | pm1 caches (×3) | Suitable for |
|---|---|---|---|---|---|
| 16 | 1,024 | 256 KB | 64 MB | 12 MB | Any GPU / CPU |
| 128 | 8,192 | 2 MB | 512 MB | 96 MB | RTX 4090 |
| 256 | 16,384 | 4 MB | 1 GB | 192 MB | RTX 4090 (tight) |
| **1024** | **65,536** | **16 MB** | **4 GB** | **768 MB** | **H100 SXM** |

---

## Artifact Format (HGZ3)

```
Magic(4B "HGZ3") + vocab_size(4B) + n_words(4B) + flags(4B)
+ sem_fwd bytes  (vocab_size × n_words × 8)   [8 MB]
+ sem_bwd bytes  (vocab_size × n_words × 8)   [8 MB]
[Total uncompressed: 16 MB]
[LZMA9 compressed: ~2–4 MB — DSV tables compress well due to structure]
```

The `SpiralDSVLanguageModel` internal codebook (vocab × n_words uint64) is **not**
stored in the artifact — it is regenerated deterministically from `seed=42` at eval
time. This keeps the artifact within the 16 MB limit.

---

## Coherence Gating (Optional)

A running document centroid that biases predictions toward tokens coherent with the
document seen so far. Enabled by `W_COHERENCE > 0.0` (default: 0.3).

```python
# Per-token coherence-augmented query:
coh_norm = coherence_pm1 / doc_token_count
h_star   = sign(sem_fwd_pm1[prev] + W_COHERENCE * coh_norm)
scores   = h_star @ codebook_pm1.T / n_bits
```

- **Artifact storage:** Zero additional bytes — `coherence_pm1` is a runtime variable
- **Compute:** One `(n_bits,)` float32 vector addition per token during eval
- **Reset:** On `is_boundary_token` document boundaries

---

## Usage

### Leaderboard run (8×H100 SXM)

```bash
# Run 1 (seed 42):
RUN_ID=spiral_dsv N_WORDS=1024 SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run 2 (seed 7):
RUN_ID=spiral_dsv N_WORDS=1024 SEED=7 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run 3 (seed 1337):
RUN_ID=spiral_dsv N_WORDS=1024 SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Convenience — run all 3 seeds sequentially

# Set for 8×H100s in RunPod as a copy-and-paste from the workspace directory in the terminal.
```bash
cd /workspace/parameter-golf-hdc-main/records && for seed in 42 7 1337; do
  echo "=== Starting seed $seed ===" && \
  RUN_ID=spiral_dsv N_WORDS=1024 SEED=$seed \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 \
      records/track_10min_16mb/2026-04-25_SpiralDSV_Eigen_DSVOnly/train_gpt.py \
      2>&1 | tee records/track_10min_16mb/2026-04-25_SpiralDSV_Eigen_DSVOnly/train_seed${seed}.log && \
  echo "=== Completed seed $seed ==="
done

```

### Local smoke test (single GPU, RTX 4090)

```bash
N_WORDS=128 SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
python train_gpt.py
```

### CPU smoke test

```bash
N_WORDS=16 SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
python train_gpt.py
```

---

## Files

| File | Description |
|---|---|
| `train_gpt.py` | Main entry point — DSV-only pipeline |
| `_semantic_layer.py` | `build_spiral_dsv()`, `eval_spiral_dsv_bpb()`, `save/load_spiral_dsv_artifact()` |
| `_spiral_dsv_lm.py` | `SpiralDSVLanguageModel`, `GoldenAxisShift`, `GOLDEN_AXES` |
| `_eigen_convergence.py` | `EigenTrainer.build_bilateral_from_tokens()` |
| `_gpu.py` | GPU acceleration helpers (cuBLAS HGEMM, scatter_add) |
| `submission.json` | Auto-updated by `train_gpt.py` after each run |
| `requirements.txt` | Python dependencies |

---

## Results

| Seed | val_bpb | val_loss | elapsed_s | artifact_bytes |
|---|---|---|---|---|
| 42 | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD |
| 1337 | TBD | TBD | TBD | TBD |

*Results to be filled after leaderboard runs.*
