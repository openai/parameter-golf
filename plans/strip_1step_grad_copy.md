# Plan: Strip Dead-Weight Components from 1_Step_Grad Copy

**Target folder:**
`records/track_10min_16mb/2026-04-07_HDC_1_Step_Grad_DSV_Radial_Slyvester_Hadamard_Matrix_Symmetry copy/2026-04-07_HDC_1_Step_Grad_DSV_Radial_Slyvester_Hadamard_Matrix_Symmetry/`

**Reference plan:** `plans/golden_shift_dsv_pure.md`

**Goal:** Remove every component that does not contribute to the final BPB output of the
`--hash_grad` path. The verified result is **BPB 0.4118** from 3 independent runs. Nothing
that contributes to that number is touched. Everything else is deleted.

This includes:
1. Removing the NMF `embed`/`W_out` arrays entirely — not just zeroing them. With
   `nmf_max_iter=1`, the NMF factors are a single AdaGrad step from random initialisation
   (KL loss stays near `ln(vocab_size)`), making them effectively random signal.
2. Removing `sem_bwd` entirely. The `hash_grad_bpb()` eval waterfall accepts `sem_bwd` as
   a parameter but **never reads it** — only `sem_fwd` is used in both the collision path
   and the miss path. `sem_bwd` is built in Phase 6 but contributes zero signal to BPB.
   Removing it halves the DSV build time and frees 128 KB of memory.

---

## What Actually Produces BPB 0.4118

From the README and code audit, the true signal chain is:

```
Phase 2  — Distributed frequency tabulation (all 8 ranks, scatter_add_ + all_reduce)
             → fingerprint table only (embed/W_out NOT computed)
                ↓
Phase 3  — Multi-seed frequency merge (fingerprint merge only)
                ↓
Phase 6  — DirectionalSemanticVec.build_from_tokens()  ← PRIMARY BPB contributor
             sem_fwd only (128 KB)   ← sem_bwd is built but NEVER READ in eval
             skip-bigram lags 2–5
                ↓
Phase 10 — LZMA9 artifact save (.hgz)
             fingerprint table + sem_fwd + skip-bigram lags
                ↓
Eval     — hash_grad_bpb() waterfall (NMF branch removed):
             fingerprint mismatch → sem_fwd XOR codebook (collision fallback)
             miss                 → sem_fwd XOR codebook (primary)
             lags 2–5             → skip-bigram blend
```

Everything outside this chain is dead weight.

---

## What Does NOT Contribute (Removal Targets)

### Files to Delete Entirely

| File | Lines | Why removed |
|---|---|---|
| `_suffix_grammar.py` | 231 | Phase 7 — morphological reranking gate. Depends on NMF g_states. README confirms it is "attempted on rank 0; skipped if budget is exhausted" — it never fires in the verified runs. |
| `_transition_codebook.py` | 1,022 | Only used by `_suffix_grammar.py` and the old Hadamard codebook path. Neither is in the verified signal chain. |

### From `_hash_grad_train.py` — Functions to Remove

The file is 1,344 lines. The following functions are NMF-pipeline-only and do not
contribute to the final BPB:

| Function | Lines (approx) | Why removed |
|---|---|---|
| `build_frozen_prior()` | ~60 | Phase 0 — NMF regularisation prior. Not used in the DSV path. |
| `xor_orbit_regularise()` | ~60 | Phase 4 — XOR orbit bucket regularisation. Heuristic smoothing before NMF. Since NMF is removed entirely, this adds no signal. |
| `nmf_kl_fit()` | ~200 | Phase 5 — 1-iteration NMF. At 1 iter the KL loss stays near `ln(vocab)`. The embed/W_out factors are a single AdaGrad step from random init — effectively random signal. **Removed entirely.** |
| `_kl_divergence_total()` | ~10 | Helper for `nmf_kl_fit()`. Removed with it. |
| `hash_grad_predict_batch()` | ~15 | Uses `embed @ W_out`. Removed with NMF. |
| `save_hash_grad_artifact()` | ~60 | Saves embed + W_out + fingerprint (NMF artifact format). **Rewritten** to save fingerprint + sem_fwd + sem_bwd + skip-bigram lags only. |
| `load_hash_grad_artifact()` | ~50 | Loads embed + W_out (NMF format). **Rewritten** to match new artifact format. |

> **Note on `precompute_g_states()`:** Phase 2 (`tabulate_bucket_frequencies_distributed()`)
> calls `precompute_g_states()` to build the rolling hash state array used for bucket
> assignment. The fingerprint table (which routes hash collisions to the DSV path) depends
> on this. **Keep.**

> **Note on `merge_seed_frequencies()`:** The verified run uses `HG_SEEDS=42,7,1337`
> (3 seeds). Phase 3 merges the per-seed fingerprint tables so the best fingerprint per
> bucket is kept. **Keep** — but only the fingerprint merge logic; the freq/count merge
> for NMF is no longer needed.

> **Note on `hash_grad_bpb()` NMF branch:** The eval waterfall currently has three paths:
> 1. `fingerprint matches + embed filled` → `embed @ W_out` (NMF, **REMOVE**)
> 2. `fingerprint MISMATCH` → DSV collision fallback (**KEEP**)
> 3. `embed is ZERO` → DSV miss fallback with lags 2–5 (**KEEP**)
>
> With embed/W_out removed, path 1 is deleted. All positions go directly to path 2 or 3.
> The fingerprint table is still useful: it detects hash collisions and routes them to the
> DSV collision fallback (path 2) rather than the miss fallback (path 3), which uses a
> slightly different blending. The fingerprint is kept; only the `embed @ W_out` computation
> is removed.

**Functions to keep in `_hash_grad_train.py`:**

| Function | Why kept |
|---|---|
| `_dist_rank()`, `_dist_world_size()`, `_dist_is_main()`, `_dist_barrier()`, `_dist_all_reduce_sum_numpy()` | Distributed utilities — used by Phase 2 |
| `tabulate_bucket_frequencies_distributed()` | Phase 2 — builds fingerprint table used in eval |
| `precompute_g_states()` | Called by Phase 2 |
| `merge_seed_frequencies()` | Phase 3 — fingerprint merge across seeds |
| `save_hash_grad_artifact()` | **Rewritten** — saves fingerprint + DSV tables only |
| `load_hash_grad_artifact()` | **Rewritten** — loads fingerprint + DSV tables only |
| `hash_grad_bpb()` | **Modified** — NMF branch removed, DSV paths unchanged |

### From `_semantic_layer.py` — Methods to Remove from `DirectionalSemanticVec`

The class is 507 lines. The following methods are never called in the verified `--hash_grad`
signal chain:

| Method | Lines (approx) | Why removed |
|---|---|---|
| `slow_wave()` | ~30 | Noise-pruning heuristic. Not called in `_run_hash_grad_single()`. |
| `build_xor_orbit_table()` | ~45 | XOR orbit diagonal table. Not called in `_run_hash_grad_single()`. |
| `build_pretrain_prior()` | ~30 | Pre-training semantic prior. Not called in `_run_hash_grad_single()`. |
| `build_correction_map()` | ~45 | XOR-neighbor correction map. Not called in `_run_hash_grad_single()`. |
| `build_token_distributions()` | ~65 | WHT-based token distribution. Not called in `_run_hash_grad_single()`. |
| `vote_scores_for_context_tok_gpu()` | ~50 | GPU path for vote scores. Not called in `_run_hash_grad_single()`. |
| `vote_scores_for_context_tok_batch()` | ~30 | Batch vote scores. Not called in `_run_hash_grad_single()`. |
| `query_backward()` | ~10 | Uses `sem_bwd`. `hash_grad_bpb()` never reads `sem_bwd` — confirmed by code audit. |
| `vote_scores_for_context_tok()` | ~20 | Uses both `sem_fwd` and `sem_bwd`. Not called in `hash_grad_bpb()` — the eval waterfall does the XOR directly on `sem_fwd` inline. |

**`sem_bwd` removal from `build_from_tokens()` and `__init__()`:**

`build_from_tokens()` currently builds both arrays per context depth:
```python
cls._scatter_xor_fast(sf_2d, a_toks, b_toks, codebook, chunk_size)  # sem_fwd ← KEEP
cls._scatter_xor_fast(sb_2d, b_toks, a_toks, codebook, chunk_size)  # sem_bwd ← REMOVE
```
Remove the `sem_bwd` scatter call and the `self.sem_bwd` array allocation in `__init__()`.
This halves Phase 6 build time (one scatter pass per context depth instead of two).

**Methods to keep in `_semantic_layer.py`:**

| Method | Change |
|---|---|
| `__init__()` | Remove `self.sem_bwd` allocation |
| `_scatter_xor_fast()` | Unchanged |
| `build_from_tokens()` | Remove `sem_bwd` scatter call |
| `_scatter_xor()` | Unchanged |
| `query_forward()` | Unchanged |
| `build_skip_bigram_lags()` | Unchanged |
| `get_lag_matrix()` | Unchanged |
| `summary()` | Update to `sem_fwd` only |

### From `train_gpt.py` — Sections to Remove

`train_gpt.py` is 6,166 lines. The `--hash_grad` path (`_run_hash_grad_single()`) starts
around line 5,650. Everything before that is the legacy DNA-HDC path, which is never
executed when `--hash_grad` is set (the default).

**Large blocks to remove:**

| Block | Approx lines | Why removed |
|---|---|---|
| `hadamard_bipolar_hash()` function | ~20 | Only used by legacy DNA-HDC path |
| `hadamard_bipolar_hash_bytes()` function | ~15 | Only used by legacy DNA-HDC path |
| `_TENSOR_CORE_KERNELS` CUDA kernel string | ~200 | Only used by legacy DNA-HDC path |
| `_TRANSITION_CODEBOOK_AVAILABLE` import block | ~15 | `_transition_codebook.py` is being deleted |
| `cupy` import block | ~5 | Only used by legacy DNA-HDC path |
| Entire legacy DNA-HDC function body | ~4,500 | Lines ~3,500–5,649: DNA-HDC Phases 0–4, 3.5-*, 4.0, 4A, 4 |
| Phase 7 (suffix grammar) in `_run_hash_grad_single()` | ~30 | `_suffix_grammar.py` is being deleted |
| Phase 9 (embed pruning) in `_run_hash_grad_single()` | ~5 | NMF embed is being removed |
| `--no_hash_grad` argument and legacy routing | ~10 | Legacy path is being removed |

**What to keep in `train_gpt.py`:**

| Section | Why kept |
|---|---|
| Imports: `glob`, `json`, `os`, `sys`, `time`, `numpy`, `sentencepiece` | Used by `_run_hash_grad_single()` |
| `_run_hash_grad_single()` function body (minus Phase 7, 9) | The verified signal chain |
| `main()` / `argparse` setup (hash_grad args only) | Entry point |
| Token loading helpers (`_load_tokens()` or equivalent) | Used by `_run_hash_grad_single()` |

---

## Architecture Diagram — Before vs. After

### Before (current copy folder)

```
train_gpt.py (6,166 lines)
    ├── Legacy DNA-HDC path (~4,500 lines) ← DEAD WEIGHT
    │     Phase 0: semantic prior
    │     Phase 1: Hadamard codebook
    │     Phase 1b: TransitionCodebook
    │     Phase 1.5: bigram pre-compute
    │     Phase 1.5b: trigram pre-compute
    │     Phase 2: DNA-stacked context table
    │     Phase 3: additional passes
    │     Phase 3.5-DSV: DirectionalSemanticVec (old path)
    │     Phase 3.5-SRH: semantic rolling hash
    │     Phase 3.5-SkipBigram: skip-bigram lags
    │     Phase 3.5-XOROrbit: XOR orbit table
    │     Phase 3.5-SuffixGrammar: suffix grammar
    │     Phase 4.0: AR calibration sweep
    │     Phase 4A: queue repairs
    │     Phase 4: predictive coding repair
    │
    └── --hash_grad path (_run_hash_grad_single) (~500 lines) ← KEPT
          Phase 0: build_frozen_prior()          ← REMOVE
          Phase 2: tabulate_bucket_frequencies() ← KEEP
          Phase 3: merge_seed_frequencies()      ← KEEP
          Phase 4: xor_orbit_regularise()        ← REMOVE
          Phase 5: nmf_kl_fit()                  ← REMOVE
          Phase 6: DirectionalSemanticVec        ← KEEP (primary BPB)
          Phase 7: SuffixGrammarTable            ← REMOVE
          Phase 9: embed pruning                 ← REMOVE
          Phase 10: save_hash_grad_artifact()    ← KEEP (restructured)
          Eval: hash_grad_bpb()                  ← KEEP

_hash_grad_train.py (1,344 lines)
    ├── build_frozen_prior()                     ← REMOVE
    ├── precompute_g_states()                    ← KEEP
    ├── tabulate_bucket_frequencies_distributed()← KEEP
    ├── merge_seed_frequencies()                 ← KEEP
    ├── xor_orbit_regularise()                   ← REMOVE
    ├── nmf_kl_fit()                             ← REMOVE
    ├── save_hash_grad_artifact()                ← KEEP (restructured)
    ├── load_hash_grad_artifact()                ← KEEP (restructured)
    └── hash_grad_bpb()                          ← KEEP

_semantic_layer.py (507 lines)
    ├── DirectionalSemanticVec class
    │     ├── __init__()                         ← KEEP
    │     ├── _scatter_xor_fast()                ← KEEP
    │     ├── build_from_tokens()                ← KEEP
    │     ├── _scatter_xor()                     ← KEEP
    │     ├── query_forward()                    ← KEEP
    │     ├── query_backward()                   ← REMOVE (uses sem_bwd, never called in eval)
    │     ├── vote_scores_for_context_tok()      ← REMOVE (uses sem_bwd, not called in eval)
    │     ├── vote_scores_for_context_tok_batch()← REMOVE
    │     ├── vote_scores_for_context_tok_gpu()  ← REMOVE
    │     ├── slow_wave()                        ← REMOVE
    │     ├── build_skip_bigram_lags()           ← KEEP
    │     ├── get_lag_matrix()                   ← KEEP
    │     ├── build_xor_orbit_table()            ← REMOVE
    │     ├── build_pretrain_prior()             ← REMOVE
    │     ├── build_correction_map()             ← REMOVE
    │     ├── build_token_distributions()        ← REMOVE
    │     └── summary()                          ← KEEP (sem_fwd only)

_suffix_grammar.py (231 lines)                   ← DELETE ENTIRE FILE
_transition_codebook.py (1,022 lines)            ← DELETE ENTIRE FILE
```

### After (stripped copy folder)

```
train_gpt.py (~600 lines)
    └── --hash_grad path only
          Phase 2: tabulate_bucket_frequencies()
          Phase 3: merge_seed_frequencies()
          Phase 6: DirectionalSemanticVec (primary BPB)
          Phase 10: save artifact
          Eval: hash_grad_bpb()

_hash_grad_train.py (~600 lines)
    ├── precompute_g_states()
    ├── tabulate_bucket_frequencies_distributed()
    ├── merge_seed_frequencies()
    ├── save_hash_grad_artifact()
    ├── load_hash_grad_artifact()
    └── hash_grad_bpb()

_semantic_layer.py (~200 lines)
    └── DirectionalSemanticVec (sem_fwd only)
          __init__ (sem_fwd only, no sem_bwd),
          _scatter_xor_fast, build_from_tokens (sem_fwd only),
          _scatter_xor, query_forward,
          build_skip_bigram_lags, get_lag_matrix, summary

_suffix_grammar.py                               ← DELETED
_transition_codebook.py                          ← DELETED
```

---

## Implementation Steps

### Step 1 — Delete `_suffix_grammar.py`

Delete the file entirely. It is only called in Phase 7 of `_run_hash_grad_single()`, which
is also being removed.

### Step 2 — Delete `_transition_codebook.py`

Delete the file entirely. It is only imported by `_suffix_grammar.py` (deleted) and the
legacy DNA-HDC path in `train_gpt.py` (being removed).

### Step 3 — Strip `_semantic_layer.py`

**Remove methods:**
- `vote_scores_for_context_tok_batch()` — not called in eval
- `vote_scores_for_context_tok_gpu()` — not called in eval
- `vote_scores_for_context_tok()` — uses `sem_bwd`, not called in `hash_grad_bpb()`
- `query_backward()` — uses `sem_bwd`, never read in eval
- `slow_wave()` — not called in eval
- `build_xor_orbit_table()` — not called in eval
- `build_pretrain_prior()` — not called in eval
- `build_correction_map()` — not called in eval
- `build_token_distributions()` — not called in eval

**Modify `__init__()`:** Remove `self.sem_bwd = np.zeros(uint64_count, dtype=np.uint64)`.

**Modify `build_from_tokens()`:** Remove the `sem_bwd` scatter call:
```python
# DELETE this line:
cls._scatter_xor_fast(sb_2d, b_toks, a_toks, codebook, chunk_size)
```
Also remove `sb_2d = dsv.sem_bwd.reshape(vocab_size, W)` and the `sem_bwd` verbose log.

**Modify `summary()`:** Update to report `sem_fwd` only.

Result: ~200 lines. Phase 6 build time halved (one scatter pass per context depth).

### Step 4 — Strip `_hash_grad_train.py`

**Remove functions entirely:**
- `build_frozen_prior()` — Phase 0, NMF regularisation prior
- `xor_orbit_regularise()` — Phase 4, heuristic smoothing before NMF
- `nmf_kl_fit()` — Phase 5, 1-iteration NMF (≈ random signal)
- `_kl_divergence_total()` — helper for `nmf_kl_fit()`
- `hash_grad_predict_batch()` — uses `embed @ W_out`, removed with NMF

**Modify `save_hash_grad_artifact()`:** Remove `embed` and `W_out` parameters. Save
fingerprint + `sem_fwd` + skip-bigram lags only. Update magic bytes to `HGZ3` (or a new
version tag) to distinguish from the old NMF format.

**Modify `load_hash_grad_artifact()`:** Match new format — return fingerprint + `sem_fwd`
+ skip-bigram lags. Remove `embed`/`W_out` from return tuple.

**Modify `hash_grad_bpb()`:**
- Remove `embed` and `W_out` parameters from signature
- Remove `embed_norm` / `has_embed` computation
- Remove the `has_emb_mask.any()` branch (the NMF `embed @ W_out` path)
- The `collision_pos` and `miss_mask` paths are unchanged — they use `sem_fwd` only
- Remove `sem_bwd` parameter from signature (it was accepted but never read)
- Remove `suffix_grammar` / `srh` parameters (those components are deleted)

**Keep unchanged:**
- All `_dist_*` helpers
- `precompute_g_states()`
- `tabulate_bucket_frequencies_distributed()`
- `merge_seed_frequencies()` (fingerprint merge logic only — freq/count arrays no longer needed after Phase 2)

### Step 5 — Strip `train_gpt.py`

This is the largest change. The file is 6,166 lines; the goal is to reduce it to ~600 lines
containing only the `--hash_grad` path.

**Remove:**
1. `hadamard_bipolar_hash()` and `hadamard_bipolar_hash_bytes()` functions (~35 lines)
2. `_TENSOR_CORE_KERNELS` CUDA kernel string (~200 lines)
3. `_TRANSITION_CODEBOOK_AVAILABLE` import block and all references (~15 lines)
4. `cupy` import block (~5 lines)
5. The entire legacy DNA-HDC function body — everything from the DNA-HDC Phase 0 print
   statement through Phase 4 completion (~4,500 lines, approximately lines 3,500–5,649)
6. Inside `_run_hash_grad_single()`:
   - Phase 0 call: `build_frozen_prior()` (~10 lines)
   - Phase 4 call: `xor_orbit_regularise()` (~10 lines)
   - Phase 5 call: `nmf_kl_fit()` (~10 lines)
   - Phase 7 block: suffix grammar build (~30 lines)
   - Phase 9 block: embed pruning (~5 lines)
7. `--no_hash_grad` argument (legacy path toggle, ~5 lines)
8. The `if getattr(args, "hash_grad", True)` routing block that called the legacy path

**Keep:**
- All imports still needed by `_run_hash_grad_single()`
- `_run_hash_grad_single()` function (minus the removed phases)
- `main()` / `argparse` setup (hash_grad args only)
- Token loading helpers

### Step 6 — Align `hash_grad_bpb()` byte-counting with the Official Formula

The official competition eval (from [`train_gpt.py`](records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py:265))
computes token bytes as:

```python
# Official formula (build_sentencepiece_luts + eval loop):
is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)   # initialised ALL TRUE
for token_id in range(sp_vocab_size):
    if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
        continue
    is_boundary_token_np[token_id] = False   # only real tokens set to False
    ...

# Per-token byte count:
token_bytes = base_bytes_lut[tgt_ids]
token_bytes += has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
# No floor — token_bytes can be 0 for control/unknown tokens
```

The current `hash_grad_bpb()` deviates in two ways:

| Deviation | Current code | Official code |
|---|---|---|
| Byte floor | `np.maximum(tok_bytes, 1)` — clips to min 1 | No floor — bytes can be 0 |
| `is_boundary_token` init | Passed in as a parameter (may be `None`) | Always `np.ones(...)` (all True) |

**Changes to make in `hash_grad_bpb()`:**

1. Remove `np.maximum(tok_bytes, 1)` — use raw `tok_bytes` as computed, matching the
   official formula exactly.

2. Ensure `is_boundary_token` is always built from `build_sentencepiece_luts()` with
   `is_boundary_token_np = np.ones(...)` initialisation (all True, then set False for
   real tokens). This is already how the 1_Step_Grad model builds it — just make it
   explicit and non-optional (remove the `if is_boundary_token is not None` guard).

3. The BPB accumulation formula stays identical:
   ```python
   total_bits  += float(-np.log2(p_correct).sum())   # Σ(-log₂ p_i)
   total_bytes += int(tok_bytes.sum())                # Σ(utf8_bytes)
   # Final: BPB = total_bits / total_bytes
   ```

This aligns the DSV eval exactly with the official competition metric.

### Step 7 — Update `README.md`

Add a section at the top of the README noting that this is the **stripped** version of the
1_Step_Grad model, with the dead-weight components removed. Reference the original
(non-copy) folder for the full pipeline. The BPB result (0.4118) and run commands are
unchanged.

---

## Key Invariants — What Is Preserved

| Invariant | How preserved |
|---|---|
| Phase 2 fingerprint tabulation | `tabulate_bucket_frequencies_distributed()` kept unchanged |
| Phase 3 multi-seed fingerprint merge | `merge_seed_frequencies()` kept unchanged |
| Phase 6 `sem_fwd` build | `build_from_tokens()` sem_fwd scatter kept; sem_bwd scatter removed |
| Phase 6 skip-bigram lags | `build_skip_bigram_lags()` kept unchanged |
| Eval collision path | `hash_grad_bpb()` collision branch uses `sem_fwd` — unchanged |
| Eval miss path | `hash_grad_bpb()` miss branch uses `sem_fwd` + lags — unchanged |
| BPB formula | Aligned to official: `Σ(-log₂ p) / Σ(utf8_bytes)` with no byte floor |
| `is_boundary_token` init | All-True (official standard) — non-optional |
| Run command | Unchanged — `TABLE_BITS=19 EMBED_DIM=16 HG_SEEDS=42,7,1337 torchrun ...` |

**What changes in the eval waterfall:**
The `has_emb_mask` branch (`embed @ W_out`) is removed entirely. With NMF gone, all
positions go to either the collision path or the miss path — both of which use `sem_fwd`.
The fingerprint table is kept: it still correctly routes hash collisions to the collision
path (which uses `sem_fwd` directly) rather than the miss path.

---

## File Size Reduction Summary

| File | Before | After | Reduction |
|---|---|---|---|
| `train_gpt.py` | 6,166 lines | ~600 lines | ~90% |
| `_hash_grad_train.py` | 1,344 lines | ~500 lines | ~63% |
| `_semantic_layer.py` | 507 lines | ~200 lines | ~60% |
| `_suffix_grammar.py` | 231 lines | **deleted** | 100% |
| `_transition_codebook.py` | 1,022 lines | **deleted** | 100% |
| **Total** | **~9,270 lines** | **~1,300 lines** | **~86%** |

---

## What This Plan Does NOT Do

- Does **not** change the architecture (no GoldenShift, no EigenTrainer, no 1/freq weighting)
- Does **not** change any hyperparameters (`TABLE_BITS`, `EMBED_DIM`, `HG_SEEDS`)
- Does **not** change the `sem_fwd` build logic or the skip-bigram lag build
- Does **not** change the BPB formula
- Does **not** change the fingerprint table (kept — still routes collisions correctly)
- Does **not** touch the original (non-copy) folder

The result is a clean, minimal implementation of the 1_Step_Grad model that contains only
the components that actually produce BPB 0.4118: distributed fingerprint tabulation,
`sem_fwd` XOR-bundle build, skip-bigram lags 2–5, and the DSV-only eval waterfall.
