# Experiment 0069_combined_k3_k4_side_memory

Parent: 0068_fourgram_pruned_side_memory (K=4 pruned side-memory)

## Question

Combine K=3 trigram (aggressively pruned to top-100K contexts) WITH K=4 4-gram (top-200K contexts) for a 3-way blend at inference: model + K=3 + K=4. Offline analysis (`scratch/blend_probe/combined_aggressive_K3.py`) shows this gives blended BPB 1.9504 (Δ -0.045 vs model 1.9956), beating K=4 alone (-0.041) by +0.004 BPB at the cost of ~1 MB additional cap.

## Hypothesis [LIKELY]

Predicted post-quant val_bpb ≈ 0064 single-seed 2.0030 - 0.045 = ~1.958 BPB.

Cap math:
- Model brotli'd: ~13.44 MB (from 0064)
- K=4 pruned (top_N=200K, top_K=2): ~1.66 MB brotli'd (from 0068 smoke)
- K=3 pruned (top_N=100K, top_K=2): ~0.72 MB brotli'd (estimated from offline)
- Total: ~15.82 MB. Within 16 MB cap with safety.

## Change

Single-file edit to `experiments/0069_combined_k3_k4_side_memory/train_gpt.py` (and/or `modules/trigram_side_memory.py`). Build TWO side-memory packs simultaneously: K=3 with TRIGRAM_TOP_N_CTX_K3 and K=4 with TRIGRAM_TOP_N_CTX_K4.

New env vars (extending 0068's):
- `TRIGRAM_K=3,4` (comma-separated; default just "3" = original 0067/0068 single-K behavior)
- `TRIGRAM_TOP_N_CTX_K3=100000`
- `TRIGRAM_TOP_N_CTX_K4=200000`
- `TRIGRAM_BLEND_WEIGHTS=0.7,0.10,0.20` (model, K=3, K=4 weights; must sum to 1.0)

When TRIGRAM_K is multi-K:
- Build separate packs for each K, store as buffers with K-suffixed names (`trigram3_keys`, `trigram4_keys`, etc.)
- Forward: look up under both K=3 and K=4, get two log-prob distributions (with bigram fallback for missing contexts in either)
- Blend: 3-way logsumexp(`log(w_m) + model_log_softmax`, `log(w_3) + K3_log_probs`, `log(w_4) + K4_log_probs`)
- Compute CE on the resulting distribution

When TRIGRAM_K is single-K (default), behavior is byte-identical to 0068.

### Smoke check

Update `_combined_smoke.py` (analogous to _fourgram_smoke.py) to:
- Build both K=3 (top_N=100K) and K=4 (top_N=200K) packs
- Use blend weights (0.7, 0.10, 0.20)
- Validate against `combined_aggressive_K3.py` reference: blended BPB 1.9504 ± 0.005
- Pack size: K=3 raw + K=4 raw + bigram fallback. Brotli'd estimate. Verify total artifact ≤ 15.8 MB.

## Disconfirming

- val_bpb_post_quant > 1.99 → blend math wrong or quantization breaking the lookup.
- Artifact > 16 MB → packs too large; reduce top_N's.
- Smoke disagrees with `combined_aggressive_K3.py` reference by > ±0.005 BPB → bug.

## Notes from execution

### Implementation summary (2026-04-28)

Extended 0068's K-parameterized side-memory to support TWO K values
simultaneously (combined K=3 + K=4 with a 3-way blend at inference). Default
behavior is byte-identical to 0067/0068 single-K paths (TRIGRAM_K="3" =
0067; TRIGRAM_K="4" + TRIGRAM_TOP_N_CTX=200000 = 0068). The combined
configuration (K_LIST=[3,4], top_N_K3=100K, top_N_K4=200K, weights
(0.7, 0.10, 0.20)) is enabled in env.sh.

Code changes (all in `experiments/0069_combined_k3_k4_side_memory/`):

- `modules/trigram_side_memory.py`:
  - Added `kgram_log2p_per_position(...)`: factors out the per-K (B, L, V)
    log2-prob lookup from `trigram_blend_loss` (same searchsorted +
    matched-context scatter, with bigram fallback for unmatched contexts
    and unigram for t==0). Lets multiple K's reuse the shared bigram
    fallback.
  - Added `trigram_blend_loss_multi_K(...)`: takes a `packs_by_K` dict and a
    `blend_weights` list, computes per-K log2-probs, and blends via N-way
    log2-space logsumexp:
      `log P_blend = logsumexp([log(w_m)+model_logp, log(w_K)+K_logp, ...])`
    Validates weights sum to 1.0; supports any number of K's in
    `K_order`. With N=1 it's bit-identical to the original
    `trigram_blend_loss` (verified directly with a small synthetic batch).
  - Original `trigram_blend_loss` and `build_trigram_pack` unchanged → 0067/
    0068 single-K behavior preserved.

- `train_gpt.py`:
  - `Hyperparameters`: `TRIGRAM_K` is now comma-separated (parsed into
    `trigram_K_list: list[int]`; single-int back-compat for 0067/0068).
    Added `TRIGRAM_TOP_N_CTX_K3` (default 100000), `TRIGRAM_TOP_N_CTX_K4`
    (default 200000), `TRIGRAM_BLEND_WEIGHTS` (comma-separated; default
    `[alpha, 1-alpha]` for single-K — preserves 0067/0068 — or
    `[0.7, 0.10, 0.20]` for two-K). Validates weights sum to 1.0 and
    length matches `1 + len(K_list)`.
  - `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`: added `trigram3_*` and
    `trigram4_*` scale/offset patterns so K-suffixed quant scalars stay
    fp32 in the artifact (mirrors 0067's `trigram_log2p_scale` etc).
  - `GPT.__init__`: added `_trigram_K_list = [3]` and
    `_trigram_blend_weights = [0.8, 0.2]` (default). Original
    `_trigram_K = 3`, `_trigram_blend_alpha` retained for single-K.
  - `GPT.forward`: when `len(self._trigram_K_list) > 1`, dispatches to
    `trigram_blend_loss_multi_K` with K-suffixed buffer names
    (`trigram3_keys`, `trigram4_keys`, ...) and the shared bigram/unigram
    buffers (un-suffixed, identical names to 0067). Single-K path
    unchanged.
  - `main()` post-training pack build: if `len(K_list) == 1`, builds
    one pack with original (un-suffixed) buffer names — byte-identical to
    0067/0068. If `len(K_list) > 1`, builds one pack per K with
    `K_to_top_n = {3: top_n_k3, 4: top_n_k4}`; verifies bigram/unigram
    are identical across K's (deterministic build); installs shared
    bigram/unigram once (un-suffixed) + K-specific keys/offsets/next/log2p
    under K-suffixed names. Sets `_trigram_K_list` and
    `_trigram_blend_weights` so the round-tripped model's forward
    dispatches correctly.

- `env.sh`:
  - Replaced `TRIGRAM_K=4 / TRIGRAM_TOP_N_CTX=200000 / TRIGRAM_BLEND_ALPHA=0.80`
    with `TRIGRAM_K=3,4 / TRIGRAM_TOP_N_CTX_K3=100000 / TRIGRAM_TOP_N_CTX_K4=200000 / TRIGRAM_BLEND_WEIGHTS=0.7,0.10,0.20`.
    Comment block updated to point at `combined_aggressive_K3.py` reference
    and `_combined_smoke.py`.

- `_combined_smoke.py` (new):
  - Builds K=3 (top_N=100K) + K=4 (top_N=200K) packs via the production
    `build_trigram_pack`.
  - Verifies shared bigram/unigram tensors are byte-identical across K's
    (precondition for installing them once under un-suffixed names).
  - Reports per-K + total raw + brotli'd pack sizes; asserts
    projected total artifact <= 15.9 MB safety target.
  - Computes per-K log2p via numpy lookup of the production pack data
    layout (mirrors `_fourgram_smoke.py`'s reference loop, generalized).
  - Computes 3-way blend in log-prob space via N-way logsumexp.
  - Verifies the production tensor-based `trigram_blend_loss_multi_K`
    matches the np-lookup blend (must be within +/- 0.001).
  - Compares against `combined_aggressive_K3.py` reference: target
    blended BPB 1.9504 +/- 0.005.

### Smoke check result

Run command:
  `/Users/tonyliu/Desktop/projects/parameter-golf-ssm/.venv/bin/python \
   experiments/0069_combined_k3_k4_side_memory/_combined_smoke.py`

Output (default config matches env.sh):
- K_LIST=[3,4], TOP_K=2, TOP_N_K3=100K, TOP_N_K4=200K, weights
  (0.7, 0.10, 0.20), build_tokens=100M, min_count=2
- model BPB (cached): 1.9956 (matches offline reference exactly)
- K=3 pack: raw 320,432 contexts → after top-N=100K pruning: 100,000 kept
  (220,432 dropped); 199,686 entries; raw 1.50 MB / brotli 0.63 MB
- K=4 pack: raw 7,545,383 contexts → after top-N=200K pruning: 200,000 kept
  (7,345,383 dropped); 395,049 entries; raw 2.99 MB / brotli 1.32 MB
  (matches 0068 K=4-only smoke exactly: same build path)
- Shared bigram + unigram: raw 1.06 MB / brotli 0.34 MB (byte-identical
  across K's as expected; verified by `torch.equal`)
- TOTAL side-memory: raw 5.55 MB / brotli est 2.29 MB
- Projected total artifact: 13.44 + 2.29 = **15.73 MB** (safety target
  15.9, hard cap 16.0; comfortably within budget)
- K=3 standalone BPB: 2.2462 (worse than K=4's 2.2083 — K=3 is more
  aggressive top-N=100K vs K=4 top-N=200K)
- K=4 standalone BPB: 2.2083 (matches 0068 smoke exactly)
- np-lookup blended BPB: **1.9505** (matches offline reference 1.9504
  within +0.0001; delta vs model: -0.0451)
- production tensor-based forward path blended BPB: **1.9505** (diff vs
  np-lookup: -0.000000 → multi-K forward wiring is correct)
- SMOKE OK

### Cross-check: multi-K with N=1 reduces to single-K

A direct synthetic-batch test (`trigram_blend_loss_multi_K` with K_order=[3]
and weights=[0.8, 0.2] vs `trigram_blend_loss` with blend_alpha=0.8) on
the same pack returned identical loss to 0.0 difference (1e-10). This
confirms the N-way logsumexp blend is consistent with the original
2-way blend math.

### Backward compatibility

- `TRIGRAM_K="3"` (default) + `TRIGRAM_TOP_N_CTX=0` (default): goes
  through the single-K branch in main() with original buffer names → produces
  the same pack tensors and the same forward path as 0067 (verified
  upstream by 0068's byte-identical check, and the multi-K blend math
  is consistent with single-K as shown above).
- `TRIGRAM_K="4"` + `TRIGRAM_TOP_N_CTX=200000`: same single-K branch with
  K=4 → byte-identical to 0068.
- `TRIGRAM_SIDE_MEMORY=0`: no buffers installed, `_use_trigram_blend=False`,
  forward path identical to parent 0064.

### Deviations from plan

- Plan estimated K=3 top_N=100K brotli at ~0.72 MB; actual is 0.63 MB
  (slightly smaller). Combined total brotli is 2.29 MB (better than the
  ~2.4 MB the plan implied), giving 15.73 MB projected vs the plan's
  15.82 MB.
- Plan said offline blended BPB target 1.9504; smoke landed at 1.9505
  (diff +0.0001), well within the ±0.005 tolerance.
- Used N-way logsumexp (max-subtract for stability) for the blend so the
  helper handles arbitrary N (not just 2 or 3) — same numerics as the
  original 2-way log2-space blend.
- Bigram + unigram tables are SHARED in the model state_dict
  (un-suffixed names). Verified that `build_trigram_pack` builds them
  identically for any K (the K-loop only counts K-grams; bigram/unigram
  are computed before the K-loop and depend only on the train slice).
  Installing once saves ~1 MB raw / ~0.34 MB brotli.
