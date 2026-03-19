# Parameter Golf Research Tracks

Priority order is dictated by the challenge rules:

1. stay under the `16,000,000` byte artifact cap
2. stay within the `10 minute / 8xH100` training budget for record attempts
3. optimize post-roundtrip `val_bpb`, not pre-quant loss

## Integrated now

- Post-compression-aware training:
  - sampled int8 reconstruction regularizer
  - optional ternary-weight regularizer
  - optional outlier suppression penalty
- Weight sharing / recurrence:
  - shared-block transformer via `NUM_UNIQUE_BLOCKS`
- Sparse attention:
  - optional sliding-window attention via `WINDOW_SIZE`
- Factorized embeddings:
  - optional `EMBED_DIM < MODEL_DIM`
- Hybrid eval-time compute:
  - optional recent-token cache bias during validation / roundtrip eval
- Local proxy iteration:
  - capped validation
  - optional skip of expensive final roundtrip eval
  - proxy sweep launcher

## Current knobs

- `NUM_UNIQUE_BLOCKS`
- `WINDOW_SIZE`
- `EMBED_DIM`
- `COMPRESSION_REG_WEIGHT`
- `COMPRESSION_GRID_REG_WEIGHT`
- `COMPRESSION_SCALE_REG_WEIGHT`
- `COMPRESSION_RANK1_REG_WEIGHT`
- `TERNARY_REG_WEIGHT`
- `OUTLIER_REG_WEIGHT`
- `EVAL_CACHE_MIX_WEIGHT`
- `EVAL_BIGRAM_MIX_WEIGHT`
- `EVAL_CACHE_SIZE`
- `FINAL_ROUNDTRIP_EVAL`
- `ROUNDTRIP_VAL_MAX_TOKENS`

## Local proxy reference point

All local comparisons below use the same quick 3090 proxy envelope:

- `MAX_WALLCLOCK_SECONDS=180`
- `TRAIN_BATCH_TOKENS=32768`
- `VAL_MAX_TOKENS=1048576`
- `FINAL_ROUNDTRIP_EVAL=0`
- baseline architecture:
  - `NUM_LAYERS=12`
  - `NUM_UNIQUE_BLOCKS=12`
  - `MODEL_DIM=384`
  - `EMBED_DIM=0`
  - `NUM_HEADS=6`
  - `NUM_KV_HEADS=3`

## Roundtrip proxy track

Use this when ranking experiments on a more faithful local objective:

- keep the same baseline architecture unless explicitly testing architecture
- enable `FINAL_ROUNDTRIP_EVAL=1`
- keep `ROUNDTRIP_VAL_MAX_TOKENS` capped so the run stays practical on a 3090
- treat this as the local approximation to the actual challenge metric

## Latest findings

- Quick local baseline:
  - run: `baseline3090_20260318_170251`
  - result: `val_bpb=2.0916`, `val_loss=3.4910`
  - total artifact: `6,831,983` bytes
  - interpretation: current local number to beat
- Hybrid eval sidecar, recent-token + bigram continuation bias:
  - run: `sidecar3090_20260318_172524`
  - knobs: `EVAL_CACHE_MIX_WEIGHT=0.03`, `EVAL_BIGRAM_MIX_WEIGHT=0.05`, `EVAL_CACHE_SIZE=16`
  - result: `val_bpb=2.0970`, `val_loss=3.5000`
  - total artifact: `6,810,819` bytes
  - delta vs baseline: `+0.0054 bpb` worse, `21,164` bytes smaller
  - interpretation: close enough to keep around for later tuning, not good enough to become the default path
- Compression-aware baseline, reconstruction regularization `0.01`:
  - run: `compress3090_20260318_174132`
  - result: `val_bpb=2.0943`, `val_loss=3.4954`
  - total artifact: `6,812,935` bytes
  - delta vs baseline: `+0.0027 bpb` worse, `19,048` bytes smaller
  - interpretation: strongest experimental branch so far
- Compression-aware baseline, reconstruction regularization `0.005`:
  - run: `compress3090_half_20260318_1750`
  - result: `val_bpb=2.0928`, `val_loss=3.4930`
  - total artifact: `6,829,073` bytes
  - delta vs baseline: `+0.0012 bpb` worse, `2,910` bytes smaller
  - interpretation: best pre-roundtrip proxy result outside the plain baseline
- Matched roundtrip-proxy baseline:
  - run: `baselinert3090_20260318_181344`
  - exact final roundtrip result: `val_bpb=2.11089617`, `val_loss=3.56464830`
  - total artifact: `6,705,058` bytes
- Matched roundtrip-proxy compression baseline:
  - run: `compressrt3090_20260318_175828`
  - knobs: `COMPRESSION_REG_WEIGHT=0.005`
  - exact final roundtrip result: `val_bpb=2.06085837`, `val_loss=3.48014999`
  - total artifact: `6,839,798` bytes
  - delta vs matched roundtrip baseline: `-0.05003780 bpb`, about `2.37%` better
  - interpretation: compression-aware training is now the leading local research branch when measured on a more faithful objective
- Sparse-attention probe on the winning compression setup:
  - run: `compressrt_sparse512_20260318_1842`
  - knobs: `WINDOW_SIZE=512`, `COMPRESSION_REG_WEIGHT=0.005`
  - exact final roundtrip result: `val_bpb=2.07004634`, `val_loss=3.49566562`
  - delta vs best compression baseline: `+0.00918797 bpb` worse
  - interpretation: not good enough to displace the dense compression-aware path; sparse attention stays experimental for later
- Focused QAT roundtrip sweep around the winning compression point:
  - sweep: `qatrtsweep_20260318_1906`
  - best result in sweep:
    - run: `qatrtsweep_20260318_1906_w0045_o0000`
    - knobs: `COMPRESSION_REG_WEIGHT=0.0045`, `OUTLIER_REG_WEIGHT=0.0`
    - exact final roundtrip result: `val_bpb=2.06804196`, `val_loss=3.49228084`
    - total artifact: `6,814,995` bytes
  - interpretation:
    - tiny outlier regularization did not help on this local roundtrip track
    - none of the focused QAT sweep runs beat the standing best dense compression-aware run at `2.06085837`
    - the dense compression-aware baseline remains the current best local result
- Recurrent/shared-block roundtrip sweep:
  - sweep: `recurtsweep_20260318_1925`
  - tested:
    - `16 layers / 8 unique / embed 0` -> `2.25452146`
    - `18 layers / 6 unique / embed 0` -> `2.28804085`
    - `16 layers / 8 unique / embed 256` -> `2.28260194`
    - `18 layers / 6 unique / embed 256` -> `2.34886036`
  - interpretation:
    - this branch cuts artifact size aggressively, but quality collapses on the current local roundtrip track
    - none of these shapes are close to the dense compression-aware baseline
    - shared-block recurrence stays interesting for the 16 MB objective, but this first pass is not competitive enough to prioritize locally
- Roundtrip sidecar revisit on top of the winning dense compression setup:
  - sweep: `sidecarrtsweep_20260318_1942`
  - best usable result in sweep:
    - run: `sidecarrtsweep_20260318_1942_c0020_b0030_s8`
    - knobs: `EVAL_CACHE_MIX_WEIGHT=0.02`, `EVAL_BIGRAM_MIX_WEIGHT=0.03`, `EVAL_CACHE_SIZE=8`
    - exact final roundtrip result: `val_bpb=2.06132482`, `val_loss=3.48093767`
    - total artifact: `6,864,315` bytes
    - delta vs best dense compression baseline: `+0.00046645 bpb` worse
  - sweep reliability notes:
    - `c0015_b0020_s8` and `c0020_b0020_s8` stopped before a usable roundtrip result was written
    - `c0020_b0020_s16` reached artifact export but never wrote `final_int8_zlib_roundtrip_exact`
  - interpretation:
    - the sidecar branch is the closest secondary idea so far
    - it still did not beat the plain dense compression-aware winner
    - keep it parked as a late-stage add-on, not the current pivot
- Conservative ternary / low-bit sweep on top of the winning dense compression setup:
  - sweep: `ternaryrtsweep_20260318_201412`
  - tested:
    - `TERNARY_REG_WEIGHT=0.0005` -> `2.07311732`
    - `TERNARY_REG_WEIGHT=0.0010` -> `2.07009530`
    - `TERNARY_REG_WEIGHT=0.0020` -> `2.07025558`
    - `TERNARY_REG_WEIGHT=0.0035` -> `2.08786263`
    - `TERNARY_REG_WEIGHT=0.0050` -> `2.07821685`
  - interpretation:
    - native low-bit pressure in this form clearly hurts the local roundtrip metric
    - very small ternary weights degrade less, but still do not approach the current leader
    - do not prioritize ternary shaping again until a stronger baseline exists or the training formulation changes
- Quantization residual-budget sweep on top of the winning dense compression setup:
  - sweep: `residualrtsweep_20260318_203241`
  - tested:
    - `residual_rank=0, residual_budget=0` -> baseline export control for this sweep
    - `residual_rank=1, residual_budget=65536` -> `2.08312093`
    - `residual_rank=1, residual_budget=262144` -> `2.08187280`
    - `residual_rank=1, residual_budget=524288` -> `2.08285302`
    - `residual_rank=1, residual_budget=1048576` -> `2.07731235`
  - interpretation:
    - spending more bytes on rank-1 residual export corrections did not improve the local roundtrip metric
    - the export-side residual mechanism is not currently a better lever than the plain dense compression-aware setup
    - quantization-budget tuning should be deprioritized for now
- Refined sidecar micro-sweep around the prior near-win:
  - sweep: `sidecarrefine_20260318_205219`
  - completed exact results:
    - `cache=0.018, bigram=0.030, size=8` -> `2.08080110`
    - `cache=0.020, bigram=0.028, size=8` -> `2.07489103`
    - `cache=0.020, bigram=0.030, size=8` rerun -> `2.08947255`
    - `cache=0.020, bigram=0.032, size=8` -> `2.07840275`
  - incomplete run:
    - `cache=0.022, bigram=0.030, size=8` reached artifact export but did not write `final_int8_zlib_roundtrip_exact`
  - interpretation:
    - the earlier `2.06132482` sidecar near-win did not reproduce
    - the sidecar branch now looks unstable on the local roundtrip track
    - measuring repeatability is more important than additional sidecar micro-tuning right now
- Corrected wallclock repeatability sweep:
  - sweep: `repeatrtsweepfix_20260318_215301`
  - dense compression-aware runs:
    - `base_a` -> `2.06761597`
    - `base_b` -> `2.07369637`
    - `base_c` -> `2.08956232`
  - sidecar near-win reruns:
    - `side_a` -> `2.05608381`
    - `side_b` -> `2.09377262`
    - `side_c` -> `2.07285932`
  - interpretation:
    - both branches show too much spread on the local `180s` wallclock track
    - the best sidecar rerun did beat the standing leader, but the worst sidecar rerun was much worse
    - the dominant local noise source now looks methodological, not architectural
    - the next step should be a fixed-step local roundtrip track, not more wallclock micro-sweeps
- Fixed-step roundtrip sweep:
  - sweep: `fixedsteprtsweep_20260318_221632`
  - dense compression-aware runs:
    - `base_a` -> `2.04299145`
    - `base_b` -> `2.04299145`
  - sidecar near-win reruns:
    - `side_a` -> `2.04300345`
    - `side_b` -> `2.04300345`
  - interpretation:
    - once wallclock variance is removed, the sidecar branch is effectively identical to the dense baseline
    - the dense compression-aware branch remains the cleanest local control
    - future local search should use fixed-step comparison first, then wallclock only as a secondary sanity check
- Export-aware fixed-step compression probe:
  - sweep: `exportaware_fixedstep_20260318_223456`
  - completed result:
    - `g010_r000` -> `2.04288777`
    - knobs: `COMPRESSION_REG_WEIGHT=0.005`, `COMPRESSION_GRID_REG_WEIGHT=0.10`, `COMPRESSION_RANK1_REG_WEIGHT=0.0`
    - total artifact: `6,663,470` bytes
    - delta vs fixed-step dense control: `-0.00010368 bpb` better
  - execution note:
    - the broader coarse sweep was aborted after the first positive signal to avoid spending more 3090 time on low-probability points
  - interpretation:
    - export-aware grid alignment is the first post-fixed-step change that improved the dense compression-aware control
    - the gain is small, but it is deterministic and points in the right direction
    - the next compression-native pivot should stay inside export-aware regularization, not revisit sidecar or architectural branches
- Scale-aware fixed-step compression sweep:
  - sweep: `scaleaware_fixedstep_20260318_224233`
  - completed results:
    - `g010_s0010` -> `2.04313626`
    - `g010_s0025` -> `2.04358127`
  - interpretation:
    - adding explicit adjacent-scale smoothing made the roundtripped result slightly worse at both tested weights
    - this version of scale-aware pressure does not improve on the grid-aligned winner
    - the next best move is to refine the grid-alignment weight itself, not add more compression-native terms yet
- Grid-refinement fixed-step sweep:
  - sweep: `gridrefine_fixedstep_20260318_225110`
  - completed results:
    - `g0080` -> `2.04396986`
    - `g0120` -> `2.04350611`
  - interpretation:
    - both nearby grid weights regressed versus the `0.10` winner
    - `COMPRESSION_GRID_REG_WEIGHT=0.10` currently looks like a real local optimum on the fixed-step track
    - the next compression-aware pivot should keep `grid=0.10` fixed and test only very small outlier pressure around it
- Tiny outlier sweep on top of the grid-aligned winner:
  - sweep: `gridoutlier_fixedstep_20260318_225946`
  - completed results:
    - `o00010` -> `2.04373218`
    - `o00025` -> `2.04372289`
  - interpretation:
    - even very small outlier pressure still regresses
    - outlier suppression should stay parked unless it becomes tensor-targeted
- Dense iso-byte frontier sweep:
  - sweep: `isobyte_fixedstep_20260318_234805`
  - completed results:
    - `b10` -> `2.02814871` at `9,683,932` bytes
    - `b12` -> `2.05262920` at `11,334,608` bytes
    - `b14` -> `2.03768242` at `13,094,288` bytes
    - `b155` -> `2.00290272` at `13,741,308` bytes
  - interpretation:
    - dense scaling dominates the small-model micro-ideas by a wide margin
    - the current best result is no longer the 6.66 MB regime; it is the larger dense `b155` run
    - the frontier is not monotonic with size alone, so geometry still matters, but the main lesson is clear: under-byte-spent local negatives were misleading
    - the next step should stay on the dense high-cap frontier and compare width-vs-depth near the byte ceiling
- High-cap dense frontier:
  - recovered / rerun results:
    - `w608_l12` -> `2.00551677` at `14,371,393` bytes
    - `w624_l12` -> `2.01128088` at `15,024,114` bytes
    - `d576_l14` -> `1.99806297` at `15,222,128` bytes
    - `w640_l12` -> `2.00505534` at `15,658,993` bytes
  - interpretation:
    - depth beat width at roughly the same byte spend in this near-cap regime
    - the first sub-`2.0` local fixed-step result came from the deeper dense model, not the wider one
    - near the byte cap, width is not obviously the best place to spend additional budget
- Tokenizer sanity check on the current best dense recipe:
  - matched local subset controls built from the same `120k` selected-doc prefix
  - SP1024 subset control:
    - `sp1024subsetbest_20260319_020125` -> `1.99806297`
    - total artifact: `15,222,128` bytes
    - dataset stats: `149,659,022` total tokens on the subset
  - SP4096 subset swap on the same trainer:
    - `sp4096best_20260319_015500` -> `1.89591231`
    - total artifact: `16,627,470` bytes
    - dataset stats: `109,783,049` total tokens on the same subset
  - interpretation:
    - moving from SP1024 to SP4096 on the same local subset improved exact roundtrip BPB by `0.10215066`, about `5.11%`
    - the same subset needed about `26.64%` fewer tokens with SP4096, which matches the expected compression benefit
    - the merged `14x576` SP4096 run broke the `16,000,000` byte cap by `627,470` bytes, so it is a strong signal but not yet a submission-shape replacement
    - tokenizer work is no longer purely deferred; it is now a real frontier lever, but it must be co-optimized with model size to stay under cap
- Iso-byte SP4096 dense sweep:
  - sweep: `sp4096isobyte_fixedstep_20260319_022236`
  - completed results:
    - `l15_d544` -> `1.90194008` at `16,090,675` bytes
    - `l14_d560` -> `1.89329916` at `15,869,071` bytes
    - `l12_d608` -> `1.89424125` at `15,844,603` bytes
  - interpretation:
    - SP4096 is now a cap-compliant win, not just an over-budget curiosity
    - the best cap-legal SP4096 point beat the SP1024 `14x576` control by `0.10476381` bpb, about `5.24%`
    - `l14_d560` is the current best local result overall
    - `l12_d608` is slightly worse on fixed-step BPB but notably faster per step, so it remains a plausible wallclock-oriented backup shape
    - the first deeper SP4096 point (`l15_d544`) pushed just over the cap, which suggests the next useful local refinement is a slightly narrower deeper sweep

## Current leader

- `sp4096isobyte_fixedstep_20260319_022236_l14_d560`
- dense attention, no sidecar, no recurrence, no factorized embedding
- `VOCAB_SIZE=4096`, tied embeddings
- `COMPRESSION_REG_WEIGHT=0.005`
- `COMPRESSION_GRID_REG_WEIGHT=0.10`
- fixed-step exact final roundtrip result: `val_bpb=1.89329916`
- total artifact: `15,869,071` bytes
- best wallclock-track reference remains `compressrt3090_20260318_175828` at `2.06085837`

## Regime correction

- The trusted local dense control is now in the near-cap regime, not the old `6.66 MB` regime.
- That is why the dense iso-byte and high-cap frontier sweeps changed the project direction so much.
- Many earlier negative results were gathered in an under-byte-spent regime and should not be treated as globally final.
- The trustworthy questions now are:
  - how should the remaining byte budget be spent near the cap?
  - which export-aware or tokenizer-aware changes still help once the dense control is already strong?
  - how should the remaining cap headroom be spent inside the stronger SP4096 regime?

## Immediate next step

- Keep the SP1024 `14x576` run as the baseline control and the SP4096 `14x560` run as the new frontier control
- keep `COMPRESSION_REG_WEIGHT=0.005` and `COMPRESSION_GRID_REG_WEIGHT=0.10`
- treat tokenizer changes as a real branch, not a deferred curiosity
- next tokenizer-aware experiments should stay near the cap:
  - refine the SP4096 depth/width trade in the `15.7 MB` to `16.0 MB` band
  - keep a close eye on step time, because `l12_d608` was materially faster than `l14_d560`
- continue ranking ideas by `final_int8_zlib_roundtrip_exact val_bpb`

## Next experiments

- SP4096 frontier refinement:
  - test slightly narrower deeper shapes like `15x528` and `16x512`
  - compare them against the current `14x560` leader and the faster `12x608` backup
  - stay under `16,000,000` bytes
- Export-side symmetry-aware permutation:
  - apply function-preserving reordering of MLP channels / attention heads before export
  - test whether the grid-alignment hint can be turned into a larger zlib win without quality loss
- Tensor sensitivity mapping / heterogeneous export allocation:
  - measure which tensors hurt roundtrip BPB most when quantized
  - spend residual / protection budget selectively instead of globally
- Export-aware compression regularizer:
  - continue aligning sampled training-time regularization with the actual export path
  - hold `COMPRESSION_GRID_REG_WEIGHT=0.10` fixed unless new evidence suggests otherwise
- Scale-aware compression regularizer:
  - parked for now after the first two weights regressed
  - revisit only if a different formulation of scale entropy or scale clustering becomes compelling
- Fixed-step compression sweeps:
  - keep using the fixed-step roundtrip track as the local ranking metric
  - only move promising compression-native changes back onto the 180s wallclock track later
- Sidecar branch is parked:
  - fixed-step results say it is not moving the needle in a reliable way
  - do not spend more 3090 time on sidecar micro-tuning for now
- Export-side ideas remain parked:
  - residual-budget tuning did not help
  - sparse attention did not help
  - shared-block recurrence did not help
- Low-bit shaping remains parked:
  - revisit only if the training objective changes materially or H100 results suggest a different regime

## Medium-term work

- Dense winner + sidecar + low-bit combined into one trainer once the individual branches are measured cleanly
- Global/shared codebook quantization across layers
- Basis-generated per-layer weights or hypernetwork-style weight generation
- Test-time adaptation with strict reset semantics
- Token-adaptive recurrent depth / halting policy

## Deferred until the model is stronger

- full tokenizer redesign beyond the SP1024 vs SP4096 sanity branch
- aggressive code-size golf
- heavy hyperparameter brute force
