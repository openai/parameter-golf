# Parameter Golf Checkpoint

## Status

This is the living checkpoint for applying the Enigma workflow to `pgolf/parameter-golf`.

Current phase:

- repo surface-area mapping,
- first-pass search strategy definition,
- Stage 1 doctrine recut around mechanism-level families,
- Stage 1 hypotheses updated against new public records,
- Stage 2 `8xH100` experiment plan defined against the public record SOTA baseline,
- level-one H100-only mutation hooks added in `train_gpt.py`,
- earlier config-heavy family marked as provisional rather than canonical.

Current assumption:

- large wins are unlikely to come from hyperparameter sweeps alone,
- Stage 1 should be mostly distinct code and data mutation neighborhoods,
- config changes are still useful, but only as anchors for broader search,
- public records should shift the priors instead of being treated as unrelated outside work.

## Current Constraints

- Track target: `10 minutes` on `8xH100`.
- Artifact target: under `16,000,000` bytes total.
- Baseline result: `1.22436570` final round-tripped `val_bpb`.
- Baseline throughput: about `43.54 ms/step`.
- Baseline step budget under the cap: about `13,780` steps.
- Baseline size headroom: about `136 KB`.

Implication:

- throughput is a core objective,
- compression is a core objective,
- code-size growth is dangerous,
- early training behavior matters disproportionately.

## Files Reviewed

Primary files:

- `pgolf/parameter-golf/README.md`
- `pgolf/parameter-golf/train_gpt.py`
- `pgolf/parameter-golf/train_gpt_mlx.py`
- `pgolf/parameter-golf/data/README.md`

Reference runs:

- `pgolf/parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`
- `pgolf/parameter-golf/records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
- `pgolf/parameter-golf/records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md`
- `pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md`
- `pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md`
- `pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md`
- `pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/submission.json`
- `pgolf/parameter-golf/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`

Working notes:

- `pgolf/parameter-golf/hypotheses.md`
- `pgolf/parameter-golf/strategy.md`
- `pgolf/parameter-golf/checkpoint.md`
- `pgolf/parameter-golf/stage1/variables.md`
- `pgolf/parameter-golf/stage1/code_scope.md`
- `pgolf/parameter-golf/stage1/hypotheses.md`
- `pgolf/parameter-golf/stage1/gaps.md`
- `pgolf/parameter-golf/stage1/portfolio.md`
- `pgolf/parameter-golf/stage2/experiments.md`
- `pgolf/parameter-golf/stage2/checkpoint.md`
- `pgolf/parameter-golf/stage2/h100_matrix_r2/portfolio.md`
- `pgolf/parameter-golf/stage2/h100_matrix_r2/run_configs.json`
- `pgolf/parameter-golf/stage2/h100_matrix_r2/orchestrate_stage2.py`

## Surface Area Summary

### `train_gpt.py`

Main editable surfaces:

- model hyperparameters,
- Muon optimizer behavior,
- Adam split and LR ratios,
- training loop schedule and stopping behavior,
- transformer block structure,
- tokenizer-aware evaluation path,
- quantization and compression rules.

### `data/`

Secondary editable surfaces:

- tokenizer rebuild path,
- dataset export path,
- train-shard prefix for cheap iteration,
- tokenizer-family changes using the fixed published docs cache.

### `records/`

Purpose:

- gives the empirical baseline,
- documents real throughput,
- shows current artifact budget,
- provides the exact command and validation cadence used for comparison.

## Early-Experiment Guidance Found In Repo

The original repo recommends:

- smoke testing locally first,
- using small train-shard prefixes for quick iteration,
- using cheaper hardware and even `1xH100` before full `8xH100`,
- keeping the full fixed validation split for anything intended to be comparable.

Enigma interpretation:

- use cheap runs to kill families, not crown winners,
- use current public records to update the family priors before spending our own compute,
- use A100 for family ranking and mechanism checks,
- only expensive aligned runs should decide promotion.

## Current Strategy Decision

The corrected Stage 1 strategy is:

1. search by mechanism family, not by random idea order,
2. count mutation neighborhoods, not knob settings,
3. keep at most `2` config-heavy families,
4. cover architecture, optimizer, tokenizer/data, systems, compression, and evaluation surfaces,
5. promote only families that survive wallclock, artifact, and round-tripped validation constraints together.

Stage 1 output now exists as:

- a scope audit,
- an evidence-updated `11`-family mechanism-level Stage 1 set,
- and a 5-slot learning-oriented portfolio.

Level-one code mutation now exists in the H100 trainer only:

- alternate-layer sharing hook for `M03`,
- adaptive Muon backend-step scheduling hook for `M07`,
- env-configurable quantizer byte/quality controls as a partial substrate for `M09`,
- baseline-preserving defaults when the new hooks are not enabled.

Important caveat:

- those hooks do not yet cover the full corrected Stage 1 space,
- they are only a partial substrate for the later family runner.

## Candidate First-Wave Families

- long-context training geometry,
- optimizer partitioning and Muon scheduling,
- wallclock-aware scheduling,
- export-aware quantization,
- evaluation-time context,
- architecture and structural follow-ons,
- tokenizer/data changes later.

## What I Am Working With Now

Primary active working file:

- `pgolf/parameter-golf/train_gpt.py`

Reason:

- it contains most of the materially editable solution space in one place.

Secondary files to touch later if justified:

- `pgolf/parameter-golf/data/tokenizer_specs.json`
- `pgolf/parameter-golf/data/download_hf_docs_and_tokenize.py`
- a future record folder under `pgolf/parameter-golf/records/`

Current active doctrine files:

- `pgolf/parameter-golf/stage1/variables.md`
- `pgolf/parameter-golf/stage1/code_scope.md`
- `pgolf/parameter-golf/stage1/hypotheses.md`
- `pgolf/parameter-golf/stage1/gaps.md`
- `pgolf/parameter-golf/stage1/portfolio.md`
- `pgolf/parameter-golf/stage1/frontier_family_r2/portfolio.md`
- `pgolf/parameter-golf/stage1/frontier_family_r2/run_configs.json`
- `pgolf/parameter-golf/stage1/h100_family_r1/portfolio.md`
- `pgolf/parameter-golf/stage1/h100_family_r1/run_configs.json`

Current active code-mutation file:

- `pgolf/parameter-golf/train_gpt.py`
- `pgolf/parameter-golf/stage1/frontier_family_r2/run_family.py`
- `pgolf/parameter-golf/stage1/h100_family_r1/run_family.py`

## Not Doing Yet

- no premature commitment to optimizer-only search,
- no full-track expensive runs without a survivor portfolio,
- no assumption that the earlier config-heavy `h100_family_r1` package is the true Stage 1 family,
- no tokenizer rewrite before the trainer-local surfaces are at least represented in the portfolio.

## Next Steps

1. Use `stage2/h100_matrix_r2/` as the active budget-aware Stage 2 runner.
2. Run `sanity` and `screen` as `8 x 1xH100` parallel phases instead of `1 x 8xH100` per idea.
3. Promote only the strongest matched-control survivor to `final_single`.
4. Use `8xH100` only for later champion confirmation if the single-GPU finalist looks real.

## Open Questions

- How much of the public frontier is training versus evaluation?
- Which long-context geometry transfers best from A100 to `8xH100`?
- Which export submove is actually carrying the win: fp16 embedding, late-K passthrough, lower-bit quantization, or schedule-shaped weights?
- How much code-size budget can realistically be spent on better compression logic before it cancels out the gain?
- When do tokenizer changes become worth the added complexity and validation burden?
