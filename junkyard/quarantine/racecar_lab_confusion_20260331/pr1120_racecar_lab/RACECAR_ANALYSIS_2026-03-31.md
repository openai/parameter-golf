# Rascal Racecar Analysis (2026-03-31)

## Scope
- Target PR: `openai/parameter-golf#1120` (Rascal, mean `1.1099`)
- Constraint: copy-only analysis workspace
- Local copies used:
  - `analysis/pr1120_racecar_lab/copies/train_gpt_rascal_pr1120.py`
  - `analysis/pr1120_racecar_lab/copies/train_gpt_rascal_sota_local.py`
  - `analysis/pr1120_racecar_lab/copies/train_gpt_rascal_master_local.py`
  - `analysis/pr1120_racecar_lab/copies/train_gpt_rascal_final_lc4_local.py`
  - `analysis/pr1120_racecar_lab/copies/train_gpt_bandit_local.py`
  - `analysis/pr1120_racecar_lab/leaderboard_copies/train_gpt_pr1060_loader_gptq.py`
  - `analysis/pr1120_racecar_lab/leaderboard_copies/train_gpt_pr1122_engramlite.py`

## Key fact check (latest landscape)
- `README.md` leaderboard in `main` is stale relative to active PR stream.
- Newer open PRs (2026-03-31) include:
  - `#1172`: `1.1015` (SLOT + split-LR + full GPTQ + XSA-all)
  - `#1184`: `0.9485` (Scylla tokenizer + modern stack)
- PR `#1120` is still a strong base-neural run but not current frontier.

## PR1120 bottlenecks (from your actual logs)
From `records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_seed{42,300,444}.log`:
- Training reaches `~6593` steps at `~91 ms/step` (good)
- Post-EMA neural quality: `~1.1332-1.1338`
- Int6 roundtrip penalty is large: `~1.1437-1.1442` (about `+0.010` to `+0.011` vs post-EMA)
- Final sliding recovers to `1.1098-1.1102`, but quantization gap is still the largest clear lever
- Explicitly skipping GPTQ: `gptq:SKIPPED (SKIP_GPTQ=1)`

## Important local discovery on disk
Your local Rascal lineage already has a GPTQ-enabled branch:
- `train_gpt_rascal_sota_local.py` and `train_gpt_rascal_master_local.py` contain:
  - Full Hessian GPTQ (`gptq_quantize_weight`, `gptq_calibrate`)
  - `GPTQ_RESERVE_MS` wallclock reservation logic
  - Mixed int6 GPTQ quantization export path
- This path is absent in PR1120 copy.
- `train_gpt_rascal_master_local.py` differs from `train_gpt_rascal_sota_local.py` mainly by `COPRIME_MAX_LOADED_SHARDS` default `1` instead of `4`.

## Local negative signal worth respecting
`experiments/Rascal_Final_Submission_LC4/results/2026-03-31_seed444_lc4_race.md`:
- Regressed to `1.11052831` (worse than PR1120 seed 444)
- Artifact overflow: `16,751,237` bytes (invalid)
- Root issue pattern: large code path + `SKIP_GPTQ=1` can break size budget

## Symbiotic techniques from leaderboard PRs

### High-confidence, directly compatible with Rascal
1. Full Hessian GPTQ inside 600s budget (`#1060`, `#1019`, `#1172` lineage)
2. Keep XSA-all and coprime loader (already in Rascal)
3. Shorter GPTQ reserve tuning (`~9-14s`, not 30s) to recover training steps

### Medium-confidence, compatible with moderate code work
1. SLOT eval adaptation (`#1172`) on top of sliding window
2. Sigmoid-gated skip blending (`#1122`, `#1172`)
3. Split layerwise Muon LR (`#1172`)

### High-impact but high-cost lane
1. Tokenizer replacement (Scylla lane, `#1143`/`#1184`)

## Practical recommendation order

### Phase 1 (do first)
- Promote your existing local GPTQ Rascal branch to race candidate.
- Start from `train_gpt_rascal_sota_local.py`/`train_gpt_rascal_master_local.py` copy path.
- Run with `SKIP_GPTQ=0` and sweep `GPTQ_RESERVE_MS` in `{9000, 12000, 14000}`.
- Use insta-cache calibration to avoid a full extra loader pass:
  - `GPTQ_INSTA_CACHE=1`
  - `GPTQ_CACHE_SEQS_PER_STEP=1` (or `2`)
  - `GPTQ_CALIB_SAMPLES=256`
- Keep `XSA_LAST_N=11`, `ROPE_DIMS=16`, `BIGRAM_VOCAB_SIZE=2048`, `SWA_EVERY=50`.
- Expectation: best immediate BPB gain likely comes from collapsing quantization error, not architecture churn.

### Phase 2
- Add SLOT eval-only delta optimization (keep model weights frozen; score-first semantics).
- Try `SLOT_STEPS=8`, `SLOT_LR=0.005` style settings.
- This is a low-risk eval-side boost after quantization is fixed.

### Phase 3
- If still chasing more, port one architecture lever at a time:
  - gated skips
  - split-LR Muon
  - bigram dim increase with artifact budget check

## What not to prioritize now
- Skip-gram extras from local notes: signal is weak/noisy vs cost.
- Muon backend step tweaks alone: mixed and unstable local evidence.
- `SKIP_GPTQ=1` on larger-code branches: can fail artifact size cap.

## Minimal race matrix (3 seeds each)
1. `R0`: PR1120 control (`SKIP_GPTQ=1`)
2. `R1`: local GPTQ branch + `GPTQ_RESERVE_MS=14000`
3. `R2`: local GPTQ branch + `GPTQ_RESERVE_MS=9000`
4. `R3`: best of `R1/R2` + SLOT eval

Use seeds: `42`, `300`, `444` for direct comparability with PR1120.
