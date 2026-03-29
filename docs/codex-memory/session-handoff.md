# Session Handoff

Date: 2026-03-29

## Current Truths

- Session 03 pre-TTT anchor port is complete.
- Sliding s64 val_bpb: `1.12904446` on `8xH100 SXM5`, `serv-3342`.
- Pre-quant EMA val_bpb: `1.14472403`.
- Int6 roundtrip val_bpb: `1.15247273`.
- Steps: `6564`, step_avg: `91.37 ms`.
- Artifact: `15751324` bytes (model `15692752` + code `58572`).
- Official leaderboard entry is record-gated. Beating `#5` quality is not enough; a submission must beat current `#1`.
- NGC 26.03 container + fscratch is the confirmed stable Pegasus path.
- Saved Pegasus `25.02` FA3 container is now a measured negative-result path, not a mainline candidate.
- Session 04 Delta 1 (GPTQ-lite clip search) is COMPLETE — FAILED.
- Session 04 Delta 2 (LeakyReLU^2) is COMPLETE — NEUTRAL.
- Session 05 mainline is now GPTQ correctness.
- The first `1xH100` Full Hessian GPTQ smoke exposed a correctness failure in the export path.

## What Matters Now

- Session 05b currently has one job: repair Full Hessian GPTQ so that the roundtrip path behaves at least plausibly.
- Do not spend more training budget until the export path is debugged on the same checkpoint.
- Working PR code is now the primary implementation source. Papers are secondary.

## Latest GPTQ Smoke Result

Experiment folder:
- `records/track_non_record_16mb/2026-03-29_full_hessian_gptq`

Measured on `1xH100`:
- stopped at `906` steps
- step_avg `662.47 ms`
- pre-quant EMA exact `1.47753094`
- roundtrip exact `1.68963326`
- Hessians collected: `67`
- GPTQ layers used: `66`
- Cholesky fallbacks: `0`
- artifact total: `7754877` bytes
- job timed out before sliding eval finished

Interpretation:
- the smoke is valid as a **mechanics test**
- it is not valid as a **quality comparison** to the `8xH100` anchor
- the export gap is catastrophic, so the quantizer is still wrong

## Confirmed divergences from working PR code

- local within-block GPTQ residual propagation used `W_block[:, j + 1:]`, while PRs `#634`, `#1019`, and `#1060` use `W_block[:, j:]`
- the old local path had no multi-percentile GPTQ search
- the old local path clamped to `[-32, 31]` instead of symmetric `[-31, 31]`
- the old local classifier pulled an extra `bigram.proj` Hessian due to broad `.proj.` matching
- the old path had no per-layer naive-vs-GPTQ export diagnostics

## Safest current conclusion

The local GPTQ implementation was not faithful enough to the known-good PR quantizer.

A PR-grounded repair is now landed in `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`, but it is not runtime-verified yet because:
- no saved checkpoint exists in the repo for same-checkpoint export-only replay
- this local shell does not have `torch`, so verification here stopped at `py_compile`

Update:
- one server-side replay was run and still failed
- `gptq_diag` reported GPTQ worse than both naive baselines on all `66/66` layers
- an export-only replay mode is now landed so the next step can reuse the saved `final_model.pt` without retraining
- a debug-only replay flag now exists: `EXPORT_SKIP_SLIDING_EVAL=1`
- that flag skips the slow sliding-window submission eval and is intended for same-checkpoint A/B replay work only

Do this next:
1. run export-only replay from the saved `final_model.pt`
2. inspect `gptq_layer_diagnostics.json`
3. if needed, run `actorder=False` / `block_size=d_col` ablations on the same checkpoint
4. only then rerun `1xH100`

## Source Of Truth Files

- `docs/campaign/AGENT_SYNC.md`
- `CLAUDE.md`
- `docs/codex-memory/decisions.md`
- `docs/codex-memory/project-state.md`
- `docs/codex-memory/next-session.md`
- `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/README.md`
- `docs/campaign/prompts/session_05b_gptq_debug_restart.md`
