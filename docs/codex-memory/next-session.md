# Next Session

## Phase

**Session 05b GPTQ correctness repair.**

The next session should behave like a code-debugging session, not a broad research session.

## Immediate next action

1. Compare the current local GPTQ export against naive int6 on the **same checkpoint**.
2. Inspect `gptq_layer_diagnostics.json` from that export.
3. If GPTQ is still bad, run the same-checkpoint ablations (`actorder=False`, `block_size=d_col`) before any retraining.
4. Only then re-run `1xH100` smoke.
5. Only after that passes, run `8xH100`.

## Current diagnosis

- Session 03 anchor (`8xH100`) is healthy at sliding s64 `1.12904446`.
- Saved-container FA3 is a clean negative result and is parked.
- Session 05b `1xH100` smoke exposed a **GPTQ export correctness failure**:
  - pre-quant EMA exact `1.47753094`
  - roundtrip exact `1.68963326`
  - gap `~0.2121`
- This smoke confirmed:
  - Hessian collection works
  - Cholesky does not fail
  - compressed artifact fits comfortably
  - the current GPTQ quantizer still reconstructs weights badly

## Most likely issue

The local GPTQ implementation drifted too far from the known-good PR code.

The code-level repair already landed:
- within-block residual update now matches the PR loop (`j:` instead of `j+1:`)
- 5-percentile reconstruction search is in place
- symmetric `[-31, 31]` clamp is in place
- `_classify_param` excludes top-level `bigram.proj`
- export writes per-layer diagnostics to `gptq_layer_diagnostics.json`

What is still missing is the **runtime check on a real checkpoint**.

Update:
- that runtime check was run once on server
- result: GPTQ was worse than both naive baselines on `66/66` layers
- next step is now replay ablations from the saved `final_model.pt`, not another training rerun

## Required workflow

1. Use PR code first:
   - PR #1060
   - PR #1019
   - PR #634
2. Use local repo code second.
3. Use papers only for ambiguous math details.
4. The PR quantizer transplant is already done; do not revert to the old custom loop.

## Concrete debug order

1. Run the export-only A/B on a real checkpoint:
   - legacy row-max int6 reconstruction MSE per layer
   - percentile-naive int6 reconstruction MSE per layer
   - GPTQ reconstruction MSE per layer
2. Inspect `gptq_layer_diagnostics.json`:
   - layers where GPTQ is worse than legacy row-max
   - layers where GPTQ is worse than percentile-naive
   - `worst_block_start`
   - `max_block_mse`
3. Run an ablation only if needed:
   - `actorder=False`
   - `block_size=d_col`
   - optionally reduce calibration samples to probe whether the issue is stable
4. After correctness is restored:
   - keep the landed PR-style 5-percentile search
   - keep the landed symmetric `[-31, 31]` clamp
   - keep the tightened hook target filtering

## Hard gates

- No more `8xH100` GPTQ runs until the smoke roundtrip gap is sane.
- No more saved-container FA3 runs.
- No TTT work.
- No broad training-stack bundling before GPTQ is healthy.

## Files to read first

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `docs/codex-memory/decisions.md`
4. `docs/codex-memory/project-state.md`
5. `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/README.md`
6. `docs/campaign/prompts/session_05b_gptq_debug_restart.md`

## Key code targets

- `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`
  - `_classify_param`
  - `collect_hessians`
  - `gptq_quantize_layer`
  - `gptq_mixed_quantize_int6`
- `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md`
- PR implementations in `openai/parameter-golf`

## Operational reminder

For the next smoke run, leave explicit post-train time budget.
The prior `1xH100` smoke hit the Slurm time limit before sliding eval finished.
Also note: this local shell has no `torch`, so verification here only reached `py_compile`; the next session should verify the repaired export path in the runtime environment that has PyTorch installed.
Replay helper env vars now exist for that server-side work:
- `EXPORT_ONLY_CHECKPOINT`
- `EXPORT_TAG`
- `GPTQ_ACTORDER`
- `GPTQ_BLOCK_SIZE`
- `GPTQ_CALIBRATION_SAMPLES`
