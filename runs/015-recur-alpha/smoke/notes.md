# Spec 015 smoke — execution notes

**Pod:** `1wcixp96k29f0e` (8×H100 SXM, AP-JP-1, $23.92/hr), stopped
**Commit:** `a9aa141` on `exp/recur-alpha` (via `fork` remote)
**Command:** `ITERATIONS=500 RECUR_ALPHA_ENABLED=1 TRAIN_LOG_EVERY=50 ... torchrun --nproc_per_node=8 train_gpt.py`
**Status:** **SMOKE HALTED — spec 015 stop-early criterion literally triggered.**

## What the smoke showed

**Correctness signals:**
- Model loaded cleanly (35,989,664 params, matches baseline)
- `recur_alpha: enabled=True num_loops=2 loop_start=3 loop_end=5 diag_p2p_cos=False` — startup log line appears as spec expected
- Training completed 500 iterations with NO NaN, NO divergence
- torch.compile succeeded (heavy upfront — ~6 minutes for compile + bucket warmup; spec open-question #3 flagged this risk)

**Failure signal (blocker):**
- **All 15 per-step `recur_alpha` log lines show `values=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] grad_norm=0.000000`** — exactly zero gradient norm for every log entry from step 50 through step 500.
- Spec 015 stop-early criterion: *"α grad norms exactly zero for 5+ consecutive log entries → halt, optimizer routing broken"* — 15 consecutive zeros observed, threshold exceeded 3×.
- α values never moved off initial 0 (expected — no gradient means no update).

**Secondary failure (not the blocker):**
- At step 500, training exited cleanly and entered `serialize()` → `_compressed_code_size()` → `subprocess.run(['pyminify', ...])` → `FileNotFoundError: 'pyminify'`.
- `pyminify` is not installed on the parameter-golf pod template. Not blocking for screening runs (we skip serialize/EMA/quant/TTT per user memory `feedback_screen_via_training_endpoint_val.md`) — but is a latent issue for full / submission runs.

## Interpretation (for research)

Execution's role stops at observing and reporting. The spec's own author flagged that this *exact* pattern ("α grad norms exactly zero for many steps") would indicate "optimizer not registering". So the spec anticipated this failure mode as a correctness gate. Handing back.

Sanity-check hypothesis (research to verify, don't trust this):
- Log is printed *after* `optimizer.step()` and grads are zeroed by the time the log computes `grad_norm`. If so, the log is cosmetically wrong but optimizer is fine — check mid-step grad snapshotting.
- OR the Parameter isn't in the autograd graph (detach somewhere in the encoder/decoder alpha_info dispatch).
- OR the `Optimizers.__init__` scalar-AdamW routing is picking up the Parameter but with `requires_grad=False`, or in a group that's never stepped.

The 500 non-NaN iters with identity-init α=0 = baseline behavior confirms the identity-at-init invariant holds (training is numerically fine, consistent with spec 008's 1.0697 trajectory shape).

## Data state

- Smoke artifacts rsynced to `runs/015-recur-alpha/smoke/` locally.
- `train.log` is the full 24KB log including the 15 grad_norm=0 lines and pyminify traceback.
- `51d90d25-....txt` — one of flash-attn's compile-cache files; can be ignored.
- JP pod stopped (`runpodctl pod stop 1wcixp96k29f0e`). Not deleted — cheap to resume for a patched smoke.

## Cost

| Item | Cost |
|---|---|
| Failed cross-region retries (JP capacity probe) | ~$0.05 |
| NA 2×H100 smoke pod (stopped before use) | ~$0.20 |
| NA prep pod (RTX PRO 6000) — still running in background for data regen | ~$0.10/hr ongoing |
| JP 8×H100 smoke (compile + 500 iters + pyminify crash) | ~$3.50 |
| **Total spec 015 smoke** | **~$3.85** |

## Parallel: NA caseops data prep still running

Separately from spec 015, the NA volume was missing caseops data. A prep pod (`q52slv996d3uqx`, RTX PRO 6000, $0.44/hr, AP NA-1) is running `prepare_caseops_data.py` in background on 15M docs to unblock future NA-side runs. Orthogonal to the α bug — will complete regardless and leave NA volume equivalent to JP for future specs.

## Handback recommendation for research

1. Inspect `train_gpt.py` around `Optimizers.__init__` and wherever recur_alpha is added as a Parameter. Most likely: the log computes `grad_norm` post-step-post-zero_grad, which would be a cosmetic issue rather than a real bug.
2. If the grad really is zero, check Parameter registration (is it `nn.Parameter` with `requires_grad=True`?) and the encoder/decoder alpha_info forward path for any `.detach()` / tensor copy.
3. Also add `pyminify` to pod environment or EXECUTION.md preflight for any future full-pipeline submission run.
