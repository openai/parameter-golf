# AGENTS.md — Parameter Golf 2 Local Runbook

This checkout is a working sandbox for non-record Parameter Golf experiments, launcher
development, and PR preparation. Use it to stage, validate, and document runs before
claiming results upstream.

## Non-Negotiables

1. Keep contest-rule claims aligned with the upstream repo rules.
2. Do not claim BPB improvements from unmatched metrics.
3. Do not let a paid RunPod pod run without bounded shutdown logic.
4. Do not write secrets to repo files or logs; use environment variables only.
5. When PR claims depend on follow-up controls, include compact machine-readable
   artifacts in the submission folder, not just scratch-space paths.

## Canonical Places

- **Scratch / live run outputs:** `results/`
- **PR-bound submission folder:** `records/track_non_record_16mb/<run>/`
- **Submission metadata:** `README.md`, `submission.json`, `train_gpt.py`, `train.log`
- **Compact follow-up evidence:** `records/.../results/followups/`

If a result is important enough to mention in a PR body or `submission.json`, copy a
small summary of it into the submission folder.

## RunPod Lessons Learned

### 1. Always prove the cheap path first

- Prefer local CPU smoke tests, then the smallest GPU rehearsal that can validate the
  exact risky behavior (startup, retrieval, eval-only flow, resume upload, etc.).
- Do not jump straight to larger paid pods until startup + retrieval are proven on a
  cheaper path.

### 2. Timed shutdown is mandatory

- Every launcher should enforce pod-side self-termination with a hard wallclock cap.
- Keep a retrieval buffer; do not spend the full allowed wallclock on training alone.
- For long runs, separate:
  - **training stop horizon**
  - **schedule horizon**
  so resumed continuations can preserve schedule semantics without changing the hard
  pod deadline.

### 3. Treat RunPod proxy retrieval as unreliable until proven otherwise

- The HTTP proxy can return transient `502`/`503`/`504` even after the science is done.
- Optional artifact downloads must retry and then fail soft, not abort the whole run.
- Large upload/download flows need generous wait windows; ~300 MB rank-local resume
  files are normal for 4-GPU resumable checkpoints.

### 4. Copy critical JSONs to `/root/rehearsal_out/`

Files written under `/root/rehearsal_out/seed42/` are not automatically available at
the root artifact URL unless you copy them there explicitly.

For any launcher that later downloads root-relative filenames, explicitly copy:

- `prequant_eval_summary.json`
- `resume_stage_decomposition.json`
- `resume_stage_batch_deltas.jsonl`
- `ttt_eval_summary.json`

from `seed42/` into `/root/rehearsal_out/`.

### 5. Use side-channel watchers for critical results

Do not rely solely on the final launcher download step for scientifically critical
outputs.

For important runs, poll nested URLs directly, e.g.:

- `https://<pod>-30000.proxy.runpod.net/seed42/train_seed42.txt`
- `https://<pod>-30000.proxy.runpod.net/seed42/prequant_eval_summary.json`
- `https://<pod>-30000.proxy.runpod.net/seed42/resume/resume_manifest.json`

Nested `/seed42/...` URLs are valid and often more reliable than waiting for copied
root files.

Use these watchers to capture:

- the critical JSON itself
- a stdout/log copy containing the final metric line

This is how to recover results when the pod finishes the computation but the launcher
later hits a proxy 5xx during cleanup/download.

### 6. Capture a fresh fallback resume snapshot near the end of long runs

If a long run reaches a new resume-save milestone near the endpoint, download the
latest manifest plus all rank-local checkpoint files before the final eval completes.

Verify:

- manifest exists
- `world_size` matches the intended resume shape
- all referenced rank files are present and non-empty

This gives you a short-rerun fallback if the final eval or artifact retrieval fails.

### 7. Use eval-only dataset download mode when possible

Eval-only flows should fetch only:

- tokenizer
- validation shards

Do **not** pull the full training shard set for:

- TTT-only sweep runs
- prequant eval-only runs
- decomposition/eval-only diagnostics

This reduces Hugging Face contention and shortens pod wallclock.

### 8. Resume safety rules

- Resume checkpoints are rank-local and manifest-driven.
- Refuse resume on incompatible:
  - `world_size`
  - architecture fingerprint
  - optimizer config
  - tokenizer path
  - data path
- Keep the resumed GPU count identical to the saved checkpoint unless the checkpoint
  format explicitly supports something else.

### 9. Scientific reporting rules for quantization / TTT

Do **not** infer GPTQ effects from:

- live non-EMA training validation
- earlier-step validation
- unmatched eval pipelines

Prefer matched controls:

- **pre-quant EMA -> quantized -> post-TTT**

When reporting quantization/TTT effects:

1. report the matched pre-quant EMA BPB
2. report the quantized BPB
3. report the post-TTT BPB
4. compute:
   - quantization tax
   - TTT gain
   - residual gap vs pre-quant EMA

If a PR claim depends on these controls, add small JSON/CSV summaries under
`records/.../results/followups/`.

## Good Defaults For Future Agents

- Assume scratch `results/` is ephemeral; promote only the compact evidence needed for
  the PR into the submission folder.
- Red-team the PR body against the README and `submission.json`; they must agree on:
  - hardware
  - cost
  - comparator type
  - exact BPB values
- If a live run finishes the science but the launcher exits nonzero during artifact
  download, treat that as a **retrieval failure**, not a failed experiment.
