# Path B Byte Eval RunPod Execution Plan and Copilot Prompt

## Human-Readable Execution Plan

This document is a future execution plan and pasteable prompt for a GitHub Copilot CLI session or Copilot agent controlling RunPod. It does **not** authorize a RunPod launch by itself. Do not run paid GPU jobs, use credentials, or access validation data until a human explicitly authorizes the specific stage and budget.

### Objective

Run a safe, fresh, non-record Path B byte-level eval for the exp_1876 PPM-D artifact after finishing real sliding-eval wiring. The eval should decompose official token-level neural probabilities into normalized byte probabilities, merge rank shards in absolute byte order, and stream the byte sequence through score-before-update PPM-D.

### Inputs and Known Constants

- Repository root: `/hpfs/scratch/gpfs/mcclec07/parameter_golf2`
- Source module to import during explicit eval: `results/exp_1876_ppmd/train_gpt_merged.py`
- Quantized artifact: `results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz`
- Absolute artifact path on the HPC/controller: `/hpfs/scratch/gpfs/mcclec07/parameter_golf2/results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz`
- Artifact size: `15,975,706` bytes
- Full validation target: `40,540,160` tokens and `151,078,222` bytes
- First 8M-token subset: `29,365,687` bytes
- 8×H100 SXM planning rate: `~$23.92/hr`

### Approved Control Path

Use **HTTP-bootstrap only** through `scripts/runpod_http_rehearsal.py` or a minimal reviewed derivative of it.

- Do not use SSH, `scp`, `rsync`, `runpodctl ssh`, or Jupyter contents/kernel APIs.
- Do not bundle `.git/`, `plans/`, `records/`, credentials, personal notes, or unrelated results.
- Bundle only the minimal eval inputs needed for the approved stage: eval script, exp_1876 source module, tokenizer/data helper files, requirements, and the compressed model artifact.
- Keep credentials only in the controller environment; never write, echo, log, or paste secret values.
- Every launch must have a pod-side timed shutdown and a local `finally` teardown path.
- Retrieve and verify artifacts before teardown whenever the pod reaches `DONE`, `FAIL`, or `TIMEOUT`.

### Validation Ladder

1. **Local dry-run and tests, no GPU cost**
   - Run the focused unit tests locally.
   - Run `scripts/eval_path_b_ppmd.py --dry-run` against the exact artifact path.
   - Confirm the dry-run JSON reports `eval_status: not_run` and does not claim BPB.

2. **1×H100 HTTP-bootstrap smoke/dry-run**
   - Use `scripts/runpod_http_rehearsal.py` with `--gpus 1`, short `--max-minutes`, and a dry-run-only command.
   - Confirm HTTP retrieval works and outputs are non-empty.

3. **1×H100 short sliding subset**
   - Only after real `--eval --eval-kind sliding` wiring is implemented.
   - Use a small token subset first, then the 8M-token subset if the small subset passes accounting checks.
   - Retrieve one rank shard, eval JSON, logs, and accounting audit.

4. **8×H100 non-record full sliding eval**
   - Only after explicit human budget approval.
   - Use `--eval-kind sliding`; do not use `ttt` unless a separate TTT accounting implementation and audit exist.
   - Require timed shutdown, retrieval buffer, and verified downloads.

### Planning Cost Bounds

| Stage | Hardware | Expected runtime | Estimated cost | Launch condition |
| --- | --- | ---: | ---: | --- |
| Local dry-run/tests | local CPU | minutes | $0 | Always allowed; no credentials. |
| 1×H100 smoke/dry-run | 1×H100 | ~5-8 min cap | ~$0.25-$0.40 | Requires explicit approval and HTTP-bootstrap. |
| Optimized Path B first 8M | 8×H100 | 3-6 min | ~$1.20-$2.40 | Only after real sliding wiring and 1× subset pass. |
| Python/reference first 8M | 8×H100 | 8-15 min | ~$3.20-$6.00 | Fallback only with approval. |
| Sliding-only full validation | 8×H100 | likely 5-10 min | ~$2.00-$4.00 | Preferred first full non-record eval after gates. |
| Full validation with TTT | 8×H100 | 9-15 min | ~$3.60-$6.00 | Blocked until TTT path and audit exist. |
| Conservative Python full validation | 8×H100 | 30-60 min | ~$12-$24 | Avoid unless explicitly approved. |

### Output Artifacts to Retrieve

Every RunPod stage should retrieve the standard launcher artifacts:

- `status.txt`
- `pgolf_exit_code.txt`
- `overall_exit_code.txt`
- `pgolf_stdout.txt`
- `http_server.log`
- `nvidia_smi.txt`
- `python_version.txt`
- `upload_manifest.txt`
- `upload_sizes.txt`
- `launcher_state.json`

Path B dry-run should additionally retrieve:

- `path_b_dry_run.json`

Path B 1×H100 short eval should additionally retrieve:

- `path_b_sliding_1x_subset.json`
- `path_b_sliding_accounting_audit.json`
- `path_b_sliding_merge_manifest.json`
- `path_b_sliding_rank0.npz`

Path B 8×H100 full eval should additionally retrieve:

- `path_b_sliding_full.json`
- `path_b_sliding_accounting_audit.json`
- `path_b_sliding_merge_manifest.json`
- `path_b_sliding_rank0.npz`
- `path_b_sliding_rank1.npz`
- `path_b_sliding_rank2.npz`
- `path_b_sliding_rank3.npz`
- `path_b_sliding_rank4.npz`
- `path_b_sliding_rank5.npz`
- `path_b_sliding_rank6.npz`
- `path_b_sliding_rank7.npz`

No BPB should be copied into a README, PR, or leaderboard note until these artifacts are retrieved and the accounting audit passes.

## Pasteable Copilot CLI / RunPod-Control Prompt

Use the following prompt for a future Copilot CLI session or Copilot agent. It intentionally contains no secrets.

```text
You are GitHub Copilot working in /hpfs/scratch/gpfs/mcclec07/parameter_golf2 on the Path B byte eval. Work autonomously but do not launch RunPod, spend money, use credentials, or run heavy eval unless the human explicitly authorizes the exact stage and budget in this session. Do not write, echo, log, or paste any API key or secret. If credentials are needed for an authorized launch, use only an environment variable supplied by the human; never store it in a file.

Goal: prepare and, only if authorized, execute a safe non-record Path B sliding eval for:
- source module: results/exp_1876_ppmd/train_gpt_merged.py
- artifact: results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz
- absolute artifact path: /hpfs/scratch/gpfs/mcclec07/parameter_golf2/results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz
- artifact size: 15,975,706 bytes
- full validation target: 40,540,160 tokens / 151,078,222 bytes
- first 8M-token subset: 29,365,687 bytes

Hard constraints:
1. Do not claim BPB until full sliding eval wiring is complete and byte accounting is audited.
2. Use HTTP-bootstrap via scripts/runpod_http_rehearsal.py as the approved RunPod control path. Do not use SSH, scp, rsync, runpodctl ssh, or Jupyter uploads/kernels.
3. Every authorized launch must have a pod-side timed shutdown, a max-minutes budget, retrieval before teardown, and local verification that retrieved files are non-empty.
4. Do not bundle .git/, plans/, records/, credentials, personal notes, or unrelated results. Bundle only the minimal eval inputs for the authorized stage: scripts/eval_path_b_ppmd.py, results/exp_1876_ppmd/train_gpt_merged.py, results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz, data/cached_challenge_fineweb.py, data/tokenizer_specs.json, and requirements.txt. If the current HTTP-bootstrap helper cannot include these exact eval files, make a minimal reviewed launcher/wrapper change before any launch and show the diff for approval.
5. Keep --eval-kind ttt blocked unless a separate TTT implementation and accounting audit are completed. The first production-size eval must be --eval-kind sliding.

Validation ladder:
A. Local: run focused tests for tests/test_path_b_ppmd_eval.py and a dry-run JSON against the exact artifact. Do not use credentials.
B. 1xH100 smoke/dry-run: only after explicit approval, use HTTP-bootstrap with --gpus 1 and a short max-minutes cap. Retrieve status.txt, pgolf_stdout.txt, launcher_state.json, nvidia_smi.txt, python_version.txt, upload_manifest.txt, upload_sizes.txt, and path_b_dry_run.json.
C. 1xH100 short sliding subset: only after real sliding eval wiring is implemented. Retrieve path_b_sliding_1x_subset.json, path_b_sliding_accounting_audit.json, path_b_sliding_merge_manifest.json, path_b_sliding_rank0.npz, and launcher artifacts.
D. 8xH100 non-record full sliding eval: only after the subset accounting audit passes and the human explicitly approves the budget. Retrieve path_b_sliding_full.json, path_b_sliding_accounting_audit.json, path_b_sliding_merge_manifest.json, path_b_sliding_rank0.npz, path_b_sliding_rank1.npz, path_b_sliding_rank2.npz, path_b_sliding_rank3.npz, path_b_sliding_rank4.npz, path_b_sliding_rank5.npz, path_b_sliding_rank6.npz, path_b_sliding_rank7.npz, plus all launcher artifacts.

Specific TODO before any production eval: finish the real scripts/eval_path_b_ppmd.py --eval --eval-kind sliding implementation. It must:
1. Dynamically import results/exp_1876_ppmd/train_gpt_merged.py only inside explicit eval mode.
2. Instantiate the exp_1876 Hyperparameters and ValidationData objects exactly as the merged training/eval code expects.
3. Set the quantized artifact path to results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz.
4. Initialize distributed CUDA only in explicit eval mode, with rank/world-size derived from torchrun environment.
5. Load the model via the exp_1876 deserialize path, not by hand-rolling state dict assumptions.
6. Mirror eval_val_sliding windowing exactly for the selected subset/full validation scope.
7. Compute next-token probabilities for target positions, then call vectorized_target_path_logprobs to emit byte-level neural logprob records.
8. Emit compact .npz rank shards named path_b_sliding_rank{rank}.npz with absolute token positions, byte offsets, byte values, and neural logprobs.
9. On rank 0, merge rank shards by absolute token position and byte offset, reject duplicates/out-of-order records, then stream the merged byte sequence through score-before-update PPM-D exactly once globally.
10. Write output JSON with audit metadata, including artifact path, artifact size, source path, git diff/status summary, denominator formula, byte denominator, token count, scored byte count, skipped/zero-byte token counts, PPM-D config, lambda/gating config, shard list, merge manifest path, runtime, and metrics. Metrics must be null or absent if any gate fails.

Accounting gates to enforce before reporting any number:
- full-validation denominator must be 151,078,222 bytes for the full run;
- first 8M-token denominator must be 29,365,687 bytes for that subset;
- SentencePiece leading-space marker handling must not double-count spaces;
- zero-byte special/control tokens must be terminal mass excluded from next-byte denominators;
- PPM-D must score before update;
- lambda/gating must not depend on the target byte;
- PPM-D state must be global after rank-shard merge, not rank-local.

When authorized to run a RunPod stage, use scripts/runpod_http_rehearsal.py with HTTP-bootstrap, a conservative --max-minutes, an explicit --results-dir, and --download listing the exact expected artifact names for that stage. The pod command should write outputs under /root/rehearsal_out so the HTTP retriever can download them. Do not terminate before retrieval unless the launcher has already retrieved status/logs or the hard shutdown fires.

At the end, summarize what was implemented or run, list retrieved artifact paths and sizes, state whether all gates passed, and explicitly say whether BPB is claim-ready. If full eval wiring or accounting audit is incomplete, say BPB is not claim-ready.
```

## Pre-Launch Checklist

- [ ] Human explicitly authorized the exact stage and budget.
- [ ] No secret values appear in commands, docs, logs, or launcher state.
- [ ] HTTP-bootstrap bundle is minimal and excludes `.git/`, `plans/`, `records/`, and credentials.
- [ ] Real `--eval --eval-kind sliding` wiring is complete before any eval beyond dry-run.
- [ ] `ttt` remains disabled unless separately implemented and audited.
- [ ] Pod-side timed shutdown and local teardown path are active.
- [ ] Expected output artifact names are listed in `--download`.
- [ ] Retrieved artifacts are verified non-empty before teardown is considered successful.
- [ ] No BPB claim is made until full eval and accounting audit pass.