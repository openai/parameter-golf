# Path B PPM-D Legality Audit — Implementation Notes

Companion to `plans/ppmd-legality-proof.md`. This file documents the Path B
**byte-level** evaluator added to `scripts/eval_path_b_ppmd.py`, the
testability surface, the remaining risks, and the explicit RunPod rehearsal
commands. **No BPB is claimed in this document.** Any number must come from a
fresh authorized run that passes the accounting audit.

---

## Current session status (latest update)

- ✅ Implementation complete: `scripts/eval_path_b_ppmd.py` (2317 lines),
  `tests/test_path_b_ppmd_eval.py` (713 lines, 30/30 passing on `.venv-smoke`,
  29/30 on `/bin/python3.8` with one torch-only test skipped).
- ✅ Local dry-run verified: writes
  `results/exp_1876_ppmd/path_b_ppmd_eval_plan.json` with `mode=dry_run`,
  `artifact_exists=True`, `source_python_exists=True`, `schema_version=1`.
- ✅ Launcher extended: `scripts/runpod_http_rehearsal.py` now accepts
  repeated `--extra-file local_path[:arcname]` so the artifact + the source
  module can be bundled without per-experiment launcher edits. Bundle build
  unit-tested locally (members: `train_gpt.py`, `cached_challenge_fineweb.py`,
  `tokenizer_specs.json`, `requirements.txt`, `train_gpt_merged.py`).
- ✅ Documented rehearsal commands corrected: prior version referenced
  unflattened paths (`/root/rehearsal_src/results/exp_1876_ppmd/...`) that
  do not exist on the pod because the bundle is flat. Current commands use
  the correct on-pod paths and include `pip install --break-system-packages`,
  `DATA_DIR=./`, and a `cp` step to surface per-rank shards into
  `/root/rehearsal_out/` for the launcher's `--download` to fetch.
- ❌ **Blocker: `RUNPOD_API_KEY` is empty in this shell session.** The repo
  `.env` deliberately stores the key empty per security policy. The key
  referenced in the prior session summary returned HTTP 400 (rotated /
  stale). No 1×H100 rehearsal was launched. To proceed, the operator must
  `export RUNPOD_API_KEY=<live key>` and re-run the documented commands
  below — in order: startup smoke first, then 1-window real-eval rehearsal,
  then verify accounting on this HPC, only then consider 8×H100 production.

---

## What was implemented

`scripts/eval_path_b_ppmd.py` previously shipped a Phase 1/2 utility layer
(token-byte tries, optimized trie tables, vectorized target-byte logprob
extraction, NPZ shard helpers, PPM-D with exclusion, streaming mixture
scoring, dry-run CLI). The explicit `--eval` path was a `NotImplementedError`.

This change adds:

1. **Window planning helpers** (CPU-only, torch-free):
   - `plan_sliding_window_starts(total_tokens, seq_len, stride)` — mirrors
     `eval_val_sliding` exactly: `[ws for ws in range(0, total_tokens, stride) if ws + (seq_len-stride) < total_tokens]`.
   - `slice_window_starts_for_rank(window_starts, rank=, world_size=)` —
     mirrors the `n*r//W : n*(r+1)//W` per-rank slicer.
2. **Filename / accounting helpers** (CPU-only):
   - `rank_shard_filename(rank)` → `path_b_sliding_rank{N}.npz`.
   - `rank_accounting_filename(rank)` → `path_b_sliding_accounting_rank{N}.json`.
   - `merged_eval_result_filename(subset_tokens=, full_eval=)`.
   - `emitted_token_byte_count(target_id, prev_id, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)`
     — mirrors the eval_val_sliding formula
     `base_bytes_lut[target] + (has_leading_space[target] & ~is_boundary_token[prev])`,
     correctly returning `0` for special/control/unknown/unused tokens (zero
     `base_bytes` and `has_leading_space=False`) and treating `prev_id < 0` as a
     boundary.
   - `expected_denominator_for_eval(subset_tokens, full_eval=)` — returns
     `151_078_222` for full eval, `29_365_687` for the audited 8M-token subset,
     `None` otherwise.
   - `filter_records_by_subset(records, subset_tokens=)`.
   - `build_per_rank_accounting(...)` and `build_merge_manifest(...)`.
3. **Result schema** — `build_sliding_eval_result(...)` produces the canonical
   final JSON, including all required fields:
   `schema_version`, `path_b_version`, `source_module_path`, `artifact_path`,
   `artifact_size_bytes`, `eval_kind`, `rank`, `world_size`, `subset_tokens`,
   `full_eval`, `scored_token_count`, `scored_byte_count`,
   `expected_denominator`, `denominator_match`, `denominator_formula`,
   `zero_byte_token_count`, `ppm_d_config`, `lambda_gating_config`,
   `shard_manifest_path`, `accounting_audit_path`, `runtime_seconds`,
   `neural_only_bpb`, `ppm_d_only_bpb`, `mixture_bpb`, `warnings`,
   `claim_ready`, `error`. All BPB fields are `None` unless every gate passes
   (no error, summary present, denominator match if expected, positive byte
   and token counts).
4. **Real distributed sliding executor** — `execute_sliding_eval(config, *, output_json_path)`:
   - Dynamically imports `results/exp_1876_ppmd/train_gpt_merged.py` via
     `importlib.util.spec_from_file_location` only inside the heavy path;
     dry-run remains torch-free.
   - Reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from torchrun env; sets the CUDA
     device and initializes NCCL only when `RANK`+`WORLD_SIZE` are present.
   - Instantiates `Hyperparameters()` from the dynamic module, overrides
     `quantized_model_path` to the `--artifact-path` value, sets
     `ppm_enabled=False` (we rebuild byte logic in Path B), and reflects rank
     metadata onto `h`.
   - Builds `ValidationData(h, device)` and `deserialize(h, device)`; toggles
     `eval_model.looping_active = True` when `h.num_loops > 0`.
   - Mirrors `eval_val_sliding` exactly: per-batch, scored slice starts at
     `0` for the first window and `context_size = seq_len - stride` otherwise.
   - For scored positions: runs `forward_logits` under `inference_mode` and
     `autocast(bfloat16)`, takes `softmax(float())` over the vocab, derives
     `mode_flag = 0 if is_boundary_token[prev] else 1`, and calls
     `vectorized_target_path_logprobs(...)` for the **Path B token-trie
     marginalization** (no token-logprob geometric mean).
   - Builds boundary/non-boundary tries from the live SP tokenizer with the
     audit semantics: leading `▁` is stripped, the literal space byte is
     emitted only in non-boundary mode, byte fallback `<0x..>` tokens emit
     their literal byte, and `is_control / is_unknown / is_unused` tokens
     become zero-byte terminals (excluded from continuable-byte denominators
     by `neural_byte_distribution`'s prefix-terminal exclusion).
   - Subset gating: `--subset-tokens N` keeps only positions with
     `absolute_token_position < N`. `--full-eval` overrides the subset filter.
   - Per-rank outputs: `path_b_sliding_rank{N}.npz` and
     `path_b_sliding_accounting_rank{N}.json` (rank, scored_token_count,
     scored_byte_count, zero_byte_token_count, min/max abs positions, shard
     path).
   - Synchronizes ranks via `dist.barrier()`.
   - Rank 0 reads all per-rank NPZ shards via `merge_record_npz_shards` (which
     rejects duplicate / out-of-order records), streams the merged sequence
     once through `score_ppmd_stream(...)` (PPM-D order from
     `--ppmd-order`, prefix-only confidence-gated lambda from
     `ppmd_prefix_lambda` — never reads the target byte), then emits:
     - `path_b_sliding_merge_manifest.json` (per-shard rank, scored_tokens,
       scored_bytes, file_path, sha256).
     - `path_b_sliding_accounting_audit.json` (denominator formula, expected
       and observed denominators, merged record count, zero-byte token count).
     - `path_b_sliding_full.json` or `path_b_sliding_subset_{N}.json`
       (canonical result schema; honors `--output-json` if provided).
5. **Failure mode**: `run_explicit_eval` wraps the executor in a try/except.
   Any exception (executor crash, denominator mismatch, missing summary,
   non-CUDA host) yields `claim_ready=false` and `null` BPB metrics, never a
   fake number.
6. **TTT remains blocked** — `--eval-kind ttt` continues to raise
   `NotImplementedError` via `guard_explicit_eval_kind`.
7. **Dry-run remains the default** and remains torch-free.

The CLI now has `--full-eval` in addition to `--subset-tokens`. Default is the
8 M-token subset matching the existing audit constant.

## Compliance with the audit's hard requirements

| # | Requirement | Where it lives |
|---|---|---|
| 1 | No BPB claimed before audit | `build_sliding_eval_result` gates all BPB fields on `claim_ready` |
| 2 | Score-before-update PPM-D | `score_ppmd_stream` (existing, unchanged) |
| 3 | Token-trie marginalization, not geometric mean | `vectorized_target_path_logprobs` (existing) |
| 4 | Denominator = true emitted target bytes only | `emitted_token_byte_count` and the rank-local accumulator |
| 5 | Full validation = 151,078,222 bytes | `expected_denominator_for_eval(full_eval=True)` |
| 6 | First 8M subset = 29,365,687 bytes | `expected_denominator_for_eval(subset_tokens=8_000_000)` |
| 7 | PPM-D state global after shard merge | Rank 0 streams merged records once through `score_ppmd_stream` |
| 8 | Lambda gating prefix-only, no target peek | `ppmd_prefix_lambda` (existing) ignores `target_byte` |
| 9 | SP `▁` not double-counted | `token_byte_sequences_from_piece` strips `▁` and re-introduces space only in non-boundary mode |
| 10 | Zero-byte specials excluded from continuable denominator | `neural_byte_distribution` denominator = subtree − terminal mass |
| 11 | No RunPod launch | This change does not touch `runpod_*` |
| 12 | TTT stays NotImplementedError | `guard_explicit_eval_kind` |

## Tests

`tests/test_path_b_ppmd_eval.py` runs 30 tests:

- 17 pre-existing primitive tests (token byte sequences, trie/optimized
  tables, vectorized logprob extraction, PPM-D, mixture, score-before-update,
  shard merge, NPZ roundtrip, output schema, eval-kind guards).
- 13 new tests for the Path B sliding eval surface:
  - `test_plan_sliding_window_starts_matches_reference`
  - `test_slice_window_starts_for_rank_partitions_evenly`
  - `test_rank_shard_filename_pattern`
  - `test_filter_records_by_subset_tokens`
  - `test_emitted_token_byte_count_matches_eval_val_formula`
  - `test_expected_denominator_for_eval`
  - `test_build_merge_manifest_schema`
  - `test_build_sliding_eval_result_claim_ready_true`
  - `test_build_sliding_eval_result_claim_ready_false_on_denom_mismatch`
  - `test_build_sliding_eval_result_claim_ready_false_on_error`
  - `test_run_explicit_eval_ttt_remains_blocked`
  - `test_run_explicit_eval_sliding_writes_failed_metrics_when_executor_raises`
  - `test_default_main_does_not_run_sliding_eval`

Local results:

- `/bin/python3.8 -m unittest tests.test_path_b_ppmd_eval` → **30 tests, 29
  pass, 1 skipped** (the vectorized-target torch test; this Python lacks
  torch).
- `.venv-smoke/bin/python3 -m unittest tests.test_path_b_ppmd_eval` →
  **30 tests, 30 pass** (skip resolved when torch is available).
- `/bin/python3.8 -m py_compile scripts/eval_path_b_ppmd.py tests/test_path_b_ppmd_eval.py` → clean.

What local validation **cannot** prove without H100s + the production
artifact: end-to-end correctness of the distributed forward pass, the
`forward_logits` softmax under bfloat16 autocast, and the actual byte-level
PPM-D mixture against a real validation stream. Those gates fire on the pod.

## Remaining risks

- **CUDA-only paths** (`forward_logits`, `softmax`, `autocast`) cannot be
  exercised on this HPC. Their correctness depends on: (a) the dynamic
  exp_1876 module loading without side effects beyond CUDA init, (b) the
  artifact deserializing under the same `Hyperparameters` defaults used at
  training time, and (c) the bfloat16 autocast not perturbing softmax enough
  to materially change Path B byte logprobs vs. fp32. Audit on the rehearsal
  pod by comparing token-level NLL from `eval_val_sliding` to the sum of
  byte-level NLLs from this evaluator across a small slice.
- **`forward_logits` is not `torch.compile`d** in this evaluator (the
  reference path compiles it). This is intentional — keeps the autocast
  surface simple. Throughput will be lower but correctness is unaffected.
- **`PATH_B_BATCH_SEQS` environment knob** defaults to 16. If the pod runs
  out of memory on the larger softmax/probabilities tensor (vs. raw logits),
  drop this to 8 or 4.
- **Per-rank shard files** are written to `--output-json`'s parent directory.
  Make sure that directory survives pod teardown (use a path that's also
  retrieved by the rehearsal manifest).
- **No partial recovery**: if a rank dies mid-eval its NPZ shard will be
  short or missing, and rank-0 merge will fail (claim_ready=false). This is
  the desired conservative behavior.
- **Numerical stability**: `score_ppmd_stream` raises if PPM-D ever assigns
  zero probability to an observed byte. With the implemented PPM-D-with-
  exclusion + uniform-fallback this should not happen, but if it does, the
  result is `claim_ready=false`, never a silent number.

## Authorized future RunPod rehearsal command

**Operator must export `RUNPOD_API_KEY` out-of-band (env var only, never
written to disk).** The repo `.env` file deliberately stores `RUNPOD_API_KEY=`
empty per security policy — see AGENTS.md.

### Launcher capability used: `--extra-file`

`scripts/runpod_http_rehearsal.py` accepts repeated `--extra-file
local_path:arcname` (or just `local_path` to use basename). Files land at
`/root/rehearsal_src/<arcname>` on the pod. Required for Path B because the
artifact + the source module live outside the default `FILES_TO_BUNDLE` set.

### Important on-pod path semantics

The bundle is **flat**: every member is extracted directly under
`/root/rehearsal_src/`. `--train-script <path>` renames the script to
`train_gpt.py` on the pod. So:

| Source on this HPC                                                 | On-pod path                                |
|--------------------------------------------------------------------|--------------------------------------------|
| `scripts/eval_path_b_ppmd.py` via `--train-script`                 | `/root/rehearsal_src/train_gpt.py`         |
| `data/cached_challenge_fineweb.py` (default `FILES_TO_BUNDLE`)     | `/root/rehearsal_src/cached_challenge_fineweb.py` |
| `results/exp_1876_ppmd/train_gpt_merged.py` via `--extra-file`     | `/root/rehearsal_src/train_gpt_merged.py`  |
| `results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz` via `--extra-file` | `/root/rehearsal_src/final_model.int6.ptz` |

### 1×H100 startup smoke (~$0.30, 6 min)

Verifies the pod boots, the bundle extracts, and the dry-run path writes a
schema-compliant JSON. No eval, no GPU work, no data download.

```bash
python3 scripts/runpod_http_rehearsal.py \
  --gpus 1 --max-minutes 6 --pod-name path-b-smoke \
  --train-script scripts/eval_path_b_ppmd.py \
  --cmd "python3 /root/rehearsal_src/train_gpt.py --dry-run \
      --output-json /root/rehearsal_out/dry_run_plan.json \
      > /root/rehearsal_out/pgolf_stdout.txt 2>&1; \
      echo \$? > /root/rehearsal_out/pgolf_exit_code.txt" \
  --download dry_run_plan.json pgolf_stdout.txt pgolf_exit_code.txt status.txt
```

### 1×H100 1-window real-eval rehearsal (~$0.60, 12 min)

Uploads the artifact + source module via `--extra-file`, downloads SP8192 val
data on-pod, runs Path B sliding eval on a 100K-token subset, retrieves all
shards and accounting JSON.

```bash
python3 scripts/runpod_http_rehearsal.py \
  --gpus 1 --max-minutes 12 --pod-name path-b-1gpu \
  --train-script scripts/eval_path_b_ppmd.py \
  --extra-file results/exp_1876_ppmd/train_gpt_merged.py:train_gpt_merged.py \
  --extra-file results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz:final_model.int6.ptz \
  --cmd "set -e; \
      pip install --break-system-packages -r /root/rehearsal_src/requirements.txt \
        > /root/rehearsal_out/pip_install.log 2>&1; \
      cd /root/rehearsal_src && DATA_DIR=./ \
        MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
        python3 cached_challenge_fineweb.py --variant sp8192 \
        > /root/rehearsal_out/data_setup.txt 2>&1; \
      RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
      python3 /root/rehearsal_src/train_gpt.py --eval --eval-kind sliding \
        --source-python /root/rehearsal_src/train_gpt_merged.py \
        --artifact-path /root/rehearsal_src/final_model.int6.ptz \
        --output-json   /root/rehearsal_out/path_b_sliding_subset_100000.json \
        --subset-tokens 100000 \
        > /root/rehearsal_out/pgolf_stdout.txt 2>&1; \
      echo \$? > /root/rehearsal_out/pgolf_exit_code.txt; \
      cp -v /root/rehearsal_src/path_b_sliding_*.npz \
            /root/rehearsal_src/path_b_sliding_*.json /root/rehearsal_out/ 2>/dev/null || true" \
  --download path_b_sliding_subset_100000.json \
             path_b_sliding_rank0.npz path_b_sliding_accounting_rank0.json \
             path_b_sliding_merge_manifest.json path_b_sliding_accounting_audit.json \
             pgolf_stdout.txt pgolf_exit_code.txt status.txt \
             data_setup.txt pip_install.log
```

**Notes on the rehearsal command:**

- `pip install --break-system-packages` is required because the
  `matotezitanka/proteus-pytorch:community` image enforces PEP 668.
- `DATA_DIR=./` is required so `cached_challenge_fineweb.py` writes to the
  expected on-pod working directory (verified during the 1876+PPM-D run).
- The eval script writes per-rank shards next to where it was invoked
  (cwd-relative). The trailing `cp -v` step copies them into
  `/root/rehearsal_out/` so the launcher's `--download` can fetch them.
- 100K subset tokens chosen to fit comfortably in 12-min wallclock with
  margin for data download (~3 min) and pip install (~30 s).

After artifact retrieval **must** verify on this HPC:

1. `path_b_sliding_accounting_audit.json` `denominator_match == true`
   (or expected_denominator is null and the reported `scored_byte_count` is
   reproducible).
2. `path_b_sliding_subset_*.json` `claim_ready == true`.
3. The reported `scored_byte_count` agrees with the merged NPZ shard
   (re-run `merge_record_npz_shards` locally and re-tally bytes).

Only after the 1×H100 rehearsal completes successfully and is fully
retrieved is an 8×H100 production eval authorized — at the user's explicit
direction with explicit budget approval.

## Disclaimer

**No BPB claim is valid until a fresh authorized run completes and the
accounting audit passes.** The implementation in this PR adds the machinery
to produce a defensible Path B BPB number; it does not by itself produce one.
The session that performs the actual production eval must (a) confirm the
denominator matches the audited expected value for the chosen subset, (b)
inspect the shard manifest, and (c) only then quote the `mixture_bpb` /
`neural_only_bpb` / `ppm_d_only_bpb` from `claim_ready=true` JSON output.
