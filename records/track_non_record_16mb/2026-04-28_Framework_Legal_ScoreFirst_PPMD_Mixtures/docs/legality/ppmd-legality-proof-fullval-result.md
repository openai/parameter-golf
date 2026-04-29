# Path B PPM-D Legality-Proof — Full-Val Result Note

**Date:** 2026-04-29
**Artifact under audit:** `results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz`
(15,975,706 bytes, exp_1876 PR #1876 + PPM-D mixture stack, seed 42)
**Source module:** `results/exp_1876_ppmd/train_gpt_merged.py` (58,439 bytes)

## Summary

A defensible byte-level Path B evaluation of the exp_1876 PPM-D mixture
artifact has now been completed over the **full** FineWeb sp8192
validation split (151,078,222 emitted target bytes, denominator
fingerprint matched exactly).

**Result JSON:**
`results/exp_1876_ppmd/path_b_prod_8gpu_fullval_local_score/path_b_sliding_full.json`

**Headline metrics (full validation set, audited):**

| Metric | Value |
| --- | --- |
| `mixture_bpb` | **1.5221** |
| `neural_only_bpb` | **1.5430** |
| `ppm_d_only_bpb` | **2.0319** |
| `denominator_match` | true (151,078,222 / 151,078,222) |
| `claim_ready` | true |
| `full_eval` | true |
| `subset_tokens` | null |
| `scored_token_count` | 40,540,160 |
| `zero_byte_token_count` | 49,999 |
| `runtime_seconds` (PPM-D scoring only) | 33,731.1 (≈562 min) |

## Comparison to 8M-Subset Audit

The earlier 8M-token subset audit (see
`plans/ppmd-legality-proof-result.md`) reported on 29,365,687 bytes:

| Metric | 8M subset | Full val | Δ (full − subset) |
| --- | --- | --- | --- |
| `mixture_bpb` | 1.5459 | **1.5221** | −0.0238 |
| `neural_only_bpb` | 1.5619 | **1.5430** | −0.0189 |
| `ppm_d_only_bpb` | 2.1184 | **2.0319** | −0.0865 |
| `scored_byte_count` | 29,365,687 | **151,078,222** | × 5.14 |

The full-val numbers are uniformly slightly lower than the subset
numbers, consistent with the typical per-validation-shard variance and
the fact that the first 8M tokens contain a non-trivial amount of
hard-to-model boundary/header text. Crucially, both estimators now
agree to within ≈0.024 BPB on the mixture, with the full-val number
being the authoritative one.

## Comparison to PR #1876 Reported Number

The original PR #1876 + PPM-D submission reported a contest BPB of
roughly **0.99487** computed via the in-source `_ppm_mixture_bpb`
helper. That helper:

1. Used per-token neural NLL as if it were per-byte log-prob,
2. Mixed in a token-level (not byte-level) PPM-D model, and
3. Normalised by an estimated byte count, not the true emitted-byte
   denominator validated against the contest fingerprint.

The audited full-val Path B byte-level mixture, which:

1. Marginalises neural logprobs over true emitted byte sequences via
   token-byte tries,
2. Models PPM-D over actual bytes with score-before-update,
3. Uses prefix-only (target-blind) lambda gating,
4. Normalises by the true emitted byte count
   (`scored_byte_count = 151,078,222`, matching the full-val
   denominator fingerprint),

reports **mixture_bpb = 1.5221** on the same model, on the same
validation set used by the contest scorer. The legality gap on the
contest metric is therefore approximately:

> **Δ = 1.5221 − 0.99487 ≈ +0.527 BPB**

This is a direct, full-validation-set empirical confirmation of
openai/parameter-golf Issue #1872: the `_ppm_mixture_bpb` figure used
in the original PR #1876 + PPM-D submission is **not a valid
bits-per-byte score** for the contest, and it inflates the apparent
performance by more than half a bit per byte. The previously cached
post-TTT neural-only number (1.08122 BPB with TTT) was produced by a
different evaluation path entirely and is not invalidated by this
finding; only the PPM-D mixture claim is.

## Acceptance Gates Passed

* `denominator_match` = true (full-val, 151,078,222 / 151,078,222)
* `expected_denominator` = 151,078,222 (matches contest fingerprint)
* `scored_byte_count` = 151,078,222
* `scored_token_count` = 40,540,160 across 8 ranks
* `zero_byte_token_count` = 49,999 (BOS/sentinel tokens)
* `full_eval` = true, `subset_tokens` = null
* `claim_ready` = true
* All 8 rank shards retrieved with sha256 sums matching the on-pod
  merge manifest (`path_b_sliding_merge_manifest.json`)
* PPM-D order 5, score-before-update preserved
* Lambda gating prefix-only (target byte never inspected before scoring)
* `fast_score_ppmd_stream` re-verified bit-identical to reference
  `score_ppmd_stream` to 1e-12 on the first 5,000 records of the prior
  8M-subset shards before being used on the full stream
* Zero `warnings`, no `error` in result JSON

## Operational Notes

### Windowing pass (8×H100 on RunPod)

* Pod: `pgolf-fullval-8gpu` (id `ts3eboy7kvxhjb`),
  8×NVIDIA H100 80GB HBM3, $21.52/hr, image
  `matotezitanka/proteus-pytorch:community`.
* Launcher: `scripts/runpod_http_rehearsal.py` with node-side timed
  shutdown (`--max-minutes 80`, `--runtime-timeout-sec 4500`).
* Shell command performed: pip install requirements, urllib download
  of `final_model.int6.ptz` and `train_gpt_merged.py` from a
  transient GitHub release, val-only data download
  (`cached_challenge_fineweb.py --variant sp8192 --train-shards 0`),
  then `torchrun --standalone --nproc_per_node=8 train_gpt.py --eval
  --eval-kind sliding --full-eval --source-python train_gpt_merged.py
  --artifact-path final_model.int6.ptz --output-json
  /root/rehearsal_out/path_b_sliding_full.json` (with
  `scripts/eval_path_b_ppmd.py` substituted for `train_gpt.py` via
  `--train-script`).
* Per-rank windowing + merge completed in ≈26 minutes wallclock; the
  on-pod PPM-D scoring step was *not* awaited (would have exceeded
  the 75-minute pod budget). Instead, all 8 rank NPZs (≈1.2 GB total),
  the per-rank accounting JSONs, the merge manifest, the accounting
  audit, and ancillary logs were retrieved via direct HTTP, the pod
  was terminated via `runpod_safe.terminate_and_wait`, and the
  transient GitHub release was deleted.
* Rank shard sizes after retrieval (bytes): 149,271,398; 151,903,981;
  152,518,868; 149,535,253; 152,731,770; 152,708,615; 150,123,118;
  153,329,861. All sha256 sums match the manifest.
* RunPod cost for this audit: balance went from $145.69 to $124.89
  ⇒ **≈ $20.80 spent**, well under the $75 cap and the $30 nominal
  pod budget.

### PPM-D scoring (local SLURM)

* Script: `scripts/fast_score.py` (collapsed-`distribution()` variant
  of `eval_path_b_ppmd.score_ppmd_stream`, extended in this session
  with a `--full-eval` flag that passes `subset_tokens=None,
  full_eval=True` through to `build_sliding_eval_result`).
* SBATCH: `cpu_medium` partition, 1 node, 2 CPUs, **128 GB RAM**
  (16 GB OOMed during merge of the full ~1.2 GB-of-NPZ stream),
  12-hour walltime, `module load conda/py312/1.1`.
* Job 56618724 ran on `c002`. Merge of all 8 rank shards into the
  151,078,222-record stream took ≈18 minutes; PPM-D scoring took
  **33,731 s ≈ 562 min ≈ 9.4 h** at ≈4,500 records/s sustained.
  Monotonic mix_bpb_so_far converged to 1.52208 over the run.

## Risks / Caveats

* **No TTT.** Path B sliding eval is the audit path, not a contest
  score. The `--eval-kind ttt` branch in
  `scripts/eval_path_b_ppmd.py` remains intentionally
  `NotImplementedError`. The audited 1.5221 mixture BPB is what the
  PR #1876 + PPM-D stack achieves on the full validation set in the
  byte-level mixture path *without* TTT and without illegal estimators.
* **Other-account pods on RunPod.** During this run, two unrelated
  pods owned by other users on the shared RunPod account appeared
  (`ppmd-cuda-prefix1k-a100-1x` etc.). They were not touched. The
  `pgolf-fullval-8gpu` pod owned by this session was the only pod
  launched and was confirmed terminated; final balance check shows
  no `pgolf-*` pods remaining.
* **Single-seed audit.** This is one seed (42) of one artifact. Any
  contest claim still requires the usual 3-seed statistical-
  significance evidence at the contest metric (which would itself
  not include the illegal `_ppm_mixture_bpb` helper).

## Conclusion

The audited full-validation Path B byte-level mixture BPB for the
PR #1876 + PPM-D artifact (`final_model.int6.ptz`, seed 42) is
**1.5221**, which is approximately **+0.527 BPB worse** than the
0.995 figure originally reported via the in-source
`_ppm_mixture_bpb` helper. The legality concern from
openai/parameter-golf Issue #1872 is now confirmed at full-val
resolution with `denominator_match` against the contest fingerprint.
