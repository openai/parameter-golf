# Path B Byte Eval Red-Team Review

## Objective

Red-team the proposed Path B non-record byte-level evaluator before any RunPod execution or BPB claim. Path B would score the existing exp_1876 PPM-D artifact with a normalized byte predictor formed from token-level neural probabilities, then mix that byte stream with a proper score-before-update PPM-D byte model.

Target artifact under review:

- Model artifact: `results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz`
- Artifact size: **15,975,706 bytes**
- Full validation target: **40,540,160 tokens**, **151,078,222 target bytes**
- First 8M-token subset: **29,365,687 target bytes**
- RunPod 8×H100 SXM rate from repo guidance: **~$23.92/hr** (`$2.99/GPU/hr`)

## Verdict

**Not claim-ready.** The utility layer implemented in `scripts/eval_path_b_ppmd.py` has useful correctness scaffolding for token-byte tries, vectorized target-byte logprobs, deterministic shard merging, and streaming PPM-D scoring. However, full model-loading sliding evaluation is still explicitly guarded and not wired, and the `ttt` path is intentionally incomplete. No `val_bpb` claim should be made until full eval wiring is complete, the byte accounting is audited end-to-end, and a fresh full-validation run produces retrievable logs and audit metadata.

Path B is reasonable to continue as an exploratory non-record eval only if the acceptance gates below are met in order. A surprisingly low BPB should be treated first as an accounting bug until proven otherwise. Tiny denominator gremlins love leaderboard trophies.

## Cost and Runtime Expectations

These estimates are planning values, not measured Path B production results.

| Path | Scope | Estimated runtime | Estimated 8×H100 cost | Red-team interpretation |
| --- | --- | ---: | ---: | --- |
| Existing quantized eval | Existing baseline timing | ~25.7s | ~$0.17 | Useful sanity bound only; not Path B. |
| Existing sliding eval | Existing baseline timing | ~126.6s | ~$0.84 | Useful reference for model forward cost. |
| Existing TTT eval | Existing baseline timing | ~510.8s | ~$3.39 | TTT accounting is not implemented for Path B. |
| Python PPM first 8M postpass | Invalid/reference postpass | ~129.7s | ~$0.86 | Demonstrates Python overhead can matter. |
| Optimized Path B first 8M | Planned subset eval | 3-6 min | ~$1.20-$2.40 | Good first real signal after wiring and audit. |
| Python/reference first 8M | Conservative subset eval | 8-15 min | ~$3.20-$6.00 | Fallback only; too slow to iterate freely. |
| Optimized full validation with TTT | Planned but not wired | 9-15 min | ~$3.60-$6.00 | Do not run/claim until TTT path and accounting exist. |
| Python/reference full validation | Conservative fallback | 30-60 min | ~$12-$24 | Budget risk; use only if explicitly approved. |
| Sliding-only/no TTT optimized | Planned full eval | likely 5-10 min | ~$2.00-$4.00 | Preferred first production-size Path B target after subset gates. |

## Critical Risks, Mitigations, and Acceptance Gates

| Risk | Why it matters | Mitigation | Acceptance gate |
| --- | --- | --- | --- |
| **Official token-boundary byte decomposition could be challenged vs all possible token segmentations.** | Path B marginalizes bytes along the official validation token stream. A reviewer could argue a normalized byte model should marginalize over every tokenizer segmentation that emits the same byte prefix, not only the official next-token distribution at known token boundaries. | Document the exact semantics: neural probabilities are conditioned on the official tokenized validation prefix, then decomposed into emitted target bytes. If claiming comparability, prove this matches the contest evaluator's tokenization assumptions or label it as a non-record exploratory metric. | Written audit section states the segmentation assumption and includes at least one independent review. No BPB claim unless this assumption is accepted for the intended track. |
| **Denominator/accounting drift.** | BPB can silently improve if the denominator uses tokens, total raw file bytes, context bytes, or emitted bytes inconsistently. | Normalize only by scored target bytes. Preserve constants: full validation `151,078,222` bytes; first 8M tokens `29,365,687` bytes. Output JSON must include byte denominator, token count, skipped bytes/tokens, and formula. | Dry-run and eval JSON expose denominator metadata; full run denominator matches the audited constant for the selected scope. |
| **SentencePiece leading-space double count.** | The `▁` marker can accidentally count the same literal space once in token LUT construction and again in per-token boundary logic. | Keep one canonical token-byte helper. Audit boundary and non-boundary modes with marker-only, leading-space, byte-fallback, and plain-token examples. | Unit tests for leading-space modes pass, and eval audit samples prove no emitted space is double-counted. |
| **Zero-byte special/control tokens.** | Special tokens can place terminal mass at the trie root; if included in next-byte denominators they distort normalized byte probabilities. | Treat special/control/unknown/unused tokens as zero-byte terminals and exclude exact terminal mass from continuable-byte denominators. | Root distribution tests pass with high special-token mass; full eval reports count and treatment of zero-byte tokens. |
| **PPM-D score-before-update ordering.** | Updating PPM-D before scoring leaks the target byte into its own probability and invalidates BPB. | Use score-before-update everywhere. Keep PPM-D scoring in a single streaming function whose tests verify old-state probability is used. | `test_score_before_update_ordering` and stream-score tests pass; eval logs state score-before-update explicitly. |
| **Rank-local PPM state invalidity.** | PPM-D is sequential over the global byte stream. Independent per-rank PPM states would produce invalid scores because each rank lacks previous bytes from other ranks. | Let ranks emit neural byte logprob shards only. Merge shards by absolute token position and byte offset, then stream the merged global byte sequence through one PPM-D state on rank 0 or an equivalent globally ordered serial pass. | Shard merge rejects duplicates/out-of-order records; production run writes a merge manifest and a single global PPM-D score summary. |
| **Target-dependent lambda/gating cheating.** | Mixture weights must not depend on the target byte being scored, or the scorer can peek at the answer. | Gate lambda from prefix-only PPM confidence or fixed config. Treat `target_byte` as validation-only input that must not affect gate selection. | Tests and code review confirm `ppmd_prefix_lambda` ignores the target symbol; audit metadata records lambda schedule parameters. |
| **Python PPM-D too slow for full eval.** | A pure-Python PPM postpass may push full validation past the intended 10-minute budget and increase RunPod spend. | Start with dry-runs and 8M subset. Profile the merged stream scorer. If needed, add a compiled/NumPy/C++/Rust PPM-D path after proving parity with Python reference. | First 8M subset completes inside the approved budget; full eval launch requires an explicit runtime estimate and timeout. |
| **Incomplete `ttt` path and incomplete full sliding wiring.** | The current CLI intentionally refuses real eval. Reporting a BPB from the skeleton would be fake. TTT accounting is even more delicate because adaptation must not train on unseen validation bytes. | Keep explicit guard until `--eval --eval-kind sliding` is fully implemented and audited. Keep `ttt` disabled unless a separate TTT accounting plan and tests are added. | `--eval --eval-kind sliding` loads the real model, emits rank `.npz` shards, merges, streams PPM-D, and writes JSON. `--eval-kind ttt` remains blocked unless separately implemented and reviewed. |
| **RunPod credential, safety, retrieval, and teardown risks.** | Lost credentials, accidental uploads, SSH/Jupyter failures, unretrieved artifacts, or idle pods can compromise security and budget. | Use only environment-provided credentials. Do not write or echo secrets. Use HTTP-bootstrap via `scripts/runpod_http_rehearsal.py`; do not use SSH, `scp`, `rsync`, or Jupyter upload APIs. Require pod-side timed shutdown, retrieval before teardown, and local verification of non-empty outputs. | Every RunPod stage has explicit approval, max minutes, expected cost, pod-side shutdown, expected downloads, and verified retrieved file sizes before considering the stage complete. |

## Acceptance Gates

### Gate 0: Local documentation and utility readiness

- Existing tests for `tests/test_path_b_ppmd_eval.py` pass locally.
- Dry-run JSON can be generated without importing torch-heavy model code.
- Red-team risks above are acknowledged in the execution prompt.
- No RunPod launch command is executed during planning.

### Gate 1: Real sliding eval wiring completed locally

- `scripts/eval_path_b_ppmd.py --eval --eval-kind sliding` is no longer a guarded skeleton.
- The real eval path dynamically imports `results/exp_1876_ppmd/train_gpt_merged.py` only in explicit eval mode.
- It instantiates the exp_1876 `Hyperparameters` and `ValidationData` objects, points to `results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz`, initializes distributed CUDA only for explicit eval, loads via `deserialize`, mirrors `eval_val_sliding` windowing, computes neural probabilities, emits byte logprob `.npz` rank shards, merges on rank 0, and streams PPM-D exactly once globally.
- Output JSON includes audit metadata, denominator formula, byte counts, rank shard manifest, artifact path, artifact size, and exact code/config versions.

### Gate 2: 1×H100 smoke and subset rehearsal

- Use HTTP-bootstrap only.
- First run is a smoke/dry-run that retrieves `status.txt`, `pgolf_stdout.txt`, dry-run JSON, and launcher metadata.
- Second run is a short eval subset that retrieves at least one `.npz` rank shard, one eval JSON, and logs.
- Retrieved files are non-empty and byte counts match the selected subset.

### Gate 3: 8×H100 non-record full eval

- Human explicitly authorizes budget and launch.
- Full eval uses `--eval-kind sliding` only unless a separate TTT audit exists.
- The launch has a pod-side hard deadline and enough retrieval buffer.
- All expected JSON/log/shard artifacts are downloaded and verified before teardown.

### Gate 4: Post-run audit before any claim

- Recompute or independently audit denominator from emitted byte records.
- Verify no leading-space double count and no target-dependent gate.
- Verify global PPM-D state was not rank-local.
- Compare a small prefix against a brute-force/reference path.
- Only after this audit may a BPB be discussed, and even then it should be labeled non-record unless the token-boundary semantics are accepted.

## Claim Policy

No BPB claim should be made until full eval wiring is complete and accounting is audited. Until then, outputs from `scripts/eval_path_b_ppmd.py` are planning, dry-run, or subset-debug artifacts only. If any run reports a surprisingly strong result, pause and audit denominator, token-byte mapping, PPM update ordering, and shard merge ordering before interpreting it as a modeling improvement.