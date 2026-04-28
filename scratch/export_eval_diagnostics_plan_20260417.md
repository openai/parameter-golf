# Export + Eval Diagnostic Plan — 2026-04-17

**Scope.** Deep-dive diagnostic plan for the EXPORT and EVAL phases of Parameter
Golf, branch `skc_competition_sp8192`. Document only; no code changes.

**Primary sources.** All line references are against the current working tree.

- `train_gpt_verbose.py` (5006 lines; canonical source)
- `build_submission.py` (240 lines)
- `triton_kernels.py` (605 lines)
- `scratch/skc_engram_diagnosis_20260417.md` (prior diagnosis)

> Submission is written to `final_model.ternary.ptz` and the code file
> `train_gpt.py` produced by `build_submission.py`. The competition scorer reads
> both; their combined size must be `<= HARD_BUDGET_BYTES` (default
> 16_000_000, `train_gpt_verbose.py:425`).

---

## 0. Architecture of the export/eval pipeline (as-is)

### 0.1 Export pipeline

1. **Budget lower-bound precheck** (before training proper completes in eval
   paths, and unconditionally at export).
   - `estimate_export_lower_bound_bytes` returns `(total_lb, ternary_lb, fp_lb,
     code_lb)` (`train_gpt_verbose.py:881-882`).
   - Precheck log and raise: `train_gpt_verbose.py:4019-4022` raises
     `Projected minimum artifact+code bytes … exceeds HARD_BUDGET_BYTES`.
2. **Quantize + pack state dict.**
   - `pack_ternary(q)` → `(packed_bytes, n_trits)` groups by 5 trits/byte
     encoding (`train_gpt_verbose.py:756`).
   - `pack_ternary_bitmask(q)` alt path at `:773`, chosen by policy.
   - Per-tensor entry stores: `{'type', 'packed', 'scale': scale.half(),
     'shape', 'padded_cols', 'group_size', 'n_trits', 'orig_shape'}` — scales
     are fp16 (`q_stats['ternary_bytes'] += len(packed_bytes) + scale.numel()*2`).
   - Which params are ternary: `export_ternary_param_names` (`:816`), which are
     kept fp16: `export_fp16_param_names` (`:824`). The filter `CTP = (...)`
     keeps small critical scalars fp (`:754`).
3. **Codec selection** — multiple codecs tried (zstd/lzma/brotli), best wins;
   `torch.save(q_obj, buf)` at `:4721`; `full_blob = codec_header + final_blob`
   (`:4742`).
4. **Write artifact** `final_model.ternary.ptz` at `:4743-4744` (and
   `final_model.competition.ptz` mirror at `:4746-4748`).
5. **Hard-budget gate post-export** (`:4754-4757`):
   `total = artifact_bytes + code_bytes` (`code_bytes = get_fresh_code_bytes(args)`
   at `:4751`).
6. **Round-trip verify** — reload blob (`:4768`), decode, dequantize,
   `load_roundtrip_state_strict(base_model, deq_sd(loaded, target_dtype=
   torch.float32))` (`:4798`). Evaluates `final_ternary_roundtrip val_loss` /
   `val_bpb` at `:4801-4803` and records `roundtrip_val_bpb`.
7. **Scoreboard** at `:4982-4984` picks
   `submission_bpb = min(roundtrip_val_bpb, augmented_val_bpb)`.
8. **Submission code emit** — `build_submission.py:166-191`:
   - `_inline_triton_kernels` inlines `triton_kernels.py` content (strips
     `__main__` and `from __future__ import annotations`), replacing
     `import triton_kernels` in-place (`build_submission.py:117-162`).
   - `minify_python` / `_strip_comments` shrink code.
   - Codec wrapper chosen (`_select_wrapper`, `:107-114`).

### 0.2 Eval pipeline

- **Sliding eval (no TTT)** `eval_val_sliding` at `train_gpt_verbose.py:3338`;
  stride defaults to 64 (`SLIDING_EVAL_STRIDE`, `:338`); `seq_len =
  args.train_seq_len`. Windows scored with `score_from = 0 if start == 0 else
  seq_len - stride`, so the FIRST window contributes the entire window and all
  subsequent windows contribute only the last `stride` tokens.
- **Legal-TTT sliding eval** `eval_val_sliding_ttt` at `:3470`.
  - Chunked by `ttt_chunk_tokens` (default 32768, `:358`).
  - Score-first, then adapt per chunk (comment at `:3512-3514`, `:3575-3583`).
  - EvalEngram absorb occurs AFTER the chunk is fully scored under
    `torch.no_grad()` (`:3575-3583`).
- **EvalEngram** class at `:1493-1549`:
  - Non-persistent buffers (`persistent=False`, `:1525`) — not in state_dict.
  - Storage per (order, head): `logit_sum` fp32 `(O, H, B, V)`; `count` int32
    `(O, H, B)`. With defaults `O=3, H=4, B≈?, V≈8200` this is
    `3*4*B*8200*4` bytes — a B=4096 choice = ~1.6 GB. A DQ risk at eval time.
- **Augmented eval path (engram-mixed logits)** at `:3199-3201` mixes
  `ee_logits` into model logits by `alpha` and `entropy_thr`.
- **Final selection** `submission_bpb = min(roundtrip, augmented)` (`:4984`).

### 0.3 Data / tokenizer

- Tokenizer path `./data/tokenizers/fineweb_8192_bpe.model` (`:190-192`,
  `:491-492`), loaded via `spm.SentencePieceProcessor` at `:3979`. **Not
  bundled** into `final_model.ternary.ptz`. The submission code file must
  locate a tokenizer at the scorer-provided path (or embed/fetch it).
- Hard fail on tokenizer/data family mismatch at `:3981-3996` (sp1024 vs sp8192
  regime guard).

---

## A. Export diagnostics

### A.1 Byte-exact size breakdown

**Objective.** Decompose `artifact_bytes` into (ternary payload, scales,
non-ternary fp16 params, codec header, pickle overhead, tokenizer-derived
constants). Current log at `:4753` only reports `ternary_params/ternary_bytes /
fp_params / fp_bytes / code`. Need per-tensor and per-category accounting.

**Current behavior issues.**

- `q_stats['ternary_bytes'] += len(packed_bytes) + scale.numel() * 2` counts
  raw payload before `torch.save` overhead and before outer codec compression.
  There is no accounting for the `torch.save` header per entry (dict keys,
  pickle metadata).
- After codec (`codec_header + final_blob`, `:4742`) actual size can differ
  from `fp_bytes + ternary_bytes` by up to tens of KB — not surfaced to the
  budget estimator at `:881-882` and `estimate_export_lower_bound_bytes`.
- `code_bytes` is `get_fresh_code_bytes(args)` (`:4751`) which invokes
  `build_submission.py` indirectly via `:891`, but build result is not
  round-trip audited for determinism (see A.2).

**What to measure.**

- Per-tensor: `param_name, shape, group_size, packed_len, scale_bytes,
  fp16_bytes, pickle_overhead, ratio_vs_fp16_dense`.
- Category totals: attention blocks, MLP, embeddings (tied?), SKC mixer
  (`mixer_diag/lowrank/conv/scale`, `:754`), capsule tables, vocab_bias,
  koopman, EMA NOT-exported sanity check.
- Final codec: name, uncompressed size, compressed size, codec_header length.

### A.2 Determinism of export

**Current behavior.** `build_submission.py` uses AST transforms; minification is
idempotent only if input `train_gpt_verbose.py` is byte-stable. No seed set
during `torch.save(q_obj, buf)` (`:4721`); pickle of dicts in CPython 3.x is
deterministic for fixed-dtype tensors, but ordering of `q_obj` keys depends on
python dict insertion order over `model.named_parameters()` — should be stable
but is not asserted.

**Tests.** Run export twice with `PYTHONHASHSEED=0` on identical weights; diff
`final_model.ternary.ptz` byte-by-byte and `train_gpt.py`.

### A.3 Round-trip fidelity (export → reload → logit diff)

**Current behavior.** `final_ternary_roundtrip val_loss` at `:4801` is the only
numerical guard. There is no *token-for-token* logit delta check between the
in-memory model (post-training, post-EMA-applied) and the reloaded quantized
model.

**Gap.** `load_roundtrip_state_strict` (`:4798`) runs with
`target_dtype=torch.float32`, but the training forward uses bf16. Any cast loss
in dequant path is hidden.

**What to measure.**

- `logit L2 delta` and `argmax agreement rate` on a fixed calibration batch for
  (a) pre-quant fp32 weights, (b) pre-quant bf16, (c) reloaded ternary, (d)
  reloaded ternary via submission `train_gpt.py` rebuild.
- Loss delta `roundtrip_bpb - final_bpb` should be ≤ 0.01 BPB for a safe
  submission.

### A.4 Kernel-inlining numerical parity (verbose vs submission)

**Current behavior.** `_inline_triton_kernels` replaces `import triton_kernels`
with the file contents, strips parity test functions via AST visitor
(`build_submission.py:125-140`), strips `__main__` blocks (`:145-147`), removes
`from __future__ import annotations` (`:156`). No automated parity test after
inlining.

**Gap.**

- If a kernel depends on a module-level symbol defined only in
  `triton_kernels.py` (e.g. `_get_hadamard`, `triton_kernels.py:31`,
  `HAS_TRITON` guard), the inlined version may silently diverge at runtime.
- `_strip_comments` (`:52-63`) is line-based; multi-line string literals that
  start with `#` inside docstrings could be incorrectly affected in edge cases.

**What to measure.** For every Triton entry in the call path (`triton_fwht_
blockwise`, `triton_parallel_scan`, `triton_ternary_dequant`,
`triton_engram_hash_gather`, `triton_spectral_decay_scan`,
`triton_rms_norm`): compare output of verbose vs inlined on matched random
inputs.

### A.5 Size-budget guardrails

Two current gates:

1. Pre-quant lower bound (`:4019-4022`) — uses
   `estimate_export_lower_bound_bytes` which does NOT include pickle/codec
   overhead, so the lower bound can be overly optimistic.
2. Post-export hard gate (`:4754-4757`) raises on breach.

**Gaps.**

- No *pre-training* simulation of export at `step=0` (optimistically packs
  random-init weights and checks budget). A late failure at minute 9 wastes 10
  minutes of compute.
- No tolerance margin: `total > hard_budget_bytes` is checked, but common DQ
  causes (`build_submission.py` output 50 KB larger than expected due to new
  code) cannot trigger an early warning.
- `get_fresh_code_bytes` invocation path can re-enter `build_submission.py`
  during training (`:891`), adding I/O jitter.

---

## B. Eval diagnostics

### B.1 Sliding-window correctness

**Current behavior (`eval_val_sliding`, `:3338-3408`).**

- `all_starts = list(range(0, total_tokens, stride))` (`:3346`).
- Window `seq_len = args.train_seq_len`, window end `min(start+seq_len,
  total_tokens)`; `wlen = end - start`.
- `score_from = 0 if start == 0 else seq_len - stride` (`:3395`).
- `nll` cross-entropy, scored `[score_from:wlen]`.
- `bpb = val_loss / log(2) * (token_count / byte_count)` (`:3406-3408`).

**Gaps / risks.**

- **Last partial window.** Windows where `wlen < seq_len` still apply
  `score_from = seq_len - stride`; if `wlen < seq_len - stride`, slice is
  empty and `token_count` underflows. Needs explicit handling (checked at
  `:3395` area — verify).
- **First-window bias.** First window contributes `seq_len` tokens; all others
  contribute `stride`. If `total_tokens < seq_len + stride`, ratio is skewed.
- **Byte accounting.** `byte_count` is accumulated via `base_bytes_lut` indexed
  by sy tokens (`:3399` area). Verify that `score_from` is applied identically
  to `sx` and `sy` (it is: `sx = x_batch[j, score_from:wlen]`, `sy =
  y_batch[j, score_from:wlen]`).

### B.2 Legal-TTT legality

**Current behavior (`eval_val_sliding_ttt`, `:3470-3600+`).**

- Chunks: `ttt_chunk = args.ttt_chunk_tokens`, `window_starts` computed at
  `:3490` with filter `min(ws + seq_len, total_tokens) - ws >= stride or ws ==
  0`.
- Each chunk: SCORE under `torch.no_grad()` first (`:3568` area:
  `score_from = 0 if ws == 0 else seq_len - stride`), THEN adapt + absorb.
- `collect_ttt_params` (`:3410`) selects which params get gradients by
  `args.ttt_scope`.

**Gaps / risks to legality.**

- **Gradient scope leakage.** No assertion that `requires_grad` is reset to
  `False` for ALL non-TTT parameters before adapt phase. If TTT_SCOPE is
  misconfigured (`'feedback'` default at `:355`, `'skc_safe'` at `:528`) and a
  shared parameter is inadvertently included, it leaks information across
  chunks in an illegal way. No hard gate in code — must add a snapshot/diff.
- **Persisted state scope.** EvalEngram tables are NOT reset between chunks
  (intentional — memory is cumulative). But also NOT reset between eval runs.
  If `final_sliding` runs first and then `legal_ttt` runs (`:4843`, `:4865`),
  engram counts could carry over unless `maybe_build_eval_engram` rebuilds from
  scratch — currently idempotent (`:3152`), so second call is a no-op and
  state persists. This is a LEGALITY risk: engram accumulated during the
  first eval observes tokens that would otherwise be "future" in the second.
- **RNG determinism.** No explicit RNG seeding inside
  `eval_val_sliding_ttt` TTT step loop. If TTT uses dropout or any stochastic
  op, reruns differ — DQ-adjacent (non-reproducible).
- **Param backup/restore.** Search for `backup_params`/`restore_params`
  returned zero hits. If TTT mutates weights in place and does not restore
  after each chunk, the eval measures a different model than the exported
  artifact.
- **Absorb boundary (commit 8a8c73b).** Code comment confirms absorb is
  post-score (`:3575-3583`). Verify that for the LAST partial chunk, absorb is
  skipped (since there is no next chunk to benefit). Currently not gated.

### B.3 EvalEngram behavior

- Build: `maybe_build_eval_engram` at `:3148-3161`.
- Logits mix: `:3199-3201` uses `alpha = _eval_engram_alpha` (default 0.05,
  `:292`), `thr = _eval_engram_entropy_thr` (2.0, `:293`).
- Laplace smoothing `:294` default 1.0.

**Gaps.**

- **Memory footprint.** `logit_sum` fp32 `(O, H, B, V)`. Even at O=3, H=4,
  B=1024, V=8200 this is ~403 MB. Needs a sanity log at build time.
- **Hit rate.** No counter of how many token positions received a non-trivial
  mixing (i.e. `count[o,h,bucket] > 0`). Without this metric we cannot tell
  whether engram gating is firing.
- **Persisted vs transient.** `persistent=False` means EvalEngram is NOT
  saved/reloaded during export. Round-trip therefore cannot validate engram
  behavior reproducibility.

### B.4 Eval wallclock & determinism

- `eval_time_ms` logged for sliding (`:4843`) and legal_ttt (`:4865`). No
  per-window or per-chunk timing.
- DDP uses `torch.distributed` reduce for loss/token/byte accumulation
  (inferred from `world_size` use at `:3395` region). Determinism depends on
  reduction ordering; at world_size=8 on H100 this is normally deterministic
  for SUM reductions but NOT guaranteed for AVG via reduce. Audit the reduce
  op used.

### B.5 Full-eval vs sliding-eval parity on dev slice

- Non-sliding eval branch at `eval_val` (near `:3332`) computes BPB via
  `val_loss / log(2) * (token_count / byte_count)`.
- Both paths should agree when `stride == seq_len` (non-overlapping). No
  regression test exists.

---

## C. Integration diagnostics

### C.1 Train/eval parameter divergence

- **EMA.** `EMAHelper` at `:3885-3902`. `apply_shadow` swaps; `restore`
  reverts. Eval uses shadow when `EMA_EVAL_APPLY=1` and
  `step_fraction(step) >= ema_start_fraction` (`:4297-4305`).
  - Risk: `apply_shadow` uses `.copy_(.to(p.device))` — if shadow dtype is
    fp32 and `p.data` is bf16, fidelity is preserved but if reversed, it
    silently downcasts.
  - Export uses `base_model` state at `:4484` (`_proxy_roundtrip_bpb`). Must
    verify EMA shadow is applied BEFORE quantize.
- **Quantization.** Quant happens at export; during training the model is
  still fp. There is no calibration-aware QAT here; mismatch between
  bf16 final weights and ternary-reloaded weights is the largest source of
  roundtrip BPB gap.
- **TTT reset.** Per-chunk state. See B.2.

### C.2 Data path

- sp8192 tokenizer at `:190-192`, `:491-492`.
- Boundary/leading-space LUTs (`base_bytes_lut`, `has_leading_space_lut`,
  `is_boundary_token_lut`) passed into every eval entry (`:4587`, `:4601`,
  `:4612`).
- Risk: LUT computation happens once per process; if sp model changes under
  the process, LUTs are stale. Add a hash-of-tokenizer sanity log.

### C.3 Arch-switch propagation

- `ARCHITECTURE=hybrid` switches model class. SKC / feedback / hybrid have
  different sets of ternary-eligible params. `export_ternary_param_names`
  (`:816`) must handle all three. No test that enumerates architectures and
  verifies no fp-dense tensor slipped in.

---

## D. Diagnostic scripts (6–10, runnable, specific)

Each script is a standalone Python file under `scratch/diag/`. None modify
model code; all import `train_gpt_verbose` directly.

### D1. `diag_export_size_decompose.py`

- **Measures.** Per-tensor bytes: `packed_len`, `scale_bytes`, pickle overhead
  (compare `torch.save(entry)` vs summed raw), category totals, outer codec
  compression ratio.
- **Plug-in.** Run after training on the final state dict; or feed a fake
  state dict at step 0 to preview budget.
- **Pass/fail.** Artifact + code ≤ `HARD_BUDGET_BYTES - 64 KiB margin`;
  per-category breakdown printed as a markdown table; no category >70% of
  total unless embeddings.

### D2. `diag_export_determinism.py`

- **Measures.** Runs `build_submission.main()` twice with
  `PYTHONHASHSEED=0`, identical weights. SHA-256 of `final_model.ternary.ptz`
  and `train_gpt.py`. Runs `torch.save` of `q_obj` twice and diffs bytes.
- **Pass/fail.** Hash equal across runs. If not, surface first differing
  offset.

### D3. `diag_roundtrip_logit_diff.py`

- **Measures.** Fixed 4096-token calibration batch. Compute logits on (a)
  in-memory model post-EMA-apply, (b) reloaded from `final_model.ternary.ptz`.
  Report L2 delta, max abs delta, argmax agreement %, per-layer L2 on hidden
  states.
- **Plug-in.** Runs after `:4801` as a non-fatal audit; write
  `roundtrip_logit_report.json`.
- **Pass/fail.** Argmax agreement ≥ 99.0% on calibration set; max |Δlogit| ≤
  0.5 in bf16.

### D4. `diag_kernel_inline_parity.py`

- **Measures.** For each Triton kernel entry, generate fixed inputs, run
  (verbose import path) and (inlined via exec of
  `_inline_triton_kernels` output) in separate subprocesses; compare tensors
  with `torch.allclose(rtol=1e-4, atol=1e-5)`.
- **Pass/fail.** All kernels match. Any deviation blocks submission.

### D5. `diag_sliding_eval_parity.py`

- **Measures.** For `stride == seq_len`, run `eval_val_sliding` and
  `eval_val` (non-sliding) on the same dev slice. Compare BPB / val_loss
  token-count / byte-count totals.
- **Pass/fail.** |Δ BPB| < 1e-4. If not, window scoring is dropping or
  double-counting tokens.

### D6. `diag_legal_ttt_legality.py`

- **Measures.** Snapshot every `p.requires_grad` and `p.data.clone()` before
  adapt phase; after adapt + before next chunk's SCORE phase, assert that (i)
  any param not in the TTT scope is bytewise unchanged, (ii) no future-token
  gradient was applied to the PREVIOUS chunk's score computation (derive by
  re-scoring chunk `i` after chunk `i` adapt — result must equal recorded
  `i`-th chunk score).
- **Pass/fail.** Bytewise equality outside scope; chunk re-score delta = 0.
- **Plug-in.** Enable via `LEGAL_TTT_AUDIT=1` env var at
  `eval_val_sliding_ttt` entry (`:3470`).

### D7. `diag_eval_engram_footprint.py`

- **Measures.** On build of `EvalEngram` (`:3155`), log
  `O*H*B*V*4` bytes expected; torch CUDA memory before/after; hit rate per
  order/head during eval; post-eval percentage of buckets with `count > 0`.
- **Pass/fail.** Footprint ≤ 2 GiB (configurable); hit rate on high-entropy
  positions > 5% (signal that mixing actually happens).

### D8. `diag_ema_apply_order.py`

- **Measures.** Assert `ema.apply_shadow(base_model)` is called before the
  quantize-and-pack export path (trace `_orig_ema_weights` lifetime through
  `:4297-4305` and `:4484-4482`). Dump a timeline.
- **Pass/fail.** Shadow is active at the moment `pack_ternary`
  (`:942`/`:945`) is called. Otherwise roundtrip measures a different model
  than the exported one.

### D9. `diag_tokenizer_fingerprint.py`

- **Measures.** SHA-256 of `fineweb_8192_bpe.model`; base-bytes LUT hash;
  vocab size; EOS id. Log at train start and at each eval entry.
- **Pass/fail.** Hash unchanged across the run; vocab size matches
  `sp_vocab_size` used at `:3264` (`table_size`).

### D10. `diag_budget_simulation_step0.py`

- **Measures.** Run `estimate_export_lower_bound_bytes` plus a mock pack of
  randomly-initialized weights at step 0 through the real codec; compare
  against `HARD_BUDGET_BYTES`.
- **Pass/fail.** mock_total + 5% safety margin ≤ cap. Abort training if not.
- **Plug-in.** Gate at `:4010-4022` before the training main loop rather than
  after.

---

## E. Phased rollout

### P0 — Must-have before next submission

1. **D10** (budget sim at step 0): prevents the worst-case DQ (budget breach
   at minute 9). Small diff at `:4010`.
2. **D3** (roundtrip logit diff): confirms the submitted artifact is
   numerically the model we trained. Runs in ~5 s.
3. **D6** (Legal-TTT legality audit): protects against eval-time DQ. Runs
   under `LEGAL_TTT_AUDIT=1`.
4. **D8** (EMA-apply order): one-line assert; quickly disproves a class of
   "trained better than submitted" regressions.
5. **D9** (tokenizer fingerprint): catches data/tokenizer drift between train
   and eval hosts.

### P1 — Should-have within one iteration

6. **D1** (size decomposition): report table for each submission, to guide
   which categories to compress further.
7. **D2** (export determinism): confirms reproducibility of scoreboard.
8. **D4** (kernel inline parity): CI check after every
   `build_submission.py` change.

### P2 — Nice-to-have

9. **D5** (sliding vs full parity): only needed if we ever change stride
   policy.
10. **D7** (engram footprint + hit rate): tuning signal for
    `EVAL_ENGRAM_*` env vars.

---

## F. Specific line-cited risks (summary)

| # | Risk | Site | Severity |
|---|------|------|----------|
| R1 | Tokenizer not bundled; submission code must locate `.model` file at runtime | `:190`, `:491`, `:3979` | DQ if scorer env differs |
| R2 | Budget lower bound ignores codec/pickle overhead | `:881-882`, `:4019-4022` | DQ — late failure |
| R3 | No roundtrip logit delta audit; only BPB | `:4798-4803` | High — silent quant regressions |
| R4 | No kernel-inline numerical parity test after `_inline_triton_kernels` | `build_submission.py:117-162` | High — kernel divergence |
| R5 | Legal-TTT has no explicit backup/restore of non-TTT params; no grep match for `backup_params`/`restore_params` | `:3470-3600+` | Legality DQ if scope misconfigured |
| R6 | EvalEngram state persists across two eval passes (idempotent build); second pass observes first pass's absorb | `:3148-3161`, `:4843` vs `:4865` | Legality DQ — second eval not independent |
| R7 | EvalEngram fp32 `(O,H,B,V)` buffer can be >1 GB with defaults | `:1525` | OOM at eval |
| R8 | EMA `apply_shadow` ordering relative to quantize-pack not explicitly asserted | `:4297-4305`, `:4484` | Train/export mismatch |
| R9 | First-window contributes `seq_len` tokens vs `stride` elsewhere — documented but not tested | `:3395` | Minor BPB bias |
| R10 | `build_submission.py` comment stripping is line-based; fragile on docstrings with `#` | `:52-63` | Low — edge case |

---

## G. Open questions (to resolve before P0 ships)

1. Does competition-eval environment provide `fineweb_8192_bpe.model` at a
   known path, or must the submission code fetch/unpack it? (affects R1).
2. Is `final_model.competition.ptz` (`:4746`) required by the scorer, or is
   `final_model.ternary.ptz` sufficient? Both are written; size of both counts
   if scorer sums all artifacts.
3. Are EvalEngram buffers legal under competition rules? `persistent=False`
   keeps them out of the artifact, but they observe eval tokens — is this
   considered TTT-equivalent? (affects legality interpretation, R5/R6).

## H. Appendix — file:line index

```
train_gpt_verbose.py
  190-192, 491-492   tokenizer / data paths
  292-294            EVAL_ENGRAM_ALPHA/ENTROPY_THR/LAPLACE
  321-324            EMA env vars
  337-338            SLIDING_EVAL, SLIDING_EVAL_STRIDE
  354-361            TTT env vars
  425, 464-465       HARD_BUDGET_BYTES
  754                CTP critical-scalar whitelist
  756, 773           pack_ternary, pack_ternary_bitmask
  816, 824           export_ternary_param_names, export_fp16_param_names
  881-882            estimate_export_lower_bound_bytes return shape
  891-903            build_submission subprocess call in get_fresh_code_bytes
  942-945            pack call sites (turbo vs bitmask)
  973-975            unpack call sites
  1493-1549          class EvalEngram (+ absorb)
  3148-3161          maybe_build_eval_engram
  3199-3201          engram logits mixing
  3264               table_size = max(sp_vocab_size, vocab_size, 50257)
  3338-3408          eval_val_sliding
  3410               collect_ttt_params
  3470-3600+         eval_val_sliding_ttt
  3512-3514, 3575-3583 Legal-TTT absorb boundary commentary and call
  3710-3767          _proxy_roundtrip_bpb
  3885-3902          EMAHelper (shadow / apply_shadow / restore)
  3979-3996          tokenizer load + regime lock
  4010-4022          budget lower-bound gate
  4262-4305          EMA lifecycle in train loop + eval swap
  4484               _proxy_roundtrip_bpb call
  4587, 4601, 4612   eval entry points
  4721               torch.save(q_obj, buf)
  4742-4748          write final_model*.ptz
  4749-4757          hard-budget gate
  4768-4803          reload + round-trip eval
  4817-4843          final_sliding eval
  4855-4866          legal_ttt eval
  4978-4984          scoreboard: submission_bpb = min(...)

build_submission.py
  52-63              _strip_comments
  107-114            _select_wrapper
  117-162            _inline_triton_kernels
  166-191            build_submission main

triton_kernels.py
  31                 _get_hadamard
  47, 136, 151, 214, 353  kernel definitions
  85, 196, 240, 331, 411, 482  triton wrappers
```

*End of plan.*
