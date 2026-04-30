# [Non-record] Online Best-Agree Infini-gram Phrase Memory — val_bpb 1.08167651

**Online variable-length causal phrase memory with purity-weighted blending on top of the cached `#1394` SP8192 base.**

> **Important note on evidence:** This submission is the lzma2+base85 single-file wrapper of the original three-file Direction 3 stack. The original 2026-04-26 8×H100 run produced the BPB numbers reported here and is included as `train_seed1337.log`. The wrapper itself was **not rerun on GPU**. The `15,998,983 bytes` total submission size is therefore a **projected** number, computed as `serialized_model_quantized+brotli (15,971,234) + len(wrapper train_gpt.py) (27,749)`, not a value emitted by a wrapper-rerun log.

---

## Real seed-1337 metrics (from `train_seed1337.log`)

| Metric | Value | Log line |
|--------|-------|----------|
| Pre-quant post-EMA val_bpb | 1.08923876 | line 125 |
| Quantized exact val_bpb | 1.10084495 | line 136 |
| Quantized sliding-window val_bpb | 1.08429171 | line 137 |
| **Online best-agree val_bpb (full val, exact)** | **1.08167651** | line 161 |
| Overlay gain vs sliding baseline (signed delta, lower is better) | **−0.00279899 BPB** | line 162 |
| Sliding eval time | 108.842 s | line 137 |
| Online best-agree eval time | 406.206 s | line 160 |
| Serialized model (int6 GPTQ + brotli) | 15,971,234 bytes | line 134 |
| Original code size (3 source files counted by `submission_code_bytes`) | 105,237 bytes | line 127 |
| Original total submission size | 16,076,471 bytes | line 135 |
| Original headroom under 16 MB cap | **−76,471 bytes (over)** | — |
| Train wallclock cap | 600 s (10 min) | hyperparameters |
| Compute | 8× H100 SXM | — |
| Seed | 1337 | — |

> **Sign convention:** The README reports the overlay gain as a **signed BPB delta** (`online_best_agree_bpb − sliding_bpb = −0.00279899`, i.e. lower is better). `submission.json` records the same quantity as a **positive improvement magnitude** (`overlay_gain_bpb: 0.00279899`). Both refer to the same `0.00279899 BPB` improvement reported on line 162 of the log.

The original three-file payload missed the cap by `76,471 bytes`. That is what motivated the wrapper packaging step below.

---

## Projected wrapper-packaging math (Option A, no GPU rerun)

The original three-file payload (`train_gpt.py`, `online_best_agree_eval.py`, `online_ngram_state.c`) was packaged into a single `train_gpt.py` lzma2+base85 wrapper. The wrapper file in this folder was built by:

1. Inlining `online_best_agree_eval.py`'s body into the merged source (skipping its `__file__`-dependent header).
2. Embedding `online_ngram_state.c` as a Python string constant inside the merged source. At startup the wrapper writes that string to a temp directory and the runtime gcc compile path picks it up there.
3. Replacing `submission_code_bytes()` so it returns `len(Path(__file__).read_bytes())` of the wrapper itself, since there is now only one source file.
4. Compressing the merged single-file source with `lzma.FORMAT_RAW + FILTER_LZMA2`, base85-encoding the result, and wrapping it in a 3-line `import lzma; exec(...)` shim.

| Quantity | Value |
|----------|-------|
| Merged single-file source (uncompressed) | 105,207 bytes |
| Wrapper `train_gpt.py` (lzma2+base85, in this folder) | 27,749 bytes |
| Serialized model (from real run) | 15,971,234 bytes |
| **Projected total = model + wrapper** | **15,998,983 bytes** |
| **Projected headroom under 16 MB cap** | **+1,017 bytes** |

These are projected numbers, not log-emitted numbers. They assume the wrapper produces an identical `final_model.pt` and an identical eval pass to the original three-file run. The wrapper has been syntax-checked (`py_compile`, `ast.parse`) and round-trip-verified (decompress → `ast.parse`), but **end-to-end equivalence to the original run has not been re-verified on GPU** in this Option-A package.

---

## What This Is

An online, variable-length, causal n-gram phrase memory layered on top of the cached `#1394` SP8192 base. At each evaluation chunk, the system:

1. Runs the base model forward pass to get `P_neural`.
2. Queries a causally-maintained n-gram phrase table built from already-scored tokens.
3. Blends `P_neural` and a phrase distribution `P_phrase` using a **purity-weighted gate** — higher-order matches with purer continuation distributions receive more weight.
4. Scores the next token against the blended distribution.

The phrase memory is fully causal: the table only contains tokens whose scores have already been locked. The blending distributes mass over the full vocabulary (not just observed continuations), which is the four-condition compliance shape from Issue #1017.

The native helper accumulates token-order and word-order phrase statistics simultaneously during the evaluation pass, so a single online state covers both `token_order=16` and `word_order=4` lookups.

---

## What Is Novel In This Submission

- **Online variable-length (infini-gram) phrase memory** with a **purity-weighted gate** layered as an eval-time overlay on top of an unmodified `#1394` SP8192 base.
- A C-backed online n-gram state that maintains both **token-order and word-order** phrase statistics in a single causal pass over the validation stream.
- Single-file lzma2+base85 packaging of a previously three-file stack so the artifact fits under the 16 MB cap without changing the trained model bytes.

This submission deliberately does **not** make claims about the relative novelty vs specific prior PRs (e.g. `#1145`, `#1379`, `#1493`). The point of the submission is the real seed-1337 result on the `#1394` base, the four-condition compliance shape, and the packaging that makes it fit.

---

## Architecture (eval-time overlay)

```
Training time  →  #1394 SP8192 transformer base (unchanged, no TTT)
                  ↓ artifact: int6 GPTQ + brotli compressed model

Eval time      →  sliding window scoring
                  ├─ forward pass    → P_neural   (base model)
                  ├─ phrase lookup   → P_phrase    (causal infini-gram from already-scored context)
                  └─ purity gate     → P_final = (1-α)·P_neural + α·P_phrase
                                       α = f(match_order, continuation_purity)
```

Key hyperparameters used in the seed-1337 run (from `train_seed1337.log`):

- `train_seq_len=2048`, `eval_seq_len=2048`, `eval_stride=64`
- `iterations=20000`, `val_loss_every=4000`
- `loop_warmup` depth-recurrence schedule from the `#1394` base
- Online eval: `token_order=16`, `word_order=4`, `chunk_tokens=131072`, `batch_seqs=32`

The "no test-time training" property is a source-level fact about this stack, not a logged hyperparameter — see the Compliance section below.

---

## Compliance

- **Causal:** the phrase table only contains tokens whose scores have already been locked.
- **Full normalized distribution:** the blended distribution is over all V tokens before each score, not just the observed continuations.
- **Score-before-update:** the score at position t is locked before the state updates with `x_t`.
- **Single left-to-right pass** over the validation stream.
- **No test-time training:** the `#1394` base used here, `online_best_agree_eval.py`, and `online_ngram_state.c` contain no TTT/test-time-training machinery — TTT is not compiled into this stack at all (verifiable by grepping `TTT` / `test_time` across the source).

---

## Run Command (original three-file run, included for reproducibility)

The run that produced `train_seed1337.log` was launched on an 8×H100 SXM pod with the original three-file source:

```bash
PYTHONUNBUFFERED=1 \
RUN_ID=direction3_best_agree_phase1_seed1337 \
SEED=1337 \
ONLINE_BEST_AGREE_EVAL=1 \
ONLINE_BEST_AGREE_PROGRESS_EVERY_CHUNKS=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The packaged wrapper in this folder is a single-file equivalent of that three-file source (see "Projected wrapper-packaging math" above). It has not been rerun under torchrun in this submission package.

---

## Files In This Folder

- `train_gpt.py` — single-file lzma2+base85 wrapper (27,749 bytes).
- `train_seed1337.log` — exact 8×H100 seed-1337 run log from 2026-04-26 that produced the reported BPB numbers.
- `submission.json` — leaderboard metadata. Note that `bytes_total_projected` is the projected wrapper total (`15,998,983`), and `bytes_total_original` is the original three-file total (`16,076,471`).
- `README.md` — this file.

---

## Author

Abhishek Kumbhar (Abhishek8108)
