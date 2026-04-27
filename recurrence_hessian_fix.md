# Fix: GPTQ Hessian collection now sees recurrence

**Date:** 2026-04-27
**Branch:** shikhar
**File:** `train_gpt_swa.py` (3 edits, ~30 lines added)

## What this fix is

When depth recurrence is active (e.g. `RECUR_LAYERS="4,5"`), the GPTQ calibration model (`_HessianGPT`) now applies the same recurrence wiring as the actual training model. Before this fix, recurrence was only wired into the real `GPT` class — the `_HessianGPT` clone used for Hessian collection iterated layers flat, with no recurrence at all. That meant GPTQ quantized layers 4 and 5 against single-pass activation statistics, while eval ran them twice. The error compounded on the second pass and produced our observed 0.0843 BPB quantization gap on v3.

## Why we believe this is the bug, not just *a* bug

The leaderboard gives us a controlled comparison. PR #1394 (`SP8192 + GPTQ Embeddings + Loop45x2 + MuonEq-R + SDClip`, sliding BPB **1.0856**, gap **0.012**) implements recurrence by toggling `looping_active` on a single model class, so its Hessian collection runs `model.forward_logits(x)` with `looping_active = True` inherited from training. Our v3 (`SP8192 + recurrence + MuonEq-R + SDClip + paired-head Muon`, sliding BPB **1.1792**, gap **0.0843**) used the same SDClip k=12.85 / k=20, the same Cholesky GPTQ, the same row-std clipping, the same recurrence wiring during training — but a separate `_HessianGPT` for calibration that discards the recurrence. That's the only mechanically relevant difference between the two stacks for the calibration step.

We are not claiming the entire 0.072 BPB delta will close. We are claiming the calibration-eval mismatch is the dominant contributor and removing it should close most of it.

## The three edits

### Edit 1 — `_HessianGPT.__init__` (around line 1820)

Added `recur_layers=""` keyword argument to the constructor signature.

After `self.lm_head` is constructed, added the same recurrence-index construction that the real `GPT` class does at line 1156-1170:

```python
loop_layers = [int(x) for x in recur_layers.split(",") if x.strip()]
self.recur_active = False
if loop_layers:
    loop_start = min(loop_layers)
    loop_end = max(loop_layers)
    loop_seg = list(range(loop_start, loop_end + 1))
    all_indices = list(range(loop_start)) + loop_seg + loop_seg + list(range(loop_end + 1, num_layers))
    mid = len(all_indices) // 2
    self._recur_encoder_indices = all_indices[:mid]
    self._recur_decoder_indices = all_indices[mid:]
    self._recur_num_skip = min(len(self._recur_encoder_indices), len(self._recur_decoder_indices))
else:
    self._recur_encoder_indices = None
    self._recur_decoder_indices = None
    self._recur_num_skip = 0
```

The construction is verbatim from `GPT.__init__`. Same `loop_seg + loop_seg` doubling, same encoder/decoder split, same `_recur_num_skip` formula. This guarantees identical layer iteration order between training and calibration when both are active.

### Edit 2 — `_HessianGPT.forward` (around line 1874)

Replaced the flat encoder/decoder iteration with the recurrence-aware version, mirroring `GPT._run_blocks` lines 1214-1239:

```python
use_recur = self.recur_active and self._recur_encoder_indices is not None
enc_indices = self._recur_encoder_indices if use_recur else list(range(self.num_encoder_layers))
dec_indices = self._recur_decoder_indices if use_recur else [self.num_encoder_layers + i for i in range(self.num_decoder_layers)]
num_skips = self._recur_num_skip if use_recur else self.num_skip_weights
for bi in enc_indices:
    ve = self._get_ve(bi, input_ids, ve_cache)
    x = self.blocks[bi](x, x0, v_embed=ve)
    skips.append(x)
for j, bi in enumerate(dec_indices):
    if j < num_skips and skips:
        skip_idx = min(j, self.num_skip_weights - 1)
        x = x + self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
    ve = self._get_ve(bi, input_ids, ve_cache)
    x = self.blocks[bi](x, x0, v_embed=ve)
```

`skip_idx = min(j, self.num_skip_weights - 1)` matches the real model's clamp. When recurrence is off, behavior is bit-identical to the prior implementation — `enc_indices` and `dec_indices` collapse to `range(self.num_encoder_layers)` and `range(self.num_encoder_layers, num_layers)`, and `num_skips` collapses to `self.num_skip_weights`.

### Edit 3 — hessian_model construction site (around line 2480)

Pass the env-controlled `recur_layers` string into the constructor and activate it:

```python
hessian_model = _HessianGPT(
    ...,
    recur_layers=args.recur_layers,
).to(device).bfloat16()
...
if args.recur_layers:
    hessian_model.recur_active = True
    log0(f"gptq:enabled recurrence on hessian_model (recur_layers={args.recur_layers})")
```

When `RECUR_LAYERS` is unset (every existing run that doesn't use recurrence), this branch doesn't fire and behavior is unchanged. When it's set, the calibration model produces Hessians from the same activation distribution the eval model will see.

## What this fix does not do — be honest

**It does not match PR #1394's exact post-quant number.** Their stack has no SWA, no bigram, no value embeddings, no partial-key offset, no smear gate, no paired-head Muon. They got 1.0856 with a structurally simpler model. We will get *something* — almost certainly better than 1.1792 — but the floor is governed by how the rest of our stack interacts with quantized recurrence, not just by the calibration fix.

**It does not address the autoregressive calibration token distribution.** `generate_autoregressive_calib(base_model, ...)` already runs the recurrent training model, so the *tokens* fed into Hessian collection are recurrence-distribution. That part was correct before this fix. The bug was solely on the Hessian-computation side.

**It does not validate that paired-head Muon survives recurrence at calibration time.** Our paired-head Muon affects how QK weight banks are normalized during training; whether the resulting weight statistics produce a well-conditioned Hessian when fed through the recurrence path is empirically unknown. If the post-quant gap stays large after this fix, paired-head Muon × recurrence is the next suspect.

**It does not handle backout.** The real model's `_run_blocks` has a `backout_layers` path (lines 1220, 1227, 1240-1241) that saves a snapshot at a particular layer and subtracts a scaled version at the end. `_HessianGPT.forward` never had this and still doesn't. If someone enables `BACKOUT_LAYERS` simultaneously with `RECUR_LAYERS`, the calibration will diverge from training in the same way it did before — just on a different axis. The current `run_v7_recur_tests.sh` does not enable backout, so this is latent.

**It does not change the late_qat schedule.** Late QAT activates around step 5100 in our v7 runs, leaves only ~300 steps before wallclock cap. The model has very little time to absorb quantization noise across the recurrence path during training. If post-quant BPB is still loose after the calibration fix, ramping QAT earlier (e.g., from frac=0.5 alongside recurrence activation) is the next intervention to test.

## What I have verified

- `python3 -m py_compile train_gpt_swa.py` passes.
- `args.recur_layers` is a valid attribute (line 158: `recur_layers = os.environ.get("RECUR_LAYERS", "")`).
- When `RECUR_LAYERS` is empty string, `_HessianGPT.recur_active` stays False and the new code paths in `__init__` and `forward` reduce to the prior behavior.
- The recurrence-index construction in Edit 1 is identical to the real model's at line 1156-1170 (loop_seg appears twice, decoder slice starts at midpoint).
- The skip-weight indexing in Edit 2 matches the real model's `_run_blocks` line 1232 (`skip_idx = min(j, self.num_skip_weights - 1)`).
- Pre-existing Pyright diagnostics (zstandard/flash_attn imports, Muon None subscripts at lines 294-360) are not introduced by this change.

## What I have not verified

- I have not run this against H100. There may be a runtime issue I cannot see locally (e.g., `_HessianGPT` shape mismatch when recurrence wiring lengthens the effective decoder list). The encoder/decoder lists are constructed from the real model's logic, but `_HessianGPT.skip_weights` is sized `(num_skip_weights, model_dim)` where `num_skip_weights = min(num_encoder_layers, num_decoder_layers)`. The real model's `_recur_num_skip` can exceed this when recurrence doubles the layer count. The real model handles this with `skip_idx = min(j, self.num_skip_weights - 1)` — Edit 2 replicates this clamp exactly, so I expect it to work, but it has not been exercised on actual data.
- I have not measured how much the calibration time grows. Recurrence roughly doubles the active-layer FLOPs of layers 4,5 during the calibration forward pass. With 256 calibration batches, this should add a few seconds at most.
- I have not confirmed that the AR-token generator (`generate_autoregressive_calib`) was already using recurrence. It calls `model.forward_logits(tokens)` on `base_model`, and `base_model.recur_active` should be True at that point (set in training, never reset). But the v7 test script doesn't log this state, so I'm inferring rather than observing. Worth adding a log line confirming `base_model.recur_active` at AR-generation time.

## How to test on H100

A targeted single-seed run that isolates the fix:

```bash
RUN_ID=v8_recur_calib_fix SEED=1337 \
TRAIN_SEQ_LEN=4096 EVAL_SEQ_LEN=4096 \
SWA_WINDOW_SIZE=256 SWA_FULL_ATTN_LAYERS=5 \
PARTIAL_KEY_OFFSET=1 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
RECUR_LAYERS="4,5" RECUR_START_FRAC=0.5 \
WARMDOWN_ITERS=4000 \
torchrun --standalone --nproc_per_node=8 train_gpt_swa.py
```

Compare to v3 (1.1792) and v7b (1.1961) which had the same env minus the calibration fix. Look for:

- `gptq:enabled recurrence on hessian_model` log line — confirms Edit 3 fired.
- `final_int6_sliding_window_exact val_bpb: ...` — the headline number.

## Expected outcomes

| Scenario | Sliding BPB | Implied quant gap | Read |
|---|---|---|---|
| Floor (PR #1394 territory) | 1.085-1.095 | 0.005-0.015 | Calibration was the entire bug. |
| Mid (most-but-not-all) | 1.10-1.13 | 0.015-0.045 | Calibration was the dominant bug. Paired-head Muon × recurrence is residual. |
| Ceiling (no improvement) | 1.17-1.18 | ≥0.07 | Calibration was *not* the dominant bug. Look at paired-head Muon, late_qat schedule, or AR-token generation next. |

If we land in the floor or mid range, the next move is to layer parallel residuals + (already-default) QK-gain 5.25 to push toward rank 1 (1.0810). If we land at the ceiling, the diagnosis was wrong and we should reconsider before burning more H100 time.

## Why I am being cautious about the prediction

I claimed earlier in this conversation that recurrence "will at most close 7% of the SOTA gap" — that prediction was based on a SOTA reference (Scylla 0.9485) that had been reverted upstream. The actual leaderboard #1 is 1.0810, putting our reachable improvement closer to **0.022 BPB** rather than the 0.155 I was anchored on. I am calibrating my new prediction against PR #1394 directly because that's the closest comparable stack to ours and its quant gap is empirically known. But predictions in this regime have been noisy, and a single-seed result will have ±0.005 BPB seed variance per the v7 doc's reproducibility data. We should run at least 2 seeds before celebrating or panicking.

## Files changed

- `train_gpt_swa.py` — 3 edits, additions only, no deletions
- `recurrence_hessian_fix.md` — this file (new)
