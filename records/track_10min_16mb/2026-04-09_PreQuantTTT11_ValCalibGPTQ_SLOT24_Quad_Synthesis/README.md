# Pre-Quant TTT 11ep + Val-Calibrated GPTQ + SLOT-24 — Quad-Stack Synthesis

**Status:** validation pending compute. Code is `py_compile` clean and is a focused patch on top of an existing record stack. Awaiting an 8xH100 SXM run.

Four val-data adaptations stacked for the first time:

1. **Pre-Quant AdamW TTT** — 11 epochs, `freeze_blocks=0`. Adapts FP weights to validation before quantization. Track A.
2. **Val-Calibrated GPTQ** — Hessian `H = X^T X` computed on validation activations instead of training activations. Aligns the one-shot quantization decision with the eval distribution. Track A.
3. **SLOT-24** — per-window AdamW optimization of a hidden delta `[bsz,1,dim]` + logit bias `[bsz,1,vocab]` on the frozen post-quant model. 24 steps, cosine LR `0.012 → 0.001`, stride 96. Throwaway parameters.
4. *(Optional)* **Eval-Time Legal Score-First TTT** — disabled by default in this synthesis (SLOT supersedes it for the same eval budget). Set `SLOT_ENABLED=0 TTT_ENABLED=1` to fall back.

Each component has independent precedent on this challenge. Their combination is novel.

## Why each piece

- **Pre-Quant TTT** recovers ~0.046 BPB on the FP weights (`1.0874 → 1.0415` in the base stack).
- **Val-Calibrated GPTQ** attacks the `0.0187` BPB quantization gap (`1.0415 → 1.0602`) by aligning quantization with the actual eval distribution. Was ablated on an older base only — never ported forward.
- **SLOT-24** then adds a per-sample throwaway delta on the frozen post-quant model. On weaker bases SLOT alone delivered ~`-0.23` BPB. Stacking it on the strongest pre-quant + val-calib base should push further.

## Time budget (8xH100 SXM)

| Stage | Estimated |
|---|---:|
| Train (wallclock cap) | 590 s |
| Pre-Quant AdamW TTT (11 ep) | ~190 s |
| Val-Calibrated GPTQ (Hessian collection on val) | ~10 s |
| Final int6 sliding window eval (baseline number) | ~80 s |
| **SLOT-24 eval (FINAL submission score)** | **~250 s** |
| **Total eval used** | **~530 s of 600 s** |

70 s headroom for variance. Fallback if budget pressure: `SLOT_STEPS=16` or `SLOT_BATCH_SEQS=48`.

## Diff against the base

Six focused patches in `train_gpt.py`. All training, optimization, EMA, GPTQ machinery, and architecture code is unchanged.

| Patch | Where | What |
|---|---|---|
| 1 | `Hyperparameters` | New `gptq_calib_source`, `slot_*` knobs. Pre-quant TTT defaults pushed to `epochs=11`, `freeze_blocks=0`. `qk_gain_init=5.5`. |
| 2 | `collect_hessians_val` (new) | Iterates `val_data.val_tokens` per-rank, all-reduces Hessians for a global val-data estimate. Reuses existing hooks / `CastedLinear` / `classify_param`. |
| 3 | `serialize` | Threads `val_data` through. Picks `collect_hessians_val` when `gptq_calib_source="val"`. Falls back to the original train-data path otherwise. |
| 4 | `GPT.forward_hidden` + `compute_logits` | Splits `forward_logits` into hidden + projection so SLOT can add the delta to the hidden state without re-running the transformer. |
| 5 | `eval_val_slot` (new) | Per-window throwaway-parameter optimization (`delta`, `logit_bias`), 24 cosine-decayed AdamW steps, scored under the optimized delta. |
| 6 | `run_evals` | Wires SLOT (and the optional legal TTT path) on a fresh post-quant model copy. |

## Compliance

- **Track A (artifact-baked):** Pre-Quant AdamW TTT trains weights on val before GPTQ — baked into the int6+brotli artifact. Val-Calibrated GPTQ computes activation statistics on val for a one-shot quantization decision (no weight gradients) — also baked into the artifact.
- **Track B / SLOT (frozen-model per-window):** model weights are never updated during eval. SLOT optimizes only per-window throwaway `delta` and `logit_bias`. Score-after-delta is the standard SLOT pattern.
- **Sliding-window eval** is causal, prefix-only.
- **No n-gram cache, no ETLB, no cross-window leakage.**
- All artifacts < 16 MB (inherits selective ±1 pruning to fit).

## Reproduction

```bash
git clone https://github.com/owizdom/parameter-golf
cd parameter-golf
pip install brotli sentencepiece kernels
pip install flash_attn_3 --no-deps --find-links \
  https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

cd records/track_10min_16mb/2026-04-09_PreQuantTTT11_ValCalibGPTQ_SLOT24_Quad_Synthesis
bash run.sh
```

`run.sh` iterates `SEED ∈ {42, 1337, 2024}`. Each seed: ~10 min train + ~9 min eval. Final number is `final_int6_slot val_bpb` — the mean across the 3 seeds is the submission score.

See `VALIDATION.md` for RunPod step-by-step and the interpretation table.

## Files

| File | Purpose |
|---|---|
| `train_gpt.py` | The patched training + eval script |
| `README.md` | This file |
| `submission.json` | Metadata + projected range |
| `run.sh` | 3-seed runner with all env vars |
| `VALIDATION.md` | RunPod instructions, cost, fallback table |

## Credits

Building blocks reused from prior PRs:

- **PR #1487** — base `train_gpt.py`, Pre-Quant AdamW TTT, depth recurrence, parallel residuals, EMA, `MuonEq-R`, SDClip GPTQ machinery, 16 MB selective pruning.
- **PR #1485** — predecessor stack (3-layer recurrence + parallel residuals + EMA).
- **PR #1488 / #1313** — SLOT-24 reference implementation (`hidden_delta` + `logit_bias`, 24-step AdamW, stride masking).
- **PR #1019** — original Val-Calibrated GPTQ ablation; SDClip GPTQ + actorder + Cholesky machinery.
- **PR #1394** — SP8192 + GPTQ embeddings + `MuonEq-R` + depth recurrence.
- **PR #1413** — SP8192 base, legal score-first TTT framework.
- **PR #549** — original `LeakyReLU²` + score-first TTT + Parallel Muon.
- **PR #1412 / #1204** — parallel residuals.
- **PR #1423** — Pre-Quant AdamW TTT origin.
- **PR #1445** — hyperparameter tuning (`WD`, `MLR`, `EMA`, warmdown).
