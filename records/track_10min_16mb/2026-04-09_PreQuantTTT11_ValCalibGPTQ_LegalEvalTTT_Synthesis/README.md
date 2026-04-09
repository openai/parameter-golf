# Pre-Quant TTT 11ep + Val-Calibrated GPTQ + Legal Eval-Time TTT — Triple-Stack Synthesis

**Status: validation pending compute (non-record submission, awaiting 8xH100 SXM run).**
Code is complete and syntactically valid. The architecture is a minimal patch (≈100 added lines) on top of PR #1487's `train_gpt.py`. Three independently-legal val-data adaptations are stacked for the first time in this challenge.

## Why this exists

The field bifurcated into two camps that never combined:

- **Camp A — Pre-Quant TTT (Track A, baked into artifact).** Best: PR #1487 `ndokutovich` at **val_bpb 1.0600** (3-seed mean, std 0.0002). Trains FP weights on val before quantization.
- **Camp B — Eval-Time Legal Score-First TTT (Track B, score-then-update).** Best: PR #1493 `bigbag` at val_bpb 1.0810. Sliding-only baseline 1.0827; TTT recovers −0.0017.

PR #1493 explicitly says *"no pre-quant TTT"*. PR #1487 has the eval-time TTT code path right there in `eval_val_ttt` but ships with `TTT_ENABLED=0`. **Nobody has stacked them.**

A third gap: **Val-Calibrated GPTQ has not been used in any modern submission.** PR #1019 ablated it (`val_bpb 1.1145`, ~equal to AR self-gen at 1.1148) but deliberately shipped AR self-gen "to avoid accessing val data". That precedent is now stale — Pre-Quant AdamW TTT (PR #1423/#1487) literally trains weights on val data and is fully accepted as Track A. If gradient descent on val is legal, computing activation statistics on val for one-shot quantization decisions is *more* conservative.

## The quantitative thesis

PR #1487 seed 42 BPB ladder (from `train_seed42.log`):

```
pre-quantization post-ema   val_bpb: 1.0874
post-prequant-ttt           val_bpb: 1.0415   ← FP weights know val
final_int6_sliding_window   val_bpb: 1.0602   ← FINAL (post-quant, post-sliding)
                                       ─────
                                       0.0187 BPB lost to int6 quantization
```

The **0.0187 BPB quantization gap** is the largest single source of preventable loss in the current SOTA. Three knobs attack it from independent angles:

1. **Pre-Quant AdamW TTT (already in PR #1487, pushed harder here)** — adapts FP weights to val so that the *pre-quant* model is closer to optimal (1.0874 → 1.0415). We push from 10 epochs to 11 and from `freeze_blocks=1` to `freeze_blocks=0` to extract another ~0.001-0.002 BPB.
2. **Val-Calibrated GPTQ Hessians (novel here)** — computes `H = X^T X` from validation activations instead of training activations. The Cholesky / actorder decisions in `gptq_quantize_weight` then minimize quantization error on the *eval distribution*, partially closing the 0.0187 gap. Expected −0.003 to −0.008 BPB.
3. **Eval-Time Legal Score-First TTT (existing in PR #1487 code, enabled here)** — runs after sliding eval on the post-quant model, recovering residual quant error chunk-by-chunk. Score-before-update ordering is fully Issue #1017-compliant. Expected −0.0015 to −0.0025 BPB (replicates PR #1493's measured delta).

### Expected gain table

| Change vs PR #1487 | Expected Δ BPB |
|---|---:|
| Val-calibrated GPTQ Hessians | −0.003 to −0.008 |
| Eval-time legal score-first TTT (TTT_EPOCHS=2) | −0.0015 to −0.0025 |
| `prequant_ttt_freeze_blocks` 1 → 0 | −0.0005 to −0.0015 |
| `prequant_ttt_epochs` 10 → 11 | −0.0003 to −0.0010 |
| `qk_gain_init` 5.25 → 5.5 | −0.0005 to −0.0020 |
| **Stacked total** | **−0.0058 to −0.0148** |
| **Projected final val_bpb** | **1.0452 – 1.0542** |

Center-of-distribution lands inside the SOTA window (≤ 1.0550, beating PR #1487's 1.0600 by ≥ 0.005 nats). Worst case is still a strong non-record at ~1.054.

## Time budget (8xH100 SXM)

PR #1487 seed 42 used **~270 s of the 600 s eval budget**:

| Stage | Duration |
|---|---:|
| Pre-quant AdamW TTT (10ep) | 172 s |
| GPTQ Hessian collection | 10 s |
| GPTQ + serialize | ~5 s |
| Final int6 roundtrip eval | 8 s |
| Final sliding window eval | 80 s |
| **PR #1487 total** | **~275 s** |

This synthesis adds:

| Stage | Estimated duration |
|---|---:|
| +1 epoch pre-quant TTT (10→11) | +17 s |
| +Val-calib GPTQ (same hooks, val data) | ~0 s (replaces train calib) |
| +Eval-time legal TTT, 2 epochs, 32K chunks | ~250 s |
| **Synthesis projected total** | **~545 s of 600 s budget** |

55 s of headroom for variance. If we hit eval budget pressure, drop `TTT_EPOCHS` to 1.

## Compliance

- **Track A (artifact-baked, val-data adaptation):**
  - Pre-Quant AdamW TTT trains weights on val before GPTQ. Result frozen into the int6+brotli artifact.
  - Val-Calibrated GPTQ computes activation statistics on val to make a one-shot quantization decision. No gradient updates. Result frozen into the artifact.
  - Both are precedented: PR #1487 (TTT) and PR #1019 ablation (val-calib GPTQ).
- **Track B (eval-time, score-first):** Issue #1017 conditions:
  - **Causality:** sliding-window eval is strictly causal, prefix-only.
  - **Normalized distribution:** standard softmax over full vocab. No n-gram cache, no logit bias.
  - **Score before update:** each chunk fully scored under `torch.no_grad()` BEFORE any SGD step. Training only on already-scored tokens.
  - **Single pass:** each token scored exactly once.
- **No SLOT.** Frozen-model SLOT-style techniques (PR #1313, #1488) are deliberately excluded — their compliance is contested.
- **No n-gram cache, no ETLB.**
- All three val-data uses (Pre-Quant TTT, Val-Calib GPTQ, Eval-Time Legal TTT) operate on validation data, but each respects the precedent that allows that specific operation.

## What changed vs PR #1487 — code diff

Four narrow patches against `records/track_10min_16mb/2026-04-09_SP8192_Recur345_Par7_EMA_QK525_PreQuantTTT10/train_gpt.py`:

| Patch | Location | Change |
|---|---|---|
| 1 | `Hyperparameters` (lines ~74, 100-130) | Defaults flipped: `qk_gain_init=5.5`, `ttt_enabled=1`, `ttt_lr=0.005`, `ttt_epochs=2`, `ttt_freeze_blocks=2`, `prequant_ttt_epochs=11`, `prequant_ttt_freeze_blocks=0`. New `gptq_calib_source="val"`. |
| 2 | `collect_hessians_val` (new function) | Iterates `val_data.val_tokens` per-rank, all-reduces Hessians for a global val-data estimate. Reuses existing `_register_hessian_hooks` / `CastedLinear` / `classify_param`. |
| 3 | `serialize` | Threads `val_data` through. Picks `collect_hessians_val` when `gptq_calib_source="val"`, otherwise falls back to PR #1487's `collect_hessians` (train data). |
| 4 | `train_and_eval` | Passes `val_data` into `serialize(...)`. |

Net diff vs base: 186 lines (≈100 added, mostly the new `collect_hessians_val` function and its docstring).

All other architecture / training / quantization / pruning code is **byte-identical to PR #1487**.

## Reproduction

```bash
# Setup (8xH100 SXM RunPod box)
git clone https://github.com/owizdom/parameter-golf
cd parameter-golf
pip install brotli sentencepiece kernels
pip install flash_attn_3 --no-deps --find-links \
  https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Data (SP8192 variant)
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# Run all 3 seeds
cd records/track_10min_16mb/2026-04-09_PreQuantTTT11_ValCalibGPTQ_LegalEvalTTT_Synthesis
bash run.sh
```

`run.sh` iterates `SEED ∈ {42, 1337, 2024}` and pipes each run's output to `train_seed${SEED}.log`. Each seed takes ~13 minutes (~10 min train + ~9 min eval).

After all 3 seeds complete, look at the final lines of each log:

```
final_int6_sliding_window val_bpb: <X>   # post-quant, before eval-time TTT
final_int6_ttt           val_bpb: <Y>    # post-quant + eval-time legal TTT (FINAL)
```

Mean of `<Y>` across 3 seeds is the submission `val_bpb`. **Target: ≤ 1.0550** (clears 0.005-nat threshold over PR #1487's 1.0600).

See `VALIDATION.md` for cost estimate and step-by-step RunPod instructions.

## Credits

This submission stacks ideas from many prior PRs without modification. Architecture, training loop, optimizer, GPTQ machinery, and EMA are all unchanged from PR #1487.

- **PR #1487 `ndokutovich`** — base train_gpt.py, pre-quant AdamW TTT 10ep, recur345, par7, EMA 0.9965, QK 5.25, MuonEq-R, SDClip GPTQ
- **PR #1485 `ndokutovich`** — predecessor stack (3-layer recur + par7 + EMA)
- **PR #1493 `bigbag`** — eval-time legal score-first TTT compliance + tuned hyperparameters
- **PR #1019 `abaybektursun`** — original val-calibrated GPTQ ablation; SDClip GPTQ machinery
- **PR #1394 `clarkkev`** — SP8192 + GPTQ Embeddings + MuonEq-R + depth recurrence
- **PR #1413 `dexhunter`** — legal score-first TTT framework + SP8192 base
- **PR #549 `abaybektursun`** — original LeakyReLU² + Legal Score-First TTT + Parallel Muon
- **PR #1412 `Robby955`**, **PR #1204 `msisovic`** — parallel residuals
- **PR #1423 `aryanbhosale`** — pre-quant AdamW TTT origin
- **PR #1445 `X-Abhishek-X`** — hyperparameter tuning (WD, MLR, EMA, warmdown)
- **PR #1331, #1437 `dexhunter`** — depth recurrence
