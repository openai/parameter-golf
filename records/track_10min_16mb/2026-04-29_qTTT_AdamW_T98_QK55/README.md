# qTTT + AdamW + Temperature 0.98 + QK-Gain 5.5 — Documented Negative Result

**3-seed mean val_bpb = 1.08902 ± 0.00041** | Artifact ≤ 15,992,907 bytes | 8×H100 SXM | Train 588s, Eval ~378s

> **Status:** This submission **does not beat the merged SOTA** of 1.0810 (PR #1493 by @bigbag).
> It is offered as a **documented negative result** that reproduces a non-trivial finding for the depth-recurrent stack: **AdamW-based TTT degrades inference**, and **qTTT (query-only TTT) acts as a strong mitigation but does not recover positive TTT gain**. The ablation in this PR isolates that effect quantitatively.

## Headline numbers

| Stage | Seed 42 | Seed 314 | Seed 999 | Mean |
|---|---|---|---|---|
| Pre-quant post-EMA | 1.08756 | 1.08830 | 1.08845 | 1.08810 |
| Quantized standard | 1.09879 | 1.09962 | 1.09965 | 1.09936 |
| Quantized sliding (s=64) | 1.08276 | 1.08353 | 1.08357 | 1.08329 |
| **Quantized TTT** | **1.08855** | **1.08921** | **1.08931** | **1.08902** |
| Artifact bytes | 15,992,266 | 15,991,439 | 15,992,907 | — |

vs PR #1493 (1.0810): **+0.0080 BPB** (worse).

## What this submission changes vs PR #1493

Five modifications to the legal score-first TTT stack of PR #1493, all gated by environment variables (defaults preserve PR #1493 behavior bit-for-bit):

| Hyperparameter | PR #1493 | This submission | Channel |
|---|---|---|---|
| `TTT_OPTIMIZER` | `sgd` (hardcoded) | `adamw` | New env var |
| `TTT_LR` | 0.005 | 0.001 | Existing env var |
| `TTT_QUERY_ONLY` | n/a (full TTT) | `1` (only `c_q.weight`) | New env var |
| `TTT_WEIGHT_DECAY` | n/a (SGD momentum=0.9) | 0.01 | New env var |
| `EVAL_TEMPERATURE` | 1.0 (raw logits) | 0.98 (logits / 0.98) | New env var, applied in `eval_val_sliding` and the scoring path of `eval_val_ttt` |
| `QK_GAIN_INIT` | 5.25 | 5.5 | Existing env var |

Pre-quantization val_bpb (1.08810 mean) is **statistically indistinguishable from PR #1493's 1.0878 pre-quant**, confirming the training path was unaffected (the only training-time change was QK-Gain init, which converges similarly).

## Key finding (the reason this PR is here)

The quantized TTT path **regressed by +0.0058 BPB** vs sliding window in this submission, instead of improving by -0.0017 BPB as in PR #1493. To isolate cause, an ablation was run on seed 42 with `TTT_QUERY_ONLY=0` (full TTT, all other env vars identical):

| Configuration | seed | Sliding BPB | TTT BPB | TTT effect |
|---|---|---|---|---|
| **No TTT** (sliding only) | — | 1.0828 | — | (baseline) |
| **PR #1493: SGD lr=0.005 + Full TTT** | 42 | 1.0827 | 1.0810 | **−0.0017 (improves)** |
| **This submission: qTTT + AdamW lr=0.001** | 42 | 1.0828 | 1.0885 | **+0.0058 (regresses)** |
| **Ablation: Full TTT + AdamW lr=0.001** | 42 | 1.0825 | **1.1885** | **+0.1060 (catastrophic)** |

Two empirical claims supported by these numbers:

1. **AdamW + lr=0.001 in full-TTT mode catastrophically degrades the quantized model on this depth-recurrent base.** Pre-quant val_bpb is healthy (1.08734), sliding window is healthy (1.0825), but applying AdamW TTT to all 35.9M parameters across 1238 chunks pushes BPB to 1.1885 — a +0.106 regression.
2. **qTTT (adapting only `c_q.weight`, ~8% of parameters) is a strong mitigation but not a recovery.** It contains the AdamW damage to +0.006 BPB instead of +0.106, but does not produce net-positive TTT gain at this learning rate.

## Hypothesized mechanism

The PR #1493 base activates 3-layer depth recurrence (loop start=3, loop end=5) at training fraction 0.35. The looped layers have their weights traversed multiple times per forward pass (encoder=[0,1,2,3,4,5,3,4], decoder=[5,3,4,5,6,7,8,9,10] — 17 virtual layers from 11 physical). This creates an implicit weight-sharing structure that AdamW's per-parameter adaptive learning rate appears to interact with poorly: each effective gradient signal at a looped-layer parameter compounds through reuse, and AdamW's variance estimate, computed online during 3 TTT epochs over 1238 chunks, may amplify these compounded updates instead of dampening them as in non-recurrent settings.

The Issue #140 live-commentary (paused April 5, 2026) noted that SGD + momentum has remained dominant for legal score-first TTT in this competition; this submission corroborates that observation empirically and quantifies the cost of switching to AdamW on this specific stack.

A natural follow-up — not pursued here due to compute budget — would be to ablate `TTT_OPTIMIZER=sgd, TTT_LR=0.005` with `TTT_QUERY_ONLY=1` to test whether qTTT alone (without AdamW) recovers or extends PR #1493's TTT gain. That test would isolate the contribution of the qTTT filter from the AdamW failure mode.

## Compliance (Track A, Issue #1017 conditions)

- ✅ **Train ≤ 600s** (588s, 4567-4573 steps reached the wallclock cap)
- ✅ **Eval ≤ 600s** (sliding ~94s + TTT ~280s = ~378s)
- ✅ **Artifact ≤ 16,000,000 bytes** (max observed: 15,992,907)
- ✅ **Score-first TTT** (each chunk scored under `torch.no_grad()` before any optimizer step; AdamW state shared across chunks but never sees ungated tokens)
- ✅ **Single pass** (each token scored once)
- ✅ **No SLOT, no pre-quant TTT, no ETLB, no n-gram cache**
- ✅ **Three seeds** (42, 314, 999) with fixed config

## Reproduction

```bash
# Setup (RunPod 8xH100 SXM template y5cejece4j or equivalent):
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# Run all three seeds:
for SEED in 42 314 999; do
  SEED=$SEED \
  QK_GAIN_INIT=5.5 \
  TTT_ENABLED=1 \
  TTT_OPTIMIZER=adamw \
  TTT_LR=0.001 \
  TTT_QUERY_ONLY=1 \
  TTT_EPOCHS=3 \
  TTT_WEIGHT_DECAY=0.01 \
  EVAL_TEMPERATURE=0.98 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 \
    | tee train_seed${SEED}.log
done

# Ablation (full TTT instead of qTTT, all other settings identical):
SEED=42 \
QK_GAIN_INIT=5.5 \
TTT_ENABLED=1 \
TTT_OPTIMIZER=adamw \
TTT_LR=0.001 \
TTT_QUERY_ONLY=0 \
TTT_EPOCHS=3 \
TTT_WEIGHT_DECAY=0.01 \
EVAL_TEMPERATURE=0.98 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 \
  | tee train_seed42_ablation_no_qttt.log
```

## Code modifications

`train_gpt.py` is the PR #1493 source with:

- 4 new environment variables in the `Hyperparameters` class: `TTT_OPTIMIZER`, `TTT_QUERY_ONLY`, `TTT_QUERY_PATTERN`, `TTT_WEIGHT_DECAY`, plus `EVAL_TEMPERATURE`.
- Conditional optimizer construction in `eval_val_ttt`: `AdamW(...)` when `TTT_OPTIMIZER=adamw`, else original `SGD(...)`.
- Conditional parameter filter in `eval_val_ttt`: when `TTT_QUERY_ONLY=1`, builds `ttt_params` from `[p for n,p in model.named_parameters() if TTT_QUERY_PATTERN in n]` and freezes all other params via `requires_grad_(False)`.
- `logits / h.eval_temperature` in two places: `eval_val_sliding` (`logits=logits_fn(x_batch)/h.eval_temperature`) and the **scoring path** (not the training loop) of `eval_val_ttt` (`logits=compiled_logits(x_batch)/h.eval_temperature`).

Defaults preserve PR #1493 behavior. Code is wrapped in the `lzma + base85` self-decompression idiom common to other submissions in this repo, keeping the file at 16,860 bytes.

## What this PR is for

This is **not a leaderboard advance**. It is documented negative evidence that:

- **AdamW-based TTT** at lr=0.001 on a depth-recurrent base **does not work** as drop-in replacement for SGD+momentum, even with low learning rate.
- **qTTT** (query-only TTT, paper arXiv:2512.13898) is **highly effective as containment** of AdamW's instability, but **insufficient on its own** to recover net TTT gain in this configuration.

If a future stack revisits qTTT, this PR provides a reference point to beat: any qTTT submission must produce TTT BPB **lower than its sliding window BPB** before claiming the qTTT mechanism works on this base, and must outperform val_bpb=1.08902 with three seeds before claiming improvement over this configuration.

## Credits

| Contribution | Author / PR / Source |
|---|---|
| Base stack (SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal score-first TTT) | @bigbag (PR #1493) |
| SP8192 + GPTQ + SDClip + MuonEq-R | @clarkkev (PR #1394) |
| Legal score-first TTT framework | @abaybektursun (PR #549), @dexhunter (PR #1413) |
| 3-layer depth recurrence | @dexhunter (PR #1331, PR #1437) |
| Parallel residuals | @Robby955 (PR #1412), @msisovic (PR #1204) |
| Hyperparameter tuning | @X-Abhishek-X (PR #1445) |
| Post-TTT temperature calibration (T=0.98) | @cmcdnd (PR #576) — referenced and adopted |
| Q-only TTT (qTTT) theoretical motivation | arXiv:2512.13898 |

## Files in this submission

| File | Purpose |
|---|---|
| `train_gpt.py` | Self-decompressing source; lzma+base85 wrapper around the actual training/eval code (16,860 bytes) |
| `submission.json` | Machine-readable metadata (results, hardware, compliance, attribution) |
| `train_seed42.log` | Full stdout for seed 42 (qTTT + AdamW config) |
| `train_seed314.log` | Full stdout for seed 314 |
| `train_seed999.log` | Full stdout for seed 999 |
| `train_seed42_ablation_no_qttt.log` | Ablation: same config but `TTT_QUERY_ONLY=0` (full TTT) — shows the catastrophic regression |
| `README.md` | This file |
