# Record: SP8192 + 3-Epoch Parallel Pre-Quant TTT + Huber WD Muon (SDPA-friendly)

**val_bpb 1.07037** (3-seed mean, std 0.00027) on the 10 min / 16 MB track.

## Summary

Record over merged SOTA (PR #1493, 1.0810) by **−0.01063 BPB** at a 3-seed mean.

This submission adapts the parallel pre-quant AdamW TTT stack (PR #1735, @AjAnubolu + PR #1738, @alertcat) for environments without FlashAttention-3 — specifically a torch 2.11+cu130 stack that has no compatible FA3 wheel and no nvcc available for source builds. On such a stack, SDPA is the only available attention backend and TTT epochs cost ~4× longer than the FA3 reference. The original 21-epoch schedule blows the 600s eval budget in that regime; this PR rebalances the schedule to fit.

The three concrete changes, each small and defensible in isolation:

1. **3-epoch pre-quant TTT with epoch-level cosine (1e-3 → 1e-4, no warm restart).**
   A dedicated ablation (seed 1337) showed that with SDPA's ~96 s warmup + ~19 s/epoch costs, the budget only fits 3 full TTT epochs. A 4-epoch cosine-warm-restart variant (cycle 1 = 3 ep, cycle 2 = 1 ep) was tried first and regressed from 1.0701 (3-epoch) to 1.0727 (4-epoch) — the restart LR jolt hurt when the follow-on cycle was too short to re-converge. Final schedule is plain `CosineAnnealingLR(T_max=3, eta_min=1e-4)`.
2. **Odd-epoch-only diagnostic eval + runtime budget guard.** Diagnostic `eval_val` calls after every epoch cost ~12s on SDPA. We run them on epochs 1, 3, 5, … and always the final epoch; a budget guard breaks TTT early if `elapsed + 150s × remaining_epochs > 600s`. Under the 3-epoch schedule the guard never triggers, but it protects long-tail variance on slower runs.
3. **Huber weight decay in the main-train Muon optimizer.** Replaces Muon's decoupled L2 decay `p ← p · (1 − lr·wd)` with a Huber variant: L2 for `|w| < δ`, L1 above, with `δ = 3/√(fan_in)`. The intent is to suppress outlier weights that cause int6 GPTQ clipping loss, without over-penalizing typical weights. Contribution to final BPB is small (within noise of the 3-epoch TTT change).

Everything else is inherited verbatim from the PR #1493 stack (SP8192 + CaseOps tokenizer + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + EMA + GPTQ SDClip + Brotli).

## 3-seed results (8× H100 80GB SXM, 10-min train / 10-min eval)

| Seed | Pre-quant post-EMA | Post-TTT pre-quant | Quantized | **Sliding BPB** | Artifact |
|------|-------------------:|-------------------:|----------:|----------------:|---------:|
| 1337 | 1.08893 | 1.07552 | 1.08762 | **1.07013** | 15,857,678 |
| 42   | 1.08872 | 1.07502 | 1.08828 | **1.07065** | 15,858,437 |
| 2025 | 1.08893 | 1.07529 | 1.08778 | **1.07033** | 15,862,994 |
| **Mean** | **1.08886** | **1.07528** | **1.08789** | **1.07037** | **15,859,703** |
| **Std** | 0.00010 | 0.00021 | 0.00028 | **0.00027** | — |

Artifact margin: worst-case 137,006 bytes under 16MB. Training uses 588s of the 600s cap across all seeds; SDPA eval uses ~300s total.

## Per-epoch TTT trajectory (seed 1337)

| Epoch | LR | val_bpb |
|-------|---:|--------:|
| 1/3 | 1.0e-3 | 1.09388 |
| 2/3 | 7.8e-4 | skipped |
| 3/3 | 3.3e-4 | 1.07589 |

The epoch-1 eval intentionally overshoots because the initial LR is at peak — the loss floor at epoch 3 (1.07589) is what matters for the quantization step that follows.

## Compliance (Issue #1017 Track A)

- ✅ **Fixed predictor**: scored artifact is int6-GPTQ + brotli, no eval-time adaptation
- ✅ **No SLOT, no RLS, no n-gram cache, no ETLB, no pre-quant TTT leakage** (TTT uses only legal held-out tokens, federated-averaged across ranks)
- ✅ **Sliding-window eval**: strictly causal, stride 64, single pass
- ✅ **Normalized softmax distribution**
- ✅ **CaseOps byte sidecar** for honest BPB accounting (Title/AllCaps/CapNext control symbols don't inflate byte counts)
- ✅ **Train < 600s** (588s), **Eval < 600s**, **Artifact < 16MB** (all three seeds)

## Relationship to pending PRs

PR #1735 (@AjAnubolu, 1.0429), PR #1738 (@alertcat, 1.0354 with CaseOps), and the kilojoules follow-up (1.0284 with LR=1e-3/freeze=0) all use FA3 and run 21 epochs of pre-quant TTT. On FA3-less hardware those scores are not reachable; this submission reconstructs the best TTT schedule that *is* reachable there, and separately adds Huber-Muon WD.

If any of those PRs merge first and become the new record baseline, this PR should be rebased or withdrawn — it does not claim improvement over them.

## Reproduction

```bash
# Data + tokenizer (PR #1729, CaseOps-v1)
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python3 cached_challenge_fineweb.py \
  --variant sp8192_lossless_caps_caseops_v1_reserved \
  --train-shards 80

# Run 3 seeds (8×H100 SXM)
for SEED in 1337 42 2025; do
  SEED=$SEED DATA_DIR=/path/to/data_caseops \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee train_seed${SEED}.log
done
```

Environment: pytorch 2.11.0+cu130, no FA3 (script falls back to SDPA). A reproduction on pytorch 2.9.1+cu128 with FA3 would finish faster but should land at the same BPB to within ~0.001.

## Attribution

- @clarkkev (PR #1394) — SP8192 + GPTQ SDClip + Brotli
- @dexhunter (PR #1331, #1437) — 3-layer depth recurrence
- @Robby955 (PR #1412) — Parallel residuals
- @bigbag (PR #1493) — QK-Gain 5.25 + Legal Score-First TTT stack (merged SOTA baseline)
- @stukenov (PR #1364) — Pre-quant AdamW TTT concept
- @AjAnubolu (PR #1735) — 8-GPU parallel pre-quant AdamW TTT
- @romeerp (PR #1729), @alertcat (PR #1738) — CaseOps lossless-case tokenizer + byte sidecar
- kilojoules (unmerged follow-up on PR #1738) — reference for LR=1e-3 / freeze_blocks=0 TTT defaults

This PR's contribution: schedule + eval-budget rebalancing for FA3-less stacks, and Huber-WD variant for Muon.
