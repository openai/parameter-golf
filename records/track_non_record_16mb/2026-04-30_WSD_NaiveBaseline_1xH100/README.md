# Non-Record Submission: WSD LR Schedule on Naive Baseline (1×H100)

A non-record submission demonstrating that the **Warmup-Stable-Decay (WSD)** LR schedule [Hu et al., 2024](https://arxiv.org/abs/2404.06395) cleanly improves the naive baseline at 1×H100 / 10-min wallclock scale. The change is ~13 lines, gated behind `USE_WSD=1`, and stacks on the existing optimizer.

**3-seed mean post-quant `val_bpb`: 1.34297** vs 1×H100 naive baseline 1.34580 (Δ **+0.00282 BPB**). All three seeds beat the baseline individually by ≥ 0.001 BPB.

This is a *signs-of-life* submission. The absolute `val_bpb` sits well above the 8×H100 leaderboard frontier (1.0810) and the canonical 8×H100 naive baseline (1.2244) because a 1×H100 only completes ~1100 of the 20,000 planned iterations inside the 10-minute wallclock cap. The improvement is hardware-agnostic in intent and warrants validation on 8×H100.

## What changed

The naive baseline schedule is a wallclock-aware linear warmdown. In the 1×H100 / 600s regime, the projected `WARMDOWN_ITERS × step_ms` exceeds the wallclock budget, so LR starts decaying almost immediately and the model never sees a real peak-LR plateau.

WSD makes the plateau explicit. With `STABLE_FRACTION=0.6`:

- **0–360s**: flat at peak LR
- **360–600s**: linear decay to `MIN_LR_FRAC × peak_lr` (0.1× peak)

Gated by `USE_WSD=1`. Default (`USE_WSD=0`) preserves the original schedule exactly.

## Per-seed results

3 seeds, identical config (`USE_WSD=1 STABLE_FRACTION=0.6 WSD_DECAY_SHAPE=linear`):

| Seed | val_loss | val_bpb (post-quant) | Steps | Artifact (int8+zlib) | Δ vs baseline |
|---:|---|---|---:|---|---|
| 1337 | 2.26603233 | **1.34207199** | 1102 | 14,269,390 B | **+0.00373** |
| 42 | 2.26661124 | **1.34241485** | 1102 | 14,279,445 B | **+0.00338** |
| 2025 | 2.27002154 | **1.34443462** | 1055 | 14,145,578 B | **+0.00136** |
| **Mean** | **2.26755504** | **1.34297382** | 1086 | 14,231,471 B | **+0.00282** |

Artifact grew 12.7 MB → 14.2 MB (mean): the WSD-trained weight distribution compresses worse via int8+zlib. Still well under the 16 MB cap.

## Configuration

```
USE_WSD=1
STABLE_FRACTION=0.6
WSD_DECAY_SHAPE=linear
MIN_LR_FRAC=0.1
# all other knobs at naive baseline defaults
```

Per-seed run command:

```bash
SEED=<seed> USE_WSD=1 STABLE_FRACTION=0.6 WSD_DECAY_SHAPE=linear \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Negative results from the same loop

WSD survived a 4-experiment loop on the naive baseline. Full log in `results.tsv`. The discarded experiments:

| Technique | Δ vs anchor | Notes |
|---|---|---|
| z-loss α=1e-4 on naive | **−0.01620** | Adds redundant regularization on top of the existing `LOGIT_SOFTCAP=30`, plus ~3% step-time overhead from the extra `logsumexp`. |
| qk-norm `off` on naive | −0.00348 | Removing the existing q/k RMSNorm hurts. |
| qk-norm `q_only` on WSD | −0.00524 | Asymmetric (norm Q only) is worse than removing both. |
| qk-norm `k_only` on WSD | −0.00081 | Tied with WSD-alone within seed-to-seed noise. |

The QK-norm baseline default (`both`) survives ablation cleanly across all three asymmetric variants. Z-loss may be worth re-testing on a stack where logit softcapping is not active.

## Hardware

Single **NVIDIA H100 80GB SXM** RunPod instance, 600s wallclock cap. Steps reached: 1055–1102 (of 20,000 planned). Peak memory: 10.0 GB.

The 8×H100 canonical setup reaches roughly 8× more steps in the same wallclock window. The score gap vs the 8×H100 leaderboard is throughput-driven; a follow-up validation on 8×H100 should clarify how much of the +0.00282 BPB transfers.

## References

- Hu, S. et al., 2024. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)

## Files

- `train_gpt.py` — code with WSD patch applied
- `train_seed{1337,42,2025}.log` — per-seed train + eval logs
- `results.tsv` — full experiment-loop log (8 rows)
- `submission.json` — leaderboard metadata
