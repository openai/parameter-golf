# SP8192 Multi-Phase Global SGD (4 phases) + Phased TTT + QK-Gain 5.25

**Score: 1.07217 bpb** (3-seed mean, seeds 42 / 0 / 1234)

| Seed | val_bpb | artifact |
|------|---------|----------|
| 42   | 1.07310 | 15,933,641 B |
| 0    | 1.07090 | 15,938,690 B |
| 1234 | 1.07250 | 15,930,318 B |
| **mean** | **1.07217** | |

## Approach

Builds directly on @jorge-asenjo's PR #1700 **Multi-Phase Global SGD + Phased LoRA TTT** stack (val_bpb 1.07219, which used num_phases=3) with two changes:

1. **`PHASED_TTT_NUM_PHASES=4`** — one extra multi-phase SGD pass at eval time. PR #1700 used 3 phases; extending to 4 gives the base-weight SGD more adaptation cycles while staying well under the 600s eval budget (3-phase eval used ~352s, so the 4th phase of ~30-50s easily fits).
2. **`QK_GAIN_INIT=5.25`** (up from PR #1700's 5.0) — matches merged SOTA PR #1493.

No architectural or code changes. All optimizer, quantization, and data settings identical to PR #1700.

## Why more phases?

In the Multi-Phase Global SGD TTT protocol, each phase = full scoring pass + SGD update on scored tokens. PR #1626 (the original MP-SGD proposal) and PR #1700 both use 3 phases. With the observed per-phase improvement trend and remaining eval budget, adding a 4th phase is a natural extension that doesn't violate Issue #1017:

- Each phase still scores under `torch.no_grad()` before any SGD update (Condition 3).
- Each token is still scored exactly once per phase (Condition 4 — no rescoring across passes).
- Attention is still strictly causal (Condition 1); softmax is unchanged (Condition 2).

## Compliance (Issue #1017 Track A)

- **Condition 1 (Causality):** standard causal attention, strict left-to-right
- **Condition 2 (Normalized distribution):** standard softmax over full vocab
- **Condition 3 (Score before update):** each phase scored under `torch.no_grad()` BEFORE SGD (inherited verbatim from PR #1700)
- **Condition 4 (Single pass):** each token scored exactly once per phase

No SLOT, no n-gram caches, no ETLB, no pre-quantization TTT.

## Reproduction

8× H100 SXM, torch 2.9.1+cu128, flash_attn_3 (Hopper wheel).

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
pip install brotli sentencepiece python-minifier numpy huggingface-hub zstandard einops ninja

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

for seed in 42 0 1234; do
  SEED=$seed \
  QK_GAIN_INIT=5.25 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=4 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${seed}.log
done
```

## Credits / Lineage

- **@jorge-asenjo** — PR #1700: full base stack (Multi-Phase Global SGD TTT, Phased LoRA TTT, VarLen attention, fused Triton MLP, SP8192 pipeline). This submission tunes two hyperparameters on top of PR #1700.
- **@bigbag** — PR #1493 (merged SOTA): originated QK-Gain 5.25 tune, 3-layer depth recurrence, parallel residuals.
- **@samacqua** — PR #1530: VarLen + fused MLP base for PR #1700 lineage.
- **@clarkkev** — PR #1394: SP8192 + GPTQ SDClip + depth recurrence base.
- **@abaybektursun** — PR #549: legal score-first TTT framework.
- **@dexhunter** — PR #1626: Multi-Phase Global SGD TTT concept.
