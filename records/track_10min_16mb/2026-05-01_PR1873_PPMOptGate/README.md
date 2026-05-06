# SP10240 Casefold + TTT + GPTQ + PPM-D with Tuned Gate (T=0.7 / H=0.99 / L=0.3) — 0.80051 BPB

**val_bpb: 0.80050966** (3-seed mean, std 0.00026321) | seeds: 42 / 314 / 999 | 8×H100 80GB SXM

## Lineage

This submission **builds directly on PR #1873 (@schattenjuwel / Liva, originally nothingLiva PR1707)** — _SP10240 Casefold + TTT + GPTQ + Byte-PPM-D Mixture, val_bpb 0.82006_. The neural network, training pipeline, GPTQ quantization, TTT phase, and PPM-D order-5 byte mixer are **byte-identical** to PR #1873. `train_gpt.py` is unchanged from PR #1873; only env-var hyperparameters differ at run time. Full credit to Liva for the underlying stack.

## What's New

I found PR #1873's hand-picked PPM-D gate hyperparameters can be improved by an offline sweep on a dumped `(tga, lpa)` from their actual NN distribution. The mixer is:

```
lam = (PPM_LLO if cf >= PPM_C else PPM_LHI)
mix = log(lam * exp(nn_token_logp) + (1 - lam) * exp(ppm_tok_lp))
```

PR #1873 used hand-picked `PPM_C=0.9, PPM_LHI=0.9, PPM_LLO=0.05`. Sweep on the dump found `PPM_C=0.7, PPM_LHI=0.99, PPM_LLO=0.3` dominates by **+19.7 mBPB on the 3-seed mean** (0.82006 → 0.80051).

| param                                        | PR #1873 | This submission | Effect                                      |
| -------------------------------------------- | -------- | --------------- | ------------------------------------------- |
| `PPM_C` (confidence threshold for switching) | 0.9      | **0.7**         | Switch to heavy PPM at lower PPM confidence |
| `PPM_LHI` (lambda when low confidence)       | 0.9      | **0.99**        | Even more NN weight when PPM uncertain      |
| `PPM_LLO` (lambda when high confidence)      | 0.05     | **0.3**         | Less aggressive PPM lock-in                 |

Net effect: a sharper gate that trusts NN more when PPM is uncertain (high `PPM_LHI`) and trusts PPM less aggressively when it's confident (higher `PPM_LLO`), with the switching threshold lowered (lower `PPM_C`) to widen the regime where PPM contributes meaningfully.

## Per-seed Results (8×H100 80GB SXM, ≤600s training)

| Seed     | val_bpb        |
| -------- | -------------- |
| 42       | 0.80076582     |
| 314      | 0.80023955     |
| 999      | 0.80052360     |
| **Mean** | **0.80050966** |
| Std      | 0.00026321     |

3-seed std 0.00026 — well below the headroom over PR #1873's 0.82006 (+19.7 mBPB / 76× std).

## Run Commands

```bash
# Seed 42
RUN_ID=optgate_seed42 SEED=42 \
  PPM_ENABLED=1 PPM_ORDER=5 \
  PPM_C=0.7 PPM_LHI=0.99 PPM_LLO=0.3 \
  TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=4 \
  MAX_WALLCLOCK_SECONDS=600 DATA_DIR=./data/ \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Seed 314
RUN_ID=optgate_seed314 SEED=314 \
  PPM_ENABLED=1 PPM_ORDER=5 \
  PPM_C=0.7 PPM_LHI=0.99 PPM_LLO=0.3 \
  TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=4 \
  MAX_WALLCLOCK_SECONDS=600 DATA_DIR=./data/ \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Seed 999
RUN_ID=optgate_seed999 SEED=999 \
  PPM_ENABLED=1 PPM_ORDER=5 \
  PPM_C=0.7 PPM_LHI=0.99 PPM_LLO=0.3 \
  TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=4 \
  MAX_WALLCLOCK_SECONDS=600 DATA_DIR=./data/ \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Compliance

- **Code**: byte-identical to PR #1873's `train_gpt.py` (no source edits).
- **Causal PPM**: score-before-update on every byte; gate `cf` computed from PPM tables before looking up the observed byte's count.
- **Single left-to-right pass** ✅
- **Token-level mixing** at probability level (sum byte log-probs into token log-prob, then mix) — same as PR #1873.
- **Artifact size**: < 16,000,000 bytes (all 3 seeds).
- **Training time**: < 600s wall clock (all 3 seeds).
- **3-seed validation** with seeds 42/314/999.

## Acknowledgments

- **Liva (@schattenjuwel / nothingLiva)** for PR #1873 — the entire underlying stack (SP10240 Casefold tokenizer, TTT, GPTQ int6/int7, brotli, PPM-D order-5 byte mixer). This submission contributes only the gate hyperparameter discovery on top of that work.
- **PR #1835 (@anmarhindi)** for the byte-PPM mixture inspiration that PR #1873 builds on.
- OpenAI for hosting the Parameter Golf challenge.
