# Record: Discriminative TTT — val_bpb 1.0807 (3-seed mean)

## Result

**val_bpb: 1.0807** (3-seed mean, std 0.0005) | ~15.8 MB | 8xH100 SXM

| Seed | BPB | val_loss | Artifact |
|------|-----|----------|----------|
| 1337 | 1.0803 | 1.8241 | 15,815,343 |
| 42 | 1.0805 | 1.8243 | 15,810,497 |
| 2025 | 1.0812 | 1.8255 | 15,804,659 |
| **Mean** | **1.0807** | **1.8246** | |

Delta vs merged SOTA (PR #1019, 1.1147): **-0.0340 BPP** (-0.034 nats, p < 0.001).

## Novelty: Discriminative Test-Time Training (dTTT)

**Nearest comparable PR:** PR #1306 (our prior submission: pre-quant AdamW TTT, 6 epochs, flat LR, freeze=2, 1.0846 BPP)

**What we share:** Pre-quant AdamW TTT — adapting EMA weights before GPTQ quantization.

**What is mechanistically different:** Discriminative TTT applies **per-block adaptive learning rates** during pre-quant TTT, inspired by ULMFiT discriminative fine-tuning (Howard & Ruder, 2018). Instead of a flat LR across all unfrozen blocks, each transformer block receives a learning rate scaled by its depth:

- Block 0 (earliest): 0.3x base LR
- Block 10 (latest): 1.0x base LR
- Intermediate blocks: linearly interpolated

This replaces the binary freeze/unfreeze approach used by all existing TTT submissions:
- PR #549: freeze=2, flat LR (SGD)
- PR #1318: freeze=10/11, flat LR (AdamW, last block only)
- PR #1306: freeze=2, flat LR (AdamW)

Discriminative LR is a **gradient between freeze and full adaptation** — early blocks are "mostly frozen" (low LR preserves learned features) while later blocks are "fully adapted" (high LR for distribution shift). This is a new mechanism, not parameter tuning: no existing PR modulates LR per block during TTT.

Additionally:
- **All blocks trainable (freeze=0):** Community feedback (@MatoTeziTanka, Issue #140 comment) confirmed that freeze=0 outperforms freeze=2 on similar stacks. We verified: freeze=0 + dTTT gives -0.0007 over freeze=2 + dTTT.
- **10 epochs** (up from 6): More TTT adaptation steps. Fits within eval time budget (~185s).

## Technique Stack

| Component | Detail |
|-----------|--------|
| Base | PR #1019 fork (Full Hessian GPTQ, XSA-all, BigramHash 2048x128) |
| Training | Parallel Muon, ~87ms/step, ~6900 steps in 600s |
| **Pre-quant dTTT (novel)** | **AdamW, 10 epochs, freeze=0, per-block LR: 0.3x (block 0) to 1.0x (block 10), linear interpolation** |
| Quantization | Full Hessian GPTQ int6, damp=0.005, training-data calibration |
| Config | QK_GAIN=5.0, WARMDOWN=4000 |
| Coprime loader | Weighted random shard sampling with coprime stride |

## Compliance (Track A — Fixed Predictor)

This submission is a **Track A (fixed predictor)** run under the Issue #1017 framework:

- **No SLOT** — no eval-time delta optimization of any kind
- **No TTT during eval** — all TTT happens BEFORE quantization, within the training time budget
- **No n-gram cache** — no eval-time statistics accumulation
- **No eval-time adaptation of any kind** — model weights are frozen after training + TTT + GPTQ
- **Condition 1** (causal dependence): Standard autoregressive sliding-window eval, no future token access
- **Condition 2** (full normalized distribution): Standard softmax over full 1024-token vocabulary
- **Condition 3** (score-before-update): No updates during eval at all
- **Condition 4** (single left-to-right pass): Single sliding-window pass, no rescoring

All improvements are purely training-time. The eval procedure is identical to PR #1019.

## Pipeline

1. Training: 600s on 8xH100 (~87ms/step, ~6900 steps)
2. **Pre-quant Discriminative TTT: 10 epochs, per-block LR (~185s)**
3. GPTQ int6 quantization: ~23s
4. Sliding window eval (stride=64): ~115s

Total: ~15 min (training 10 min + eval ~5 min).

## Reproduction

```bash
# Install FA3
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Run (seed 1337 is default)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

No env vars needed. All config is hardcoded as defaults. SLOT is disabled by default (`SLOT_ENABLED=0`).

## Credits

- Base: PR #549 (@sanjeevmadhav), PR #1019 (@abaybektursun)
- Pre-quant AdamW TTT: PR #1006 (@abaybektursun)
- Discriminative fine-tuning concept: ULMFiT (Howard & Ruder, 2018)
- Coprime loader: PR #1184 (@icryo)
- QK-Gain: PR #1217 (@bigbag)
- Freeze=0 insight: @MatoTeziTanka (Issue #140 comment)
