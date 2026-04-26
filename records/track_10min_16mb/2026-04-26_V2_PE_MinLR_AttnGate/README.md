# Record: SP8192 + PE + MIN_LR + SmearGate + AttnOutGate + 4ep TTT — val_bpb 1.0770 (3-seed mean)

**val_bpb = 1.0770** (3-seed mean, std 0.0004) | **~15.98 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Steps | Sliding BPB | **TTT BPB** | Artifact (bytes) |
|------|-------|-------------|-------------|-------------------|
| 1337 | 4631 | 1.0785 | **1.0772** | 15,982,989 |
| 42 | 4637 | 1.0777 | **1.0765** | 15,984,317 |
| 2024 | 4633 | 1.0784 | **1.0772** | 15,985,404 |
| **Mean** | **4634** | **1.0782** | **1.0770** | **15,984,237** |
| **Std** | | 0.0004 | **0.0004** | |

Delta vs previous SOTA (1.0783): **-0.0013 BPB**

## Changes from previous SOTA (2026-04-12)

### Training improvements
- **Polar Express NS coefficients** — 5 per-iteration minimax-optimal tuples + row normalization (was: fixed 3.4445/-4.775/2.0315)
- **MIN_LR=0.10** warmdown floor (was: 0.0 — LR dropped to zero)
- **QK_GAIN_INIT=5.25** (was: 5.0)
- **GPTQ_RESERVE_SECONDS=0.5** (was: 12.0)
- **VAL_LOSS_EVERY=0** — skip periodic val during training

### Architecture additions
- **SmearGate** — causal content-gated residual, zero-init transparent
- **Attention Output Gate** — per-head sigmoid gate on attn output (width=12), zero-init

### TTT improvement
- **4 epochs** (was: 3) of score-first SGD TTT

## Architecture (unchanged from base)

```
SP8192 tokenizer, 11 physical / 17 virtual layers
512 dim, MLP 4x (2048 hidden), GQA 8Q/4KV, head_dim=64
Parallel residuals L7+, QK-Gain 5.25, XSA all 11 layers
LeakyReLU(0.5)², skip gates, logit softcap 30
MuonEq-R (lr=0.022, wd=0.095, momentum=0.97) + AdamW
EMA 0.997, warmdown 66.7%, loop at 35%
SDClip GPTQ int6 (k=12.85) + int8 embed (k=20) + brotli
Score-first TTT: SGD lr=0.01, mom=0.9, 4ep, 32K chunks
Hash embedding: 16384x512, zero-init, trained in TTT
~36M params, ~15.98MB artifact
```

## Compliance (Track B — Score-First TTT)

Per Issue #1017:
- **Condition 1:** Hash key uses prefix tokens only
- **Condition 2:** Full normalized softmax distribution
- **Condition 3:** Each chunk scored under no_grad() before TTT update
- **Condition 4:** Single left-to-right pass, no rescoring

No SLOT, no pre-quant TTT, no n-gram caches, no CaseOps, no global TTT, no multi-phase.

## Reproduction

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
SEED=1337 TTT_ENABLED=1 HASH_EMBED_ENABLED=1 TTT_LR=0.01 TTT_EPOCHS=4 TTT_OPTIMIZER=sgd MUON_MOMENTUM=0.97 GLOBAL_TTT_ENABLED=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
