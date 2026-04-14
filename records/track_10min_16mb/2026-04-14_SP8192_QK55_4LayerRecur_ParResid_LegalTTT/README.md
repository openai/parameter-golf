# Attempt: SP8192 + QK-Gain 5.5 + 4-Layer Recurrence + Parallel Residuals + Legal TTT

**⚠️ PLACEHOLDER — logs and val_bpb to be filled after training runs**

**Target val_bpb < 1.0760** (must beat SOTA 1.0810 by ≥0.005)

## Hypothesis

The current SOTA (PR #1493, bigbag, 1.0810) noted **monotonic improvement from QK-Gain 4.0 → 5.25**, suggesting the optimum may not yet be reached. This submission tests:

1. **QK-Gain 5.5** — pushing the learnable per-head query scale one step further
2. **4-Layer Depth Recurrence** (layers 2,3,4,5) — extending the 3-layer recurrence from layers 3-5 to include layer 2, increasing virtual depth to 18 from 11 physical layers
3. All other techniques inherited from PR #1394 / PR #1493 stack unchanged

## Architecture Changes vs SOTA

| Param | SOTA (1.0810) | This attempt |
|-------|---------------|--------------|
| QK-Gain | 5.25 | **5.5** |
| Recurrence layers | 3,4,5 (3-layer) | **2,3,4,5 (4-layer)** |
| Recurrence activation frac | 0.35 | **0.30** (earlier to allow more loop steps) |
| Everything else | — | unchanged |

## How to Run

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Seed 42
SEED=42 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  RECUR_LAYERS="2,3,4,5" RECUR_FRAC=0.30 \
  torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/train_gpt.py

# Seed 314
SEED=314 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  RECUR_LAYERS="2,3,4,5" RECUR_FRAC=0.30 \
  torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/train_gpt.py

# Seed 999
SEED=999 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  RECUR_LAYERS="2,3,4,5" RECUR_FRAC=0.30 \
  torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/train_gpt.py
```

## 3-Seed Results (PLACEHOLDER — fill after runs)

| Seed | Sliding BPP | TTT BPP | Artifact |
|------|-------------|---------|----------|
| 42   | TBD         | TBD     | TBD      |
| 314  | TBD         | TBD     | TBD      |
| 999  | TBD         | TBD     | TBD      |
| **Mean** | **TBD** | **TBD** | **TBD** |
| **Std**  | **TBD** | **TBD** |          |

## Key Techniques (inherited from SOTA stack)

1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0)
2. **4-Layer Depth Recurrence** (layers 2,3,4,5, activate at frac=0.30) — 18 virtual layers from 11 physical
3. **Parallel Residuals** (layers 7+) — GPT-J style
4. **QK-Gain 5.5** — extended from 5.25, testing monotonic improvement hypothesis
5. **Legal Score-First TTT** — SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk
6. **Tuned Hyperparameters** — WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72
7. **LZMA code wrapper** — ~16.6KB code

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: loops layers 2-5 (activated at step ~1728). Parallel residuals from layer 7.

## Compliance Checklist (to verify after runs)

- [ ] train_under_600s
- [ ] artifact_under_16mb  
- [ ] eval_under_600s
- [ ] no_slot
- [ ] no_pre_quant_ttt
- [ ] no_etlb
- [ ] no_ngram_cache
- [ ] score_first_ttt
- [ ] three_seeds

## Attribution

- **@clarkkev** — SP8192 + GPTQ SDClip + MuonEq-R (PR #1394)
- **@dexhunter** — depth recurrence framework (PR #1331, #1437), legal TTT (PR #1413)
- **@abaybektursun** — score-first TTT (PR #549)
- **@Robby955** — parallel residuals (PR #1412)
- **@X-Abhishek-X** — hyperparameter tuning (PR #1445)
- **@bigbag** — 3-layer recurrence + QK-Gain 5.25 (PR #1493)
