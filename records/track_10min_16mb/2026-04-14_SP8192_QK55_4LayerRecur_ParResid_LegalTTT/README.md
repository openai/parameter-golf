# Attempt: SP8192 + QK-Gain 5.5 + 4-Layer Recurrence + Parallel Residuals + Legal TTT

> ⚠️ **TRIAL RUN — Not on required 8xH100 SXM hardware**
> This run was executed on a single **1xA100 SXM** GPU as a proof-of-concept and compute grant application. The model was stopped early at step 682/20000 due to the 600s wallclock cap. Results below are from this single-GPU trial. Full 8xH100 runs with 3 seeds pending compute grant approval.

## Results (1xA100 Trial Run)

| Metric | Value |
|--------|-------|
| val_bpb (fp32) | **1.4050** (step 682) |
| val_bpb (int8+zlib roundtrip) | **1.4100** |
| val_loss (fp32) | 2.3723 |
| val_loss (int8+zlib roundtrip) | 2.3807 |
| Steps completed | 682 / 20000 |
| Train time | 600.25s (wallclock cap hit) |
| Eval time | 23.1s |
| Artifact size (int8+zlib) | **11,160,217 bytes** (11.16 MB ✅ < 16 MB) |
| GPU | 1× A100 SXM (trial; target: 8×H100 SXM) |

## Hypothesis

The current SOTA (PR #1493, bigbag, 1.0810) noted **monotonic improvement from QK-Gain 4.0 → 5.25**, suggesting the optimum may not yet be reached. This submission tests:

1. **QK-Gain 5.5** — pushing the learnable per-head query scale one step further
2. **4-Layer Depth Recurrence** (layers 2,3,4,5) — extending the 3-layer recurrence from PR #1493 to include layer 2, increasing virtual depth to 18 from 11 physical layers
3. All other techniques inherited from PR #1394 / PR #1493 stack unchanged

## Architecture Changes vs SOTA

| Param | SOTA (1.0810) | This attempt |
|-------|---------------|--------------|
| QK-Gain | 5.25 | **5.5** |
| Recurrence layers | 3,4,5 (3-layer) | **2,3,4,5 (4-layer)** |
| Recurrence activation frac | 0.35 | **0.30** (earlier to allow more loop steps) |
| Everything else | — | unchanged |

## Training Log (1xA100 Trial)

```
step:0/20000    val_loss:6.9344  val_bpb:4.1069  train_time:0ms
step:1/20000    train_loss:6.9357  train_time:891ms
step:10/20000   train_loss:5.9437  train_time:9091ms  step_avg:909ms
step:200/20000  train_loss:2.7328  train_time:173253ms step_avg:866ms
step:400/20000  train_loss:2.3531  train_time:350663ms step_avg:877ms
step:600/20000  train_loss:2.4464  train_time:527465ms step_avg:879ms
step:682/20000  val_loss:2.3723   val_bpb:1.4050  train_time:600254ms
stopping_early: wallclock_cap
peak memory: 14000 MiB allocated / 14218 MiB reserved
Serialied model: 67,224,578 bytes
Code size: 48,310 bytes
Total submission size: 67,272,888 bytes
int8+zlib model: 11,111,907 bytes
Total int8+zlib: 11,160,217 bytes
final_int8_zlib_roundtrip val_loss:2.3807 val_bpb:1.4100 eval_time:23096ms
```

## How to Run (Full 8xH100)

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

## 3-Seed Results (Pending — requires 8xH100 compute grant)

| Seed | val_bpb | Artifact |
|------|---------|----------|
| 42   | pending | pending  |
| 314  | pending | pending  |
| 999  | pending | pending  |
| **Mean** | **pending** | **pending** |
| **Std**  | **pending** | — |

## Key Techniques (inherited from SOTA stack)

1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0)
2. **4-Layer Depth Recurrence** (layers 2,3,4,5, activate at frac=0.30) — 18 virtual layers from 11 physical
3. **Parallel Residuals** (layers 7+) — GPT-J style
4. **QK-Gain 5.5** — extended from 5.25, testing monotonic improvement hypothesis
5. **Legal Score-First TTT** — SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk
6. **Tuned Hyperparameters** — WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72
7. **LZMA code wrapper** — ~16.6KB code

## Architecture

11L × 512d × 8H / 4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: loops layers 2-5 (activated at step ~1728). Parallel residuals from layer 7.

## Compliance Checklist

- [ ] train_under_600s — *(not tested on 8xH100; trial hit cap at 682 steps on 1xA100)*
- [x] artifact_under_16mb — 11.16 MB ✅
- [ ] eval_under_600s — *(eval was 23s on 1xA100, expected well under 600s on 8xH100)*
- [x] no_slot
- [x] no_pre_quant_ttt
- [x] no_etlb
- [x] no_ngram_cache
- [x] score_first_ttt
- [ ] three_seeds — *(pending compute grant)*

## Attribution

- **@clarkkev** — SP8192 + GPTQ SDClip + MuonEq-R (PR #1394)
- **@dexhunter** — depth recurrence framework (PR #1331, #1437), legal TTT (PR #1413)
- **@abaybektursun** — score-first TTT (PR #549)
- **@Robby955** — parallel residuals (PR #1412)
- **@X-Abhishek-X** — hyperparameter tuning (PR #1445)
- **@bigbag** — 3-layer recurrence + QK-Gain 5.25 (PR #1493)
