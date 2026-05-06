# Record: SP8192 + Improved Parallel Residuals + Muon 0.97 + LR 0.03 + Legal TTT

**val_bpb = 1.07785** (3-seed mean, std 0.00047) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPP | **TTT BPP** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.07880     | **1.07718** | 15,990,780 |
| 314  | 1.07959     | **1.07810** | 15,987,449 |
| 999  | 1.07963     | **1.07826** | 15,987,550 |
| **Mean** | **1.07934** | **1.07785** | **15,988,593** |
| **Std**  | **0.00039** | **0.00047** | |

Merged SOTA (PR #1493, our previous): **1.0810 BPP**. Delta: **-0.0032 BPP**.

## Key Techniques

1. **Improved Parallel Residuals** (from PR #1529 @msisovic) -- cross-lane routing where attention and MLP outputs route to BOTH lanes via learned scalars. 66 new scalar params (`par_post[11,2,2]` + `par_resid[11,2]`). Final output = MLP lane (lane1). Starts at layer 7.

2. **Muon Momentum 0.97** (from PR #1514 @dexhunter) -- reduced from 0.99. Shorter memory horizon (~33 steps) better tracks the rapidly changing loss surface during warmdown.

3. **MATRIX_LR = 0.03** -- re-tuned for momentum 0.97 (higher LR pairs with lower momentum). Sweep: 0.022 → 1.0797, 0.03 → 1.0795, 0.04 → 1.0811.

4. **3-Layer Depth Recurrence** (L3-5, activate at frac=0.35) -- 17 virtual layers from 11 physical.

5. **QK-Gain 5.25** -- monotonic improvement from 4.0 to 5.25.

6. **Legal Score-First TTT** -- SGD (lr=0.005, mom=0.9), 3 epochs per 32K-token chunk, cosine LR decay.

7. **SP8192 + GPTQ SDClip** -- int6 matrices (k=12.85), int8 embeddings (k=20.0), Brotli-11 compression.

8. **Tuned Hyperparameters** -- WD=0.095, EMA=0.9965, warmdown=0.72.

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10]. Improved parallel residuals from layer 7: attention reads from lane0, MLP reads from lane1, both outputs route to both lanes via learned `par_post` and `par_resid` scalars. Skip gates (sigmoid-gated U-Net connections).

## Compliance (Track B)

Per Issue #1017:
- **Condition 1 (Causality):** Sliding-window eval, prefix only
- **Condition 2 (Normalized):** Standard softmax, no n-gram/logit bias
- **Condition 3 (Score before update):** Each chunk scored under `torch.no_grad()` BEFORE SGD
- **Condition 4 (Single pass):** Each token scored once, no rescoring

No SLOT, no pre-quant TTT, no ETLB, no n-gram cache. All artifacts < 16MB, train < 600s, eval < 600s.

## Reproduction

```bash
SEED=42 QK_GAIN_INIT=5.25 MUON_MOMENTUM=0.97 MATRIX_LR=0.03 \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@msisovic** -- Improved parallel residuals (PR #1529, #1204)
- **@clarkkev** -- SP8192 + GPTQ + SDClip + MuonEq-R (PR #1394)
- **@dexhunter** -- Muon 0.97 (PR #1514), depth recurrence (PR #1331, #1437), TTT on SP8192 (PR #1413)
- **@abaybektursun** -- Score-first TTT framework (PR #549)
- **@X-Abhishek-X** -- Hyperparameter tuning (PR #1445, #1471)
- **@Robby955** -- Parallel residuals on SP8192 (PR #1412)

## Acknowledgements

Thanks to OpenAI's Advanced Competitor grant ($500 compute credit via RunPod).

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`
