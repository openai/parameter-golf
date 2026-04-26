# Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT + V-Gated

**val_bpb = 1.0796** (3-seed mean, std 0.00025) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPP | **TTT BPP** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.08112468     | **1.07985553**  | 15985814 |
| 314  | 1.08057225      | **1.07927887**  | 15983675 |
| 999  | 1.08110726    | **1.07973035**  | 15986648 |
| **Mean** | **1.0809** | **1.0796** | **15985379** |
| **Std** | **0.00025** | **0.00025** | |

## Key Techniques

1. Based on **SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT** (PR #1493 by @bigbag).
2. Added a learnable **final norm scale** and a **Smear gate** to make representations smoother and slightly more compression-friendly, reducing the compressed artifact size by about **40 KB** and freeing space for additional parameters.
3. Added a **per-head V-Gate**, where the V projection jointly determines both **what information is fed into attention** and **how much each head contributes to the output**. This significantly improved model performance.
4. Improved the quantized compression pipeline with **per-matrix automatic layout selection**, giving a small further reduction in final artifact size of about **10 KB**.
5. Performed extensive hyperparameter search, including settings such as `MUON_BACKEND_STEPS=4` and `TTT_LR=0.01`. This made training more stable and improved reproducibility.

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at step ~2016). Parallel residuals from layer 7: attention and MLP operate on same pre-residual input. Skip gates (sigmoid-gated U-Net connections). Final norm scale, Smear gate and V-Gate.

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=314 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.01 TTT_EPOCHS=3 \
MLP_MULT=4 MUON_BACKEND_STEPS=4 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@bigbag** — SP8192 + 3-Layer Recurrence + Parallel Residuals + QK_GAIN_INIT=5.25 (PR #1493)
- **@clarkkev** — SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + depth recurrence (PR #1394)
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549, merged precedent)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445, #1471)
- **@kellerjordan** -- SmearGate concept (originally from modded-nanogpt)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`