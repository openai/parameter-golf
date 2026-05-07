# Non-record: SP8192 + LoRA on tied embedding

**val_bpb = 1.07994** (1 seed only, seed 42) | **~15.99 MB** | 8xH100 SXM

## 1-Seed Result

| Seed | Sliding BPP | **TTT BPP** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.0818      | **1.07994** | 15,993,732 |

Merged SOTA (bigbag PR #1493): **1.0810** (3-seed mean). Delta vs bigbag seed 42 (1.08079): **−0.00085 BPB**. Below the 0.005-nat threshold, so submitted as **non-record**.

## Key Techniques

All of bigbag's stack (PR #1493), unchanged, plus two small additions applied only to the tied token embedding at GPTQ time:

1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0), zero selective pruning (PR #1394 @clarkkev)
2. **3-Layer Depth Recurrence** (layers 3,4,5, activate at frac=0.35) — 17 virtual layers from 11 physical (PR #1331, #1437 @dexhunter)
3. **Parallel Residuals** (layers 7+) — GPT-J style (PR #1412 @Robby955, PR #1204 @msisovic)
4. **QK-Gain 5.25** — learnable per-head query scaling
5. **Legal Score-First TTT** — SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk (PR #549 @abaybektursun, PR #1413 @dexhunter)
6. **Tuned Hyperparameters** — WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 (PR #1445 @X-Abhishek-X)
7. **Rank-1 int8 LoRA residual on tok_emb** — NEW in this PR
8. **Hessian-weighted shrinkage in GPTQ rounding** — NEW in this PR

## Architecture

Identical to bigbag: 11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10]. Parallel residuals from layer 7. Skip gates.

## Training

Unchanged from bigbag: MuonEq-R + AdamW, 4608 steps in 588s on 8xH100 SXM, linear warmdown 0.72, EMA 0.9965.

## Quantization

Inherits bigbag's full-Hessian GPTQ with SDClip (int6 matrices, int8 embeddings, byte-shuffle + Brotli-11). **New on top:**

- **LoRA residual on tok_emb.** After int8 GPTQ rounding, compute SVD of the residual `E = W_fp - W_deq` and keep rank-1 `(A, B)` stored as int8 with per-row/col fp16 scales. Dequantization returns `W = W_deq + A @ B`. Recovers ~14% of residual Frobenius energy at ~8 KB net cost.
- **Hessian-weighted shrinkage.** Columns with below-mean Hessian diagonal get an extended zero-zone during rounding (thresh 0.55, H-cutoff 0.5 × mean). Low-impact columns compress better with minimal BPB cost.

Controlled by `LORA_EMBED_RANK=1`, `SHRINKAGE_THRESH=0.55`, `SHRINKAGE_H_CUTOFF=0.5`.

## TTT (Test-Time Training)

Unchanged from bigbag — score-first, chunk-based SGD at eval time, 3 epochs per 32K chunk, cosine LR decay, gradient clipping 1.0. Total TTT eval time ~350 s.

## Compliance

Per Issue #1017 (Track B — legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()` BEFORE any SGD update.
- **Condition 4 (Single pass):** Each token scored exactly once.

Additional:
- No SLOT, no pre-quant TTT, no ETLB, no n-gram cache
- Artifact under 16,000,000 bytes (15,993,732 B)
- Training under 600s (588 s)
- Eval (sliding + TTT) under 600s (~468 s)

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  SHRINKAGE_THRESH=0.55 SHRINKAGE_H_CUTOFF=0.5 LORA_EMBED_RANK=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

Inherits all of bigbag's credits (PR #1493):

- **@clarkkev** — SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + depth recurrence (PR #1394)
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning (PR #1445, #1471)
- **@bigbag** — combined stack (PR #1493)
- **@yijieyuan (this PR)** — rank-1 int8 LoRA residual on tied tok_emb + Hessian-weighted shrinkage

## Notes

Single-seed only due to compute budget. A 3-seed mean would be needed for a record claim — marked non-record accordingly.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
