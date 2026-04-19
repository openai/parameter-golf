# Record: SP8192 + Hadamard Rotation + AWQ + Layer-wise Precision + Hessian-Aware Calibration + Legal TTT

**val_bpb = 1.0785** (3-seed mean, std 0.0001) | **~15.98 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPP | **TTT BPP** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.0791      | **1.0783**  | 15,978,456 |
| 314  | 1.0789      | **1.0785**  | 15,979,234 |
| 999  | 1.0787      | **1.0787**  | 15,977,892 |
| **Mean** | **1.0789** | **1.0785** | **15,978,527** |
| **Std** | **0.0002** | **0.0001** | |

Merged SOTA (PR #1493): **1.0810 BPP**. Delta: **-0.0025 BPP**. Improves upon leaderboard #1.

## Key Techniques

1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0), zero selective pruning (PR #1394 @clarkkev)

2. **Hadamard Rotation** — Orthogonal transformation for outlier removal before quantization, reduces quantization noise by ~2-3%, applied to activation tensors during QAT

3. **AWQ (Activation-aware Weight Quantization)** — Significance-aware quantization that preserves important weights with higher precision, computed from activation statistics over calibration data

4. **Layer-wise Precision Allocation** — Mixed-precision quantization:
   - Embeddings: Int8 (most sensitive)
   - Attention layers: Int8 for Q/K/V, Int6 for output
   - MLP layers: Int6 for FC1, Int4 for FC2 (less sensitive)
   - Residual connections: Int4 (least sensitive)

5. **Hessian-Aware Calibration** — Uses Fisher information matrix (diagonal approximation) to determine per-layer quantization ranges, aligns quantization with model sensitivity

6. **3-Layer Depth Recurrence** (layers 3,4,5, activate at frac=0.35) — 17 virtual layers from 11 physical (PR #1331 @dexhunter, PR #1437 @dexhunter)

7. **Parallel Residuals** (layers 7+) — GPT-J style, attention and MLP read from same input (PR #1412 @Robby955, PR #1204 @msisovic)

8. **QK-Gain 5.25** — learnable per-head query scaling, monotonic improvement from 4.0 to 5.25

9. **Legal Score-First TTT** — SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk, cosine LR decay. Score-before-update ordering. (PR #549 @abaybektursun, PR #1413 @dexhunter)

10. **Tuned Hyperparameters** — WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 (PR #1445 @X-Abhishek-X)

11. **LZMA code wrapper** — ~16.6KB code, saves ~43KB vs uncompressed

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at step ~2016). Parallel residuals from layer 7: attention and MLP operate on same pre-residual input. Skip gates (sigmoid-gated U-Net connections).

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps), AdamW for embeddings/scalars. 4550 steps in 588s on 8xH100 SXM. Linear warmdown to LR=0 over final 72% of training. EMA decay 0.9965.

## Quantization

**Hadamard-Rotated AWQ with Hessian-Aware Calibration:**
- Hadamard rotation applied to activation tensors to orthogonalize before quantization
- AWQ computes importance scores from activation statistics: `importance = mean(|activation|)`
- Layer-wise precision determined by Hessian sensitivity: `sensitivity = sqrt(fisher_diag)`
- Quantization ranges adjusted per-layer: `range = base_range * (1 + sensitivity_norm)`
- Byte-shuffle + Brotli-11 compression. Zero selective pruning needed -- model fits natively under 16MB.

## TTT (Test-Time Training)

Score-first, chunk-based SGD adaptation at eval time:
- Chunk val tokens into 32K-token chunks
- For each chunk: (1) score all sliding windows under `torch.no_grad()`, (2) train model on scored chunk tokens with SGD
- 3 epochs per chunk, cosine LR decay across chunks
- Gradient clipping at 1.0, distributed all-reduce for multi-GPU
- Total TTT eval time: ~370s (within 600s eval budget)

## Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()` BEFORE any SGD update. Training only on already-scored tokens.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:
- No SLOT (standard or causal)
- No pre-quant TTT on val data (model quantized once during training, TTT adapts at eval time)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes on all 3 seeds
- Training under 600s on all 3 seeds (~588s actual)
- Eval (sliding + TTT) under 600s on all 3 seeds (~500s actual)

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  HADAMARD_ROTATION_ENABLED=1 AWQ_ENABLED=1 HESSIAN_AWARE_CALIBRATION=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@clarkkev** — SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + depth recurrence (PR #1394)
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549, merged precedent)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445, #1471)
- **@Victory963** — Hadamard rotation, AWQ, layer-wise precision, Hessian-aware calibration

## Acknowledgements

Thanks to OpenAI's Advanced Competitor grant ($500 compute credit via RunPod) -- this was instrumental in running the experiments that led to this result.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`
