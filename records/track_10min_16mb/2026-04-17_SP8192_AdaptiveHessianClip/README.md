# Record: SP8192 + Adaptive Hessian-Sensitivity GPTQ Clipping

**val_bpb = 1.0822** (3-seed mean, std 0.0009) | **~15.91 MB** | 8xH100 SXM

## 3-Seed Results

| Seed     | Sliding BPB | Artifact       |
|----------|-------------|----------------|
| 1337     | **1.0811**  | 15,906,928     |
| 42       | **1.0826**  | 15,909,023     |
| 999      | **1.0828**  | 15,911,535     |
| **Mean** | **1.0822**  | **15,909,162** |
| **Std**  | **0.0009**  |                |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0325 BPP**. Clears the 0.005-nat threshold.

## Novel Contribution: Adaptive Hessian-Sensitivity GPTQ Clipping

Standard GPTQ uses a global `clip_sigmas` parameter (e.g., 12.85) for all weight matrices. This submission replaces that with **per-tensor adaptive clipping** derived from Hessian sensitivity analysis.

**Key insight:** Weight matrices with higher Hessian sensitivity (measured as `H_diag.mean() * row_variance`) suffer more from quantization error. These layers should use wider clipping windows (higher clip_sigmas) to preserve precision, while less-sensitive layers can tolerate tighter clipping for better compression.

**Algorithm:**
1. Collect full Hessian matrices from calibration data (same as baseline GPTQ)
2. For each weight matrix, compute sensitivity: `sens = mean(diag(H)) * mean(var(W, dim=1))`
3. Compute log-space raw clip_sigmas: `log_cs = -0.15 * log(sens)` (conservative exponent)
4. Binary search for an additive offset in log-space such that the numel-weighted mean of `log(clamp(exp(log_cs + offset), 6.0, 24.0))` equals `log(12.85) + 0.012` (baseline budget + compression margin)
5. Final per-tensor clip_sigmas: `clamp(exp(log_cs + offset), 6.0, 24.0)`

**Result:** Clip_sigmas range from ~8.5 (early layers, high sensitivity) to ~19.0 (deep decoder layers, low sensitivity), while the numel-weighted geometric mean preserves the compression budget of the global baseline.

Example per-tensor clip_sigmas from seed 1337:
- `blocks.0.mlp.proj.weight`: 8.46 (high sensitivity, tight clipping)
- `blocks.2.attn.proj.weight`: 14.05
- `blocks.8.attn.proj.weight`: 19.02 (low sensitivity, wide clipping)

## Other Techniques

1. **SP8192 + GPTQ SDClip** -- int6 matrices, int8 embeddings, zero selective pruning (PR #1394 @clarkkev)
2. **3-Layer Depth Recurrence** (layers 3,4,5, activate at frac=0.35) -- 17 virtual layers from 11 physical (PR #1331 @dexhunter, PR #1437 @dexhunter)
3. **Parallel Residuals** (layers 7+) -- GPT-J style, attention and MLP read from same input (PR #1412 @Robby955, PR #1204 @msisovic)
4. **QK-Gain 5.25** -- learnable per-head query scaling
5. **Tuned Hyperparameters** -- WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 (PR #1445 @X-Abhishek-X)
6. **zlib+base85 code wrapper** -- 61KB source compressed to ~19.8KB

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at step ~2038). Parallel residuals from layer 7: attention and MLP operate on same pre-residual input. Skip gates (sigmoid-gated U-Net connections).

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps), AdamW for embeddings/scalars. ~4612 steps in 588s on 8xH100 SXM. Linear warmdown to LR=0 over final 72% of training. EMA decay 0.9965.

## Quantization

Full-Hessian GPTQ with **adaptive per-tensor SDClip**: clip_sigmas derived from Hessian sensitivity (see above). int6 for attention/MLP matrices, int8 for token embeddings. Byte-shuffle + Brotli-11 compression.

## Compliance

- **Training under 600s:** ~588s on all seeds
- **Artifact under 16MB:** All seeds under 15,912,000 bytes (15.91 MB)
- **Eval under 600s:** Sliding window eval ~91s per seed
- **No SLOT, no ETLB, no n-gram cache**
- **Three seeds:** 1337, 42, 999

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@clarkkev** -- SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + depth recurrence (PR #1394)
- **@dexhunter** -- 3-layer depth recurrence (PR #1331, #1437)
- **@Robby955** -- Parallel residuals on SP8192 (PR #1412)
- **@msisovic** -- Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** -- Hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` (zlib+base85 compressed wrapper, 19,846 bytes)
- `train_seed1337.log`
- `train_seed42.log`
- `train_seed999.log`
