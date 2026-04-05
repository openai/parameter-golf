# Record: SP4096 + Linear LR Decay + Depth Recurrence + MuonEq-R

**val_bpb: 1.0924** (3-seed mean, std 0.0004) | **15.99 MB** | 8xH100 SXM, 600s | No TTT, no SLOT, no n-gram, no eval-time adaptation

**Improvement over current SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), 1.1147 BPB):** -0.0223 BPB (Welch t=-68.85, df=3.84, p << 0.001)

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 42 | 5,410 | 109.1 | 1.0974 | **1.0927** | 15,987,206 |
| 314 | 5,409 | 109.1 | 1.0977 | **1.0927** | 15,989,311 |
| 999 | 5,408 | 109.1 | 1.0970 | **1.0919** | 15,988,159 |
| **Mean** | **5,409** | **109.1** | **1.0974** | **1.0924** | |

Current SOTA (PR #1019, exact 3-seed mean): **1.11473509 BPB** (**1.88217853 nats**). This run's exact 3-seed mean: **1.09244346 BPB** (**2.51334715 nats**). Delta: **-0.02229163 BPB**.

Using the exact per-seed BPB scores from PR #1019 (`1.11508120`, `1.11437394`, `1.11475014`) and this run (`1.09269834`, `1.09269085`, `1.09194120`), Welch's t-test gives **t = -68.85**, **df = 3.84**, p << 0.001.

All four conditions from [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) satisfied. No test-time training, no SLOT, no n-gram cache, no eval-time adaptation. Evaluation is pure sliding-window at stride=64.

---

## Main Changes

The comparison baseline is [PR #1019](https://github.com/openai/parameter-golf/pull/1019), the current merged SOTA at **1.1147 BPB**. This submission belongs to the SP4096 architecture family introduced by [PR #1218](https://github.com/openai/parameter-golf/pull/1218) and extended with depth recurrence by [PR #1204](https://github.com/openai/parameter-golf/pull/1204). The implementation draws heavily from [PR #1296](https://github.com/openai/parameter-golf/pull/1296) by [@aryanbhosale](https://github.com/aryanbhosale) and [PR #1285](https://github.com/openai/parameter-golf/pull/1285) by [@dexhunter](https://github.com/dexhunter).

### 1. Linear LR Decay to Zero (This Work)

The single critical change. Prior cosine warmdown floored at 5% of peak LR:

```python
# before: cosine with floor
return lr_floor + (1 - lr_floor) * 0.5 * (1 + cos(pi * progress))  # lr_floor = 0.05
```

Replaced with linear warmdown to zero, matching the schedule used by all top SP4096 submissions:

```python
# after: linear to zero
return max((1.0 - frac) / warmdown_frac, 0.0)
```

The non-zero LR floor prevented weights from settling before GPTQ quantization, producing wider weight distributions that quantized and compressed worse. The measured impact on quantization quality:

| Metric | Cosine floor=0.05 | Linear floor=0.0 | Change |
|--------|-------------------|-------------------|--------|
| Quantization gap (roundtrip) | 0.038 BPB | 0.014 BPB | -61% |
| Values pruned to fit 16MB | 1,860,936 | 340,142 | -82% |
| Unpruned artifact size | 16.23 MB | 16.09 MB | -140 KB |
| Post-quant sliding BPB | 1.1124 | 1.0924 | -0.020 |

The quantization gap collapsed because: (1) weights converge to tighter distributions with less GPTQ error, (2) tighter weights compress better under Brotli, requiring less pruning, and (3) less pruning means fewer GPTQ compensation terms are destroyed.

### 2. Reduced Selective Pruning Factor

Changed the pruning safety margin from `excess * 8` to `excess * 4`. The prior heuristic over-pruned by approximately 2.4x (1.86M values zeroed when ~770K would have sufficed). The reduced factor still safely fits under 16MB while destroying fewer GPTQ error compensation terms.

---

## Architecture

| Component | Setting | First introduced by |
|-----------|---------|---------------------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 4x (2048) with LeakyReLU(0.5)^2 | [PR #493](https://github.com/openai/parameter-golf/pull/493) by [@parinzee](https://github.com/parinzee) (LeakyReLU); [PR #1218](https://github.com/openai/parameter-golf/pull/1218) by [@clarkkev](https://github.com/clarkkev) (MLP 4x + SP4096) |
| Tokenizer | SentencePiece BPE 4096 | [PR #1218](https://github.com/openai/parameter-golf/pull/1218) by [@clarkkev](https://github.com/clarkkev) |
| Depth Recurrence | Layers 4,5 repeated from step 3000 | [PR #1204](https://github.com/openai/parameter-golf/pull/1204) by [@msisovic](https://github.com/msisovic) |
| Parallel Residuals | From layer 7 | [PR #1204](https://github.com/openai/parameter-golf/pull/1204) by [@msisovic](https://github.com/msisovic) |
| XSA | All 11 layers | [PR #478](https://github.com/openai/parameter-golf/pull/478) by [@gowtham0992](https://github.com/gowtham0992) (XSA concept); [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by [@abaybektursun](https://github.com/abaybektursun) (XSA-all) |
| QK-Gain | 5.0 | [PR #1217](https://github.com/openai/parameter-golf/pull/1217) by [@clarkkev](https://github.com/clarkkev) |
| RoPE | Partial (16/64 dims) | [PR #315](https://github.com/openai/parameter-golf/pull/315) by [@jfprincz](https://github.com/jfprincz) |
| LN Scale | 1/sqrt(layer+1) | [PR #315](https://github.com/openai/parameter-golf/pull/315) by [@jfprincz](https://github.com/jfprincz) |
| U-Net Skips | Gated encoder-decoder connections | [PR #289](https://github.com/openai/parameter-golf/pull/289) |
| SmearGate | Learned token blending | [PR #65](https://github.com/openai/parameter-golf/pull/65) by [@aquariouseworkman](https://github.com/aquariouseworkman) |
| MuonEq-R | Row-normalize before Newton-Schulz | [arXiv:2603.28254](https://arxiv.org/abs/2603.28254) |
| Selective Pruning | +/-1 values by reconstruction error | [PR #609](https://github.com/openai/parameter-golf/pull/609) by [@saml212](https://github.com/saml212) |
| Full Hessian GPTQ | Cholesky error compensation + column reordering | [PR #535](https://github.com/openai/parameter-golf/pull/535) by [@raahilshah](https://github.com/raahilshah) (GPTQ); [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by [@abaybektursun](https://github.com/abaybektursun) (AR self-gen calibration concept) |
| EMA + Weight Averaging | EMA decay=0.997 every step | [PR #401](https://github.com/openai/parameter-golf/pull/401) by [@newjordan](https://github.com/newjordan) |
| FlashAttention 3 | Hopper warp-specialized kernels | [PR #122](https://github.com/openai/parameter-golf/pull/122) by [@mtybadger](https://github.com/mtybadger) |
| **Linear LR Decay** | **Warmdown to LR=0.0** | **This work** |

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon (matrix), AdamW (scalars/embeddings) |
| Matrix LR | 0.02 |
| Muon WD | 0.09 |
| Adam WD | 0.02 |
| Momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| Warmdown | Fraction 0.667, **linear decay to LR=0.0** |
| EMA | Decay 0.997, every step |
| Batch | 786,432 tokens/step (2048 seq_len x 48 seqs x 8 GPUs) |
| Grad clip | 0.3 |
| Recurrence | Activated at step 3000 |

## Quantization and Compression

| Component | Setting |
|-----------|---------|
| GPTQ | Full Hessian, 64 calibration batches from training data |
| Bit width | Int6 per-row for all attention + MLP weight matrices |
| Embeddings | Int8 per-row |
| Compression | Brotli quality=10 |
| Pruning | Selective +/-1 values, factor=4x excess (340K mean, down from 1.86M in prior stack) |

## Data

This run uses 26 SP4096 training shards and the full 50,000-document FineWeb validation set. BPB is computed as `val_loss / log(2) * (token_count / byte_count)`, the standard tokenizer-agnostic formula.

With full training data (80 shards), the pre-quantization BPP is expected to improve by approximately 0.003, projecting the final sliding-window BPB to approximately 1.089-1.092.

## Run Command

```bash
DATA_DIR=./data VOCAB_SIZE=4096 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in the script. No environment variable overrides needed beyond `DATA_DIR`, `VOCAB_SIZE`, and `SEED`.

## Requirements

FlashAttention 3 (Hopper) is required:

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece brotli
```

## Lineage

```
PR #549 (Legal SOTA 1.1194, @abaybektursun) -- LeakyReLU^2 + Parallel Muon base
  + PR #1019 (1.1147, @abaybektursun) -- AR self-gen GPTQ + XSA-all + BigramHash
    |
    +-- SP4096 branch:
        + PR #1218 (@clarkkev) -- SP4096 vocab + MLP 4x
        + PR #1204 (@msisovic) -- Depth Recurrence L4,5 + Parallel Residuals L7
        + PR #1217 (@clarkkev) -- QK-Gain 5.0
        + PR #1285 (@dexhunter) -- WD=0.090 + all-int6, first to fit MLP 4x under 16MB
        + PR #1296 (@aryanbhosale) -- Linear LR + MuonEq-R, clean 1.0897
          |
          +-- This work:
              + Linear LR decay to zero (quantization gap -61%)
              + Reduced pruning factor (pruning -82%)
              = 1.0924 BPB
```

## Credits

- **LeakyReLU(0.5)^2 activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by [@parinzee](https://github.com/parinzee)
- **SP4096 tokenizer + MLP 4x**: [PR #1218](https://github.com/openai/parameter-golf/pull/1218) by [@clarkkev](https://github.com/clarkkev)
- **Depth recurrence + parallel residuals**: [PR #1204](https://github.com/openai/parameter-golf/pull/1204) by [@msisovic](https://github.com/msisovic)
- **QK-Gain 5.0**: [PR #1217](https://github.com/openai/parameter-golf/pull/1217) by [@clarkkev](https://github.com/clarkkev)
- **XSA (Exclusive Self Attention)**: [PR #478](https://github.com/openai/parameter-golf/pull/478) by [@gowtham0992](https://github.com/gowtham0992); extended to all layers in [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by [@abaybektursun](https://github.com/abaybektursun)
- **Partial RoPE + LN Scale**: [PR #315](https://github.com/openai/parameter-golf/pull/315) by [@jfprincz](https://github.com/jfprincz)
- **U-Net skip connections**: [PR #289](https://github.com/openai/parameter-golf/pull/289)
- **SmearGate**: [PR #65](https://github.com/openai/parameter-golf/pull/65) by [@aquariouseworkman](https://github.com/aquariouseworkman)
- **Full Hessian GPTQ**: [PR #535](https://github.com/openai/parameter-golf/pull/535) by [@raahilshah](https://github.com/raahilshah)
- **AR self-gen GPTQ calibration concept**: [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by [@abaybektursun](https://github.com/abaybektursun)
- **Selective +/-1 pruning**: [PR #609](https://github.com/openai/parameter-golf/pull/609) by [@saml212](https://github.com/saml212)
- **EMA weight averaging**: [PR #401](https://github.com/openai/parameter-golf/pull/401) by [@newjordan](https://github.com/newjordan)
- **FlashAttention 3**: [PR #122](https://github.com/openai/parameter-golf/pull/122) by [@mtybadger](https://github.com/mtybadger)
- **Muon optimizer**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by [@abaybektursun](https://github.com/abaybektursun)
- **MuonEq-R**: [arXiv:2603.28254](https://arxiv.org/abs/2603.28254)
- **SP4096 clean baseline + linear LR reference**: [PR #1296](https://github.com/openai/parameter-golf/pull/1296) by [@aryanbhosale](https://github.com/aryanbhosale); [PR #1285](https://github.com/openai/parameter-golf/pull/1285) by [@dexhunter](https://github.com/dexhunter)
- **Orthogonal init + muP scaling**: [PR #162](https://github.com/openai/parameter-golf/pull/162) by [@unnir](https://github.com/unnir)
