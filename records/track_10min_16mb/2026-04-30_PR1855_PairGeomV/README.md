# Pair-Geometric Value Projection on PR #1855
**val_bpb: 1.07018705** (2 completed seeds, std 0.00017627) | **~15.31 MB** | 8xH100, 600s wallclock | legal phased TTT eval

This submission starts from the accepted PR #1855 stack and replaces the dense
attention value projection with a structured pair-geometric value projection.
The goal is not a new SOTA claim; it is a controlled architectural alteration
showing that the dense trained/stored value projection can be replaced by a
pair-based geometric carrier while remaining close to the accepted baseline.

For comparison, our reproduced accepted PR #1855 standard control scored
`1.06021565` BPB at seed 42 (`db_id=4311`). PairGeom-V trails that reproduced
control by about `0.00997` BPB on the two completed seeds, while producing a
smaller artifact and removing the dense `W_v` value matrix.

## Results
| Seed | Steps | Pre-quant val_bpb | Quantized val_bpb | **Post-TTT val_bpb** | Artifact bytes | Notes |
|------|-------|-------------------|-------------------|----------------------|----------------|-------|
| 42 | 4,981 | 1.07441666 | 1.08410724 | **1.07006241** | 15,304,981 | full raw log |
| 43 | 4,996 | 1.07519691 | 1.08481881 | **1.07031169** | 15,312,945 | recovered summary log |
| **Mean** | | | | **1.07018705** | | std 0.00017627 |

Seed 43 completed, but we ran out of time before recovering the full remote
stdout log, so this submission includes a recovered summary log instead.

## PairGeom-V Change
In the baseline PR #1855 attention block, Q, K, and V are dense learned
projections. This submission keeps Q and K dense, but replaces the dense value
projection:
```text
standard: v = W_v x
```

with a pair-geometric value path over the normalized hidden state:
```text
base = rms_norm(x)
a = base[:kv_dim]
b = base[kv_dim:2*kv_dim]
d = a - b
s = a + b
v = a*w0 + b*w1 + d*wd + s*ws
```

The validated setting uses `PAIRGEOM_V_COLLAPSE=1`, which algebraically reduces
the signed rule to learned per-dimension coefficients on the two hidden halves:
```text
v = a*(w0 + wd + ws) + b*(w1 - wd + ws)
```

An absolute-distance variant is available via `PAIRGEOM_V_ABSDIFF=1`, but it
was not used for the submitted runs.

## Architecture
| Component | Setting | Source |
|-----------|---------|--------|
| Base stack | Accepted PR #1855, BOS-fixed PR #1797-derived CaseOps/LQER/SparseAttnGate/SmearGate stack | [#1855](https://github.com/openai/parameter-golf/pull/1855) |
| Layers | 11 layers, 512d, 8 heads, 4 KV heads | PR #1855 |
| Attention scores | Standard FA3 attention, GQA 2:1 | PR #1855 |
| Value projection | **PairGeom-V structured pair-geometric value projection** | this submission |
| MLP | LeakyReLU(0.5)^2 fused MLP | [#493](https://github.com/openai/parameter-golf/pull/493), [#1530](https://github.com/openai/parameter-golf/pull/1530) |
| XSA | All 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) |
| RoPE/LN scale | Partial RoPE plus layerwise LN scale | [#315](https://github.com/openai/parameter-golf/pull/315) |
| Recurrence/parallel residuals | PR #1855 settings inherited | [#1344](https://github.com/openai/parameter-golf/pull/1344), [#1530](https://github.com/openai/parameter-golf/pull/1530) |
| Sparse attention gate / SmearGate | Inherited from PR #1855 | [#1667](https://github.com/openai/parameter-golf/pull/1667), [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| Quantization/compression | GPTQ/LQER/per-group compression inherited from PR #1855 | [#1797](https://github.com/openai/parameter-golf/pull/1797), [#1855](https://github.com/openai/parameter-golf/pull/1855) |
| Tokenizer | sp8192 lossless caps CaseOps v1 reserved | [#1729](https://github.com/openai/parameter-golf/pull/1729) |
| TTT | Legal phased TTT inherited from PR #1855 | [#1610](https://github.com/openai/parameter-golf/pull/1610), [#1626](https://github.com/openai/parameter-golf/pull/1626), [#1736](https://github.com/openai/parameter-golf/pull/1736) |

## Claim Boundary
PairGeom-V is a substantive replacement of the attention value projection path.
It does **not** replace dense Q/K projections, attention score dot products,
attention output projection, or MLP matrix products. The legal TTT path uses the
same PairGeom-V base value path and then applies the inherited low-rank LoRA TTT
adapters.

## Requirements
See `requirements.txt`. FlashAttention 3 must be installed separately:
```bash
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

Validated environment:
```text
PyTorch 2.9.1+cu128
CUDA 12.8
8xH100 80GB
lrzip system binary installed
```

## Reproducing
The defaults in `train_gpt.py` reproduce the submitted PairGeom-V configuration
when run from this record directory with the CaseOps data prepared. The core
settings are:

```bash
DATA_DIR=./data \
VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
PAIRGEOM_V_ENABLED=1 PAIRGEOM_V_RMS=1 PAIRGEOM_V_COLLAPSE=1 PAIRGEOM_V_ABSDIFF=0 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files
- `train_gpt.py` - PR #1855 plus PairGeom-V.
- `train_seed42_jarvis_pairgeomv.log` - full seed-42 run log.
- `train_seed43_jarvis_pairgeomv_summary.log` - recovered seed-43 summary log.
- `submission.json` - structured metadata.
- `requirements.txt` - Python dependencies, with FA3/lrzip notes.
- `lossless_caps.py` - CaseOps infrastructure.
- `prepare_caseops_data.py` - CaseOps shard prep and byte sidecar generation.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - SentencePiece model.

## Credits
This submission is intentionally built on the accepted PR #1855 stack and keeps
its recipe aligned except for the PairGeom-V value projection replacement.
Thanks to the Parameter Golf community contributors whose work this stack builds
on, especially:

- [PR #1855](https://github.com/openai/parameter-golf/pull/1855) by @codemath3000 - accepted BOS-fixed SmearGate + LQER + SparseAttnGate + 9-hparam stack.
- [PR #1797](https://github.com/openai/parameter-golf/pull/1797) by @dexhunter - SmearGate + LQER asymmetric rank-4 stack.
- [PR #1787](https://github.com/openai/parameter-golf/pull/1787) by @nprime06 - Polar Express NS, MIN_LR, sparse attention gate, fused softcapped CE.
- [PR #1736](https://github.com/openai/parameter-golf/pull/1736) - CaseOps + GatedAttn + QuantGate + Loop4-5 + phased TTT integration.
- [PR #1729](https://github.com/openai/parameter-golf/pull/1729) by @romeerp - lossless CaseOps tokenizer.
- [PR #1667](https://github.com/openai/parameter-golf/pull/1667) by @MarioPaerle - SmearGate and attention output gate lineage.
- [PR #1626](https://github.com/openai/parameter-golf/pull/1626) by @dexhunter and [PR #1610](https://github.com/openai/parameter-golf/pull/1610) - phased/global TTT lineage.
- [PR #1530](https://github.com/openai/parameter-golf/pull/1530) by @samacqua - variable-length attention, fused MLP, parallel residuals, doc-based LoRA TTT.
- [PR #1344](https://github.com/openai/parameter-golf/pull/1344) - Polar-Express Newton-Schulz coefficients and depth recurrence lineage.
- [PR #493](https://github.com/openai/parameter-golf/pull/493), [PR #478](https://github.com/openai/parameter-golf/pull/478), [PR #315](https://github.com/openai/parameter-golf/pull/315), and [PR #289](https://github.com/openai/parameter-golf/pull/289) for inherited activation, XSA, RoPE/LN, and U-Net skip components.
