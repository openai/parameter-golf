# Canonical reproduction support: SP8192 LQER + SparseGate + BOS-fixed SmearGate stack

**Single-seed reproduced val_bpb: 1.05985469** | **val_loss: 2.31935492** | **15,898,155 byte submission** | 8xH100 SXM | strict 600s train cap | TTT eval

This folder is a reproduction/support submission for the public stack in
`2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611`, not a claim of a
new architecture. The run uses that record's `train_gpt.py`, tokenizer, CaseOps
pipeline, compression path, and default hparam stack. The key difference from our
earlier failed attempts was using the canonical pretokenized CaseOps shards from
`romeerp/parameter-golf-caseops-v1` instead of locally re-tokenized raw docs.

The artifact was produced at `2026-05-01 00:34 +0200`, which is
`2026-04-30 22:34 UTC`.

## Result

| Seed | Steps | Train cap | Pre-quant val_bpb | Quantized val_bpb | Post-TTT val_bpb | Eval time | Artifact |
|------|-------|-----------|-------------------|-------------------|------------------|-----------|----------|
| 42 | 4,935 | 599.570s | 1.06383424 | 1.07239257 | **1.05985469** | 539.328s | 15,898,155 bytes |

The serialized quantized model was `15,866,055` bytes. The total submission size
with code wrapper was `15,898,155` bytes, under the 16,000,000 byte limit.

Artifact SHA256:

```text
47d7339fa803c52559a3acfbe4c682332c3871560fe55bea1cb112923d43c298
```

The full run log is included as `train_seed42_canonical_reproduction.log`.

## What changed versus the source record

- No ML technique change is claimed here.
- The source record's seed-42 line reports `1.05989454` BPB; this canonical
  rerun obtained `1.05985469` BPB.
- This run documents the exact dataset handling that mattered operationally:
  canonical `romeerp/parameter-golf-caseops-v1` shards with 39 train shards, one
  validation token shard, and one validation byte-sidecar shard.

## Architecture and training stack

This reproduction uses the source record's stack:

- 11-layer 512d transformer, 8 GQA heads, 4 KV heads.
- XSA on all layers, partial RoPE + YaRN, LN scale, U-Net skips, parallel decoder,
  and depth recurrence.
- LeakyReLU-square MLP, fused MLP Triton kernel, fused softcapped CE Triton kernel.
- Sparse attention head-output gate and BOS-fixed SmearGate.
- Polar-Express Newton-Schulz Muon.
- GPTQ int6 matrices, int7 embeddings, int8-per-row attention gate, and LQER
  asymmetric int4 rank-4 quantization-error correction.
- Per-group `lrzip` ZPAQ + brotli compression.
- Phased TTT eval with 3 cumulative phases, prefix 2500 docs, LoRA rank 80.

See the original source record README for the complete architecture table,
hparam stack, and lineage.

## Reproducing

Use the same command shape as the source record. The important dataset inputs are:

```bash
DATA_DIR=./data \
VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
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

System requirements match the source record: PyTorch 2.9.1+cu128, CUDA 12.8,
FlashAttention 3, 8xH100 80GB SXM, and the `lrzip` system binary.

## Credits

Primary credit belongs to Benjamin Hadad / `codemath3000` for the source record
`2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611`, plus the
community PR lineage credited there. This folder is submitted as independent
reproduction evidence and operational support for that stack.
