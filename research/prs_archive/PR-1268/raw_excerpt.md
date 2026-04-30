# PR 1268 — Mamba3-GPTQ-Long-Context: Mamba3 Hybrid with PTQ

**Author:** samquiring (per batch metadata)
**Claimed BPB:** Pre-GPTQ 1.1755, Post-GPTQ 1.1875
**Artifact size:** 15.51 MB
**Seeds:** 1337 (submission shows seeds 314, 42, 7 in filenames)
**Hardware:** 8xH100, ~108ms/step, 5557 steps

## Files retrieved
- `records__track_non_record_16mb__2026-04-02_Mamba3_Long_Context_GPTQ__README.md`
- `records__track_non_record_16mb__2026-04-02_Mamba3_Long_Context_GPTQ__submission.json`
- `records__track_non_record_16mb__2026-04-02_Mamba3_Long_Context_GPTQ__train_gpt.py`

## Run command (from README)
```
NCCL_TIMEOUT=1800 USE_GPTQ=1 USE_LZMA=1 USE_MAMBA3=1 \
NUM_LAYERS=10 ATTN_LAYERS=4,9 MODEL_DIM=512 MLP_MULT=2 TRAIN_SEQ_LEN=16384 \
TRAIN_BATCH_TOKENS=524288 VAL_BATCH_SIZE=131072 \
MATRIX_LR=0.02 SCALAR_LR=0.02 WARMDOWN_ITERS=4000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 SEED=1337 VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Claimed changes (from README, verbatim)

Mamba3-Hybrid 10-Layer Hybrid (8 Mamba3 + 2 Attention):
- 10 layers: 8 Mamba3 SISO + 2 GQA attention (layers 4, 9)
- model_dim=512, mlp_mult=2, vocab_size=1024 (sp1024 BPE)
- U-Net skip connections across all layer types
- train_seq_len=16384

Compression pipeline:
1. Train bf16 10 min on 8xH100 (~108ms/step, ~5557 steps)
2. Apply EMA (decay=0.997)
3. AR self-generation: 64 x 2048 calibration tokens (temp=0.8, seeded)
4. Full Hessian GPTQ: int6, block-128, column reordering, 5-percentile clip search
5. Int8 serialization + LZMA preset=9
6. Sliding window roundtrip validation

Key findings:
- Pure Mamba doesn't work — hybrid does (2 attention at U-Net boundaries).
- Hardware problem: FA3 gives attention SOTA ~86ms/step vs Mamba3-Hybrid's ~108ms/step. SOTA gets ~7000 steps, Mamba3-Hybrid gets ~5557.
- LR=0.02 optimal (0.03 noticeably worse, 0.04 much worse).
- BigramHash + SmearGate + OrthoInit hurt (made worse: 1.2166 vs 1.2047).
- Mamba3 > Mamba2: ~0.01 BPB improvement AND smaller compressed size.
- EMA helps post-GPTQ more than pre-quant (-0.0024 vs -0.0009).
- Mamba3-Hybrid wins early convergence (steps 1000-2000) but SOTA pulls ahead later.
