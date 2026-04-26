# SP8192 Multi-Phase Global SGD + Phased TTT

**Score: 1.07219 bpb** (3-seed mean, seeds 42 / 0 / 1234)

| Seed | val_bpb | artifact |
|------|---------|----------|
| 42   | 1.07332 | 15,930,192 B |
| 0    | 1.07115 | 15,939,461 B |
| 1234 | 1.07211 | 15,930,004 B |
| **mean** | **1.07219** | |

## Approach

Multi-phase global SGD at test-time: the validation set is split into phases. Within each phase, chunks are first fully scored under `torch.no_grad()` (score-first), then base model weights are updated with SGD on the scored tokens. This cycles across phases, letting the model progressively adapt its base weights to the validation distribution while remaining legal under Issue #1017.

Combined with:
- **SP-8192 tokenizer** (8192-vocab SentencePiece BPE)
- **Phased TTT LoRA** within each chunk
- **Int7 embedding quantization** (SDClip σ=15.0)
- **Per-layer GPTQ** with sigma clipping (MLP σ=12.0, Attn σ=13.0)
- **Muon optimizer** (momentum=0.97, matrix_lr=0.026)
- **Depth recurrence** (layers 3–5 looped, warmup at step 35%)
- **VarLen flash attention** (flash_attn_3)
- **Fused triton MLP**
- **Brotli compression** of weights + code

## Hyperparameters

```bash
PHASED_TTT_ENABLED=1
PHASED_TTT_PREFIX_DOCS=2000
PHASED_TTT_NUM_PHASES=3
MLP_CLIP_SIGMAS=12.0
ATTN_CLIP_SIGMAS=13.0
EMBED_BITS=7
EMBED_CLIP_SIGMAS=15.0
MATRIX_LR=0.026
GPTQ_RESERVE_SECONDS=4
GPTQ_CALIBRATION_BATCHES=16
```

## Reproduction

```bash
# Requires 8x H100 SXM, torch 2.9.1+cu128, flash_attn_3
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

for seed in 42 0 1234; do
  SEED=$seed \
  NCCL_NET=Socket \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${seed}.log
done
```
