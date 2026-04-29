# SP8192 #1855 Base + Asymmetric Logit Rescale

**Score: 1.06577 bpb** (3-seed mean, seeds 42 / 0 / 1234)

| Seed | val_bpb | artifact |
|------|---------|----------|
| 42   | 1.06533 | 15,902,200 B |
| 0    | 1.06643 | 15,899,693 B |
| 1234 | 1.06554 | 15,907,523 B |
| **mean** | **1.06577** | **15,903,139 B avg** |

## Approach

Verbatim PR #1855 stack with one orthogonal addition: **asymmetric logit rescale** on the eval path.

Two learnable scalars `softcap_pos` and `softcap_neg` replace the single `logit_softcap` in `forward_logits` and `forward_ttt`. Logits split by sign:

```python
torch.where(logits > 0,
            softcap_pos * tanh(logits / softcap_pos),
            softcap_neg * tanh(logits / softcap_neg))
```

Both scalars are initialized to `logit_softcap` (30.0) so the eval path is identity at the start. They are trained inside Phased TTT (passed through the global SGD loop as part of `model.parameters()`) and end up at slightly different values for the positive and negative tails of the next-token distribution.

The fused softcapped-CE Triton kernel on the training path is left unchanged (single softcap), so train-time numerics match #1855 exactly. The two scalars receive no gradient during training and stay at init until eval.

Artifact cost: 8 bytes (2 fp16 scalars in the passthrough float16 list). Compressed cost is below the lrzip block overhead, effectively free.

## Stack inherited from #1855

- **SP-8192 tokenizer** with CaseOps lossless casing
- **LQER asymmetric int4** (rank 4, group 64, top-3 layers) on tok_emb + selected MLP weights
- **Sparse Attn Gate** on attention out
- **BOS-Fixed SmearGate** in `_forward_hidden` and `forward_ttt` (cross-doc leak fix)
- **9-hparam greedy stack**: `MLP_CLIP_SIGMAS=11.5`, `EMBED_CLIP_SIGMAS=14.0`, `WARMDOWN_FRAC=0.85`, `BETA2=0.99`, `TTT_BETA2=0.99`, `TTT_WEIGHT_DECAY=0.5`, `TTT_LORA_RANK=80`, `SPARSE_ATTN_GATE_SCALE=0.5`, `PHASED_TTT_PREFIX_DOCS=2500`
- **Phased Multi-Phase Global SGD TTT** (3 phases, prefix=2500 docs)
- **Per-group lrzip ZPAQ** compression
- **8× H100 SXM**, FA3, fused softcapped CE, torch 2.9.1+cu128

## Hyperparameters

```bash
ASYM_LOGIT_RESCALE=1
HASH_EMBED_ENABLED=0      # gated; 16384x512 entry exceeds 16MB cap

# Below = #1855 verbatim
CASEOPS_ENABLED=1
PHASED_TTT_ENABLED=1
PHASED_TTT_PREFIX_DOCS=2500
PHASED_TTT_NUM_PHASES=3
EMBED_BITS=7
MATRIX_LR=0.026
MIN_LR=0.1
MLP_CLIP_SIGMAS=11.5
ATTN_CLIP_SIGMAS=13.0
EMBED_CLIP_SIGMAS=14.0
GRAD_CLIP_NORM=0.3
TTT_CHUNK_SIZE=48
WARMUP_STEPS=20
MUON_BACKEND_STEPS=5
GLOBAL_TTT_MOMENTUM=0.9
WARMDOWN_FRAC=0.85
BETA2=0.99
TTT_BETA2=0.99
TTT_WEIGHT_DECAY=0.5
TTT_LORA_RANK=80
SPARSE_ATTN_GATE_SCALE=0.5
GPTQ_RESERVE_SECONDS=0.5
GPTQ_CALIBRATION_BATCHES=16
GATED_ATTN_QUANT_GATE=1
SPARSE_ATTN_GATE_ENABLED=1
GATE_WINDOW=12
SMEAR_GATE_ENABLED=1
LQER_ENABLED=1
LQER_ASYM_ENABLED=1
LQER_RANK=4
LQER_FACTOR_BITS=4
LQER_ASYM_GROUP=64
LQER_TOP_K=3
FUSED_CE_ENABLED=1
COMPRESSOR=pergroup
NCCL_NET=Socket
```

## Reproduction

```bash
# 8x H100 SXM, torch 2.9.1+cu128, flash_attn_3, lrzip
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
apt-get install -y lrzip

for seed in 42 0 1234; do
  SEED=$seed \
  ASYM_LOGIT_RESCALE=1 \
  CASEOPS_ENABLED=1 PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
  EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
  GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
  GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
  TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 SPARSE_ATTN_GATE_SCALE=0.5 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
  SMEAR_GATE_ENABLED=1 \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${seed}.log
done
```

## Attribution

Builds directly on:

- **codemath3000** — PR #1855 (full stack baseline, 1.06108)
- **dexhunter** — PR #1797 (LQER asym + SmearGate base) and PR #1736 (CaseOps + GatedAttn + QuantGate)
- **nprime06** — PR #1787 (Polar Express NS + MIN_LR + Sparse Attn Gate + Fused CE)
- **MarioPaerle** — PR #1667 (SmearGate origin)
- **renqianluo** — PR #1767 (LoRA TTT improvements)
- **classiclarryd** — modded-nanogpt PR #181 (Asymmetric Logit Rescale, the addition over #1855)
- **Jorge Asenjo** — PR #1700 (Phased Multi-Phase Global SGD TTT, retained from #1855 stack)
