# Record: Per-Layer Adaptive GPTQ Clip + int7 Embeddings + MATRIX_LR=0.026

**val_bpb: 1.07493** (3-seed mean, std 0.00078) | **2.77666 nats** | **~15.93 MB** | 8xH100 SXM, 600s | TTT (doc-independent LoRA)

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, doc-TTT LoRA)

| Seed | Steps | ms/step | Pre-Quant BPB | Post-Quant BPB | Post-TTT BPB | TTT gain | Eval time | Artifact |
|------|-------|---------|---------------|----------------|--------------|----------|-----------|----------|
| 42   | 4888  | 120.1   | 1.07298       | 1.08495        | **1.07437**  | -0.01058 | 220s      | 15,934,100 |
| 0    | 4905  | 119.7   | 1.07184       | 1.08567        | **1.07460**  | -0.01107 | 218s      | 15,937,217 |
| 1337 | 4904  | 119.7   | 1.07310       | 1.08663        | **1.07582**  | -0.01081 | 213s      | 15,928,721 |
| **Mean** | **4899** | **119.8** | **1.07264** | **1.08575** | **1.07493** | **-1.08e-2** | **217s** | **15,933,346** |
| **Std** | | | | | **0.00078** | | | |

### Supplemental Diagnostics

| Seed | Pre-Quant BPB | Post-Quant BPB | Post-TTT BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|---------------|----------------|--------------|-----------------|-----------|------------------|------------|-----------|
| 42   | 1.07298       | 1.08495        | 1.07437      | 2.77521         | 26,845    | 15,934,100       | 587s       | 220s      |
| 0    | 1.07184       | 1.08567        | 1.07460      | 2.77579         | 26,845    | 15,937,217       | 587s       | 218s      |
| 1337 | 1.07310       | 1.08663        | 1.07582      | 2.77897         | 26,845    | 15,928,721       | 587s       | 213s      |

**Merged SOTA**: PR #1493 (@bigbag) at val_bpb=1.0810 (2.78932 nats). **Delta: -0.01266 nats** (clears 0.005-nat bar by 2.5x).

## Key Innovation: Per-Layer Adaptive GPTQ Clip

Standard GPTQ uses a single `clip_sigmas` for all weight matrices, forcing a one-size-fits-all trade-off between quantization quality and artifact size. We observe that **MLP and attention layers have fundamentally different weight distributions** and respond differently to clipping:

- **MLP layers** benefit from tighter clipping (`MLP_CLIP_SIGMAS=12.0`) -- preserving more precision for the information-dense feedforward weights
- **Attention layers** tolerate looser clipping (`ATTN_CLIP_SIGMAS=13.0`) -- saving bytes while maintaining attention pattern quality
- **Embeddings** use int7 with tight clip (`EMBED_BITS=7, EMBED_CLIP_SIGMAS=15.0`) -- saves ~530 KB vs int8 with minimal quality loss

```python
# Per-layer adaptive clip in GPTQ quantization
if "tok_emb" in name:
    cs = h.embed_clip_sigmas   # 15.0 -- tight for embeddings
elif "mlp" in name:
    cs = h.mlp_clip_sigmas     # 12.0 -- tight for quality
elif "attn" in name:
    cs = h.attn_clip_sigmas    # 13.0 -- looser for byte savings
else:
    cs = h.matrix_clip_sigmas  # 12.85 -- fallback

bits = h.embed_bits if "tok_emb" in name else h.matrix_bits  # int7 embed, int6 matrices
```

This adaptive scheme yields **better BPB than uniform clip=12.85** while using fewer bytes for attention weights.

## Changes from Baseline (PR #1530 v2)

| Parameter | PR #1530 v2 default | This submission |
|-----------|---------------------|-----------------|
| MLP_CLIP_SIGMAS | N/A (uniform) | **12.0** |
| ATTN_CLIP_SIGMAS | N/A (uniform) | **13.0** |
| EMBED_BITS | 8 | **7** |
| EMBED_CLIP_SIGMAS | 20.0 | **15.0** |
| MATRIX_LR | 0.025 | **0.026** |
| WARMDOWN_FRAC | 0.72 | **0.75** |
| TTT_CHUNK_SIZE | 32 | **48** |

## Architecture

11L x 512d x 8H / 4KV, MLP 4x (Triton fused), LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0.

- **Triple recurrence**: layers 3-5 looped 2x (NUM_LOOPS=2), activated at 35% of training
- **Parallel residuals**: GPT-J style from layer 8+
- **VarLen attention**: Flash Attention 3 variable-length for document boundaries
- **Skip gates**: sigmoid-gated U-Net connections

## Training

MuonEq-R optimizer (row-normalized, Newton-Schulz 5 steps), AdamW for embeddings/scalars. ~4900 steps in 587s on 8xH100 SXM. MATRIX_LR=0.026 (6-point sweep optimum), linear warmdown over final 75% of training. EMA decay 0.9965.

## Quantization

Full-Hessian GPTQ with **per-layer adaptive clip**:
- int6 for MLP matrices (clip_sigmas=12.0 -- tight)
- int6 for attention matrices (clip_sigmas=13.0 -- loose)
- int7 for token embeddings (clip_sigmas=15.0, saves ~530 KB vs int8)
- Brotli-11 compression

## TTT (Test-Time Training)

Doc-independent LoRA adaptation at eval time:
- LoRA rank 96 on K, MLP, and O projections
- Per-document: score chunk under `torch.no_grad()`, then adapt LoRA weights
- Adam optimizer (lr=0.0001, chunk_size=48)
- Weight decay 0.5 for regularization
- ~217s eval time (well within 600s budget)

## Rule Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** All attention is strictly causal. VarLen attention respects document boundaries. Each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab (8192 tokens). No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each document fully scored under `torch.no_grad()` BEFORE any LoRA update. Training only on already-scored tokens. Doc-independent: each document gets fresh LoRA weights.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:
- No SLOT (standard or causal)
- No pre-quant TTT on val data
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes on all 3 seeds (max: 15,937,217)
- Training under 600s on all 3 seeds (587s actual)
- Eval (TTT) under 600s on all 3 seeds (max: 220s)

## Requirements

```
torch>=2.9.0
flash-attn-3
sentencepiece
brotli
triton
numpy
```

## Run Command (3-seed loop)

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

for SEED in 42 0 1337; do
  SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${SEED}.log
done
```

## Lineage

PR #1530 v2 (@samacqua) -> PR #1523 (@EthanYangTW) -> PR #1493 (@bigbag) -> PR #1394 (@clarkkev) -> PR #549 (@abaybektursun)

## Credits

- **@samacqua** -- VarLen attention, Triton fused MLP, doc-independent LoRA TTT, triple recurrence (PR #1530)
- **@EthanYangTW** -- Parameter banking, fused MLP TMA, triple recurrence (PR #1523)
- **@bigbag** -- Current merged SOTA (PR #1493)
- **@clarkkev** -- SP8192, GPTQ embeddings, SDClip, MuonEq-R, depth recurrence (PR #1394)
- **@abaybektursun** -- Score-first TTT framework (PR #549)

## Included Files

- `train_gpt.py` -- Self-contained training + GPTQ + TTT eval script
- `submission.json` -- Machine-readable results
- `train_seed42.log`, `train_seed0.log`, `train_seed1337.log` -- Full training + eval logs
- `README.md` -- This file
