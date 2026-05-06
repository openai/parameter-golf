# Record: VarLen Attention + Triton Fused MLP + Doc-TTT + Warmdown 0.75 + Chunk 48

**val_bpb = 1.07406** (3-seed mean, std 0.00132) | **2.77441 nats** | **~15.99 MB** | 8xH100 SXM, 600s

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

### Core Results

| Seed | Steps | ms/step | Pre-TTT BPB | Post-TTT BPB | TTT Gain | TTT Time | Artifact |
|------|-------|---------|-------------|--------------|----------|----------|----------|
| 42   | 4918  | 119.4   | 1.08400     | **1.07352**  | -0.01048 | 213s     | 15,994,146 |
| 0    | 4900  | 119.8   | 1.08363     | **1.07310**  | -0.01053 | 221s     | 15,997,570 |
| 1337 | 4908  | 119.6   | 1.08619     | **1.07556**  | -0.01063 | 219s     | 15,988,610 |
| **Mean** | **4909** | **119.6** | **1.08461** | **1.07406** | **-0.01055** | **218s** | **15,993,442** |
| **Std**  |      |         |             | **0.00132** |          |          |            |

### Supplemental Diagnostics

| Seed | Post-EMA BPB | Quantized BPB | Post-TTT BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|--------------|---------------|--------------|-----------------|-----------|------------------|------------|-----------|
| 42   | 1.07134      | 1.08400       | 1.07352      | 2.77301         | 2843 lines | 15,994,146      | 587.1s     | 213s      |
| 0    | 1.07160      | 1.08363       | 1.07310      | 2.77193         | 2843 lines | 15,997,570      | 587.1s     | 221s      |
| 1337 | 1.07339      | 1.08619       | 1.07556      | 2.77829         | 2843 lines | 15,988,610      | 587.1s     | 219s      |

Merged SOTA (PR #1493 @bigbag): **1.0810 BPB** (2.78932 nats). Delta: **-0.01491 nats** (clears 0.005 bar by **3.0x**).

## Key Innovation

**Warmdown fraction and TTT chunk size tuning** on top of PR #1530's VarLen + Triton fused MLP + doc-TTT stack:

- **warmdown_frac = 0.75** (up from 0.72 default) -- extends the cosine decay phase by 3%, allowing the model to settle into a slightly lower-loss basin before quantization. This alone gives ~0.001 BPB improvement.
- **TTT_CHUNK_SIZE = 48** (up from 32 default) -- larger document chunks provide more context per TTT gradient step, improving LoRA adaptation quality at a small compute cost. Combined with warmdown tuning, yields ~0.002 BPB total gain.
- **Muon momentum 0.97** -- shorter memory horizon (~33 effective steps) tracks the rapidly changing loss surface better during the extended warmdown phase.

### Changes from PR #1530 v2 baseline

| Parameter | PR #1530 v2 | This submission |
|-----------|-------------|-----------------|
| warmdown_frac | 0.72 | **0.75** |
| TTT_CHUNK_SIZE | 32 | **48** |
| MUON_MOMENTUM | 0.95 | **0.97** |

## Architecture

11L x 512d x 8H / 4KV, MLP 4x with Triton fused kernel (LeakyReLU(0.5)^2), Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. VarLen attention via Flash Attention 3 `flash_attn_varlen_func` for document-aware batching. Triple depth recurrence: layers 3-5 looped 3x (17 virtual layers from 11 physical, activates at frac=0.35). Parameter banking with batched Newton-Schulz orthogonalization. Parallel residuals from layer 8 with mean lane fusion. Skip gates (sigmoid-gated U-Net connections).

**Optimizer**: Muon (momentum=0.97, 5-step Newton-Schulz, row-normalized) for matrix params + Adam (beta1=0.9, beta2=0.95) for scalars/embeddings. Split LR: matrix=0.022, embed=0.6, head=0.008, scalar=0.02. EMA decay=0.9965. Gradient clipping at 0.3.

**Quantization**: Full Hessian GPTQ with int6 matrices (clip_sigmas=12.85), int8 embeddings (clip_sigmas=20.0), Brotli-11 compression.

**TTT**: Doc-independent LoRA (rank=96) on K, MLP, and O projections. Adam optimizer (lr=0.0001, beta2=0.999), weight decay=0.5, chunk_size=48. Score-first: each chunk scored under `torch.no_grad()` before gradient update.

## Rule Compliance

Per Issue #1017:
- **Condition 1 (Causality):** VarLen attention with per-document `cu_seqlens` ensures strict causal masking within documents. No cross-document information leakage.
- **Condition 2 (Normalized):** Standard softmax over full vocabulary. No n-gram bias, no logit manipulation.
- **Condition 3 (Score before update):** Each TTT chunk scored under `torch.no_grad()` BEFORE any LoRA gradient update. Score-first ordering verified.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass evaluation.

No SLOT, no pre-quant TTT on val data, no ETLB, no n-gram cache, no hashed n-gram. All artifacts < 16 MB, train < 600s, eval < 600s. Compile warmup uses random tokens (not val data).

## Requirements

- Python 3.10+
- PyTorch 2.9.1+cu128
- flash-attn-interface (Flash Attention 3)
- sentencepiece
- triton
- brotli
- numpy

## Run Command

```bash
# 3-seed verification loop (defaults baked into train_gpt.py)
for SEED in 42 0 1337; do
  SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee train_seed${SEED}.log
done
```

## Lineage

PR #1530 v2 (@samacqua) -> warmdown/chunk/momentum tuning (this work)

Built on:
- PR #1530 (@samacqua) -- VarLen attention, Triton fused MLP, doc-independent LoRA TTT, triple depth recurrence, parameter banking
- PR #1523 (@EthanYangTW) -- triple recurrence (NUM_LOOPS=2), parameter banking, fused MLP TMA
- PR #1514 (@dexhunter) -- Muon momentum 0.97
- PR #1493 (@bigbag) -- merged SOTA baseline
- PR #1394 (@clarkkev) -- SP8192 + GPTQ + SDClip + MuonEq-R foundation

## Credits

- **@samacqua** -- VarLen attention, Triton fused MLP, doc-independent LoRA TTT, triple depth recurrence, parameter banking (PR #1530)
- **@EthanYangTW** -- Triple recurrence, parameter banking, fused MLP TMA (PR #1523)
- **@dexhunter** -- Muon momentum 0.97 (PR #1514), warmdown/chunk/momentum tuning (this work)
- **@bigbag** -- Merged SOTA baseline (PR #1493)
- **@clarkkev** -- SP8192 + GPTQ + SDClip + MuonEq-R (PR #1394)
- **@abaybektursun** -- Score-first TTT framework (PR #549)

## Acknowledgements

Thanks to OpenAI's Advanced Competitor grant ($500 compute credit via RunPod).

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed0.log`
- `train_seed1337.log`
