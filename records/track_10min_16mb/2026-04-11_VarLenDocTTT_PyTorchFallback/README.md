# Record: SP8192 + VarLen Attention + Doc-Independent LoRA TTT + Banking + Muon 0.97 — val_bpb 1.07747 (3-seed mean)

**val_bpb: 1.07747** (3-seed mean, std 0.00064) | **2.78321 nats** | **~15.99 MB** | 8xH100 SXM, 600s | Doc-Independent LoRA TTT

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, Doc-Independent LoRA TTT)

### Core Results

| Seed | Steps | ms/step | Pre-TTT BPB | **Post-TTT BPB** | TTT gain | TTT time | Artifact |
|------|-------|---------|-------------|------------------|----------|----------|----------|
| 42   | 4826  | 121.9   | 1.08732     | **1.07687**      | -0.01045 | 252.2s   | 15,992,482 |
| 0    | 4820  | 122.0   | 1.08756     | **1.07719**      | -0.01037 | 251.0s   | 15,995,179 |
| 1337 | 4817  | 122.1   | 1.08885     | **1.07835**      | -0.01050 | 249.6s   | 15,995,796 |
| **Mean** | **4821** | **122.0** | **1.08791** | **1.07747** | **-0.01044** | **250.9s** | **15,994,486** |
| **Std** | | | | **0.00064** | | | |

### Supplemental Diagnostics

| Seed | Post-EMA BPB | Quantized BPB | TTT BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|-------------|---------------|---------|-----------------|-----------|------------------|------------|-----------|
| 42   | 1.07591     | 1.08732       | 1.07687 | 2.78166         | 25,659    | 15,992,482       | 588.1s     | 559.5s    |
| 0    | 1.07671     | 1.08756       | 1.07719 | 2.78249         | 25,659    | 15,995,179       | 588.1s     | 431.9s    |
| 1337 | 1.07734     | 1.08885       | 1.07835 | 2.78549         | 25,659    | 15,995,796       | 588.1s     | 427.1s    |

Merged SOTA (PR #1493): **1.0810 BPB**. Delta: **-0.0035 BPB / -0.0091 nats**. Clears the 0.005-nat threshold.

## Key Innovation

### 1. VarLen Flash Attention (Within-Document Only)

Uses `flash_attn_varlen_func` to compute attention within document boundaries only, eliminating cross-document attention leakage during training. Documents are packed with `cu_seqlens` tracking document boundaries, so attention is strictly masked to the current document.

### 2. Doc-Independent LoRA TTT

At eval time, each document gets its own independent LoRA adaptation:
- **Rank 96 LoRA** on K, O, and MLP projections
- Each document scored independently — LoRA weights reset between documents
- Score-before-update: tokens are scored, then LoRA is updated from the loss
- Adam optimizer with `lr=0.0001`, `beta2=0.999`, `weight_decay=0.5`
- Documents sorted by length (longest first) for efficient batching

### 3. Parameter Banking (Depth Recurrence)

Layers 3-5 are reused with `num_loops=2`, creating an encoder-decoder pattern:
- Encoder path: `[0, 1, 2, 3, 4, 5, 3, 4]`
- Decoder path: `[5, 3, 4, 5, 6, 7, 8, 9, 10]`
- Banking activated at training fraction 0.35 with gradual warmup

### 4. PyTorch MLP Fallback (No Triton/CUTLASS)

Replaces the Triton fused MLP kernel from PR #1530 with a pure PyTorch implementation using `F.silu` gating. This eliminates the Triton/CUTLASS dependency while maintaining competitive throughput (~122 ms/step).

## Changes from PR #1530 Baseline

| Aspect | PR #1530 (@samacqua) | This submission |
|--------|---------------------|-----------------|
| MLP kernel | Triton fused TMA | PyTorch `F.silu` fallback |
| Muon momentum | 0.95 (default) | 0.97 (from PR #1514) |
| Triton dependency | Required | None |

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, SiLU gating (PyTorch), Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activated at frac=0.35). Parallel residuals from layer 7. Skip gates (sigmoid-gated U-Net connections). VarLen Flash Attention with document boundary tracking.

## Training

MuonEq-R optimizer (row-normalized Muon, momentum=0.97, Newton-Schulz 5 steps), AdamW for embeddings/scalars. ~4821 steps in 588s on 8xH100 SXM. Linear warmdown to LR=0 over final 66.7% of training. EMA decay 0.997.

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k * std(row)` for principled rate-distortion. int6 for attention/MLP matrices (k=12.85), int8 for token embeddings (k=20.0). Brotli-11 compression. 64 calibration batches.

## TTT (Doc-Independent LoRA)

Doc-independent LoRA adaptation at eval time:
- Sort val documents by length (longest first), batch by size
- For each document: apply LoRA (rank 96) to K, O, MLP projections
- Single gradient step per chunk (chunk_size=64 tokens)
- Adam optimizer: lr=0.0001, beta2=0.999, weight_decay=0.5
- LoRA weights reset between documents — no information leaks across docs
- Total TTT eval time: ~251s (within 600s eval budget)

## Rule Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** VarLen attention is strictly causal within documents. No cross-document attention.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Doc-independent LoRA scores each chunk before updating. No same-token adaptation.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:
- No SLOT (standard or causal)
- No pre-quant TTT on val data (model quantized once during training, LoRA adapts at eval time)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes on all 3 seeds
- Training under 600s on all 3 seeds (~588s actual)
- Eval (TTT LoRA) under 600s on all 3 seeds (max 559.5s with compile warmup)

## Requirements

```
torch>=2.9.0
flash-attn-3 (flash_attn_interface with varlen support)
sentencepiece
brotli
numpy
```

No Triton or CUTLASS required.

## Run Command (3-seed loop)

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

for SEED in 42 0 1337; do
  SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Lineage

PR #1530 (@samacqua, varlen + doc TTT) + PR #1523 (@EthanYangTW, banking/recurrence) + PR #1514 (@dexhunter, Muon 0.97) + PR #1394 (@clarkkev, SP8192 GPTQ SDClip baseline)

## Credits

- **@samacqua** — VarLen Flash Attention + Doc-Independent LoRA TTT framework (PR #1530)
- **@EthanYangTW** — Parameter banking / depth recurrence pattern (PR #1523)
- **@dexhunter** — Muon momentum 0.97 (PR #1514), depth recurrence (PR #1331, #1437)
- **@clarkkev** — SP8192 + GPTQ SDClip + MuonEq-R baseline (PR #1394)
- **@abaybektursun** — Original TTT framework (PR #549, merged precedent)
- **@Robby955** — Parallel residuals concept (PR #1412)
- **@msisovic** — Parallel residuals (PR #1204)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed0.log`
- `train_seed1337.log`
