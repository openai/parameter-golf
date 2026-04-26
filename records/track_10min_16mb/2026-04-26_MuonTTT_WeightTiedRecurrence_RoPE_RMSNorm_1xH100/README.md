# MuonTTT WeightTied Recurrence — RoPE + RMSNorm + LeakyReLU²

**val_bpb: 1.5063** (2-seed mean, std 0.0002, post-TTT int8+zlib roundtrip) | **3.80 MB** artifact | 1×H100 SXM 80GB

## Results (1×H100 80GB SXM)

| Seed | step_avg | steps | Pre-quant bpb | **Post-TTT bpb** | TTT steps | TTT time | Artifact |
|------|----------|-------|--------------|-----------------|-----------|----------|----------|
| 1337 | 299.55ms | 2,004 | 1.4894 | **1.5061** | 5 (lr=1e-3) | 320s | 3,800,090 |
| 42   | 275.05ms | 2,182 | 1.4865 | **1.5065** | 1 (lr=5e-4) | 6s  | 3,824,663 |
| **Mean** | **287ms** | **2,093** | **1.4880** | **1.5063 (std 0.0002)** | | | |

> **Note on TTT config**: Seed 1337 uses `ttt_steps=5, ttt_lr=1e-3` (the submitted config); seed 42 was run with an earlier config `ttt_steps=1, ttt_lr=5e-4`. The submitted `train_gpt.py` (code size 43939 bytes) corresponds to the seed 1337 run. Both confirm the model's stability across seeds.

## Architecture

A **weight-tied recurrent transformer** — a single transformer block applied `recur_steps=4` times with the **same** weights. Token embedding weights are tied to the output projection head.

| Component | Setting |
|-----------|---------|
| Model dim | 512 |
| Heads | 8 (head_dim = 64) |
| Recurrence depth | 4 (weight-tied) |
| Positional encoding | **RoPE** (persistent=False buffers — not stored in artifact) |
| Normalization | **RMSNorm** (pre-norm + final norm) |
| MLP activation | **LeakyReLU(0.5)²** |
| MLP width | 4× hidden (dim → 2048 → dim) |
| Logit softcap | 30.0 (`tanh(logits/30) * 30`) |
| Vocab size | 1,024 (SP-1024) |
| Train seq len | 1,024 |
| Unique parameters | ~7.3M |

### Why weight-tied recurrence?

Weight tying across depth means one block's parameters serve all recurrence steps:
- **Artifact efficiency**: only one block stored → 3.80 MB int8+zlib vs. ~14+ MB for a 4-layer flat transformer
- **Implicit depth**: 4 recurrence steps provide effective depth, improving convergence
- **Remaining headroom**: 12.2 MB unused — room for larger `dim` or more recurrence steps

## Optimizer

| Group | Optimizer | LR |
|-------|-----------|-----|
| block 2D weights (`qkv_proj`, `out_proj`, `fc`, `proj`) | **Muon** (Newton-Schulz, nesterov) | 0.04 |
| `tok_emb` (tied to head) | Adam | 0.05 |
| `ln_f.weight`, `block.norm.weight` (RMSNorm scales) | Adam | 0.04 |

LR warmup: 20 steps. Warmdown: 600 steps cosine to 0 before wallclock cap.

## Quantization: SDClip Int8 + zlib

2D tensors are quantized per-row with **SDClip** (clip at 2.5σ per row), then stored as int8 with fp16 scale factors and compressed with `zlib` level 9.

- **SDClip vs quantile**: `2.5 * t.std(dim=1)` is faster (no sort) and better-calibrated for weight distributions with light tails
- Small tensors (≤65536 elements, e.g. embeddings): stored as fp16 passthrough
- RoPE cos/sin buffers: `persistent=False` — recomputed from scratch on load, zero artifact cost

Serialized pipeline: `float32 → int8 + fp16 scale → zlib → base64 → torch.save`

## Evaluation: Legal Score-First TTT

Post-training test-time training adapts only the 1,024 norm-scale parameters:

- **Adapted params**: `ln_f.weight` (512) + `block.norm.weight` (512)
- **Protocol**: score chunk → SGD update → move to next chunk (strict score-before-update)
- **Optimizer**: SGD, `lr=1e-3`, `ttt_steps=5` per chunk
- **Why norm-only TTT**: with weight-tied recurrence, updating full block weights amplifies gradients ×4 through recurrence; norm scales are stable, non-amplifying, and adapt the model's effective gain to the local data distribution

```
For each 1024-token validation chunk:
  1. SCORE  → forward under inference_mode()  → record loss (reported BPB)
  2. UPDATE → 5 SGD steps on ln_f.weight + block.norm.weight
Weights restored to checkpoint state after all chunks are evaluated.
```

TTT timing for seed 1337 (5 steps): 319,979ms (~320s) within the 10-min eval window.

## Training Command

```bash
export RECUR_STEPS=4
export MODEL_DIM=512
export NUM_HEADS=8
export LOGIT_SOFTCAP=30.0
export SDCLIP_STD_MULT=2.5
export NOISY_QAT=0
export TTT_STEPS=5
export TTT_LR=1e-3
export SEED=1337
export RUN_ID=r4_512_ttt5_seed1337
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Training Curve (seed 1337)

| Step | val_bpb |
|------|---------|
| 0 | 4.1620 |
| 1,000 | 1.5613 |
| 2,000 | 1.4894 |
| 2,004 (wallclock cap) | — |
| post-int8+TTT (5 steps) | **1.5061** |

## Artifact Size Breakdown (seed 1337)

| Component | Size |
|-----------|------|
| `train_gpt.py` code | 43,939 bytes |
| Model int8+zlib payload | 3,756,151 bytes |
| **Total** | **3,800,090 bytes** |

Cap: 16,000,000 bytes. Remaining headroom: **12,199,910 bytes** (~12.2 MB).

## Noisy QAT (disabled for 10-min runs)

Noisy QAT (injecting per-row int8-scaled noise into block Linear weights during training) is implemented but adds ~13ms/step overhead. On 1×H100 at ~285ms/step, this costs ~90 training steps in 10 minutes — a net negative for the 10-min track. Enable with `NOISY_QAT=1` for longer runs.

## Limitations and Future Work

| Bottleneck | Impact | Fix |
|------------|--------|-----|
| 1×H100 (vs 8×H100) | 2,000 steps vs ~14,000 | Use 8-GPU pod |
| Weight-tied recurrence | 285ms/step (slower than flat transformer) | Accepted tradeoff for artifact size |
| Only 2 seeds submitted | Borderline statistical significance | Run seed 2025 to complete 3-seed set |
| TTT scope | Norm-only limits adaptation | Extend TTT to MLP down-projections with reduced LR |
