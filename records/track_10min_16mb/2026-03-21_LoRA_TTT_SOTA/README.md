# LoRA TTT on SOTA — Negative Result

**Status:** Complete. Non-record submission (artifact 16.9MB > 16MB; TTT eval 20.4 min > 10 min budget).

## Results

| Metric | Value |
|--------|-------|
| Base val_bpb (int8+zlib, sliding window) | **1.15402184** |
| TTT val_bpb (LoRA adapted) | **1.34789270** (+0.194 regression) |
| Artifact size (int8+zlib) | 16,907,757 bytes (over 16MB limit) |
| TTT eval time | 1225s (~20.4 min, over 10 min eval budget) |
| Training steps | ~4917 |

## Why TTT Regressed

The LoRA adapters (rank=8 on c_q+c_v) trained with Adam from scratch per document at lr=0.01. The early documents show 1.10 bpb but by document 500 it shoots to 1.35 and stays there — Adam is overshooting on short documents and the adapters never recover within the stride=256 window. The base model (1.1540 sliding) is already well-calibrated; TTT is destabilizing it rather than helping.

## Method

Training is **100% identical** to the current SOTA (`2026-03-20_10L_Int5MLP_MuonWD04_SWA50`, 1.1428 bpb).

Evaluation adds per-document LoRA test-time training:

1. After training + quantization roundtrip, wrap `base_model` in `GPTTTT` (per-block LoRA adapters on `c_q` and `c_v`, rank=8).
2. Find document boundaries in `val_tokens` via BOS token (id=1).
3. For each document:
   - Reset LoRA to zero-delta state.
   - Slide through with `ttt_stride=256` token steps.
   - Per step: forward → score new tokens → backward → Adam step.
4. Accumulate BPB across all scored tokens.

**No leakage:** scoring always happens before the TTT update on the same window.

## Architecture

Identical to SOTA:
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA), MLP 3x (relu²)
- SmearGate + BigramHash(10240, dim=128) + U-Net skips
- Int5 MLP + Int6 attn + FP16 embeddings + zstd-22

## TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| lora_rank | 8 |
| ttt_lr | 0.01 |
| ttt_stride | 256 |
| ttt_opt | Adam (β₁=0.9, β₂=0.95) |
| LoRA targets | c_q, c_v in all 10 blocks |

## Inspiration

> "Some things are destroyed by direct gaze. The quality in a piece of writing that vanishes when you try to name it."
> — *peripheral-attention* skill (McGilchrist / Polanyi)

The base model captures global patterns. But each document has a specific register, style, and implicit context — an "atmosphere" that the globally-trained model can only partially access. LoRA TTT lets the model adapt its attention (Q and V projections) to each document's implicit structure before evaluating it. The adaptation is the model's "peripheral attention" to the document's texture.

## Run

```bash
# Setup (once on RunPod)
cd /workspace && git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train + evaluate (matches SOTA training, adds TTT eval)
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Ablations via env vars
TTT_LORA_RANK=4 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
TTT_STRIDE=128 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
TTT_ENABLED=0  SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py  # baseline
```

## Expected Result

The LoRA TTT record gained ~0.004 bpb on the naive baseline (1.1928 → 1.1910). Applied to the stronger SOTA model, we expect a similar or slightly larger gain given better base representations. Targeting **< 1.139 bpb**.
