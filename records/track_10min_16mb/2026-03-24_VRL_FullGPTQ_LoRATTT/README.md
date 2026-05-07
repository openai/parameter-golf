# 11L VRL + Full GPTQ + LeakyReLU² + Batched LoRA TTT

**Author:** Atharva Date (ADIITJ)
**Status:** Non-record (pending 3-seed H100 validation)
**Expected bpb:** ~1.08–1.10 (conservative), potentially lower if TTT eval time is unconstrained

---

## What's New

Stacks all improvements from PR #414 (1.1228 SOTA) + PR #569 (VRL+FullGPTQ+LeakyReLU², 1.1175) and adds batched LoRA TTT at eval time.

| Component | Source | Gain |
|-----------|--------|------|
| Full GPTQ (Hessian Cholesky int6) | PR #569 (gowtham0992) | -0.0026 bpb |
| LeakyReLU(0.5)² | PR #569 / PR #518 | -0.0015 bpb |
| VRL (Value Residual Learning) | PR #569 (arxiv:2410.17897) | -0.015 bpb |
| Batched LoRA TTT (2 ep cosine) | PR #512/#548 (MatoTeziTanka/LoquiAuris) | -0.04–0.07 bpb est. |

## Architecture

- **11 transformer layers**, dim=512, 8 heads (4 KV, GQA)
- **XSA** (Exclusive Self-Attention) on all 11 layers
- **VRL** (Value Residual Learning): layer 0's V output added to all subsequent layers via learned sigmoid gates
- **LeakyReLU(0.5)²** MLP activation
- **Partial RoPE** (16/64 dims), **LN Scale** (1/sqrt(l+1)), **VE128** (layers 9,10)
- **SmearGate** + **BigramHash(2048×128)**
- **U-Net skip connections**
- **EMA**(0.997) + **Tight SWA** (every 50 steps when scale<0.2)
- **Late QAT** (STE int6 when LR scale<0.15)

## Quantization

- **Full GPTQ**: Hessian-aware int6 with Cholesky error compensation (IST-DASLab ICLR 2023)
- GPTQ-lite fallback for layers where Cholesky fails
- Int6 per-row for all MLP + attention weights
- Int8 per-row for embeddings
- **2% magnitude pruning** post-quant (improves zstd compressibility)
- zstd-22 compression

## LoRA TTT at Evaluation

Per-document batched LoRA TTT with cosine LR decay:

1. Find document boundaries via BOS token (id=1)
2. For each batch of 64 documents (sorted by chunk count):
   - Initialize fresh LoRA adapters: rank=8 on Q, V, LM head in all 11 layers
   - For each epoch `ep` in [0, 1]:
     - LR = `0.01 × 0.5 × (1 + cos(π × ep / 2))`
     - For each chunk of 256 tokens:
       - **Score first** (accumulate NLL only if `ep == 1`)
       - Then train: backprop through chunk loss, Adam step
   - Reset LoRA adapters before next batch

**Fairness:** Every token is scored before any training on it, in every epoch. No cross-document leakage.

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| lora_rank | 8 |
| ttt_epochs | 2 |
| ttt_lora_lr | 0.01 (cosine → ~0) |
| chunk_size | 256 tokens |
| eval_seq_len | 2048 tokens |
| batch_seqs | 64 docs/GPU |
| optimizer | Adam (β₁=0.9, β₂=0.95, ε=1e-10) |
| targets | Q, V in all 11 layers + LM head |

## Attribution

- PR #414 (signalrush): current SOTA training stack (1.1228 bpb)
- PR #569 (gowtham0992): VRL + Full GPTQ + LeakyReLU² (1.1175 bpb)
- PR #512 (MatoTeziTanka): batched LoRA TTT protocol
- PR #548 (LoquiAuris): batched LoRA TTT implementation (246s eval time)
- PR #77 (samacqua): original LoRA TTT

## Run Command

```bash
cd records/track_10min_16mb/2026-03-24_VRL_FullGPTQ_LoRATTT/

SEED=42 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
