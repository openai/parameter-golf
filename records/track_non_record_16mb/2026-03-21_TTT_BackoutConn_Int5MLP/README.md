# 10L Int5-MLP + BigramHash + TTT + Backout Connection (Non-Record)

**val_bpb: 1.4463** (1xH100, 869 steps — 8xH100 run pending)

This is a non-record submission demonstrating two techniques (TTT + Backout Connection) stacked on the current #1 record base. Full 8xH100 results pending compute availability.

## Run Command

```bash
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Hardware | Steps | val_bpb | Artifact Size |
|----------|-------|---------|---------------|
| 1xH100 (RunPod) | 869 | 1.4463 | 15.5MB |
| 1xA100 (Northeastern HPC) | 423 | 1.6760 | 15.5MB |
| 8xH100 SXM | TBD | TBD | TBD |

Note: The 1xH100/A100 scores reflect severe undertraining (~869/423 steps vs ~7000+ on 8xH100). The purpose of these runs was to verify end-to-end correctness.

## Approach

Built on thwu1's #1 record (1.1428 bpb), adding two techniques:

### 1. Backout Connection (inspired by PR #339)
A learned residual subtraction that removes redundant mid-layer information from the final representation. Captures hidden state at layer `num_layers // 2` (layer 5) and subtracts `lambda * h_mid` from the output before the final RMSNorm. Adds exactly 1 parameter (a learned scalar, init 0.2). Zero computational cost.

The intuition: U-Net skip connections create redundancy between encoder and decoder representations. A targeted subtraction sharpens the signal for the LM head.

### 2. Test-Time Training (inspired by PR #338)
After quantization roundtrip, performs 3 epochs of SGD fine-tuning directly on validation tokens. This adapts the quantized model to recover from quantization degradation. First 2 transformer blocks are frozen to preserve low-level features.

- Optimizer: SGD, lr=0.002, momentum=0.9
- Epochs: 3
- Grad clip: 1.0
- Frozen: first 2 blocks
- Estimated time on 8xH100: ~47s (well within eval budget)

## Architecture (inherited from thwu1)
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- U-Net skip connections, tied embeddings
- Mixed int5 (MLP) / int6 (attention) quantization + zstd-22
- 3% magnitude pruning
- SWA (start_frac=0.4, every=50 steps)
- Sliding window eval (stride=64)
- Backout connection at layer 5 (lambda init=0.2)
- TTT: 3 epochs SGD on val tokens post-quantization

## Training Hyperparameters
- Muon: matrix_lr=0.02, WD=0.04, momentum=0.99 (warmup from 0.92)
- seq_len=2048, batch=786K tokens, warmdown=3000
- grad_clip=0.3
