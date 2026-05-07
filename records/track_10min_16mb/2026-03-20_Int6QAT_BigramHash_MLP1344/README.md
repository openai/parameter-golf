# 10L Int6 QAT + BigramHash + Zstd MLP2.6x Muon0.99

## Summary

Builds on the int6 QAT baseline with a novel BigramHash embedding layer:

1. **BigramHash embedding**: 4096-bucket hash table (dim=128, projected to 512) injecting token-pair context. Hash: `XOR(36313 * curr, 27191 * prev) % 4095`. Zero-initialized with learned scale (init 0.05). Added to token embeddings before RMSNorm.

2. **STE int6 QAT**: Straight-through estimator fake quantization during training. Zero quant gap.

3. **Full int6 quantization** [-31,31] + **zstd-22** compression.

4. **MLP hidden 1344** (2.625x model_dim).

5. **FP16 tied embedding passthrough**.

6. **10 transformer layers**, seq_len 2048.

7. **Muon momentum 0.99**, MATRIX_LR=0.02, SCALAR_LR=0.02, grad clip 0.3, warmdown 3600.

8. **Sliding window evaluation** stride=64.

## Configuration

```bash
MLP_HIDDEN=1344 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requires: `pip install zstandard`

## Results

| Seed | Steps | val_bpb (standard) | val_bpb (sliding) | Artifact size |
|------|-------|--------------------|--------------------|---------------|
| 1337 | 8,494 | 1.1800 | 1.1589 | 15,504,131 |
| 42 | ~8,490 | ~1.1800 | 1.1593 | ~15,504,000 |
| 3 | ~8,490 | ~1.1800 | 1.1597 | ~15,504,000 |

**Mean val_bpb (sliding): 1.1593** (std: 0.00040)
**Mean val_loss (sliding): 1.9574** (std: 0.00067)

Statistical significance vs baseline (2.0727 val_loss):
- Improvement: 0.1153 nats, t=-286.2, p << 0.01

Hardware: 8xH100 80GB HBM3, PyTorch 2.8.0+cu128, ~71ms/step.

## Included Files

- `train_gpt.py` (modified training script)
- `train_seed1337.log`, `train_seed42.log`, `train_seed3.log`
- `submission.json`
