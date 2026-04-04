# Modified Mamba SSM + Byte-Level Tokenization

## Summary

First SSM/Mamba submission to Parameter Golf. Uses a modified Mamba architecture with the
"Achilles' Heel" fix from Chen et al. (NeurIPS 2025 Spotlight, arXiv 2509.17514v2), combined
with byte-level tokenization (vocab=260) to minimize embedding overhead.

## Architecture

- **14 Mamba blocks** (d_model=512, d_inner=1024, d_state=16, d_conv=4)
- **Achilles' Heel fix**: Learnable residual bypass around Conv1d, allowing the SSM to access
  raw (pre-convolution) token representations. This addresses the asymmetry bias introduced by
  Mamba's nonlinear convolution.
- **U-Net skip connections**: Encoder-decoder pattern (7+7 layers) with learned skip weights
- **Byte-level tokenization**: 260 vocab (4 special + 256 UTF-8 bytes), tiny embedding table
- **Tied embeddings**, logit softcap
- **Pure-PyTorch selective scan**: No custom CUDA kernels; relies on torch.compile

## Key Innovation

The paper "Achilles' Heel of Mamba" shows that Mamba's Conv1d + SiLU before the SSM fuses
token information asymmetrically, hurting the model's ability to recognize symmetric patterns.
Their fix — a residual connection bypassing the convolution — restores this capability.

We combine this with byte-level tokenization: at 260 vocab, the embedding table is just 133K
params (vs 524K for sp1024), freeing budget for more layers. Mamba's O(n) complexity handles
the longer byte-level sequences naturally.

## Training

```bash
DATA_PATH=./data/datasets/fineweb10B_byte260 \
TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json \
VOCAB_SIZE=260 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Parameter Budget

~24M parameters, estimated ~13MB compressed (int8+zlib). Well within 16MB.
