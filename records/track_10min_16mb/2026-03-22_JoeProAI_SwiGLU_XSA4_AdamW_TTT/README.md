# Record: SwiGLU + XSA4 + U-Net + AdamW TTT (3-seed mean val_bpb=1.0672)

**3-seed mean val_bpb: 1.0672** | Best seed: 1.0658
Verified on 8xH100 80GB, 10-minute wall-clock budget.

## Approach

Novel architecture discovered through GEPA (Gemini-driven Evolutionary Parameter Architecture search) combined with community-proven techniques. Built over 5 days across 6 waves of experiments on Modal H100s.

### Architecture (discovered by GEPA)
- **SwiGLU FFN** with Star-ReLU activation
- **U-Net skip connections** with learned gating
- **BigramHash embeddings** (8192 buckets, 128 dim)
- **SmearGate** on embeddings
- 11 layers, 512 dim, 8 heads, 8 KV heads, MLP hidden=1792, tied embeddings

### Training techniques (adopted + tuned)
- **XSA4** (cross-sequence attention on last 4 layers) -- credited to @felipe-parodi (#398)
- **EMA** (decay=0.9985) -- credited to @felipe-parodi (#398), decay tuned by us
- **AdamW TTT** (lr=0.0005, 10 epochs, wd=0.0) -- credited to @sjp611 (#442)
- **Partial RoPE** (16 dims) -- credited to @felipe-parodi (#398)
- **LN Scale** (1/sqrt(layer_idx+1)) -- credited to @felipe-parodi (#398)
- **Late QAT** (threshold 0.15) -- credited to @fbedev (#410)
- Muon optimizer (matrix_lr=0.025, wd=0.04, momentum=0.99)
- Warmdown: 6000 steps
- Int6 quantization + zstd-22 compression

## 3-Seed Results

| Seed | val_bpb |
|------|---------|
| 42 | 1.06733191 |
| 123 | 1.06833018 |
| 7 | 1.06579646 |
| **Mean** | **1.06715285** |
| **Std** | **0.00104211** |

## Comparison to prior SOTA

| Submission | Mean BPB | Best BPB |
|-----------|----------|----------|
| **Ours** | **1.0672** | **1.0658** |
| @sjp611 (#442) | 1.1027 | 1.0992 |
| @felipe-parodi (#398) | 1.1221 | 1.1213 |
| @thwu1 (#180, merged) | 1.1428 | -- |

## Key finding

AdamW TTT produced a 0.053 bpb improvement on our architecture vs 0.019 on the standard architecture (PR #398). This suggests SwiGLU + U-Net skip connections create a loss landscape that AdamW navigates significantly better than SGD during test-time training.

## Credits

- **@felipe-parodi** (#398): EMA, TTT, XSA4, Partial RoPE, LN Scale
- **@sjp611** (#442): AdamW TTT
- **@fbedev** (#410): Late QAT
- **@thwu1** (#180): 11-layer architecture direction
- Compute provided by **Modal**

Built by [@JoePro](https://x.com/JoePro) (GitHub: [@JoeProAI](https://github.com/JoeProAI)) with AI agent assistance: [OpenClaw](https://openclaw.ai) (Claude Opus), Codex (GPT-5.4), Claude Sonnet, Gemini 2.5 Pro, and Paperclip agent coordination.

## Run command

```bash
# Default seed
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Specific seed
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are set as defaults in `train_gpt.py`.
