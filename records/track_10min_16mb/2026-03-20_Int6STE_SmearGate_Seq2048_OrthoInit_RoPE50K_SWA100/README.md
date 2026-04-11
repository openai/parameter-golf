# Int6 STE + SmearGate + Seq2048 + OrthoInit + RoPE50K + NorMuon WD + SWA/100

**Mean val_bpb: 1.1507** (3 seeds, p<0.001)

Evolved over 31 AIDE2 optimization steps from baseline 1.1607. Each step evaluated on 8xH100 with 600s training cap + sliding window evaluation.

## Technique Stack

1. **Int6 STE (Straight-Through Estimator)**: Fake int6 per-row quantization ([-31,31]) every forward pass with STE gradient bypass. Model learns to cope with quantization noise. Only +0.002 bpb gap from post-training quantization.
2. **NorMuon Optimizer + Decoupled Weight Decay (0.02)**: Row-normalized Newton-Schulz updates with per-row second-moment normalization. Weight decay improves quantization-friendliness and generalization.
3. **3x MLP Width (1536 hidden)**: Wider MLP enabled by int6 compression savings. Significant capacity increase within 16MB budget.
4. **SmearGate**: Learned gating vector (~512 params) blends each token embedding with its predecessor's representation, capturing sequential context at the embedding level with minimal parameter overhead.
5. **Orthogonal Initialization**: All non-zero-init linear layers initialized with orthogonal weights (gain=1.0). Preserves gradient norms and improves early convergence.
6. **Sequence Length 2048 + RoPE Base 50K**: 2x training context length with adjusted rotary position embedding base frequency for better position allocation.
7. **SWA every 100 steps**: Stochastic Weight Averaging with more frequent checkpoint collection during warmdown. Averages ~7 checkpoints for smoother loss landscape.
8. **FP16 Tied Embedding Passthrough**: Embedding tensor stored in fp16, never int6 quantized. Protects the most quantization-sensitive tensor.
9. **Sliding Window Evaluation (stride=64)**: Every scored token gets ~1984 tokens of preceding context. ~0.023 bpb improvement over standard non-overlapping eval.
10. **Zstd-22 Compression**: Superior compression ratio vs zlib for quantized weights.
11. **U-Net Skip Connections**: Encoder-decoder structure with learnable per-layer per-dim skip weights.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact |
|------|----------|---------|-------|---------|----------|
| 1337 | 1.9421 | 1.1502 | 10613 | 56.53 | 14,555,057 |
| 42 | 1.9433 | 1.1509 | 10610 | 56.53 | 14,791,593 |
| 7 | 1.9434 | 1.1510 | 10610 | 56.53 | 14,562,412 |
| **Mean** | **1.9429** | **1.1507** | | | |

## Key Hyperparameters

- 9 layers, 512 dim, 8 heads, 4 KV heads (GQA), vocab 1024
- train_seq_len=2048, train_batch_tokens=524288
- matrix_lr=0.021, scalar_lr=0.020, tied_embed_lr=0.030
- NorMuon: momentum=0.99, beta2=0.95, weight_decay=0.02, warmup 1500 steps from 0.92
- warmdown_iters=3000, warmup_steps=20
- SWA: start at 50% warmdown, every 100 steps
- ROPE_BASE=50000, logit_softcap=30.0, qk_gain_init=1.5

## Dependencies

Standard PyTorch + zstandard (for zstd compression).
