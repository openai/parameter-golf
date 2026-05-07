# Int5/Int6 + Zstd + 3x MLP + 10L + Seq4096 + Sliding Window

## Result

| Seed | val_bpb | Artifact |
|------|---------|----------|
| 1337 | 1.1773 | 15,708,798 B |
| 42   | 1.1732 | 15,700,116 B |
| 7    | 1.1750 | 15,711,843 B |
| **mean** | **1.1752** | |
| **std**  | **0.0017** | |

All artifacts ≤ 16,000,000 bytes ✓

## Key changes

This submission stacks several compression and architecture improvements on top of the
sliding window + seq4096 baseline:

1. **Int5 quantization for MLP weight matrices** — tighter 5-bit packing frees ~1.5MB
   budget, allowing a wider MLP (3x expansion) without exceeding 16MB.

2. **Int6 quantization for attention matrices** — 6-bit precision for Q/K/V/proj weights,
   balancing model quality and size.

3. **Zstd compression** — replaces zlib for artifact packaging. Zstd achieves better
   compression ratios on quantized integer arrays, saving ~0.3–0.5MB vs zlib at the same
   decompression speed.

4. **MLP expansion 3x** — wider MLP (3×512=1536 hidden units) vs baseline 2x (1024).
   The budget freed by int5/int6+zstd enables this without size violation.

## Hyperparameters

```
num_layers     = 10
train_seq_len  = 4096
eval_stride    = 64
warmdown_iters = 3600
mlp_mult       = 3
matrix_lr      = 0.04  (default)
muon_momentum  = 0.95  (default)
```

## Quantization scheme

| Layer type | Quantization |
|-----------|-------------|
| Embedding (tok_emb) | int6 + zstd |
| Attention Q/K/V/proj | int6 + zstd |
| MLP up/down/gate | int5 + zstd |
| Norms, scales, biases | float32 (passthrough) |
| Lm_head | tied to embedding |

## Hardware

Modal 8×H100 SXM, `torchrun --standalone --nproc_per_node=8`
Training capped at 600 seconds (`MAX_WALLCLOCK_SECONDS=600`).
