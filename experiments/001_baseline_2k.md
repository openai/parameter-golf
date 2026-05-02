# Experiment 001: Baseline 2K Steps on 1xH100

## Status: COMPLETE

## Hypothesis
Run the unmodified baseline to establish ground truth BPB at 2000 steps on a single H100.
This gives us a local reference to compare all future experiments against at the same step count.

## Configuration
- **Model**: 9 blocks, dim=512, 8 heads, 4 KV heads, vocab=1024, logit_softcap=30
- **Training**: 2000 iterations, 524K tokens/step, seq_len=1024
- **GPU**: 1x H100 (Thunder Compute prototyping mode)
- **Optimizer**: Muon (lr=0.04) for matrices, Adam (eps=1e-8) for embeddings/scalars
- **Warmdown**: 1200 steps
- **wandb**: disabled (initial run)
- **Parameters**: ~17M

## Results

### Training Curve
| Step | train_loss | val_loss | val_bpb | train_time |
|------|-----------|----------|---------|------------|
| 0    | 6.94      | 6.94     | 4.1077  | 0ms        |
| 100  | 3.32      | -        | -       | 21s        |
| 500  | 2.50      | 2.50     | 1.4805  | 240s       |
| 1000 | 2.35      | 2.32     | 1.3760  | 479s       |
| 1500 | 2.26      | 2.24     | 1.3262  | 718s       |
| 2000 | 2.24      | 2.19     | 1.2963  | 958s       |

### Final Numbers
- **val_bpb (pre-quant)**: 1.2963
- **val_bpb (post-quant int8+zlib)**: 1.2978
- **Quantization cost**: 0.0015 BPB (much less than 0.03 expected at 20K steps)
- **Artifact size**: 15,012,154 bytes (well under 16MB)
- **Peak memory**: 10,936 MiB
- **Step avg**: ~464ms/step (1x H100, grad_accum=8)
- **Total training time**: 958s (~16 min)

## Key Observations
1. At 2K steps, we get val_bpb=1.2963, which is already close to the 20K-step SOTA of 1.2244
2. Quantization cost at 2K steps is tiny (0.0015 BPB) vs expected 0.03 at 20K — suggests quant cost grows with training
3. Artifact is ~15.0MB, leaving ~1MB headroom for wider models
4. 2K steps is a good screening length — takes ~16 min on single H100, captures most of the learning
5. Loss is still decreasing at step 2000, suggesting more steps would help
