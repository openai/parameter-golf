## Depth Recurrence + Cross-Repeat Skip + Value Embeddings

Beats naive baseline (1.2244) by 0.005 bpb using 3.1x fewer training steps through stateful depth recurrence.

val_bpb = 1.2196 (sliding window eval on int8+zlib roundtrip model, stride=256)
val_bpb = 1.2533 (standard int8+zlib roundtrip)

### Architecture

Replaced the baseline's 9 unique transformer blocks with 3 shared blocks repeated 4 times (12 effective layers). Trades unique parameters for effective depth.

Changes from baseline:
- Depth recurrence: 3 blocks x 4 repeats = 12 effective layers (vs 9 in baseline)
- Cross-Repeat Skip (original): each block gets a weighted residual of its own output from the previous repeat, turning stateless recurrence into stateful. Per-repeat learned scales, ~7.5K params total.
- Value Embeddings: 2 extra embedding tables mixed into the residual stream at each effective layer with learned scales. From snimu's modded-nanogpt record.
- Loop Embedding: learned per-layer vector added before each block as depth-wise positional encoding.
- Model dim 832 (vs 512), 8 heads, 4 KV heads, MLP 2x
- Removed U-Net skip connections (Cross-Repeat Skip covers this role)
- 17.14M params, 12.83MB artifact

### Training

LR x0.3 from baseline — recurrence amplifies gradients through 4 passes, so optimal LR is much lower. Found via sweep of 10 configs on RTX 3060.

MATRIX_LR=0.012, SCALAR_LR=0.012, TIED_EMBED_LR=0.015, GRAD_CLIP_NORM=0.3, WARMDOWN_ITERS=3000, TRAIN_SEQ_LEN=1024.

Tested train@2048 but 1024 gives more steps (133ms vs 253ms/step) which matters more for this architecture. Standard Muon + Adam.

### Evaluation

Sliding window eval: window=1024, stride=256 on the int8+zlib roundtrip model. Eval time 209s on 8xH100.

### Results (8xH100, 600s wallclock)

4494 steps, 133ms/step avg. Pre-quant 1.2487, roundtrip 1.2533, sliding window 1.2196. Artifact 12.83MB, quant degradation 0.005 bpb, peak memory ~29GB/GPU.

### Ablations (RTX 3060, 2000 steps each)

- Cross-Repeat Skip: -0.041 bpb
- Value Embeddings (2 tables): -0.079 bpb
- LR x0.3: -0.052 bpb
- Sliding window eval: -0.034 bpb
- WARMDOWN_ITERS=3000: -0.027 bpb

### Development

All experiments, ablations, and hyperparameter sweeps done on a single RTX 3060 12GB. Cloud GPUs (1xH200, 6xH100) used only for validation. Final run on 8xH100.

### Command

```
RUN_ID=submission_8xh100 \
QUANT_LEVELS=127 \
TTT_STEPS=0 \
EVAL_STRIDE=256 \
EVAL_SEQ_LEN=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
