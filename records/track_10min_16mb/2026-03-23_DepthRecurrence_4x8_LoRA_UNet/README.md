# Depth Recurrence: 4 Blocks x 8 Iterations + LoRA + U-Net Skips

## Score: TBD (not yet trained)

Trained on 8xH100 SXM in 600 seconds. Artifact size TBD (int6+zstd-22).

## Approach

Builds on the Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA submission, replacing 9 unique transformer layers with **4 shared blocks looped 8 times** (32 effective layers). Per-iteration LoRA adapters differentiate behavior at each loop pass.

### 1. Depth Recurrence (4 blocks x 8 iterations)
4 unique transformer blocks are cycled through 8 times: `[0,1,2,3,0,1,2,3]`. This gives 32 effective layers while using only ~44% of the original parameter budget for the transformer body. Weight sharing provides strong regularization.

### 2. Per-Iteration LoRA Adapters (rank 2)
Each of the 8 iterations has a dedicated rank-2 LoRA adapter on the attention output projection (`W_o`). Down-project A (dim, 2) initialized from N(0, 0.01), up-project B (2, dim) initialized to zeros. Total LoRA cost: ~16K params (~16KB FP16). This allows each iteration to specialize its attention write pattern.

### 3. U-Net Skip Connections Across Iterations
Iterations 0-3 are the "encoder", iterations 4-7 are the "decoder". Skip connections pair encoder->decoder symmetrically:
- iteration 0 -> iteration 7
- iteration 1 -> iteration 6
- iteration 2 -> iteration 5
- iteration 3 -> iteration 4

Each skip uses a learnable scalar weight (sigmoid-gated, initialized at 0.5).

### 4. Layer Scale for Deep Recurrence Stability
Per-iteration learnable scalars initialized to 1/8 = 0.125 that gate the block output before residual addition. Prevents activation explosion over 32 effective layers. The update rule is: `x = x + layer_scale[i] * (block_out - x)`.

### 5. Gradient Checkpointing
`torch.utils.checkpoint` applied to each block call within the loop, trading recomputation for memory to handle 32 effective layers.

### 6. Adjusted Training Hyperparameters
- Gradient clipping: tightened from 0.3 to **0.1** (deeper effective network)
- LR warmup: **2,000 steps** linear warmup (recurrence sensitive to early instability)
- Momentum warmup: extended to **2,500 steps** (0.92 -> 0.99)

### Preserved from Previous Submission
All other techniques unchanged: Per-row int6 quantization + zstd-22, 3x MLP expansion, SmearGate, BigramHash embedding, orthogonal init, Muon with WD=0.04, SWA every 50 steps, sliding window eval at stride 64.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_unique_blocks | 4 |
| num_loop_iters | 8 |
| lora_rank | 2 |
| model_dim | 512 |
| mlp_mult | 3.0 (hidden=1536) |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| warmdown_iters | 3000 |
| lr_warmup_steps | 2000 |
| matrix_lr | 0.02 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| muon_momentum | 0.99 (warmup from 0.92 over 2500 steps) |
| muon_weight_decay | 0.04 |
| adamw_weight_decay | 0.01 |
| grad_clip_norm | 0.1 |
| eval_stride | 64 |
| swa_every | 50 |
| swa_start_frac | 0.5 |
| bigram_vocab_size | 4096 |
| bigram_dim | 128 |
| compressor | zstd (level 22) |

## Parameter Budget

| Component | Previous (9 layers) | This (4 blocks x 8 iters) |
|-----------|---------------------|---------------------------|
| Transformer body | 9 x ~2.1M = ~19M | 4 x ~2.1M = ~8.4M |
| LoRA adapters | - | ~16K (FP16) |
| Skip weights | 4 x 512 = 2K | 4 scalars |
| Layer scales | - | 8 scalars |
| Embeddings, bigram | ~1.5M | ~1.5M (unchanged) |
