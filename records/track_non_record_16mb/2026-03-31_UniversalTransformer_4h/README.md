# Universal Transformer (Unlimited Compute Track, 4h on 8xH100)

Non-record submission implementing a **Universal Transformer** for the unlimited compute track, directly responding to the [Requests for PRs](https://github.com/openai/parameter-golf/blob/main/README.md#requests-for-prs) item:

> Universal transformer - We have lots of depth recurrence submissions, but I'd love to see one 4 hour

## Core Idea

Instead of N unique transformer blocks, use **a single shared block iterated many times**. Weight sharing frees the parameter budget for a wider model, while learned step embeddings let the shared block differentiate its behavior across iterations.

| Property | Baseline (9L x 512d) | This UT (1x1024d x 24 iter) |
|----------|----------------------|------------------------------|
| Unique block params | ~16.5M (9 blocks) | ~8.9M (1 block) |
| Embedding params | ~524K (1024 x 512) | ~1.05M (1024 x 1024) |
| Effective depth | 9 | 24 |
| Model width | 512 | 1024 |
| MLP expansion | 2x (1024 hidden) | 3x (3072 hidden) |
| Heads / KV heads | 8 / 4 | 16 / 4 |
| Total unique params | ~17M | ~10M |
| Compressed size (est.) | ~15.9MB | ~9.3MB |

The compressed size is well under 16MB, leaving room for future additions (BigramHash embeddings, ValueEmbeddings, etc.) once the base architecture is validated.

## Why This Should Work

1. **PR #1204 proved depth recurrence improves BPB** (1.1063 BPB, pending SOTA) by reusing layers 4-5 in the middle of the stack. Our UT takes this to its logical extreme.

2. **Step embeddings solve the "all layers are identical" problem.** Prior negative results with full recurrence (PR #1186: +0.20 BPB) likely failed because the shared block couldn't differentiate early vs. late iterations. Our learned step embeddings (one per iteration, added to the residual stream) give the block positional information about its current depth.

3. **Width compensates for sharing.** With only 1 block's worth of unique parameters, we go 2x wider (1024d vs 512d). Each iteration of the wider block has ~4x the representational capacity of a baseline block.

4. **4 hours of training is enough.** Even with ~3x slower step time (wider model, more iterations), 4 hours on 8xH100 yields ~36K+ steps -- more than enough for convergence.

## Architecture Details

- **Shared Block**: Single transformer block with GQA attention (16 query heads, 4 KV heads, 64d head dim) + 3x MLP with LeakyReLU(0.5)^2 activation
- **Iteration**: Block is called 24 times in sequence with residual connections
- **Step Embeddings**: Learned 1024d vector per iteration, added to the residual stream before each block call
- **Positional Encoding**: Standard RoPE (applied fresh in each iteration since attention is re-computed)
- **Activation**: LeakyReLU(0.5)^2 from PR #549 (proven ~0.002 BPB improvement over relu^2)
- **Optimizer**: Muon for matrix params, Adam for embeddings/scalars (same as baseline)
- **Quantization**: int8 + zlib (baseline method)
- **Sequence Length**: 2048 tokens

## Running

```bash
RUN_ID=ut_4h \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=14400 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To experiment with different iteration counts:
```bash
NUM_ITERATIONS=16 torchrun --standalone --nproc_per_node=8 train_gpt.py
NUM_ITERATIONS=32 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Future Directions

- **Adaptive Computation Time (ACT)**: Learn a per-token halting probability so the model can allocate more iterations to harder tokens
- **Progressive depth during training**: Start with fewer iterations and increase over training (curriculum for depth)
- **BigramHash embeddings**: The compressed size leaves ~6MB of headroom for hash embeddings
- **Int6 GPTQ quantization**: Could allow an even wider model within 16MB
- **Combine with 10-min track techniques**: XSA, EMA/SWA, TTT

## Included Files

- `train_gpt.py` -- Full training script with Universal Transformer architecture
- `README.md` -- This file
- `submission.json` -- Leaderboard metadata (pending training results)

## Author

Shifat Islam Santo (@oneKn8)
