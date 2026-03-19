Depth recurrence + SwiGLU submission for the 10-minute 16MB track.

The core idea is to reuse the same transformer blocks multiple times per forward pass. Instead of 9 unique blocks at width 512, this uses 5 unique blocks and runs them 3 times each, giving 15 effective layers while only storing 5 blocks worth of parameters. The freed budget goes into width (672 vs 512) and a SwiGLU MLP activation, which outperforms relu-squared at matched parameter count.

The argument for depth recurrence in this setting is straightforward. With 17M parameters trained on 7B+ tokens, the baseline is massively over-trained relative to Chinchilla scaling (420 tokens/param vs the optimal ~20). In this regime, tokens are cheap and model quality is the bottleneck. Depth recurrence trades tokens for effective compute per token. Each forward pass does ~3x more work through the same parameters, so the model sees roughly 2.5B tokens instead of 7B, but processes each one through 15 layers at width 672 instead of 9 layers at width 512. Total training FLOPs are comparable to baseline. Scaling law reasoning suggests this is a favorable trade when you are this far past the Chinchilla-optimal token count.

Each block has a learned resid_mix that blends current hidden state with the original embedding x0, acting as a gradient highway across all 15 effective layers. The SwiGLU gating (silu(gate) * up, then project down) replaces relu-squared and gives better loss per FLOP at the same parameter count. Hidden dimension is int(2/3 * mlp_mult * dim) = 896 to match relu-squared param count with 3 projections instead of 2.

Other tuning for the shorter step budget (~4700 steps on 8xH100 vs baseline's ~13800): warmdown reduced from 1200 to 400 iterations to maintain the same ~8.5% training fraction, Muon momentum warmup cut from 500 to 250 steps, gradient clipping at 1.0 for stability through 15 effective layers.

Evaluation uses seqlen 2048 (rules allow any length). This gives each token more context for prediction at zero parameter cost, with RoPE base 10000 handling 2x extrapolation cleanly.

Configuration: `NUM_LAYERS=5 RECURRENCE=3 MODEL_DIM=672 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 EVAL_SEQ_LEN=2048 WARMDOWN_ITERS=400 GRAD_CLIP_NORM=1.0`

Run command:
```bash
RUN_ID=depth_recurrence_5x3_d672 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
