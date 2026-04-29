# Clonal Selection: Vocabulary-Aware Parameter Refresh

## Biological inspiration
When a B cell successfully neutralizes an antigen, it clones and hypermutates toward
the target. Cells that fail are pruned. The immune system continuously specializes.
Opposite of standard fine-tuning.

## Architecture
During warmdown phase:
1. Identify K tokens with highest per-token validation loss ("antigens").
   K = round(vocab_size / φ⁵) ≈ 96 for vocab_size=1024. (φ⁵ ≈ 11.09)
2. Allocate small dedicated parameter deltas (residual expert weights) for those tokens.
   Specialist: CastedLinear(model_dim, model_dim) per antigen token = 96 × 384² ≈ 14M
   (too large — scale down: 96 × 64 × 384 ≈ 2.4M, a bottleneck specialist)
3. Base model frozen at SWA average; only specialist weights train on hard tokens.
4. At eval: if input token is an "antigen", add specialist residual to hidden state.

This focuses extra capacity exactly where the model is weakest.
φ bonus: K = vocab_size / φ⁵ ≈ 96. φ⁵ ≈ 11.09, so specialists cover ~9% of vocab.

## Key hyperparameters
- CLONAL_ENABLED = 1
- CLONAL_K_TOKENS = 96         (= round(1024 / φ⁵))
- CLONAL_BOTTLENECK_DIM = 64   (specialist rank)
- CLONAL_WARMDOWN_LR = 0.025

## Implementation notes
Requires:
1. A per-token loss eval pass to identify top-K hard tokens (run once at warmdown start)
2. nn.Embedding(vocab_size, bottleneck_dim) + nn.Linear(bottleneck_dim, model_dim)
   for the specialist residuals (sparse — only activated for hard tokens)
3. Hooks during warmdown training to route hard tokens through specialists

Most complex to implement correctly. Needs 2-pass architecture.
Can approximate: always compute specialist for all tokens, just don't train non-hard ones.

## Buildability: ★★☆☆☆ — needs post-training analysis pass, 2-phase training
