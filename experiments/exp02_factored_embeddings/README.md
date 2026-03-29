# Experiment 02: Factored Embeddings

This mini-project keeps the baseline training and export path mostly intact, but replaces the full token embedding with a factorized embedding plus projection.

## Change

- input path: `Embedding(vocab, factorized_dim)` then `Linear(factorized_dim, model_dim)`
- tied output path: hidden states are projected back down through the transpose of the same projection before the final tied vocabulary projection

## Why This Exists

This tests whether some embedding bytes can be moved into the model body or optimizer budget without breaking convergence.

## Main Knobs

- `FACTORIZED_EMBED_DIM`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `WARMDOWN_ITERS`
- `WARMUP_STEPS`

## Optuna-Friendly Surface

Start with:

- `FACTORIZED_EMBED_DIM` in something like `{96, 128, 160, 192}`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `WARMDOWN_ITERS`

Keep `VOCAB_SIZE` and tokenizer fixed at first so the result stays attributable to factorization alone.
