# Experiment 0023_tied_embed_init_std_02_on_winner

Parent: canonical (env-var sibling of `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_16k_matrix_lr_06`)

## Question
0022 showed that TIED_EMBED_LR is more LR-sensitive than MATRIX_LR. The
hypothesis was that tok_emb's small init (std=0.005) makes it fragile to
LR scaling — relative updates are huge against tiny initial values.

Direct test: increase the init scale to TIED_EMBED_INIT_STD=0.02 (4×).
Now embeddings start farther from zero. The effective relative-update
size at the same TIED_EMBED_LR=0.05 is ~4× smaller. Does this:
1. Move tok_emb LR sensitivity (allowing future TIED_EMBED_LR scaling)?
2. Improve val_bpb at the same effective config?
3. Hurt because the larger init dominates over the gradient signal?

The forward-pass behavior: tok_emb is followed by RMSNorm. RMSNorm
*normalizes* — so the magnitude of init doesn't directly affect forward
outputs (tok_emb shape is unchanged after RMSNorm). But for tied
embeddings, the *backward* path also reuses tok_emb at the lm_head.
Logits = x @ tok_emb.T. With bigger tok_emb, initial logits are bigger
(though still small in absolute terms — sqrt(512)*0.02 = 0.45 per row).
Step 1 forward loss may be different.

## Hypothesis [CONJECTURE]
Δ vs 0021 winner (2.20996) ≈ -0.005 to +0.010. Could go either way:
- Helps if the small init was indeed a relative-update problem.
- Hurts if the bigger init introduces forward-pass noise that
  the model has to compensate for.

## Change
`env.sh`:
- `LR_WARMUP_STEPS=30`
- `WARMDOWN_ITERS=300`
- `MLP_MULT=4`
- `TRAIN_BATCH_TOKENS=16384`
- `MATRIX_LR=0.06`
- `TIED_EMBED_INIT_STD=0.02`

## Disconfirming
- Δ < -0.005: bigger init noise hurts. Stay at 0.005.
- Δ ≤ +0.005 (noise band): init scale isn't a meaningful lever; the
  tied lm_head dominates the embedding learning.
- Δ ≥ +0.010 (clear win): init scale was suboptimal. Could try further
  scaling (e.g. 0.05).

## Notes from execution
<!-- Filled after the run. -->
