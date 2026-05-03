# JEPA — `KS_JEPA_WEIGHT`

OpenAI Requests-for-PRs item: *"JEPA"* (Joint-Embedding Predictive
Architecture, LeCun et al.).

## What this is

`ToyJEPAHead` adds an auxiliary loss: predict the *embedding* of the
next token via a small linear head on the hidden state, with MSE
against the (detached) target embedding. Mixed into the total loss
with weight `KS_JEPA_WEIGHT`:

```
total_loss = ce_loss + KS_JEPA_WEIGHT * mse(proj(h_t), tok_emb(y_t).detach())
```

The head is a single `Linear(dim, dim, bias=False)` — adds ~`dim²`
params (~256k at `dim=512`).

## Toy vs real

- **Toy:** the prediction target is just the token embedding lookup,
  not a stop-gradient teacher network's representation as in I-JEPA /
  V-JEPA. There's no separate context/target encoder pair. The loss is
  applied at every position in parallel with the standard CE.
- **Real:** would need (a) separate target encoder with EMA-only
  updates, (b) masked or block prediction targets in latent space,
  (c) a careful study of whether JEPA helps autoregressive eval at all
  — JEPA's gains are typically demonstrated in representation learning,
  not next-token prediction perplexity.

## Why it's still here

JEPA-style auxiliary losses *can* regularize the embedding space and
have shown small perplexity wins in some LM contexts (e.g. SimCSE-style
contrastive auxiliaries). At our 16 MB budget, the extra `dim²` params
might be too expensive once GPTQ-quantized. The toggle lets future work
sweep `KS_JEPA_WEIGHT ∈ {0.01, 0.02, 0.05, 0.1}` on the existing stack.

## Limits

The auxiliary head and its `dim²` weights need to either (a) live in
the artifact at quantization-friendly precision, or (b) be discarded
post-training (used only for regularization). The toggle as written
keeps the head in the model — for recordable use, a flag should
discard it before serialization.
