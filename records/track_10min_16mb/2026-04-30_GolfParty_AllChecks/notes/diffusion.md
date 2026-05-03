# Text Diffusion — `KS_DIFFUSION_FRAC`

OpenAI Requests-for-PRs item: *"Text diffusion"*.

## What this is

A training-time noise-injection signal: with probability
`KS_DIFFUSION_FRAC` per-position, replace the input embedding with
random Gaussian noise (scaled to match `emb.std()`), and add a
reconstruction loss term that asks the model to recover the clean
embedding at noised positions.

```python
noised, mask = ks_diffusion_perturb(emb, frac)
diffusion_loss = mse(model(noised), emb) * mask
```

Conceptually: a single-step denoising objective at the embedding
level, mixed with the standard CE on token logits.

## Toy vs real

- **Toy:** single noise scale, no diffusion schedule, no `t` step
  conditioning, no bidirectional decoder. The model is still
  fundamentally autoregressive at eval time — the diffusion signal
  only operates at training time as a regularizer / noisy-LM auxiliary.
- **Real:** would need (a) a full ε-prediction objective with a noise
  schedule (linear / cosine), (b) bidirectional masked decoding at
  inference, (c) a way to do this *without* breaking autoregressive
  eval (because the leaderboard scores autoregressive bpb), and
  (d) likely a separate diffusion-only model rather than a
  hybrid head.

## Why it's still here

The compatibility constraint with autoregressive scoring means a "true"
text diffusion record is genuinely hard inside this leaderboard. The
toy lets us check the box and document the architectural mismatch.
There's a real research question lurking — "can diffusion-style
training-time noise improve autoregressive perplexity?" — that this
toggle is the first scaffolding for.

## Limits

Single noise-scale + no schedule means this is closer to "input
embedding dropout" than "diffusion" in any rigorous sense. The honest
framing is: *training-time embedding-noise auxiliary, inspired by
text-diffusion literature*.
