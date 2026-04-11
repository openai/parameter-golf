# LeakyMixer — 2026-03-26

took the baseline train_gpt.py and made a few changes that seem to be working well across the leaderboard. main thing is swapping relu^2 for leaky_relu(0.5)^2 and bumping to 11 layers. also added a backoff n-gram mixer that runs at eval time — it builds up a cache of token n-grams as it scores the val set and mixes those predictions with the neural model's logits.

## what i changed

**architecture tweaks:**
- 9 → 11 layers (fits in 16MB with int8+zlib at ~13.5MB)
- leaky_relu(0.5)^2 activation in MLP (saw a bunch of people on the leaderboard using this, seems to help)
- warmdown 1200 → 3500 steps (longer LR decay)

**eval-time n-gram mixer:**
- builds a multi-order backoff n-gram table (orders 1-7) as it walks through the val set
- for each token, mixes the neural model's log-probs with the n-gram predictions using entropy-adaptive alpha (when the neural model is uncertain, trust n-grams more)
- backward-looking only — only uses tokens already scored, so it's legal under the TTT rules
- implemented in C (hash table based) for speed, compiles at eval time
- on 200K tokens locally this gave ~29% BPB improvement over neural-only. on the full 62M val set with a properly trained model, expecting ~0.45-0.50 BPB drop

## results

ran on 4xA100 SXM because i was low on runpod credits. got 849 steps in the 600s wallclock:

| metric | value |
|--------|-------|
| val_loss (int8+zlib roundtrip) | 2.3800 |
| val_bpb (neural only) | 1.4096 |
| artifact size | 13.49 MB |
| steps | 849 |

on 8xH100 this should get ~1700 steps and land around ~1.20 BPB neural-only. with the n-gram mixer on top, estimating ~0.70-0.75 BPB total.

## how to run

```bash
# train (8xH100)
torchrun --nproc_per_node=8 train_gpt.py

# the n-gram mixer is baked into the eval — it auto-compiles ngram_mixer_fast.c
# and runs after the standard int8+zlib roundtrip eval
```

## files

- `train_gpt.py` — training script with all the changes
- `ngram_mixer_fast.c` — C implementation of the n-gram cache + mixer (compiled at eval time)

## param budget

```
11 layers × (attn + mlp + scalars)  ≈ 20.2M params
tok_emb (tied):                        524K
skip weights:                          2.8K
────────────────────────────────
total:                               ~20.7M params
int8+zlib:                            13.49 MB  (under 16MB)
```
