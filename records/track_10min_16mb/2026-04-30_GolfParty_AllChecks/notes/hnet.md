# H-net Hierarchical Tokenization — `KS_HNET_CHUNK`

OpenAI Requests-for-PRs item: *"H-net tokenization"*.

## What this is

`ks_hnet_pool(h, chunk)` mean-pools the hidden representation in
chunks of `KS_HNET_CHUNK` tokens, returning a coarse `(B, T/chunk, D)`
tensor that a downstream layer can run cheaply over. Hierarchical
chunking gives the model a "summary" view of the sequence at lower
resolution, complementing the per-token attention.

## Toy vs real

- **Toy:** mean-pool only, no learned tokenization. Drop-in scaffolding
  for a coarse-grained pass — the actual coarse attention layer that
  would consume the pooled tensor is not wired in. The intent is to
  show the *plumbing* for hierarchical processing, not to claim a real
  H-net.
- **Real H-net** as in Wu et al. would need (a) a learned chunking /
  segmentation module, (b) a separate coarse-grained transformer on
  top of the pooled tokens, (c) a way to broadcast coarse
  representations back to the fine-grained per-token layer, and
  (d) a pretraining curriculum that exercises the hierarchy.

## Why it's still here

CaseOps (PR #1729) and our **CaseDigitOps** + **CaseDigitWsOps**
extensions already explore the *bijective lossless tokenizer*
direction, which is one half of the H-net spirit. The other half —
*hierarchical* tokenization — is what `KS_HNET_CHUNK` opens the door
to. A future PR could pair them: bijective byte-transforms at the
character level + learned chunking at the token level.

## Limits

The mean-pool is a very weak summary. A real implementation would
prefer (a) attention-pool with a learned `[CLS]`-style token per chunk,
or (b) a small RNN aggregator. Mean-pool is the "say the line" version.
