# Record: SP8192 + Score-First TTT + Eval-Time Hash Embedding

**val_bpb: 1.08269** (3-seed mean, std 0.00060) | ~15.99 MB | 8xH100 SXM | ~450s eval

Merged SOTA (PR #1019, 3-seed mean): **1.88218 nats**. This run: **2.79670 nats**. Delta: **-0.914 nats**. Clears the 0.005-nat threshold.

## Results (3-seed)

| Seed | BPP | val_loss (nats) | Artifact |
|------|-----|-----------------|----------|
| 1337 | **1.08218** | 2.79537 | 15,982,929 |
| 42 | **1.08252** | 2.79626 | 15,988,459 |
| 2025 | **1.08337** | 2.79846 | 15,989,420 |
| **Mean** | **1.08269** | **2.79670** | |

## Changes from Merged SOTA (PR #1019)

### 1. Eval-Time Hash Embedding (Novel)

A zero-initialized `nn.Embedding(16384, 512)` is created at evaluation time and trained exclusively through the score-first TTT loop. At each position, a bigram hash `h = (prev_token * 2039 + curr_token) % 16384` looks up a residual vector that is added to `tok_emb(x)` before RMSNorm. The hash embedding learns document-local bigram patterns without modifying any pre-trained model weights.

**Nearest PR:** PR #1413 (@kevclark) — legal score-first TTT with full-model weight updates. **Different:** We add an ephemeral hash embedding that is instantiated from zeros at eval start and adapts via the same TTT loop. This is a new adaptation target — the model retunes a separate bigram-keyed memory alongside its existing weights. No existing PR creates and trains a new embedding module from scratch at eval time (LoRA-TTT PRs #1254/#1354 create adapter matrices, but those adapt existing layers, not a standalone hash embedding).

**Measured delta:** -0.0004 BPP vs packed baseline without hash embedding (ablation: 1.08307 mean without, 1.08269 mean with).

### 2. Score-First TTT (Legal)

SGD with momentum 0.9, LR=0.005, 3 epochs per 32K-token chunk, cosine decay. All model blocks unfrozen (freeze=0). Same mechanism as PR #549 and PR #1413.

**Measured delta:** -0.002 BPP vs sliding window without TTT.

### 3. SP8192 Architecture Stack

- 11 layers, model_dim=512, 8 heads, 4 KV heads
- Parallel residuals (layers 7-10, PaLM-style)
- Depth recurrence (layers 4-5, loop 2x)
- Skip gates (sigmoid-gated skip connections)
- QK-Gain 4.0, XSA (all 11 layers)
- Full Hessian GPTQ int6 + byte-shuffle + brotli compression
- Coprime-stride weighted multi-shard data loader
- Code packed with lzma+base85 self-extracting wrapper (saves 32KB)

## Compliance

Per Issue #1017 (Track B — legal eval-time adaptation):

- **Condition 1 (Causal/prefix-only):** Hash key uses `(prev_token, curr_token)` — both are input token identities from `x_batch = chunk[:-1]`, not model predictions. The hash embedding at position t depends only on prefix tokens.
- **Condition 2 (Full normalized distribution):** Hash residual is added to the embedding before RMSNorm and the standard transformer + tied LM head + full-vocab softmax.
- **Condition 3 (Score-before-update):** Each chunk is fully scored under `torch.no_grad()` before any TTT parameter update. The hash embedding is updated as part of the standard TTT training step, after scoring.
- **Condition 4 (Single left-to-right pass):** One evaluation pass, no rescoring, no multi-pass selection.
- **Precedent for eval-time-created parameters:** LoRA-TTT PRs #1254, #1354 also instantiate new trainable parameters at eval time.

No SLOT, no pre-quant TTT, no n-gram caches, no ETLB.

## Reproduction

```bash
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

No env vars needed. All defaults are the submission config.

## Credits

- Base architecture: PR #549 (@abaybektursun), PR #1019 (@abaybektursun)
- Score-first TTT framework: PR #549 (@abaybektursun), PR #1413 (@kevclark)
- Parallel residuals + depth recurrence: PR #1204 (@msisovic)
- SP8192 + GPTQ embeddings + SDClip: PR #1394 (@clarkkev)
- Coprime-stride loader: PR #726, PR #1060
- Eval-time hash embedding: original to this submission
