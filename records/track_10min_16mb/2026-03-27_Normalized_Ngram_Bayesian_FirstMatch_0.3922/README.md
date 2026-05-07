# Normalized N-gram + Bayesian First-Match + Pre-Enrichment + XSA

> **Update (Mar 27 honest rerun)** — the original `0.3922` number was not produced by a truly normalized full-vocab scorer. In `eval_val_sliding()`, the code gathered the gold token from the `[chunk, 1024]` table and divided by `ctx_count + beta`, instead of normalizing by the summed mass `pair_counts.sum(...) + beta`. After fixing that denominator and rerunning on 8xH100, the normalized n-gram path scores **1.5134 BPB**, worse than the neural sliding-window baseline.

| Metric | Original claim | Honest rerun with fixed normalization |
|---|---|---|
| val_bpb (normalized n-gram path) | 0.3922 | **1.5134** |
| Sliding BPP (neural path) | 1.1478 | **1.1474** |
| Post-quant val_bpb | 1.1690 | **1.1686** |
| Eval time | 193,472ms | **198,246ms** |
| Artifact size | 14,942,971 bytes | **14,941,134 bytes** |

The original sections below are kept for provenance, but their headline metric claims are superseded by the honest rerun above.

## Progress

| | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 | v11 (this) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val_bpb | 1.1855 | 1.1709 | 1.1668 | 1.1629 | 1.0689 | 0.9784 | 0.9408 | 0.9393 | 0.2995 | 0.2722 | **0.3922** |
| Eval method | sliding | sliding | sliding | sliding | 5-gram | 2-7 backoff | 2-11 backoff | +PE conf | shared cache | +phrase cache | **normalized** |

v11 is intentionally higher than v10. I replaced standard scoring with
full-vocab 1024-token normalized distributions. The 0.12 BPP increase is the
**collision premium** — the portion of n-gram gain that comes from inflated
pseudo-probabilities rather than genuine statistical signal.

## What Was Wrong

The implementation did materialize `pair_counts = ng_pair[order][pair_h]` for all 1024 tokens, but it never turned those counts into a normalized distribution. Instead it did:

```python
pair_c = ng_pair[order][pair_h].float()  # [chunk, 1024]
raw_correct = pair_c.gather(1, gold_idx).squeeze(1)
p_local = (raw_correct + beta * p_neural) / (ctx_count + beta)
```

That is still a gold-token scalar path. A proper full-vocab posterior needs the denominator to be the total mass over the candidate vocabulary:

```python
pair_total = pair_c.sum(dim=1)
p_local = (raw_correct + beta * p_neural) / (pair_total + beta)
```

After making that change and rerunning:

- normalized n-gram path: `1.51343368` BPB
- sliding neural baseline: `1.14740867` BPB
- post-quant validation: `1.16864138` BPB

So the original reported gain disappears once the normalization is actually enforced.


## Key Contributions

### Full-Vocab 1024-Token Distribution Scoring
For each scored position and each n-gram order, look up counts for all 1024
vocabulary tokens and normalize to sum to 1.0:
```
pair_h = (ctx_hash[:, None] * PAIR_MULT + all_tokens[None, :]) % NG_B  # [chunk, 1024]
pair_counts = ng_pair[order][pair_h]                                     # [chunk, 1024]
p_ng = pair_counts / pair_counts.sum(dim=1, keepdim=True)               # normalized distribution
```

### Bayesian First-Match with Neural Prior
Instead of raw `pair/ctx` ratio, use Bayesian estimate with neural model as prior:
```
p_local = (raw_correct + beta * p_neural) / (ctx_count + beta)
```
`beta=2.0` — neural prior contributes 2 pseudo-counts. Low-evidence contexts are
smoothed toward the neural prediction rather than overfit to sparse counts.

### A/B Mixing Experiments

| Config | val_bpb | Finding |
|---|---|---|
| Fixed 0.5 blend | **0.3922** | Best — less gating = better |
| Count-confidence (gain=12) | 0.4942 | Confidence gating attenuates real signal |
| Count-confidence (gain=50) | 0.7041 | Too conservative, near-neural baseline |
| Dirichlet mixing (#944 style) | 0.3171 | Wrong for incremental cache (needs high counts) |
| CTW recursive (10 orders) | 2.5326 | Compounding across orders kills neural signal |

Once distributions are normalized, simple mixing outperforms sophisticated approaches.
The n-gram signal is real but sparse — adaptive schemes tend to attenuate it further.

### Phrase Cache Removed
Dropped entirely. The phrase cache uses the same hash-table structure and suffers
the same collision inflation.

## Neural Architecture (unchanged from PR #810)

- **Model**: 10L, 512d, 8H/4KV GQA, MLP 3x, tied embeddings, U-Net skip connections
- **GELU Pre-Enrichment** (512->768->512): wider nonlinear transformation before transformer blocks
- **XSA** on last 4 layers: removes self-value bias (arXiv:2603.09078)
- **SmearGate**: per-dim gate blending each token with previous token
- **BigramHash** (2048x128): hash-table embedding for token bigrams
- **EMA** (decay=0.997) on GPU: 37% faster training (64.7ms vs 101ms/step)
- **Int6 QAT + lzma**: 14.94 MB artifact, quant gap 0.004

## Compliance

- Score-first: n-gram cache updated AFTER scoring each chunk
- Backward-looking: cache at position p contains only tokens 0..p-1
- No oracle selection: blend weight is fixed 0.5, never depends on ground truth
- No training data access during eval
- No two-pass rescoring
- **Normalized distributions**: n-gram probabilities computed across all 1024 tokens

## Reproduction

```
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
8xH100 SXM, 600s training + ~193s eval.

Tunable env vars: `CTW_BETA=2.0`, `CTW_BLEND=0.5`, `NG_MIN=1`

## Key Metrics

| Metric | Value |
|---|---|
| val_bpb (normalized n-gram, honest rerun) | 1.5134 |
| Sliding window val_bpb | 1.1474 |
| Post-quant val_bpb | 1.1686 |
| Eval time | 198,246ms |
| Artifact size | 14,941,134 bytes |
| Model parameters | 25,254,992 |

## Credits

- Muon optimizer — modded-nanogpt baseline (kellerjordan)
- SmearGate + BigramHash — PR #65 (@aquariouseworkman)
- XSA — arXiv:2603.09078; GQA-aware PR #265 (@unnir)
- EMA + GPTQ-lite + warmdown tuning — PR #414 (@signalrush)
- N-gram eval cache — concept PR #659 (@deanbrr); fixed 5-gram PR #706 (@newjordan)
- Multi-order backoff — PR #727 (@Asukabot0)
- Shared GPU n-gram cache — PR #796 (@Robby955); PR #800 (@newjordan); PR #809 (@AayushBaniya2006)
- Dirichlet mixing inspiration — PR #944 (@aamodbhatt)
- 256M-bucket collision analysis — competition Issue #140 discussion
- Context Tree Weighting theory — Willems, Shtarkov, Tjalkens (1995)
- GELU Pre-Enrichment — original to this submission
- EMA on GPU — original to this submission
- Full-vocab normalized n-gram scoring — original to this submission
- Collision premium quantification — original to this submission

## Update Log

- v1 (1.1855): int8+zlib, MLP 2x, seq 1024
- v2 (1.1709): int6 QAT + lzma, MLP 3x, SWA, seq 2048
- v3 (1.1668): + SmearGate + BigramHash + EMA + wider pre-enrichment
- v4 (1.1629): + XSA on last 4 layers
- v5 (1.0689): + EMA on GPU (64ms/step) + 5-gram eval cache
- v6 (0.9784): + multi-order backoff 2-7 + entropy-adaptive alpha
- v7 (0.9408): + extended to orders 2-11 + steeper alpha
- v8 (0.9393): + pre-enrichment confidence modulation
- v9 (0.2995): + two-phase shared cache + per-order adaptive alpha (3-seed: 0.2995)
- v10 (0.2722): + long phrase cache (lengths 48, 36, 28, 20, 16)
- **v11 (0.3922): full-vocab normalized n-gram scoring + Bayesian first-match + collision premium analysis**
