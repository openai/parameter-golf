# Record: BackoffNgramMixer + Drift-Free TTT (3-seed mean val_bpb=0.6683)

**val_bpb: 0.6683** (3-seed mean, std 0.0024) | **<16 MB** | 8xH100 SXM, 600s

## Results (8xH100 80GB SXM)

| Seed | Pre-TTT bpb | Post-TTT bpb | Eval time | Artifact |
|------|-------------|--------------|-----------|----------|
| 1337 | 1.1258 | **0.6663** | 371s | 15.63 MB |
| 42 | 1.1258 | **0.6710** | 371s | 15.78 MB |
| 2024 | 1.1258 | **0.6675** | 372s | 15.48 MB |
| **Mean** | 1.1258 | **0.6683** | 371s | |
| **Std** | | **0.0024** | | |

## Background

We introduced the first n-gram eval cache in this competition (PR #659, val_bpb=1.0920, March 22 2026). That original approach used a 5-gram cache with fixed mixing and an oracle safety gate that was subsequently ruled illegal by organizers (comparing mixed vs original NLL peeks at the target).

This submission replaces the illegal oracle gate with entropy-adaptive mixing and multi-order backoff, combined with a drift-free TTT configuration.

## Technique

### 1. Multi-order N-gram Backoff (orders 2-7)

Instead of a single fixed n-gram order, we try the highest order first and cascade down on miss. Each order uses 4M hash buckets to reduce collisions. This dramatically improves coverage: a fixed 7-gram misses when the exact 6-token context has not been seen, but backoff to 6, 5, 4, 3, 2-gram catches those cases.

N-gram counts are accumulated from already-scored tokens only. Updated after scoring each chunk.

### 2. Entropy-Adaptive Alpha
```
alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))
```

where H is the neural model's own entropy over its output distribution. When the model is uncertain (high entropy), we trust n-gram statistics more. When confident (low entropy), we trust the model. This depends solely on the model's output distribution, never on the true target. No oracle selection.

The mixed probability is always applied:
```
p_mixed = (1 - alpha) * p_neural + alpha * p_ngram
```

### 3. Drift-Free TTT Configuration

Standard TTT configurations suffer from late-chunk drift: BPB bottoms around chunk 21 then climbs as cumulative adaptation becomes destructive. We use a conservative configuration that produces monotonic improvement through all 60 chunks:

| Parameter | Setting |
|-----------|---------|
| Unfrozen params | Q projections only (QTTT=1) |
| Mixer eta | 0.02 |
| TTT LR | 0.00003 |
| Chunk size | 1M tokens (60 chunks) |
| Epochs per chunk | 1 |
| Adaptive LR | Disabled |
| Polyak averaging | Disabled |

The most impactful hyperparameters are mixer eta and TTT learning rate. Reducing eta from 0.1 to 0.02 prevents expert weight runaway. Reducing TTT LR from 1e-4 to 3e-5 prevents destructive late-chunk weight updates. Together these eliminate the drift pattern entirely: BPB drops monotonically from 1.15 at chunk 1 to 0.67 at chunk 60, never reversing.

## Ablation

| Configuration | val_bpb | Delta |
|---------------|---------|-------|
| Base model (no mixer, no TTT) | 1.1363 | baseline |
| TTT only (no mixer) | 1.1369 | -0.000 |
| Mixer only (no TTT) | 0.6712 | -0.465 |
| **Full system** | **0.6663** | **-0.470** |

The ablation is unambiguous: the BackoffNgramMixer is the dominant innovation, contributing 99% of the total improvement (-0.465 of -0.470 BPB). TTT alone with drift-free settings contributes essentially nothing in isolation. When combined with the mixer, TTT adds a marginal 0.005 BPB through slightly improved base predictions that the entropy-adaptive alpha can exploit.

The practical implication: the n-gram backoff with entropy-adaptive mixing is a general technique applicable to any language model evaluation. It does not require TTT, architectural changes, or retraining. It is a pure eval-time improvement that treats BPB as a compression problem and applies adaptive compression statistics from already-scored tokens.

## Compliance

- **Score-first TTT:** Each chunk scored under `torch.inference_mode()` before any training on that chunk
- **Backward-looking n-gram:** Counts from already-scored tokens only, updated after scoring
- **No oracle selection:** Alpha depends on model entropy, never compares mixed vs original NLL
- **No training data at eval:** Naive int5 per-row quantization only. No Hessian calibration, no training data access during eval
- **Token count verified:** ratio_scored = 1.000000 (window-start fix applied)
- **No cross-GPU n-gram sync:** Each GPU maintains independent cache

## Reproduction
```bash
pip install zstandard
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
USE_MIXER=1 MIXER_ETA=0.02 \
QTTT=1 TTT_EPOCHS=1 TTT_FREEZE_BLOCKS=1 TTT_LR=0.00003 \
TTT_CHUNK_TOKENS=1048576 ADAPTIVE_LR=0 USE_POLYAK=0 \
EVAL_STRIDE=64 CROWN_Q_LAMBDA=0.01 PRUNE_PCT=0.08 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Architecture

11L, 512d, GQA 8H/4KV, MLP 3x, LeakyReLU(0.5)^2, XSA all 11 layers, Value Residual, Gated Attention, SmearGate, BigramHash(4096), Partial RoPE(16/64), LN Scale, EMA(0.997). Tied embeddings. Muon optimizer. ~5850 steps in 600s.

## Credits

- **PR #700 RoyiRa** - Base architecture, TTT framework, stride=64 eval
- **PR #606 gowtham0992** - int5 + Soft-Round QAT model
- **PR #727 Asukabot0** - Multi-order backoff concept, entropy-adaptive alpha formula
- **PR #461 Christopher-Lee-McClendon** - TTT recipe foundations
- **PR #518 sofiabod** - LeakyReLU(0.5)^2, cosine TTT scheduling
- **Dean Barr (this author)** - Original n-gram eval cache concept (first in competition, PR #659), drift-free TTT discovery, backoff+TTT combination, BackoffNgramMixer implementation
