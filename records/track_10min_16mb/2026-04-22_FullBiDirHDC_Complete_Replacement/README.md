# FullBiDirHDC Complete Replacement — Bilateral Joint Manifold Engine

> **Complete replacement** of the hash-addressed NMF + `DirectionalSemanticVec` pipeline
> with the `FullBiDirHDC` joint manifold engine from the ARC-AGI-3 submission.

---

## Results (to be filled after leaderboard runs)

| Run | Seed | BPB | Val Loss | Time | Artifact bytes | Size check |
|-----|------|-----|----------|------|----------------|------------|
| 1 | 42 | TBD | TBD | TBD | TBD | TBD |
| 2 | 7 | TBD | TBD | TBD | TBD | TBD |
| 3 | 1337 | TBD | TBD | TBD | TBD | TBD |

**Mean BPB:** TBD · **Mean Val Loss:** TBD · **Mean Time:** TBD

---

## Architecture

This submission completely replaces the prior pipeline with the `FullBiDirHDC` joint manifold engine:

### What was replaced

| Old component | New component |
|---|---|
| Rolling XOR hash `G[p]` (loses token identity) | `GoldenAxisShift` codebook (metric-preserving, lossless) |
| `DirectionalSemanticVec` (forward-only `sem_fwd`) | `SpiralDSVLanguageModel` (bilateral `sem_fwd` + `sem_bwd`) |
| Hash-addressed NMF (1-iteration, secondary signal) | `RelationshipMemory` (resonance-weighted rule learning) |
| Fingerprint 8-bit collision detection | `consistency = cosine(fwd, bwd)` bilateral agreement gate |
| No temporal signal | `ZSignal` predictive coding spine |
| No memory | `SpiralPointerMemory` hierarchical memory |
| `.hgz` artifact (16 MB embed + W_out) | `.bdhgz` artifact (~4 MB codebook + rule_bundle) |

### New pipeline

```
Training data (500M tokens from fineweb_train_*.bin)
        │
        ├── Phase 1: Build bigram freq table P[a,b] = P(b|a)
        │   O(N) pass, rank 0 only, ~2s
        │
        ├── Phase 2: Distributed training
        │   Each rank processes N/8 tokens
        │   FullBiDirHDC.train_on_tokens()
        │   reward = bigram_freq[prev_t, next_t] × 100
        │   dist.all_reduce(SUM) on rule_bundle
        │
        ├── Phase 3: SpiralDSV bilateral build (optional)
        │   GoldenAxisShift codebook
        │   sem_fwd + sem_bwd XOR-bundle tables
        │   ctx_len=4, remaining time budget
        │
        └── save_bidi_artifact() → .bdhgz (LZMA9)

Eval waterfall (bidi_bpb):
        For each position p with prev_token t:
            query_hv = codebook[t] XOR rule_bundle
            fwd_scores[v] = cosine(query_hv, codebook[v])  for all v
            bwd_scores[v] = cosine(codebook[v] XOR rule_bundle, codebook[t])
            consistency[v] = (fwd_scores[v] + bwd_scores[v]) / 2
            probs = softmax(consistency)
            p_correct = probs[target_token]
        BPB = Σ(-log₂ p_correct) / Σ(utf8_bytes(token))
```

### Why bilateral is superior

| | Existing pipeline | This submission |
|---|---|---|
| **Prediction direction** | Forward-only `sem_fwd` | Joint fwd+bwd — `consistency = cosine(fwd, bwd)` |
| **Context encoding** | Rolling XOR hash (lossy) | `GoldenAxisShift` (metric-preserving, lossless) |
| **Rule learning** | None (stateless eval) | `RelationshipMemory` resonance-weighted EMA |
| **Temporal signal** | None | `ZSignal` predictive coding spine |
| **Confidence gate** | 8-bit fingerprint only | Full bilateral agreement score |

---

## BPB Formula Verification

The BPB formula is **identical** to the reference [`train_gpt.py`](../../../train_gpt.py):

```python
# Reference (train_gpt.py:275-278):
bits_per_token = val_loss.item() / math.log(2.0)
tokens_per_byte = val_token_count.item() / val_byte_count.item()
return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# This submission (_bidi_train.py):
total_bits  += float(-np.log2(np.clip(p_correct, 1e-30, 1.0)).sum())
total_bytes += int(tok_bytes.sum())
return float(total_bits / total_bytes), float(total_nats / total_toks)
# Algebraically identical: (Σ bits_i / N) × (N / Σ bytes_i) = Σ bits_i / Σ bytes_i
```

The `[BiDirHDC BPB audit]` block printed at the end of each run shows:
- `total_tokens` — number of val tokens evaluated
- `total_utf8_bytes` — sum of UTF-8 byte lengths
- `avg bytes/token` — `total_bytes / total_tokens`
- `bits/token` — `total_bits / total_tokens`
- `nats/token (loss)` — `total_nats / total_tokens`
- `BPB = bits/token / bytes/token` — the final metric

---

## Setup

```bash
# Install dependencies (run from repo root)
pip install -r records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/requirements.txt

# Download and tokenise FineWeb data (once; run from repo root)
python data/cached_challenge_fineweb.py --variant sp1024
```

---

## Run Commands — 3 Independent Runs (Leaderboard Verification)

Each run uses a **different seed** to provide statistical variance across the 3 required independent runs.
All configuration is passed via environment variables — no extra flags required.

```bash
cd /workspace/parameter-golf

# Run 1 (seed 42):
RUN_ID=bidi_hdc N_WORDS=512 SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_gpt.py \
    2>&1 | tee records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_seed42.log

# Run 2 (seed 7):
RUN_ID=bidi_hdc N_WORDS=512 SEED=7 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_gpt.py \
    2>&1 | tee records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_seed7.log

# Run 3 (seed 1337):
RUN_ID=bidi_hdc N_WORDS=512 SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_gpt.py \
    2>&1 | tee records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_seed1337.log
```

Convenience — run all 3 seeds sequentially:
# Set for 8xH100s in Runpod as a copy and paste from the workspace directory in the terminal. 
```bash
cd /workspace/parameter-golf-hdc-main/records && for seed in 42 7 1337; do
  echo "=== Starting seed $seed ===" && \
  RUN_ID=bidi_hdc N_WORDS=512 SEED=$seed \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 \
      records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_gpt.py \
      2>&1 | tee records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_seed${seed}.log && \
  echo "=== Completed seed $seed ==="
done

```

---

## Key Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `N_WORDS` | `512` | HV width in uint64 words (512 → 32,768 bits) |
| `SEED` | `42` | Single random seed for this run |
| `MAX_WALLCLOCK_SECONDS` | `600` | Training time cap |
| `DATA_PATH` | `../../../data/datasets/fineweb10B_sp1024` | FineWeb data directory |
| `TOKENIZER_PATH` | `../../../data/tokenizers/fineweb_1024_bpe.model` | SentencePiece model |
| `VOCAB_SIZE` | `1024` | Vocabulary size |

**Budget identity:** `n_words × vocab_size × 8 bytes × 3 (codebook + fwd + bwd) ≤ 16 MB`

| `n_words` | HV bits | Artifact raw | LZMA9 est. |
|---|---|---|---|
| 16 | 1,024 | ~400 KB | ~120 KB |
| 128 | 8,192 | ~3.1 MB | ~1 MB |
| **512** | **32,768** | **~12.6 MB** | **~4 MB** |
| 1024 | 65,536 | ~25.2 MB | ~8 MB |

---

## Files

| File | Role |
|---|---|
| [`train_gpt.py`](train_gpt.py) | Main entry point (`--bidi_hdc` flag, distributed init, token loading, training, eval, auto-generates `submission.json`) |
| [`_bidi_hdc_engine.py`](_bidi_hdc_engine.py) | `FullBiDirHDC` + `Codebook` + `ManifoldAxes` + `ZSignal` + `ResonanceSignal` + `RelationshipMemory` + `ChainManifold` (adapted from `bidi_hdc_full.py`) |
| [`_bidi_train.py`](_bidi_train.py) | `build_bigram_freq()`, `train_bidi_model()`, `bidi_bpb()`, `save_bidi_artifact()`, `load_bidi_artifact()` |
| [`_spiral_dsv_lm.py`](_spiral_dsv_lm.py) | `GoldenAxisShift` + `SpiralPointerMemory` + `SpiralDSVLanguageModel` (adapted from `_spiral_dsv.py`) |
| [`requirements.txt`](requirements.txt) | `numpy`, `torch`, `sentencepiece`, `cupy-cuda12x`, `zstandard` |

---

## Submission Checklist

- [ ] 3 independent run logs (`train_seed42.log`, `train_seed7.log`, `train_seed1337.log`)
- [ ] Each log shows `Artifact size check: PASS (limit: 16,000,000 bytes)`
- [ ] Average BPB across 3 runs computed and recorded above
- [ ] Each run launched via `torchrun --standalone --nproc_per_node=8` on 8×H100 SXM
- [ ] Each run completes in under 10 minutes
- [ ] `submission.json` updated with actual `val_bpb`, `val_loss`, `artifact_bytes`, `elapsed_s`
- [ ] No validation data accessed during training (pipeline reads only `fineweb_train_*.bin`)
