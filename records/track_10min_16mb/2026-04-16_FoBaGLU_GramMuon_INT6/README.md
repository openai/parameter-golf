# FoBa-GLU + GramMuon + INT6 QAT + Sliding Window Eval

**Author:** Ajinkya Mulay (`ajinkyamulay`)
**Track:** `records/track_10min_16mb`
**Target val_bpb:** < 1.08

## Summary

This submission combines a novel sparse MLP activation (FoBa-GLU) with
Gram-corrected Muon optimization, INT6 quantization-aware training, and
sliding-window evaluation on 11 layers with 3× MLP expansion.

The key novel contribution is **FoBa-GLU**: an MLP block that replaces
ReLU² with a sparse pursuit-gated activation inspired by Forward-Backward
Pursuit (FoBa) from compressed sensing. The top-k gate selection creates
structured sparsity that improves zlib compressibility, while the
block-diagonal initialization makes weights compressible from training step 0.

## Architecture

| Parameter | Value |
|---|---|
| Layers | 11 |
| Model dim | 512 |
| Heads | 8 (4 KV, GQA) |
| MLP mult | 3× (FoBa-GLU, top-k=50%) |
| Vocab | 1024 (SP1024) |
| Pos. emb. | RoPE |
| Norm | RMSNorm (no learnable scale) |
| Tied emb. | Yes (std=0.005) |
| Logit cap | 30.0 |
| Seq len | 4096 |

Total parameters: ~26M (11 independent layers, no weight tying)

## Techniques

### FoBa-GLU (novel)
Replaces the baseline ReLU²-MLP with a sparse gated MLP:
- Gate projection computes neuron relevance scores
- Top-k selection (k = 50% of hidden dim) keeps only the most relevant neurons
- Selected neurons get SiLU activation, rest are zeroed (structured sparsity)
- Block-diagonal sparse initialization clusters weights for better zlib compression
- Motivation: OMP/FoBa atom selection → each hidden neuron is an "atom" in a learned dictionary

### GramMuon (novel variant)
Extension of the Muon optimizer with Gram correction:
- Newton-Schulz orthogonalization normalized by Frobenius norm + epsilon guard
- Prevents NaN divergence for small-magnitude weight matrices (e.g. tied embeddings)
- Embedding table routed to SGD path (not NS), avoiding the orthogonalization mismatch
- WD=0.04 on matrix parameters, no WD on scalars/embeddings

### INT6 QAT with straight-through estimator
- After step 100: round all weight matrices to 64-level INT6 grid
- Storage: INT8 bytes, but only multiples of 4 → 64 distinct values vs 256
- This creates regularity that zlib can exploit: significantly smaller compressed artifact
- Per-tensor GPTQ-lite clipping at 15% extreme values before quantizing
- STE gradient: backward flows through unquantized values

### Sliding window evaluation (stride=64)
- Each validation token is evaluated with up to 4096 tokens of context
- Dramatic improvement over fixed-window eval, especially for later tokens
- Zero artifact cost — purely an inference strategy

### EMA weight averaging
- Exponential moving average (decay=0.999) starting at step 2000
- EMA weights used for all validation and final eval
- Cleaner loss surface vs SWA (no bf16 accumulation precision issues)

### Other improvements over baseline
- 11 layers (depth > width per leaderboard analysis)
- 3× MLP multiplier (more FFN capacity per param)
- seq_len=4096 (4× more context per training step)
- Warmdown schedule (linear decay to ~0 LR in final 3500 steps)
- Encoder-decoder skip (resid_mix) in every block

## Compression Budget

With INT6 + zlib-9 on a 11L 512d 3×MLP model:
- ~26M parameters × 1 byte/param = 26MB raw
- zlib compression at ~45% = ~14.3MB model
- Code: ~30KB
- Total: ~14.33MB (well under 16MB limit)

## Training Setup

```bash
# Download data (SP1024 tokenizer)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Run (8×H100, 10 minutes)
RUN_ID=foba_glu_grammuon_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
N_LAYERS=11 \
MLP_MULT=3.0 \
FOBA_K_RATIO=0.5 \
TRAIN_SEQ_LEN=4096 \
WEIGHT_DECAY=0.04 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-16_FoBaGLU_GramMuon_INT6/train_gpt.py
```

## Results

<!-- Fill in after RunPod run — show 3-seed mean and std -->

| Seed | Pre-quant val_bpb | Post-INT6 val_bpb |
|------|-------------------|-------------------|
| 1    | TBD               | TBD               |
| 2    | TBD               | TBD               |
| 3    | TBD               | TBD               |
| Mean | TBD               | TBD               |
| Std  | TBD               | TBD               |

## Connection to Research Background

The FoBa-GLU activation is directly motivated by the author's PhD work on
Forward-Backward Pursuit (FoBa) for sparse recovery. The top-k gate selection
mirrors FoBa's forward atom selection step: given an input x, identify the k
"atoms" (neurons) most correlated with the residual signal and activate only
those. The structured sparsity this creates has a natural connection to the
Restricted Isometry Property — sparse activations preserve geometric distances
in the embedding space, which should benefit downstream representation quality.

The GramMuon optimizer extends Muon with a Gram correction factor motivated
by the same compressed sensing theory: the Newton-Schulz iteration approximates
the polar factor of the gradient matrix, but needs proper spectral normalization
to converge stably when applied to varied weight matrix sizes.

## Negative Results / What Didn't Work

- **ALBERT-style layer tying (3 unique / 11 total):** Reduced parameter count
  but hurt val_bpb by ~0.01 without the benefit of fitting more layers. Dropped.
- **FoBa k_ratio < 0.4:** Too sparse — gradient flow becomes too narrow and
  training slows significantly without quality gain.
