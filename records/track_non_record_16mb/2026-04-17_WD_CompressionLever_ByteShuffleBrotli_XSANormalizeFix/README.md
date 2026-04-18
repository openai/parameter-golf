# Non-Record: Three Findings on the Compression Side of the Pipeline

**Author:** Victor Fiz ([@Bananakin1](https://github.com/Bananakin1))
**Hardware:** single RTX 3070 (8 GB VRAM), WSL2
**Local best:** 1.1470 int6 bpb (sliding-window stride=64) / 1.1812 standard eval — 7000 steps, 10 shards
**Artifact:** 15.96 MB (under the 16 MB cap)
**Track:** non-record — local-only numbers, not yet validated on 8×H100

---

## Why this is a non-record

I do not have access to 8×H100. Everything below was developed and measured on a single RTX 3070, using a 10-shard slice of FineWeb. The local numbers land in the SOTA neighborhood (current merged record: 1.1147 bpb), but the dataset is small enough that each token is seen ~35 times during training, which produces high-entropy memorized weights that compress poorly. So the score gap to SOTA reflects compression headroom lost to memorization, not architectural shortcomings — at least, that's the working hypothesis.

I'm submitting this as a non-record to document **one verified finding** and **two hypotheses with single-point evidence**, all of which should transfer to a real H100 run. I'd rather be honest about the evidence I have than padded.

## TL;DR

1. **Verified — XSA `F.rms_norm` over-subtracts by `head_dim`.** Several R3 experiments (and at least one prior PR baseline I forked from) used `F.rms_norm(v, (head_dim,))` where `F.normalize(v, dim=-1)` was intended. With `head_dim=64`, the bug causes XSA to subtract 64× the correct projection. Verified in code: see `_xsa` at line 733 of `train_gpt.py` for the corrected version. This is the only finding I can prove with the artifacts in this submission.

2. **Single-point evidence — `byte-shuffle(stride=2) + brotli q=11` produced a smaller artifact than LZMA-9 on the exp15 checkpoint.** ~500 KB delta on one checkpoint. I have not measured it across multiple checkpoints, so this could be checkpoint-specific. The serializer prefixes a 4-byte `BSHF` magic so eval-side deserialization can branch correctly.

3. **Hypothesis only — Muon weight decay as a compression-aware knob.** Across exp13/14/15 I bumped `muon_wd` from 0.04 → 0.08 → 0.085 alongside other architectural changes. Each step trended toward a smaller artifact, but every step also changed a second variable (bigram-int6 quantization, byte-shuffle, MLP4×). I do not have a clean WD-only ablation yet. I'm including it here as a knob worth ablating, not as a measured result.

I'd rather list these honestly at three different evidence levels than present them as a uniform table. If reviewers want to dismiss (3) until I have isolated data, that's correct. (1) and (2) are the actionable items.

---

## 1. The XSA `F.rms_norm` vs `F.normalize` bug (verified)

XSA (cross-subtraction attention) removes the projection of the attention output `y` onto a normalized value vector `v_norm`. The projection-removal math requires `v_norm` to be a **unit** vector:

```
dot = (y · v_norm).sum(dim=-1)        # scalar projection
y   = y - dot * v_norm                # remove that component
```

Both the dot product and the subtraction multiply by `v_norm`. If `v_norm` is off by a constant factor `c`, the operation removes `c²` times the intended projection.

### The two functions

- `F.normalize(v, dim=-1)` returns `v / sqrt(sum(v_i^2))` → magnitude **1**.
- `F.rms_norm(v, (D,))` returns `v / sqrt(sum(v_i^2) / D + eps)` → magnitude **sqrt(D)**.

So substituting `rms_norm` for `normalize` makes `v_norm` a factor of `sqrt(D)` too large, and the XSA correction subtracts `D × (correct projection)` instead of `1 × (correct projection)`. With `head_dim = 64`, that's a 64× over-subtraction.

### Where it shows up in this repo

```python
# experiments/r3/exp9_full_7000.py — buggy version
v_norm = F.rms_norm(v_expanded, (v_expanded.size(-1),))

# experiments/r3/exp15_qkgain5_mlp4x.py — fixed version (line 743)
v_norm = F.normalize(v_expanded, dim=-1)
```

Same `_xsa` shape, same input layout. The change is a one-liner; the effect is whether XSA is doing roughly what the math says or doing something orders of magnitude larger.

### Why this matters for ablations

In R3 exp1 I tried "XSA on all layers + QK gain 4.0" and the run blew up (1.65 bpb at 500 steps vs ~1.40 at the same step count for the baseline). At the time I attributed the failure to "XSA-all + high QK gain" being incompatible. After the fix, XSA-all alone runs cleanly. The earlier failure was the bug interacting with the QK gain change, not the architectural combination — two unrelated changes hidden behind one observable failure. Anyone running an "XSA layer count" ablation against a `rms_norm`-based baseline is comparing two configs that differ in an unintended dimension.

If anyone in the competition has used `F.rms_norm` here in a merged or cited PR, I think it's worth flagging — please tell me and I'll credit / cross-link in a v2.

---

## 2. Byte-shuffle (stride=2) + brotli q=11 — single observation

The serialization for the exp15 checkpoint went through a whole-blob byte-shuffle (stride=2) followed by `brotli.compress(..., quality=11)` and produced a **15.96 MB** artifact. The same checkpoint compressed with `lzma.compress(..., preset=9)` produced a larger artifact (~500 KB more) on that one comparison.

### Why I'm not claiming a table

I only have the one checkpoint's worth of data on this. I have not:

- measured zlib/zstd/brotli/LZMA on multiple checkpoints from different runs,
- separated the contribution of the byte-shuffle from the contribution of brotli vs LZMA,
- tested how the win scales with weight entropy (I expect LZMA to catch up at low entropy).

### What I do have

- The serializer in `train_gpt.py` (search for `_byte_shuffle` and `BSHF` magic).
- The fallback path: if `brotli` is not importable, the artifact falls back to `lzma.compress(..., preset=9)` without the shuffle — verified by toggling the import.
- The output of one run on one checkpoint, where the brotli+shuffle path was smaller.

If you're choosing a compressor for your own submission, I'd suggest measuring both on your checkpoint rather than trusting one data point of mine.

---

## 3. Muon WD as a possible compression knob (hypothesis)

The `train_gpt.py` here uses `muon_wd = 0.085`. Across the three experiments leading up to it:

| Experiment | `muon_wd` | Other changes (vs previous row) | Artifact (MB) | int6 bpb (3070, ~7K steps, 10 shards) |
|------------|-----------|---------------------------------|---------------|---------------------------------------|
| exp13      | 0.04      | + bigram-int6 quantization      | (not recorded in the commit message) | not recorded |
| exp14      | 0.08      | + byte-shuffle + brotli q=11    | 13.94         | not recorded |
| exp15      | 0.085     | + MLP4×, + QK-gain 5, + slide   | 15.96         | 1.1812 (1.1470 slide-64) |

Each row changes more than `muon_wd`, so this is not a clean WD ablation. Reading across the three rows, the artifact size moved with WD-and-other-things in the direction I'd expect from theory (higher WD → bounded weight RMS → cleaner int6 → smaller compressed payload), but I can't separate the WD contribution from the others.

What I'd want to run on H100, in priority order:

1. Hold all of exp15 fixed; sweep `muon_wd ∈ {0.0, 0.04, 0.08, 0.085, 0.12}` for ≥7K steps; measure both bpb and artifact size.
2. Same sweep at 20K steps under the official compute budget — WD often interacts with longer schedules and EMA differently than at short ones.
3. Re-quantize the same checkpoint with multiple compressors (zlib-9, zstd-22, LZMA-9, brotli q=11, brotli+shuffle) — the compressor comparison should be a one-shot post-training measurement, much cheaper than the WD sweep.

## How the three findings compose

- (1) is a **correctness change** at training time. It changes what your model is.
- (2) is a **post-training change**. Worth a try; verify on your checkpoint first.
- (3) is a **knob** at training time. Worth ablating before committing.

All three are low-conflict with the standard SOTA stack (XSA-all, EMA/BEMA, BigramHash, SmearGate, partial RoPE, LeakyReLU², linear warmdown, GPTQ-lite). The four-way ablation is the obvious next step on H100.

## Architecture used to produce these numbers

Defined in `train_gpt.py`. Key knobs (from the script's environment-variable defaults):

```
NUM_LAYERS=11  MODEL_DIM=512  MLP_MULT=4
NUM_HEADS=8    NUM_KV_HEADS=4
QK_GAIN_INIT=5.0
ROPE_DIMS=16   ROPE_BASE=10000
BIGRAM_BUCKETS=3072  BIGRAM_DIM=112
TIE_EMBEDDINGS=1
LeakyReLU(0.5).square() activation
XSA on ALL 11 layers (with the F.normalize fix — see `_xsa`)
VRL: V_first blended into all subsequent layers
SmearGate + BigramHash
U-Net skips, OrthoInit
Muon(matrix_lr=0.025, momentum=0.99, wd=0.085) + Adam(0.6 / 0.008 LR groups)
Linear warmdown
Sliding-window eval stride=64
Int6 GPTQ-lite + FP16 embeddings
Byte-shuffle(stride=2) + brotli q=11 (with LZMA-9 fallback)
```

## Reproducing locally (RTX 3070, ~12 hours for 7K steps)

```bash
# Download the data (10 train shards is enough to reproduce the relative numbers)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Train (matching the row in submission.json)
ITERATIONS=7000 \
WARMDOWN_ITERS=2500 \
TRAIN_BATCH_TOKENS=524288 \
python3 train_gpt.py
```

For the H100 setup, override `ITERATIONS=20000 WARMDOWN_ITERS=3500 MAX_WALLCLOCK_SECONDS=600` and run with `torchrun --standalone --nproc_per_node=8 train_gpt.py`.

## What I'd want validated on H100

- Re-derive (1)'s impact at the SOTA scale: same architecture, swap `F.normalize` ↔ `F.rms_norm`, measure bpb.
- Run the multi-compressor measurement on a real H100 checkpoint to confirm or invalidate (2).
- Run the WD-only sweep described above to confirm or invalidate (3).

## Acknowledgments

Built on top of the public competition baseline (`train_gpt.py`) and the architecture lineage of merged record/non-record PRs in this repo. The compression-side framing was prompted by reading PR comment threads on entropy/compression interactions.

The XSA bug was identified during my R3 ablation; if anyone else has noticed it, please mention so I can credit you in a v2 of this writeup.
