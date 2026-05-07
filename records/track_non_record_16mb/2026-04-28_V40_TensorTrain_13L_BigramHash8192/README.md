# V40 Non-Record Submission: Tensor-Train (TT/MPS) Decomposition + 13L + BigramHash 8192

**Final post-quant `val_bpb = 1.16910`** via sliding window (stride=64) evaluation, under the **15.04 MB** artifact (out of 16 MB cap, FITS with ~957 KB headroom). Single-seed run on **8×H100 80GB**, 10-min wallclock cap.

This submission introduces **Tensor-Train (TT/MPS) decomposition** to the leaderboard — no other entry currently uses this technique. TT compresses square `dim×dim` linear layers (here `attn.c_q` and `attn.proj` at d_model=512) by 51× per matrix at rank 8, mode_shape (8,8,8). The freed parameter budget is reinvested into +2 transformer layers (13 vs 11 baseline) and a doubled bigram embedding vocabulary (8192 vs 4096).

## Summary

The baseline V29-stack (11L GQA Transformer with int6 QAT, BigramHash, sliding eval) hit ~1.128 BPB and used ~15.85 MB of the 16 MB artifact budget — leaving little room for architectural growth. V40's TT decomposition frees ~2.83M parameters (per layer: c_q 512×512 → 5K cores instead of 262K weights, same for attn.proj), which we reinvest into:

1. **+2 layers** (13L vs 11L) for more effective depth at constant artifact size
2. **Bigram vocabulary x2** (8192 vs 4096) for finer-grained context-aware token features
3. **Full V34 SafeBundle** — proven leaderboard wins (GPTQ-lite, partial RoPE, layerwise LN scale)

Result: **1.1691 sliding BPB** (single seed, 8×H100, 600s wallclock). Not record-breaking against the SOTA 1.1194, but contributes a **novel architectural technique** (TT/MPS for parameter compression in attention layers) that has not yet appeared on the leaderboard.

## Architecture

### Tensor-Train (TT/MPS) decomposition

For a square dense weight matrix `W ∈ ℝ^{512×512}`, instead of storing 262,144 fp32 parameters (or 196 KB int6 quantized), we represent W as a chain of three 3D cores:

```
W[i₁i₂i₃, j₁j₂j₃] = G₀[1, (i₁, j₁), r₁] · G₁[r₁, (i₂, j₂), r₂] · G₂[r₂, (i₃, j₃), 1]
```

where `(i_t, j_t) ∈ {0..7} × {0..7}` (mode_shape = (8, 8, 8); 8·8·8 = 512), and the bond dimensions are `r₁ = r₂ = 8` (TT_MAX_RANK). The total parameter count per TT-decomposed matrix is `1·64·8 + 8·64·8 + 8·64·1 = 5120` — a **51× compression** vs the original 262,144 weights.

Forward pass materializes W via sequential einsum contraction (sub-millisecond cost per layer), then standard `F.linear`. The TT-cores are exported as `nn.ParameterList` entries in the state_dict (no `.weight` key), avoiding the V32-style overflow bug where naive serialization of the materialized matrix would re-bloat the artifact.

**Initialization**: TT-SVD from a freshly orthogonal-initialized W (provides good starting point in TT manifold). Alternative random init also supported via `TT_INIT_MODE=random`.

**Optimizer**: Muon Newton-Schulz orthogonalization on 3D cores via reshape-to-2D (`(r_in·m) × (m·r_out)`), then back to 3D after the step. Verified to remain stable through 5608 cloud steps without NaN.

**TT scope**: Applied to `attn.c_q` and `attn.proj` (both square 512×512 with our config). Not applied to `attn.c_k`/`attn.c_v` (rectangular GQA, 512→256 — would need rectangular TT, deferred to v2). Not applied to MLP fc/proj (rectangular 512→1536, also v2).

### Other architectural choices

- **13 transformer blocks** (NUM_LAYERS=13): 2 more than the 11L V29 baseline. Funded by TT-savings on attention.
- **GQA**: 8 query heads, 4 KV heads
- **MLP 3×** (mlp_mult=3): 1536 hidden dim
- **U-Net skip connections** (encoder layers 0..5, decoder layers 6..12 with learnable skip weights)
- **Tied embeddings** (1024 BPE vocab)
- **BigramHashEmbedding** (vocab=8192, dim=128 → projected to 512)
- **SmearGate** (learnable temporal mixing on input embeddings)
- **XSA on last 4 layers** (eXclusive Self-Attention: subtract self-value projection)
- **LeakyReLU(0.5)² activation** in MLP

### V34 Safe Bundle (proven leaderboard wins, all enabled by default)

| Component | Source | Effect |
|---|---|---|
| **GPTQ-lite** | signalrush (1.1228) | Per-row clip-percentile sweep selects scale by min MSE → ~−0.002 BPB |
| **Partial RoPE 16/64** | jfprincz (1.1248) | Rotary positional only on first 16/64 head dims, pass remainder unchanged → ~−0.001 BPB |
| **Layerwise LN scale 1/√(layer+1)** | jfprincz (1.1248) | Per-layer norm output scale → ~−0.001 BPB |
| **EMA + warmdown** | many | Exponential moving average copied at end of warmdown phase |
| **Logit temperature calibration** | standard | Post-train scalar T scan minimizing CE on a small holdout |

### Disabled (incompatible or out-of-scope)

- `TTT_ENABLED=0` — Test-time training not used (avoids potential rule-edge cases combined with TT)
- `PARALLEL_MUON_ENABLED=0` — Parameter banking is incompatible with TTLinear in v1 (3D bank reshape conflict). Compatible variant deferred to v2.
- `RESIDUAL_QUANT_ENABLED=0` — During development we ported V35-style residual int4 cascade and discovered the original V35 implementation had a `clamp_min(1/7)` bug that collapsed q1/q2 residuals to zeros, which compressed via zstd to ~0 bytes but degraded reconstruction quality to single-int4 level. After fixing the bug, real cascade artifacts are 3-5× larger than int6 (uncompressible int4 entropy). The cascade idea is mathematically sound but not practical at this matrix scale. Closed.

## Run command (one-line)

All submission-ready parameters are baked into `train_gpt.py` as Hyperparameters defaults. No env-overrides required.

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

(With `2>&1 | tee logs/v40_final_$(date +%Y-%m-%d_%H%M).log` for log capture.)

## Setup

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install -r requirements.txt
pip install zstandard            # required for the zstd-22 compression step
python data/cached_challenge_fineweb.py --variant sp1024
# overlay this submission's train_gpt.py on top of the repo's
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

```
step:5608/20000 (wallclock cap at 600s)
final_int6_roundtrip   val_loss:2.0106  val_bpb:1.1908
final_int6_sliding(64) val_loss:1.9740  val_bpb:1.1691   ⭐
peak_memory:           24,347 MiB / 80 GiB H100
step_avg:              107.01 ms
artifact size:         14,916,215 bytes (int6 + zstd-22)
total submission:      15,042,800 bytes (artifact + 126,585 code)
budget check:          FITS (<16,000,000)
```

Eval timing (within the 10-minute eval budget):

- Roundtrip: 6.555 sec
- Sliding window (stride=64): 214.380 sec (~3.6 min)

## Comparison vs leaderboard

```
Position    val_bpb     Author              Notes
1.1194      abaybektursun  SOTA top
1.1228      signalrush     GPTQ-lite + EMA
1.1248      jfprincz       Partial RoPE + LN scale
...
1.1630      aquariouseworkman  Mixed quant
1.1691      THIS V40       ⭐ TT decomposition (13L + bigram x2)
1.1748      notapplica     Spectral embed
1.1925      Matthew Li     Sliding window eval
1.2244      Naive Baseline
```

V40 sits in the middle of the leaderboard (~12th-13th by val_bpb) but is the **only entry using TT/MPS decomposition**. The technique is orthogonal to other quantization tricks and could be combined with them in future submissions for further gains.

## Why V40 is novel

Tensor-Train decomposition originates from quantum many-body physics (DMRG / matrix product states) and was applied to neural network compression by Novikov et al. 2015 (arXiv:1509.06569) and Garipov et al. 2016 for fully-connected layers. It has been used in computer vision and recommender systems but, to the best of our knowledge, **has not been applied to attention layers in language modeling at the scale of competitive parameter-golf submissions**. This submission demonstrates:

1. TT can be applied to specific attention sub-matrices (c_q and attn.proj, both square dim×dim)
2. TT-SVD initialization from orthogonal weights gives a good starting point
3. Muon optimizer can be adapted to 3D TT cores via reshape-to-2D for Newton-Schulz orthogonalization
4. State-dict export via `nn.ParameterList` avoids the materialization-overflow trap
5. The ~3M-parameter savings can be productively reinvested into depth and bigram vocabulary

## Negative result: V40-B 15L was too big

We initially tested V40 at 15 layers (V40-B). It achieved a better sliding val_bpb (1.3181 vs 13L's 1.3375 on 1×H100) but the artifact was 16.47 MB through zstd-22 — over the 16 MB cap. We dropped to 13 layers (V40-D config baked into this submission) which fits with ~1 MB headroom while retaining most of the architectural advantage. See cloud A/B/D log analysis in `train.log`.

## Reproducibility

- Single seed (1337) — full statistical-significance ablation (3 seeds, p<0.01) skipped due to compute budget. Single-seed result documented above.
- All hyperparameters baked into `train_gpt.py` Hyperparameters class — no env-overrides required for reproduction.
- `requirements.txt` declares `zstandard` (critical) and other dependencies.
- Run on RunPod 8×H100 80GB SXM with PyTorch 2.9.1+cu128.

## Files

- `train_gpt.py` — model + training script with all V40 + V34-SafeBundle defaults baked in (126,585 bytes UTF-8)
- `final_model.int6.ptz` — trained model artifact, int6 quantized + zstd-22 compressed (14,916,215 bytes)
- `train.log` — full training log including warmup, training, roundtrip eval, and sliding window eval
- `submission.json` — metadata
- `requirements.txt` — Python dependencies
