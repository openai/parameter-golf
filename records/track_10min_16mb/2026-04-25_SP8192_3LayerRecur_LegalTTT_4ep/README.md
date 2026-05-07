# Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Score-First TTT (4 epochs) + Tuned MLP WD

**val_bpb = 1.07290** (3-seed mean, std 0.00016) | ~16.00 MB | 8xH100 SXM

## 3-Seed Results

| Seed | Pre-quant BPP | Quantized BPP | Sliding BPP | **TTT BPP** | Artifact (B) | Train (s) | Eval (s) |
|------|---------------|---------------|-------------|-------------|--------------|-----|----------|
| 42   | 1.08086       | 1.09131       | 1.07455     | **1.07303** | 15,995,398   | 600 | 577.5    |
| 314  | 1.08058       | 1.09105       | 1.07424     | **1.07272** | 15,999,207   | 600 | 575.7    |
| 999  | 1.08078       | 1.09119       | 1.07443     | **1.07295** | 15,995,751   | 600 | 586.4    |
| **Mean** | **1.08074**   | **1.09118**   | **1.07441** | **1.07290** | **15,996,785** | **600** | **579.9** |
| **Std**  | **0.00014**   | **0.00013**   | **0.00016** | **0.00016** | — | — | — |

Prior SOTA (2026-04-09 PR `SP8192_3LayerRecur_ParResid_QK525_LegalTTT` by @bigbag):
**1.0810 BPP** (3-seed mean, std 0.0002).

**Δ = −0.00810 BPP**, well above the official 0.005-nat threshold.
**Welch t = −54.93** (df = 3.80), **one-sided p < 1e-7**.

## Key Techniques (delta vs prior 04-09 record)

1. **TTT epochs 3 → 4** — extra adaptation budget per 32K-token chunk under
   the same score-first protocol. The dominant source of the val_bpb gain.
2. **Split MLP weight decay** — `muon_wd_mlp = 0.115` vs `muon_wd = 0.095`
   for non-MLP matrices. Stronger regularization on the largest matrices
   reduces post-quant degradation.
3. **Per-head attention-output gate** — sigmoid gate over attention output
   ([H, W_g=12] zero-init Parameter; raw gate = 2·sigmoid(0) = 1 at step 0,
   transparent in early training).

All other components inherited from the 04-09 record, listed below.

## Architecture (inherited from 04-09 record)

11L × 512d × 8H / 4KV (GQA), MLP 4× (hidden 2048), LeakyReLU(0.5)² MLP
activation, Partial RoPE (16/64 dims), tied embeddings, logit
softcap = 30.0. Depth recurrence: encoder/decoder layer indexing
includes loops over layers 3-5 (`num_loops=2`), activated at training
fraction 0.35. Parallel residuals from layer 7. XSA (exclusive
self-attention: subtract normalized-V projection of output) on all
11 layers. Layer-wise LN scale `1/sqrt(layer+1)`, with looped layers
additionally divided by `sqrt(num_loops+1)` for residual variance
balancing.

Total ~35.9M parameters.

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps,
nesterov), AdamW for embeddings and scalars. Time-fractional schedule:
linear warmdown over the final 72% of the 600 s wallclock. Time-
fractional Muon momentum warmup over the first 22% (peak momentum 0.99,
warmup-start 0.92). EMA decay 0.9965 throughout training. ~4640 steps
in 600 s on 8×H100 SXM at peak ~7.7 M tok/s.

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k · std(row)` for principled
rate-distortion. int6 for attention/MLP matrices (k=12.85), int8 for
token embeddings (k=20.0). Per-row fp16 scales. Block size 32 with
multiplicative Hessian damping factor 1.01.

## Compression

Byte-shuffle (stride-2) followed by Brotli quality-11 with
`lgwin=24`. The byte-shuffle separates the high-bit / low-bit byte
planes of the int8-stored quantized values, exposing the redundancy
of the unused upper bits to brotli. Final blob ≈ 15.97 MB across
seeds.

## Test-Time Training (Score-First, 4 epochs/chunk)

Per Issue #1017 Track B "legal eval-time adaptation":

- **Condition 1 (Causality)**: Sliding-window eval is strictly causal;
  each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution)**: Standard softmax over the
  full SP8192 vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score-before-update)**: Each 32K-token chunk is fully
  scored under `torch.no_grad()` BEFORE any SGD update. Training only
  on already-scored tokens.
- **Condition 4 (Single pass)**: Each token scored exactly once. No
  rescoring, no multi-pass selection.

Inner loop: SGD with momentum 0.9, nesterov; per-chunk cosine LR
decay starting from `ttt_lr = 0.005`; 4 epochs per chunk; gradient
clipping at norm 1.0; distributed all-reduce of gradients at
`ReduceOp.AVG`. Total TTT eval time ~310 s per seed (within the
600 s eval budget).

## Compliance

- No SLOT (standard or causal)
- No pre-quant TTT on val data (model is quantized once during eval
  setup; the only TTT pass is the legal score-first adaptation
  described above)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- No tokenizer changes (default SP8192 BPE on FineWeb10B)
- No external data, sidecar files, or eval-time downloads
- All 3 artifacts < 16,000,000 bytes (worst margin: 793 B on seed 314)
- Train time < 600 s on all 3 seeds (capped at 600.1 s by
  `MAX_WALLCLOCK_SECONDS=600`)
- Eval time < 600 s on all 3 seeds (worst: 586.4 s on seed 999)

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
    python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=4 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
```

(Repeat with `SEED=314` and `SEED=999` to reproduce the 3-seed mean.)

## Credits

This submission is a small two-knob delta on top of the 2026-04-09
record by **@bigbag**
(`SP8192_3LayerRecur_ParResid_QK525_LegalTTT`, val_bpb 1.0810).
Everything other than `ttt_epochs=4` and `muon_wd_mlp=0.115` is
inherited unchanged from that stack.

For the full upstream contributor chain (SP8192, GPTQ SDClip, depth
recurrence, parallel residuals, QK-Gain, MuonEq-R, score-first TTT
framework, hyperparameter tuning), see @bigbag's record README:
`records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md`.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` (lzma-RAW + base85 self-extracting wrapper around
  the full source; 68,778 B raw → 19,646 B packed)
- `seed42.log`, `seed314.log`, `seed999.log`
