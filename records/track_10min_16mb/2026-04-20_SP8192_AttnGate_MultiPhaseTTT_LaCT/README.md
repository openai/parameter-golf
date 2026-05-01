# 2026-04-20 — SP8192 · AttnGate · Multi-Phase TTT · LaCT

**Track:** 10-min / 16 MB  
**Baseline BPB (to beat):** 1.0810  (2026-04-09 seed-42 run, single-phase TTT)  
**Primary eval path:** quantized model + multi-phase 4-phase score-first TTT  
**Hardware target:** 8 × H100 SXM5 (RunPod)

---

## Overview

This candidate stacks three independent improvements on top of the accepted
April-9 baseline:

| Component | Expected mechanism | Bytes cost |
|---|---|---|
| **Attention output gate** | Learned per-channel rescaling of attention residual | +512 × fp32 = +2 KB (negligible) |
| **Multi-phase TTT** | Curriculum over TTT update scope and LR; better convergence without extra passes | 0 |
| **LaCT fast-weight adapter** | SwiGLU adapter on hidden states updated chunk-by-chunk during eval | 0 (GPU only) |

Everything else — tokenizer, depth recurrence, parallel residuals, QK-gain,
MuonEq-R, EMA, entropy-constrained GPTQ — is identical to the baseline.

---

## Architecture

### Base model (unchanged from 2026-04-09)

- **Tokenizer:** SP8192 sentencepiece, 8192 vocab  
- **Layers:** 11, dim 512, 8H / 4KV GQA, MLP mult 4×  
- **Depth recurrence:** 3-layer loop (layers 3–5), activates at training fraction 0.35  
- **Parallel residuals:** from layer 7 onward  
- **QK-gain:** 5.25 (init), learned scalar  
- **Optimizer:** MuonEq-R (row-normalized Muon) for matrices, AdamW for scalars/embed  
- **EMA:** 0.9965 decay, used for eval and GPTQ calibration  
- **Compression:** Brotli-11 + byte-shuffle + LZMA code wrapper via entropy-constrained GPTQ  

### Attention output gate (new)

A zero-initialized channelwise vector `g ∈ ℝ^dim` is added to each attention
block:

```
attn_out = attention(x)
attn_out = attn_out * (1 + g)   # g is zero at init → identity
x = x + attn_out                 # standard residual
```

At initialization this is a pure identity; the model discovers per-channel
rescaling through gradient descent. This loosely follows the direction of PR
#1667 in the parameter-golf repo. The gate adds 512 float32 parameters (≈ 2 KB
serialized), which is absorbed by the entropy allocator's 16 MB budget with no
observable impact.

The gate is treated as a *control scalar* (like `mlp_scale`, `resid_mix`) for
purposes of the multi-phase TTT Phase A, where only low-rank / scalar control
parameters are updated. The code path still supports that narrow warm-up phase,
but the current default profile disables it (`TTT_PHASE_A_END=0.00`) and goes
straight into full-parameter adaptation.

### Multi-phase TTT (new)

Replaces the single-pass score-first TTT from the April-9 baseline. The
eval document is split into `N` chunks; TTT is organized into 4 phases
defined over the chunk index:

| Phase | Chunk range | Params updated | LR scale |
|---|---|---|---|
| A | 0% | Control scalars only (`attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `attn_out_gate`) | 1.0× |
| B | 0–80% | All parameters | 1.0× |
| C | 80–95% | All parameters | 1.0× |
| D | 95–100% | All parameters | 0.5× |

**Score-first invariant is preserved throughout.** Each chunk is fully scored
under `torch.no_grad()` before the TTT update is computed. This is structurally
identical to the April-9 legal TTT; the only difference is the update scope and
LR schedule.

The rationale for the current default curriculum:

- **Phase A** remains available for ablations, but is disabled in the default
  profile so the scored path starts with full-parameter updates immediately.
- **Phase B** uses the full LR for most of the document, maximizing useful
  adaptation time under the 10-minute budget.
- **Phase C** keeps all parameters active but delays LR tapering until late in
  the document.
- **Phase D** halves the LR for the last 5% of chunks to reduce late overshoot.

Total wall time is expected to be within ±5% of single-phase TTT because the
total number of gradient steps and chunk passes is identical — only the
parameter masks and LR change between phases.

### LaCT fast-weight adapter (optional additional eval)

LaCT (Layer-adaptive Continuous-TTT) is proposed in *Test-Time Training Done
Right* (Zhang et al., arXiv 2505.23884). The key insight is to separate the
TTT update target from the base model: instead of fine-tuning model weights
directly, a **lightweight fast-weight adapter** is attached to the hidden
states immediately before the output projection. This adapter:

1. Lives entirely in GPU memory during eval — it is **never serialized** and
   contributes 0 bytes to the artifact.
2. Receives gradient updates chunk-by-chunk (score-first, same causal contract
   as legal TTT).
3. Is re-initialized fresh for each validation document.

#### Architecture

```
h = model.forward_hidden(input_ids)          # (B, T, D)
h = h + lact_adapter(h)                      # SwiGLU branch
logits = model.logits_from_hidden(h)
```

The SwiGLU adapter:

```
w1 ∈ ℝ^{D × S},  w3 ∈ ℝ^{D × S},  w2 ∈ ℝ^{S × D}
out = scale * (silu(h @ w1) * (h @ w3)) @ w2
```

where `S = 128` (state_dim) and `scale = 0.08`. `w1`, `w3` are randomly
initialized; `w2` is zero-initialized (so the adapter is a no-op at the start
of each document).

#### Why LaCT can help over standard TTT

Standard TTT updates the base model weights, which creates several tensions:

- **Gradient noise** from the quantized model's Hessian landscape can make TTT
  updates diverge.
- **Catastrophic interference**: updating attention and MLP weights changes the
  representation geometry, making each chunk's signal less transferable to
  later chunks.
- **Artifact budget**: in a 16 MB challenge, every quantization bit matters;
  TTT updates on base weights force the quantized model to "absorb" fine-tuning
  signal it was not quantized to accommodate.

LaCT sidesteps these by providing a dedicated low-rank update target:

- The adapter has only `2 × 512 × 128 + 128 × 512 = 196608` parameters
  (≈ 750 KB in fp32), updated in fp32 regardless of quantization precision.
- The base model remains frozen during eval; the adapter bears all of the
  domain-shift correction. This is cleaner than fine-tuning the quantized
  weights directly.
- The SwiGLU nonlinearity allows the adapter to express multiplicative
  interactions (e.g., "suppress token X in this context") that a linear adapter
  cannot, matching the expressiveness of the base model's MLP blocks.

#### Expected contribution from LaCT

On the FineWeb validation set, which is largely i.i.d. with the training set,
the expected gain from LaCT alone is modest (0.002–0.010 BPB). The gain would
be more significant on out-of-distribution or long-document corpora where
domain shift is larger. For the parameter-golf track we therefore treat LaCT as
an **optional exploratory eval** (`LACT_TTT_ENABLED=0` by default) and rely on
multi-phase TTT as the primary path.

If LaCT is enabled (`LACT_TTT_ENABLED=1`), the adapter is updated on top of
the already-TTT-adapted base model, making it a second pass that corrects
residual domain-shift the base TTT missed.

---

## Hyperparameter reference

All values controllable via environment variable. Defaults shown below are
the record-profile values.

### Training caps

| Variable | Default | Notes |
|---|---|---|
| `MAX_WALLCLOCK_SECONDS` | 600 | Hard training cut-off including GPTQ |
| `GPTQ_RESERVE_SECONDS` | 12 | Seconds reserved for GPTQ at end of training |

### New flags

| Variable | Default | Notes |
|---|---|---|
| `ATTN_GATE_ENABLED` | 1 | Zero-init per-channel gate on attn output |
| `MULTIPHASE_TTT_ENABLED` | 1 | 4-phase curriculum TTT (primary eval) |
| `TTT_ENABLED` | 0 | Single-phase TTT legacy path |
| `LACT_TTT_ENABLED` | 0 | LaCT adapter (optional, slow) |

### Multi-phase TTT

| Variable | Default | Notes |
|---|---|---|
| `TTT_PHASE_A_END` | 0.00 | Fraction of chunks ending Phase A |
| `TTT_PHASE_B_END` | 0.80 | Fraction of chunks ending Phase B |
| `TTT_PHASE_C_END` | 0.95 | Fraction of chunks ending Phase C |
| `TTT_PHASE_C_LR_SCALE` | 1.0 | LR multiplier in Phase C |
| `TTT_PHASE_D_LR_SCALE` | 0.5 | LR multiplier in Phase D |
| `TTT_LR` | 0.005 | Base TTT LR (cosine within each phase) |
| `TTT_EPOCHS` | 3 | SGD steps per chunk |
| `TTT_CHUNK_TOKENS` | 32768 | Tokens per TTT chunk |

### LaCT adapter

| Variable | Default | Notes |
|---|---|---|
| `LACT_FAST_WEIGHT` | swiglu | Adapter type: `swiglu` or `linear` |
| `LACT_STATE_DIM` | 128 | Inner rank S |
| `LACT_SCALE` | 0.08 | Output scaling factor |
| `LACT_LR` | 0.02 | Adapter update LR |
| `LACT_UPDATE` | muon | Update rule: `muon` or `sgd` |
| `LACT_EPOCHS` | 1 | Update steps per chunk |
| `LACT_BASE_TTT` | 1 | Run base TTT before LaCT |
| `LACT_BATCH_SEQS` | 16 | Sequences per LaCT chunk |

### Entropy GPTQ allocator

| Variable | Default | Notes |
|---|---|---|
| `EXPORT_ALLOCATOR` | entropy | `entropy` or `uniform` |
| `ARTIFACT_TARGET_BYTES` | 16000000 | Max artifact size |
| `ALLOCATOR_ATTN_BITS` | 6,7 | Allowed bit-widths for attention tensors |
| `ALLOCATOR_MATRIX_BITS` | 5,6,7 | Allowed bit-widths for MLP/other matrices |
| `ALLOCATOR_EMBED_BITS` | 8 | Bit-width for embeddings |
| `ALLOCATOR_LAMBDAS` | 10 values | λ sweep for size/quality tradeoff |

### Architecture (fixed for this record profile)

| Variable | Value |
|---|---|
| `VOCAB_SIZE` | 8192 |
| `NUM_LAYERS` | 11 |
| `MODEL_DIM` | 512 |
| `NUM_HEADS` | 8 |
| `NUM_KV_HEADS` | 4 |
| `QK_GAIN_INIT` | 5.25 |
| `NUM_LOOPS` | 2 |
| `LOOP_START` / `LOOP_END` | 3 / 5 |
| `ENABLE_LOOPING_AT` | 0.35 |
| `PARALLEL_RESIDUAL_START` | 7 |
| `EMA_DECAY` | 0.9965 |
| `MUON_WD` | 0.095 |

---

## Score-first compliance proof

The competition requires that during test-time training on the validation set,
each token's loss is measured **before** any update that could have seen that
token causally. This implementation satisfies the invariant as follows:

1. `val_data` is divided into chunks of `TTT_CHUNK_TOKENS` tokens.
2. For chunk `i`, `_score_chunk_windows` is called inside `torch.no_grad()`.
   This accumulates the BPB contribution of every token in chunk `i` using the
   current model weights, without any gradient computation.
3. Only after scoring is complete does `_ttt_update_chunk` compute gradients
   and run `optimizer.step()`.
4. The resulting updated weights are used to score chunk `i+1`, never chunk `i`.

Phase transitions (A→B→C→D) only change which parameters are in
`requires_grad=True` and the LR. They do not alter the score-then-update
ordering.

The LaCT adapter follows the same protocol: `lact_adapter` is applied during
the no-grad scoring pass (adapter is read-only), then updated via an SGD/Muon
step on the scored tokens.

---

## Eval sequence

The training script runs these evals in order after GPTQ:

1. **pre-quantization** — upper bound, FP EMA model, no TTT
2. **quantized** — quantized model, no TTT, no sliding window
3. **quantized_sliding_window** — quantized + overlapping windows
4. **quantized_ttt_multiphase** ← **primary scored path** (reported BPB)
5. **quantized_ttt** — legacy single-phase (for comparison, disabled by default)
6. **quantized_lact_ttt** — LaCT on top of multi-phase TTT (optional)

The **primary scored path** is always `quantized_ttt_multiphase` when
`MULTIPHASE_TTT_ENABLED=1`.

---

## Suggested sweep order

1. **Phase A ablation** — `TTT_PHASE_A_END` ∈ {0.00, 0.10, 0.20}
2. **Late taper search** — `TTT_PHASE_B_END` ∈ {0.75, 0.80, 0.85}, `TTT_PHASE_C_END` ∈ {0.90, 0.95}
3. **Phase D LR scale** — `TTT_PHASE_D_LR_SCALE` ∈ {0.3, 0.5, 0.7}
4. **Attn-gate only ablation** — `ATTN_GATE_ENABLED=0` vs `1` (same seed)
5. **LaCT state dim** — `LACT_STATE_DIM` ∈ {64, 128, 256}
6. **LaCT scale** — `LACT_SCALE` ∈ {0.04, 0.08, 0.12}

Run seeds 42, 314, 999 before submitting any candidate record.

---

## Expected BPB targets

| Configuration | Expected BPB | vs baseline |
|---|---|---|
| Baseline (Apr 9, single-phase TTT) | 1.0810 | — |
| + Attn gate only | ~1.079 | −0.002 |
| + Multi-phase TTT (no gate) | ~1.077 | −0.004 |
| + Both (this record) | ~1.075 | −0.006 |
| + LaCT on top (`LACT_TTT_ENABLED=1`) | ~1.073 | −0.008 |

These are rough estimates. The attn gate contribution is small but reliable
(zero-init → no risk). The multi-phase TTT gain depends on how much useful
adaptation signal can be exploited before the late taper begins.

---

## RunPod reproduction

### Prerequisites

- Pod with 8 × H100 SXM5, ≥ 80 GB VRAM each
- PyTorch 2.4+ with CUDA 12.x
- Python 3.10+
- ~50 GB free disk for dataset shards

### Step-by-step

```bash
# 1. SSH into your RunPod instance (replace with your pod address)
ssh root@<pod-ip> -p <port>

# 2. Clone the repo
git clone https://github.com/<your-fork>/parameter-golf-fork.git
cd parameter-golf-fork

# 3. Move into the record directory
cd records/track_10min_16mb/2026-04-20_SP8192_AttnGate_MultiPhaseTTT_LaCT

# 4. Install FlashAttention using the same wheel path as the April 9 record
pip install flash_attn_3 --no-deps --find-links \
  https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# 5. Verify the backend is importable
python3 -c "from flash_attn.flash_attn_interface import flash_attn_func; print('FlashAttention OK')"

# 6. (Optional) set HF token if the dataset repo requires it
export HF_TOKEN=<your_huggingface_token>

# 7. Run with default seeds
SEED=42  bash run.sh 2>&1 | tee logs/seed42.log
SEED=314 bash run.sh 2>&1 | tee logs/seed314.log
SEED=999 bash run.sh 2>&1 | tee logs/seed999.log

# 8. Check final BPB lines
grep "quantized_ttt_multiphase" logs/seed*.log | grep "bpb"
```

This record path no longer falls back to SDPA. If FlashAttention is missing,
`flash_attn_interface.py` raises an explicit error telling you to install the
same wheel family used by the April 9 record.

### Key environment overrides

```bash
# Disable multi-phase TTT (use legacy single-phase for ablation)
MULTIPHASE_TTT_ENABLED=0 TTT_ENABLED=1 bash run.sh

# Enable LaCT for exploratory eval
LACT_TTT_ENABLED=1 bash run.sh

# Ablate attention gate
ATTN_GATE_ENABLED=0 bash run.sh

# Custom phase boundaries
TTT_PHASE_A_END=0.20 TTT_PHASE_B_END=0.60 bash run.sh
```

## RunPod `@Endpoint`

The endpoint path is designed for **Serverless / `@Endpoint`** style execution,
not manual SSH sessions. It keeps the training logic in `train_gpt.py`
unchanged and adds only the worker wrapper and containerization needed for
RunPod.

### Files added for endpoint mode

- `Dockerfile` pre-bakes Python deps, PyTorch 2.9.1 CUDA 12.8, `flash_attn_3`,
  and the RunPod SDK.
- `handler.py` wraps the existing `torchrun` call, streams stdout/stderr back as
  aggregated endpoint output, and persists logs/artifacts/summaries on the
  attached network volume.

### Build

Build from the repo root so the Docker context includes the whole tree:

```bash
docker build \
  -f records/track_10min_16mb/2026-04-20_SP8192_AttnGate_MultiPhaseTTT_LaCT/Dockerfile \
  -t parameter-golf:apr20-attngate-multiphase-lact .
```

### Endpoint configuration

- GPU type: `H100 SXM`
- GPUs per worker: `8`
- Execution timeout: `1400` seconds
- Network volume: attach one and mount it at RunPod's default Serverless path
  `/runpod-volume`

The handler expects the network volume and fails explicitly if it is absent. It
stores persistent state under:

- `/runpod-volume/parameter-golf/data`
- `/runpod-volume/parameter-golf/endpoint_results/2026-04-20_SP8192_AttnGate_MultiPhaseTTT_LaCT`

The first request can populate the dataset onto that volume. Later requests
reuse it.

### Request pattern

With a `1400s` timeout, do **one seed per request**. The competition still
requires seeds `42`, `314`, and `999`, so run three jobs with the same
`run_group_id`.

Example request body for seed `42`:

```json
{
  "input": {
    "seed": 42,
    "run_group_id": "apr20-lact-runpod-001",
    "prepare_dataset_if_missing": true
  }
}
```

Repeat that request with `seed=314` and `seed=999`. The handler includes the
latest valid record (`2026-04-09`, TTT BPB `1.0810`) in the streamed response
and writes a per-seed `summary.json` plus a shared `summary_index.jsonl` so
metrics stay directly comparable against the latest valid record.

### Streamed logs and saved outputs

`handler.py` is a streaming handler with `return_aggregate_stream=True`, so:

- `/stream` receives live log chunks as they are produced
- `/run` and `/runsync` receive the aggregated streamed output

Each job also writes:

- `combined_response_log.txt` — exact handler response text you can copy to a
  local `.txt`
- `logs/<run_id>.txt` — training log captured from the script itself
- `artifacts/final_model.int6.ptz` — persisted quantized artifact
- `summary.json` — parsed metrics and deltas vs the latest valid record

### Artifact inspection

After training, the compressed artifact is at `final_model.int6.ptz`. Size
must be ≤ 16,000,000 bytes:

```bash
wc -c final_model.int6.ptz
```

The training log at `logs/<run_id>.txt` contains:
- Per-step loss and learning rate
- Artifact size after each λ candidate in the entropy sweep
- Final BPB for each eval path

---

## Files

| File | Description |
|---|---|
| `train_gpt.py` | Main training + eval script (2000+ lines) |
| `flash_attn_interface.py` | Strict FlashAttention wrapper; fails explicitly if backend is missing |
| `handler.py` | RunPod streaming endpoint wrapper around `torchrun` |
| `Dockerfile` | RunPod worker image with pre-baked deps and FlashAttention |
| `requirements.txt` | Python dependencies (sentencepiece, huggingface_hub, brotli) |
| `run.sh` | End-to-end RunPod launch script |
| `README.md` | This file |

---

## Reference

Zhang, T. et al. *Test-Time Training Done Right*. arXiv:2505.23884 (2025).  
[https://arxiv.org/abs/2505.23884](https://arxiv.org/abs/2505.23884)
