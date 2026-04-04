# Mamba-3 SSD + Attention Hybrid with QAT

## Summary

First parameter-golf submission using **Mamba-3 SISO SSD** (pure Triton parallel scan) as the primary block in a hybrid architecture with a single attention layer.

- **1.5633 bpb** post-INT6+QAT+TTT (seed 1337, 8×H100)
- Training bpb: **1.2413** (BF16, before quantization)
- Model size: **10.8MB** INT6+zlib-9
- Total submission: **10.9MB** (well within 16MB limit)

---

## Architecture

8-layer hybrid: 7× Mamba-3 SISO blocks + 1 attention layer at layer 4.

```
[Mamba-3 SSD] + [MLP]   ×7 layers
[Attention]   + [MLP]   ×1 layer (at layer 4)
dim=512, d_state=64, mlp_mult=3, seq_len=4096
```

- **Mamba-3 SISO:** pure Triton kernels (no CUDA C++ deps), `torch.compile`-compatible, chunked SSD, `expand=2`, `headdim=64`
- **Attention:** causal GQA (8 heads / 4 KV heads), RoPE, `q_gain` per head (init=1.5), GLU values (`v = v * v.sigmoid()`)
- **MLP:** LeakyReLU² hidden layer, `mlp_mult=3`
- **Other:** U-net skip connections, SmearGate, BigramHash, tied embeddings, Muon optimizer

SSD is ~2× faster than FlashAttention-2 at seq_len=4096, enabling 2× more training tokens per 10-minute run vs a pure transformer.

---

## Phase 1: Architecture Sweep (1×H100, 10 min each)

Fixed: `MODEL_DIM=512, NUM_LAYERS=8, MAMBA3_D_STATE=64, TRAIN_SEQ_LEN=4096, TRAIN_BATCH_TOKENS=1M`

| Config | attn layers | mlp_mult | params | val_bpb | steps | ms/step | HBM |
|--------|-------------|----------|--------|---------|-------|---------|-----|
| **S1** | **1** | **3** | **26.2M** | **1.6972** | **737** | **815** | **28GB** |
| S2 | 2 | 3 | 25.3M | 1.7094 | 725 | 829 | 27GB |
| S3 | 3 | 3 | ~24M | 1.7331 | 706 | 850 | 26GB |
| S4 | 1 | 5 | ~33M | 1.7761 | 659 | 911 | 33GB |
| S5 | 2 | 5 | ~31M | 1.7988 | 643 | 934 | 31GB |
| S6 | 3 | 5 | ~30M | 1.8142 | 629 | 955 | 30GB |

**Winner: S1 — 1 attention layer, mlp_mult=3**

Fewer attention layers and smaller MLP both win — at a fixed wall-clock budget, underfitting capacity is worse than having less of it. Each extra attention layer costs ~15ms/step = ~70 fewer steps = ~12–24 mBPB. mlp_mult=5 runs at 911ms/step vs 815ms — 78 fewer steps = ~80 mBPB worse.

---

## Phase 2: Technique Ablation (1×H100, S1 config)

Techniques stacked sequentially, each run adds one change on top of the previous winner.

| Run | Change | val_bpb | Δ | Verdict |
|-----|--------|---------|---|---------|
| S1 baseline | — | 1.6937 | — | — |
| **T1: Muon SOTA params** | `WEIGHT_DECAY=0.04, MUON_MOMENTUM=0.99, MATRIX_LR=0.025` | 1.6889 | -4.8 mBPB | ✓ keep |
| **T3: GLU Values** | `v = v * v.sigmoid()` in attention | **1.6322** | **-56.7 mBPB** | ✓ keep |
| T4: LN Scale Init | scale output params by `1/sqrt(layer_idx+1)` | 1.6500 | +17.8 mBPB | ✗ drop |
| T5: Value Embedding | `VE_ENABLED=1, VE_DIM=64` | 1.6413 | +9.1 mBPB | ✗ drop |

**Best: T1 + T3 → 1.6322 bpb**

GLU Values is the dominant gain. LN Scale Init hurts because Mamba-3 blocks rely on their output scale for residual mixing. Value Embedding doesn't transfer from pure transformers — with only 1/8 attention layers, VE has too little interaction surface.

### Comparison to SOTA on 1×H100

| Model | training val_bpb | steps (10 min) | tokens seen | ms/step |
|-------|-----------------|----------------|-------------|---------|
| SOTA pure transformer (PR #549) | 1.7398 | 443 | ~340M | ~1357 |
| **Our hybrid T1+T3** | **1.6322** | **742** | **~742M** | **~811** |

Our hybrid sees 2× more tokens in the same wall-clock. The SSD throughput advantage at seq_len=4096 is the main driver.

---

## Phase 3: 8×H100 Full-Scale Runs

### Clean BF16 Run (no QAT)

| Metric | Value |
|--------|-------|
| Steps | 5,189 |
| Step time | 115.64ms |
| Training val_bpb | **1.2087** |
| INT6+zlib-9 size | 9,747,981 bytes |
| Post-INT6 bpb | 1.8617 (+65.3 mBPB) |
| INT8+TTT bpb | 1.8222 (+61.4 mBPB) |

TTT (1 epoch, 474 chunks) helps by ~40 mBPB. The large gap between training and post-quant bpb is from quantization noise, not TTT degradation.

### Critical Bug: WARMDOWN_ITERS LR Mismatch

The time-based warmdown schedule computes `warmdown_ms = warmdown_iters × step_ms`. On different hardware:

| Hardware | step_ms | WARMDOWN_ITERS=3000 | Effect |
|----------|---------|---------------------|--------|
| 1×H100 | 815ms | warmdown_ms=2445s > 600s | Warmdown from step 1 ✓ |
| 8×H100 | 114ms | warmdown_ms=342s < 600s | Full LR (1.0) until step 2263 ✗ |

Default WARMDOWN_ITERS=3000 on 8×H100 gave 7× higher effective LR → loss explosion at step ~1300.

**Fix:** `WARMDOWN_ITERS=22000` on 8×H100.
**Rule:** `WARMDOWN_ITERS = target_warmdown_duration_ms / step_ms_for_hardware`

### Quantization Root Cause

INT8 ≈ INT6 post-quant bpb — going from 6 to 8 bits doesn't help. The bottleneck is **outlier parameters in Mamba-3 SSD projections**. `in_proj` carries merged B, C, dt, A projections with very different dynamic ranges; per-row quantization scales are set by outliers, leaving poor resolution for the rest of the row.

**QAT fix:** The standard QAT only covers `CastedLinear` layers (attention + MLP). Mamba-3's `in_proj` and `out_proj` are `nn.Linear` — accounting for ~47% of quantized parameters — and were never fake-quantized during training.

Fix: replace with `CastedLinear` in `Mamba3Layer.__init__` so QAT, float32 master weights, and fake-quantization all apply automatically:

```python
for attr in ("in_proj", "out_proj"):
    src = getattr(self.mamba3, attr)
    dst = CastedLinear(src.in_features, src.out_features, bias=src.bias is not None)
    dst.weight = src.weight
    setattr(self.mamba3, attr, dst)
```

### QAT Run (this submission)

| Metric | No QAT | With QAT (this run) |
|--------|--------|---------------------|
| Training val_bpb | 1.2087 | 1.2413 (+32 mBPB) |
| Post-INT6 bpb | 1.8617 | **1.5633** |
| Quantization gap | +65 mBPB | **+32 mBPB** (51% reduction) |
| Model size | 9.75MB | 10.8MB |
| Steps | 5,189 | 5,193 |
| Step time | 115.64ms | 115.56ms |

The +32 mBPB training regression is expected QAT noise cost. Net post-quant improvement: **+30 mBPB**.

---

## Setup

The pre-built `mamba-ssm` wheel does not include Mamba-3 files. After installing `requirements.txt`, copy the Mamba-3 source files from the [mamba repo](https://github.com/state-spaces/mamba):

```bash
pip install -r requirements.txt --no-deps

PKG=$(python3 -c "import mamba_ssm; print(mamba_ssm.__path__[0])")
mkdir -p $PKG/ops/triton/mamba3

cp mamba/mamba_ssm/modules/mamba3.py          $PKG/modules/mamba3.py
cp mamba/mamba_ssm/ops/triton/angle_cumsum.py $PKG/ops/triton/angle_cumsum.py
cp mamba/mamba_ssm/ops/triton/mamba3/*.py     $PKG/ops/triton/mamba3/

printf '__version__ = "2.3.1"\nfrom mamba_ssm.modules.mamba3 import Mamba3\n' > $PKG/__init__.py
python3 -c "from mamba_ssm.modules.mamba3 import Mamba3; print('OK')"
```

## Configuration

```bash
MODEL_DIM=512 MAMBA3_D_STATE=64 NUM_LAYERS=8 TRAIN_SEQ_LEN=4096 TRAIN_BATCH_TOKENS=1048576 \
MAX_WALLCLOCK_SECONDS=600 NUM_ATTN_LAYERS=1 MLP_MULT=3 \
WEIGHT_DECAY=0.04 MUON_MOMENTUM=0.99 MATRIX_LR=0.025 \
WARMDOWN_ITERS=22000 \
QUANT_BITS=6 QAT_START_FRAC=0.15 SWA_ENABLED=0 TTT_ENABLED=1 \
torchrun --nproc_per_node=8 train_mamba3_hybrid.py
```

## What Limits Performance

At 1.5633 bpb, the gap to SOTA (1.1194) is large. The main limiters:
1. **Architecture**: Mamba-3 hybrid at 27M params underperforms pure transformers at this compute budget
2. **Remaining quant gap**: +32 mBPB — per-layer precision (keep SSD A/dt in FP16) may help
3. **LR schedule**: warmdown-from-start (cosine, reaches 0 at end) — an LR floor or step-based warmdown could recover some training quality
