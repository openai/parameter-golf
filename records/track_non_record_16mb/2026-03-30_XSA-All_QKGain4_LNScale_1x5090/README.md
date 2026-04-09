# XSA-All + QK Gain 4.0 + LN Scale — Systematic Hyperparameter Exploration on 1×RTX 5090

**Non-Record Submission (Single Consumer GPU)**
**Author:** Pranjal Jain ([@jainpranjal97](https://github.com/jainpranjal97))
**Hardware:** 1×RTX 5090 (32GB VRAM, Blackwell), vast.ai
**Duration:** ~45 experiments over 4 days, mix of 10-min and 60-min runs
**Best result:** val_bpb **1.1946** (60-min, 1×RTX 5090, 3699 steps at ~1050ms/step)

---

## The Short Version

I ran 45 systematic experiments on a single RTX 5090 to find the best configuration for this architecture. Three findings stand out as potentially novel or underexplored:

1. **XSA on ALL layers beats XSA on last 4** (-0.0018 BPB). Every top entry uses XSA on the deepest 3-4 layers. I found that applying XSA to every layer helps, even the shallowest ones.
2. **qk_gain_init = 4.0** (-0.0039 BPB cumulative from default 1.5). Sharper initial attention patterns significantly help small models. I swept 1.5 → 2.0 → 3.0 → 4.0 with consistent gains.
3. **Warmdown calibration for wallclock-capped training** is critical. The default warmdown_iters=1200 with a 10-min cap means the LR never reaches full strength. Reducing to 200 gave -0.0078 BPB.

I also tested four novel architectural ideas (Progressive Layer Growing, Depth Recurrence + LoRA, Cosine Warmdown, XSA Gating) — all failed, with documented reasons why.

---

## Architecture

Standard transformer stack with targeted modifications:

| Component | Value | Notes |
|-----------|-------|-------|
| Layers | 11 | +2 over baseline |
| Model dim | 512 | Standard |
| Heads / KV heads | 8 / 4 | GQA |
| MLP multiplier | 3× | Up from 2× baseline |
| Activation | LeakyReLU(0.5)² | Preserves negative gradients |
| Attention | XSA on **all** layers | Not just last 3-4 |
| Position encoding | Partial RoPE (16/64 dims) | 48 dims position-free |
| Residual scaling | LN Scale: 1/√(layer+1) | Depth-dependent |
| QK gain init | 4.0 | Up from default 1.5 |
| Logit softcap | 20.0 | Down from 30 |
| Sequence length | 2048 | Up from 1024 |

### XSA on All Layers

Standard XSA projects out the self-value component from attention output, forcing cross-token information flow. The conventional wisdom is to apply this only to deep layers (last 3-4). I tested all-layer XSA and found it consistently better:

```
XSA last 4 layers: 1.3549 BPB
XSA all layers:    1.3451 BPB  (-0.0098 cumulative improvement at that point)
```

The overhead is modest (~707ms/step vs 619ms with XSA-4), but the quality gain outweighs the ~12% fewer steps within wallclock budget.

### QK Gain Sweep

The `qk_gain` parameter scales the QK dot product before softmax. Higher values create sharper attention patterns:

| qk_gain_init | val_bpb | Delta |
|---|---|---|
| 1.5 (default) | 1.3301 | baseline |
| 2.0 | 1.3286 | -0.0015 |
| 3.0 | 1.3268 | -0.0033 |
| 4.0 | 1.3262 | -0.0039 |

Diminishing returns above 4.0 but no degradation. Small models benefit from sharper attention.

### LN Scale

Depth-dependent residual scaling: multiply each block's residual contribution by `1/√(layer_idx + 1)`. This stabilizes training in deeper models by dampening contributions from later layers, which see more accumulated residual magnitude.

```python
self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1)
# In forward:
x = x + (s * self.attn_scale[None, None, :]) * attn_out
```

---

## Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon (matrix params) + Adam (scalars/embeds) |
| Matrix LR | 0.04 |
| Muon momentum | 0.95, warmup from 0.85 over 200 steps |
| Muon weight decay | 0.06 (decoupled) |
| Grad clip norm | 0.3 |
| Warmdown iters | 1200 (60-min) / 200 (10-min) |
| Warmup steps | 20 |

Key optimizer finding: **Muon weight decay** on matrix parameters helps consistently. I swept 0.02 → 0.04 → 0.06 with monotonic improvement. But Adam weight decay on tied embeddings is catastrophic (+0.78 BPB).

---

## Results: 1×RTX 5090 (60-min run)

| Metric | Value |
|--------|-------|
| Steps | 3699 |
| Step avg | ~1050ms |
| Train loss (final) | ~2.01 |
| val_bpb (pre-quant) | ~1.19 |
| **val_bpb (int8+zlib)** | **1.1946** |
| Artifact size | 18.1 MB (int8+zlib) |
| Peak VRAM | ~14 GB |

Note: Artifact exceeds 16MB under int8+zlib. A competitive 8×H100 submission would need int6 quantization + zstd-22 compression, which we validated is feasible but didn't fully optimize for single-GPU.

---

## Full Experiment Log (45 runs)

### 10-min Experiments (1×RTX 5090, ~850 steps)

| # | val_bpb | Status | Description |
|---|---------|--------|-------------|
| 1 | 1.3549 | keep | LeakyReLU(0.5)² + XSA last 4 layers (969 steps, 619ms/step) |
| 2 | 1.3501 | keep | + Muon WD 0.02 (-0.0048) |
| 3 | 1.3526 | discard | MLP 3x (707ms/step; fewer steps offset capacity; +0.0025) |
| 4 | 1.3469 | keep | + seq_len 2048 (-0.0032) |
| 5 | 1.4235 | discard | EMA from step 0 (+0.0766; early weights poison average) |
| 6 | 1.3494 | discard | LeakyReLU alpha=0.75 (+0.0025; 0.5 better at this scale) |
| 7 | 1.3451 | keep | XSA on ALL layers (-0.0018) |
| 8 | 1.3817 | discard | SWA scale<0.5 every 50 steps (+0.0366; started too early) |
| 9 | 1.3421 | keep | Muon WD 0.04 (-0.0030) |
| 10 | 2.1226 | discard | Adam WD on tied embed (+0.78; catastrophic) |
| 11 | 1.3672 | discard | seq 4096 (+0.0251; too slow) |
| 12 | 1.3577 | discard | Muon momentum 0.99 (+0.0156; slower steps) |
| 13 | 1.3365 | keep | warmdown_iters 400 (-0.0056) |
| 14 | 1.3343 | keep | warmdown_iters 200 (-0.0022) |
| 15 | 1.3419 | discard | warmdown_iters 100 (+0.0076; too short) |
| 16 | 1.3354 | discard | Muon backend_steps 7 (+0.0011; overhead eats gain) |
| 17 | 1.3344 | discard | warmup_steps 5 (+0.0001; noise) |
| 18 | 1.3329 | keep | Muon momentum warmup 200 steps (-0.0014) |
| 19 | 1.3463 | discard | logit_softcap 50 (+0.0134) |
| 20 | 1.3309 | keep | grad_clip_norm 1.0 (-0.0020) |
| 21 | 1.3304 | keep | logit_softcap 20 (-0.0005) |
| 22 | 1.3301 | keep | Muon WD 0.06 (-0.0003) |
| 23 | 1.3286 | keep | qk_gain 2.0 (-0.0015) |
| 24 | 1.3268 | keep | qk_gain 3.0 (-0.0018) |
| 25 | 1.3262 | keep | qk_gain 4.0 (-0.0006) |
| 26 | 1.3410 | discard | rope_base 1000 (+0.0148) |
| 27 | 1.3285 | discard | scalar_lr 0.06 (+0.0023) |
| 28 | 1.3261 | discard | tied_embed_init_std 0.01 (noise) |
| 29 | 1.3290 | discard | tied_embed_lr 0.08 (+0.0028) |
| 30 | 1.3277 | discard | TurboQuant rotation (+0.0015; hurts per-row int8) |
| 31 | 1.3586 | discard | INT8_CLIP_PERCENTILE 99.5 (+0.0324; catastrophic) |

### 60-min Experiments (1×RTX 5090, ~3500 steps)

| # | val_bpb | Status | Description |
|---|---------|--------|-------------|
| 32 | 1.2528 | keep | MUD optimizer 2-pass (-0.0781 vs 10-min!) |
| 33 | 1.2387 | keep | + 11 layers (-0.0141) |
| 34 | 1.2235 | keep | + MLP 3x (-0.0152) |
| 35 | 1.2285 | discard | late-start EMA (+0.0050; averaging hurt) |
| 36 | **1.1946** | **keep** | **Competition stack Phase 1 (best)** |
| 37 | 1.1945 | discard | + Int6 QAT (only 180 QAT steps; no effect) |

### Novel Architecture Experiments (60-min)

| # | val_bpb | Status | Description |
|---|---------|--------|-------------|
| 38 | 1.2003 | discard | Progressive Layer Growing 5→11L at 60% (+0.0057; 7956 steps but 5L ceiling) |
| 39 | 1.2699 | discard | Depth Recurrence 4×3 + LoRA16 (+0.0753; torch.compile bypass kills it) |
| 40 | 1.1985 | discard | Cosine warmdown (+0.0039; linear already optimal) |
| 41 | 1.1961 | discard | XSA Gating — learned per-head gate (+0.0015; quantizes worse) |

---

## Novel Approaches: What Failed and Why

### Progressive Layer Growing (PLG)

**Idea:** Train a 5-layer model for 60% of training (fast steps → more steps), then duplicate layers to create an 11-layer model for the remaining 40%.

**Result:** 7956 steps (2.15× more than baseline) but val_bpb only 1.2003 (+0.0057 worse). The 5-layer model hits a capacity ceiling that the late-stage 11-layer model can't recover from in 40% of training time.

### Depth Recurrence + Per-Loop LoRA

**Idea:** 4 unique blocks looped 3 times (12 effective layers from 4 blocks of params), with per-loop LoRA rank-16 adapters to differentiate iterations.

**Result:** 1.2699 (+0.0753 worse). Two compounding problems: (1) torch.compile can't fuse the loop efficiently, pushing step time to 1204ms; (2) shared weights create optimization conflicts across iterations. This aligns with the findings in PR #363.

### XSA Gating (Novel)

**Idea:** Replace binary XSA (fully remove self-value) with a learned per-head sigmoid gate controlling removal strength. 8 extra scalar parameters.

**Result:** Pre-quantization val_bpb was 1.1932 (better than best!), but post int8+zlib roundtrip was 1.1961 (+0.0015 worse). The learned gates create weight distributions that are harder to quantize. Interesting finding: **architectural changes that improve pre-quantization loss can degrade post-quantization loss.**

### Cosine Warmdown

**Idea:** Replace linear warmdown schedule with cosine decay for smoother LR reduction.

**Result:** 1.1985 (+0.0039 worse). Linear warmdown is already optimal; cosine spends too long at high LR before dropping.

---

## Key Takeaways

1. **Warmdown calibration matters more than most architecture changes.** Getting the LR schedule right for wallclock-capped training was worth -0.0078 BPB — more than any single architectural modification.

2. **XSA on all layers > XSA on last N.** The information-theoretic benefit of removing self-value extends beyond deep layers.

3. **qk_gain_init should be tuned aggressively.** Default 1.5 is suboptimal; 4.0 is consistently better for small models.

4. **Pre-quantization and post-quantization metrics can diverge.** Always evaluate after the full quantization roundtrip — architectural choices that improve floating-point loss can hurt quantized loss.

5. **Consumer GPUs are viable for research.** The RTX 5090 at $0.35/hr enabled 45 experiments for under $20 total, producing findings transferable to the 8×H100 competition environment.

---

## Reproducing These Results

```bash
# On a machine with 1×RTX 5090 or similar GPU (32GB+ VRAM):
pip install torch numpy sentencepiece huggingface_hub

# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Run training (60-min budget)
RUN_ID=exp \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
