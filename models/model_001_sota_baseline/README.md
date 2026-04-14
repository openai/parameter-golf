# Model #001: SOTA Baseline

**Status**: Ready to train

**Target BPB**: 1.1147 (replica of 2026-03-25 SOTA)

**Hardware**: 8×H100 SXM, 600s max

**Key Configuration**:
- 11 layers, 512d model, 8 attention heads (GQA: 4 KV heads)
- 3× MLP expansion (1536 hidden)
- Full Hessian GPTQ quantization with AR self-generated calibration
- XSA on all 11 layers (cross-position mixing)
- BigramHash 3072×112 (co-occurrence patterns)
- EMA(0.997) + SWA every 50 steps
- Sliding window eval (stride=64) for free +0.019 BPB
- Muon optimizer with momentum=0.99
- 4000-step cosine warmdown

---

## What This Model Does

This is an exact replica of the current leaderboard SOTA (PR #1019, 2026-03-25). It serves as:

1. **Validation baseline** — Run it locally to confirm your setup works
2. **Reference for technique breakdown** — See which techniques contribute to the 1.1147 BPB
3. **Starting point for experiments** — Copy to `model_002_`, tweak `para.py`, test a hypothesis

---

## How to Use

### 1. See the Launch Command

```bash
python para.py
```

Prints:
```
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 ... torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 2. Launch One Seed (Local Test or First Run)

```bash
python para.py --seed 314
# Copy the command, run it on your RunPod/cluster
```

### 3. Launch All 3 Seeds (Statistical Validation)

```bash
python para.py
# Run each command sequentially (or in parallel with different GPUs)
# Welch's t-test requires all 3 seeds for significance
```

### 4. Record Results

After training:
1. Find the final `Sliding BPB` in `train_gpt.py` output
2. Note the artifact size and steps taken
3. Edit `docs/training-log.md` manually (add final row for each seed)

---

## Technique Breakdown

| Technique | Contribution | Notes |
|-----------|--------------|-------|
| Baseline (Muon + Int6 + GQA + Cosine warmdown) | 1.2244 → 1.1200 | Foundational |
| Sliding window eval | +0.019 BPB | Free gain |
| EMA(0.997) + SWA(50) | +0.002 BPB | Weight regularization |
| BigramHash 3072×112 | +0.004 BPB | Co-occurrence patterns |
| XSA on all 11 layers | +0.003 BPB | Cross-position mixing |
| Full Hessian GPTQ | +0.005 BPB | Post-training quant |
| LZMA compression | +0.001 BPB | Better compression ratio |
| **Total** | | **1.1147 BPB** |

---

## For the Next Model

When you create `model_002_experiment`:
1. Copy this entire folder: `cp -r models/model_001_sota_baseline models/model_002_experiment`
2. Edit `para.py` — change `MODEL_ID`, `MODEL_NAME`, and tweak parameters
3. Run `python para.py` to see the new command
4. Training log auto-records what changed vs SOTA

Example tweaks:
- `BIGRAM_VOCAB_SIZE = 4096` — larger bigram table
- `XSA_LAST_N = 7` — XSA on fewer layers
- `NUM_LAYERS = 10` — shallower model
- `ROPE_DIMS = 32` — more RoPE dimensions

---

## References

- **SOTA Submission**: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`
- **Leaderboard Guide**: `docs/leaderboard-guide.md`
- **Technique Analysis**: `docs/complete-techniques-mapping.md`
- **Training Log**: `docs/training-log.md`

---

## Troubleshooting

**"command not found: torchrun"** — Install PyTorch with distributed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**"CUDA out of memory"** — Reduce `TRAIN_BATCH_TOKENS` in `para.py`

**Flash Attention error** — SOTA requires Flash Attention 3 (H100 only):
```bash
pip install flash-attn --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
```

---

**Last updated**: 2026-04-14
