## PR315 Recipe Reproduction on 1xH100 PCIe

Reproduction of PR #315 recipe on a single H100 PCIe GPU (RunPod, $2.39/hr).
Uses Flash Attention 2 instead of FA3.

### Results

| Metric | Value |
|--------|-------|
| Sliding-window BPB | **1.8338** |
| Pre-quant BPB | 1.4192 |
| Post-quant roundtrip BPB | 1.8398 |
| Steps | 492 |
| Wallclock | 600s (10min) |
| Artifact size | 10.0MB (int6+zstd) |
| Peak memory | 20785 MiB |

### Notes

- 1xH100 PCIe limits training to ~492 steps (vs ~6200 on 8xH100 SXM)
- QAT enabled at step 452 with only ~40 steps of adaptation, causing higher quantization loss (+0.42 BPB)
- On 8xH100 SXM this recipe achieves ~1.13 BPB with ~6200 steps and proper QAT convergence

### Configuration

- 11 layers, 512 dim, 8 heads, 4 KV heads, 3x MLP
- XSA (last 4 layers), EMA 0.997, Partial RoPE (16/64 dims)
- LN Scale, Late QAT (threshold 0.1)
- BigramHash 2048, SmearGate
- Muon optimizer + weight decay 0.04

### Run Command

```bash
pip install zstandard flash-attn

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=200 WARMDOWN_ITERS=400 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Hardware

- RunPod 1xH100 PCIe 80GB ($2.39/hr)
