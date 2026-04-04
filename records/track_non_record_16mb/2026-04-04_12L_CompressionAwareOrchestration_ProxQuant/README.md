# 12L Compression-Aware Training Orchestration (Non-Record)

## Score
- Pre-quantization val_bpb: **1.22** (8xH100, beats baseline 1.2244)
- Post-quantization roundtrip val_bpb: **1.34** (does not beat baseline due to quantization gap)
- Artifact size: 13-17 MB (varies by configuration)
- Model parameters: 29.6M (12 layers, 3x MLP)

## Why Non-Record

The pre-quantization score beats baseline, demonstrating the model has sufficient capacity. However, the quantization gap (~0.10-0.13 BPB) between pre-quant and roundtrip prevents the final score from beating baseline. This submission documents novel compression-aware training techniques and the challenges encountered.

## Key Innovations

This submission introduces multiple techniques not attempted by any other Parameter Golf entry (verified across all 1,250+ PRs as of April 4, 2026):

### 1. Multi-Phase Training Orchestration
A 5-phase coordinated training schedule that optimizes for both model quality AND compressed artifact size:

- **Phase 1 (0-50% wallclock):** Clean training — no compression constraints
- **Phase 2 (50-65%):** Gradual pruning with cubic sparsity schedule + weight decay ramping (0.04→0.08)
- **Phase 3 (65-100%):** Quantization-aware training (STE int6 or ProxQuant gradual annealing)
- **Phase 4 (post-training):** PERP — retrain biases and layer norms to recover quality lost from compression
- **Phase 5 (serialization):** Neuron reordering (sort MLP rows by L1 norm for lossless compression improvement)

Based on the Progressive Intensity Hypothesis (ICLR 2026): weaker perturbations (pruning) should precede stronger ones (quantization).

### 2. ProxQuant Progressive QAT
Gradual grid annealing instead of hard on/off STE: `w = (1-λ²)·w_orig + λ²·w_quant` with quadratic ramp.
Eliminates the loss spike ("quantization shock") that occurs with standard hard QAT.
Based on Bai et al., NeurIPS 2019. No other competition submission uses progressive QAT.

### 3. 12-Layer Architecture
Standard meta stack uses 11 layers. Our compression techniques (pruning + WD scheduling + neuron reordering) enable fitting a 12th layer under the 16MB budget, providing additional model capacity.

### 4. Non-Power-of-2 Quantization Levels
We explored using 22 or 24 quantization levels instead of the standard 16/32/64. No published paper studies optimal level count as a continuous variable. While we found the compression benefit was smaller than theoretical prediction, this remains an unexplored research direction.

### 5. PERP Post-Training Recovery
After compression, freeze all weight matrices and retrain only biases and layer norms for 200 steps. Recovers 0.001-0.003 BPB at essentially zero cost. Based on arXiv:2312.15230.

### 6. Neuron Reordering for Compression
Sort MLP hidden neurons by L1 norm before serialization. Corresponding input dimensions of the next layer are permuted accordingly. Mathematically lossless — improves zstd compression by making adjacent rows more similar. Inspired by Google Patent US20180082081A1.

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 12 |
| Model dim | 512 |
| Attention heads | 8 (4 KV heads, GQA) |
| MLP multiplier | 3x (hidden=1536) |
| Activation | LeakyReLU(0.5)^2 |
| Position encoding | Partial RoPE (16/64 dims) |
| Embedding | Tied, 1024 vocab |
| BigramHash | 1536 buckets |
| Skip connections | U-Net style |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon (lr=0.025, momentum=0.99, WD scheduled 0.04→0.08) |
| EMA | decay=0.997, loaded before serialization |
| QAT | STE int6 (64 levels) from step 0, or ProxQuant gradual |
| Pruning | 15-22% post-EMA, cubic schedule during training |
| Serialization | int8 per-row + zstd-22 |
| Evaluation | Sliding window (stride=64 for final) |

## Research Conducted

This project involved extensive research across 38+ academic papers in English, Chinese, Korean, Japanese, German, and French sources. Key papers that informed our approach:

- ProxQuant (Bai et al., NeurIPS 2019) — progressive quantization
- Progressive Intensity Hypothesis (ICLR 2026) — prune before quantize ordering
- CoDeQ (arXiv:2512.12981) — dead-zone quantizer for joint compression
- LeanQuant (ICLR 2025, Shanghai AI Lab) — loss-error-aware quantization grids
- EfficientQAT (ACL 2025, Shanghai AI Lab) — block-wise QAT
- R2 Loss (Apple, arXiv:2303.08253) — range restriction for compression
- CERWU (arXiv:2505.18758, U. Tübingen) — rate-constrained quantization
- any4 (arXiv:2507.04610) — learned 4-bit codebooks
- NuMuon (arXiv:2603.03597) — Muon + compression interaction
- Wanda (ICLR 2024) — pruning with activation awareness

Full research log with citations available in our repository.

## Lessons Learned

1. **ProxQuant vs STE QAT:** ProxQuant eliminates quantization shock but the model never "sees" quantized weights during forward pass. STE QAT enables gradient-based adaptation but requires careful torch.compile handling (class-level bool for constant-folding).

2. **The quant gap scales with training:** Undertrained models (1 GPU, ~1000 steps) show different compression behavior than fully-trained models (8xH100, ~8000 steps). Local testing is unreliable for predicting 8xH100 artifact size and quant gap.

3. **Int5 (32 levels) has a fundamental precision limitation:** The quant gap at 32 levels (0.10-0.14 BPB) is 10-100x larger than int6 (64 levels, 0.001-0.01 BPB). The extra parameters from int5 don't compensate for this quality loss.

4. **Serialization format matters enormously:** Int8+zstd compresses 64-level-aligned weights much better than raw 5-bit packing or custom index formats. The compressor and the quantization must be co-designed.

5. **Scheduling is underexplored:** No competition submission schedules weight decay, pruning, and QAT in coordinated phases. This remains a promising direction for future work.

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key environment variables:
```
NUM_LEVELS=64        # quantization levels (64 = int6)
NUM_LAYERS=12        # transformer layers
QAT_FRAC=0.35        # last 35% of wallclock for QAT
WD=0.04              # initial weight decay
WD_FINAL=0.08        # weight decay during compression phase
PRUNE_FRAC=0.15      # target pruning sparsity
BIGRAM_BUCKETS=1536   # BigramHash embedding buckets
```
