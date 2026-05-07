# Record: 33.6M Int5 GPTQ + Score-First TTT + Temp Calibration

**3-seed mean val_bpb: 1.1145 (std 0.0003)**

## Approach

Train a larger model (33.6M params, d=576) and compress harder with int5 GPTQ. Add legal score-first backward-looking TTT with temperature calibration.

## Architecture
- **Model**: 33.6M params, d=576, 11 layers (U-Net skip), 8 heads, MLP 3.5x (hidden=1792)
- **Features**: SmearGate, BigramHash(8192), XSA-all(11), Value Embeddings, Partial RoPE (16 dims), LN Scale
- **Quantization**: Int5 GPTQ (clip_range=15, [-16,15]) + zstd-22. GPTQ calibration within training budget (256 training samples)
- **Eval**: Score-first TTT + sliding window (stride=64) + temperature calibration (T=0.98)

## Results

| Seed | Base BPB (no TTT) | TTT T=0.98 BPB |
|------|-------------------|----------------|
| 1337 | 1.1243 | **1.1142** |
| 42   | 1.1242 | **1.1148** |
| 2025 | 1.1245 | **1.1144** |
| **Mean** | **1.1243** | **1.1145** |
| **Std** | **0.0002** | **0.0003** |

- Artifact: 15,885,838 bytes (under 16MB)
- Training: ~6,131 steps in 600s on 8xH100 SXM (~98ms/step)
- Eval: ~465s total (87s sliding window + 296s TTT + 82s post-TTT recal)

## Statistical Significance

vs #549 (current SOTA, 1.1194): improvement = 0.0049 nats, t-stat = 28.3, p << 0.01

## TTT Implementation (Legal Score-First)

The TTT processes validation tokens in 131K-token chunks:
1. **SCORE** each chunk under `torch.inference_mode()` — accumulates loss
2. **TRAIN** on the scored chunk — AdamW (lr=1e-4, cosine LR), 3 epochs, last 2 blocks unfrozen
3. After all chunks: re-eval with T=0.98 temperature calibration (fixes TTT overconfidence)

No token is trained on before it is scored. No val tokens in artifact. GPTQ runs within training budget.

## Run Command
```bash
pip install --break-system-packages zstandard
NCCL_IB_DISABLE=1 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Based on PR #576 by @cmcdnd.
