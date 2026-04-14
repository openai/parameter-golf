# Warmdown-Quantization: Training for Compression

## Score
**val_bpb = 1.2154** (baseline: 1.2244, improvement: 0.009 BPB / 0.017 nats)

## Key Insight

On 8xH100, the dominant bottleneck isn't model quality — it's quantization quality. The post-training int8 quantization penalty (0.014 BPB with default settings) is larger than most hyperparameter improvements combined. We attack this bottleneck from multiple angles.

## Novel Contributions

### 1. Always-Decaying Learning Rate Schedule (WARMDOWN_ITERS=20000)

Setting WARMDOWN_ITERS far beyond the actual training steps (~12,200) produces dramatically better post-quantization quality. The LR decays linearly from 61% of peak at step 0 to near-zero at the final step.

Aggressive LR decay produces tighter weight distributions with fewer outliers. Since int8 quantization error is proportional to the weight range per row, smoother weights map to the int8 grid with much less damage.

Post-quant penalty drops from 0.014 BPB (WD=1200 default) to 0.005 BPB (WD=20000). We mapped the full curve across 10 warmdown values, finding the sweet spot at WD=20000 where the entire training run is in the decay phase. WD=30000 overshoots — too little high-LR learning.

### 2. FP16 Tied Embeddings

The tied embedding matrix (tok_emb.weight) serves dual roles as input lookup and output projection. Int8 quantization causes disproportionate damage because small errors affect both input representation quality AND output logit accuracy. Keeping it in fp16 during quantization reduces the remaining post-quant penalty from 0.005 to ~0.001 BPB at a cost of ~500KB (offset by reducing MLP hidden from 1024 to 992).

### 3. Optimal NTK-RoPE Extrapolation for Well-Trained Models

The optimal eval sequence length depends on training convergence:
- Undertrained models (1xH100, ~1,600 steps): eval@2048 gives +0.048 BPB
- Well-trained models (8xH100, ~12,200 steps): eval@2048 is neutral-to-negative; eval@1408 (1.375x) is optimal (+0.007 BPB)

Well-trained models develop precise position-dependent patterns that aggressive NTK extrapolation distorts. Moderate extrapolation provides useful extra context without excessive distortion.

### 4. Optimizer-Warmdown Interaction

MUON_BACKEND_STEPS=5 outperforms 7 when combined with aggressive warmdown (WD=20000), despite 7 outperforming 5 at default warmdown (WD=2400). When warmdown already produces smooth weights, more training steps are more valuable than better per-step gradient quality.

## Configuration

```
WARMDOWN_ITERS=20000 MATRIX_LR=0.06 TIED_EMBED_LR=0.07 SCALAR_LR=0.06
GRAD_CLIP_NORM=1.0 MUON_BACKEND_STEPS=5 EVAL_SEQ_LEN=1408
```
- FP16 tied embedding (tok_emb.weight kept in fp16 during int8 export)
- MLP_HIDDEN=992 (offset FP16 embedding overhead)

## Reproduction

```bash
WARMDOWN_ITERS=20000 MATRIX_LR=0.06 TIED_EMBED_LR=0.07 SCALAR_LR=0.06 \
GRAD_CLIP_NORM=1.0 MUON_BACKEND_STEPS=5 EVAL_SEQ_LEN=1408 MLP_HIDDEN=992 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware Note

Results obtained on RunPod 8xH100 SXM (47-48ms/step vs baseline's 43.5ms/step). Scores should improve when re-evaluated on OpenAI's faster hardware due to additional training steps within the 10-minute window.
