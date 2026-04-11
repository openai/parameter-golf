# Mixed INT5/INT6 Quantization-Aware Training From Step 1

**Non-Record Submission**
**Author:** Denis ([@BruhTheMomentum](https://github.com/BruhTheMomentum))
**Best result:** 1.3039 val_bpb (post-quantization), 15.4MB artifact
**Hardware:** 8xH100 80GB HBM3, 600s wallclock
**Runs:** 6 runs (~$60 total)

---

## Approach

The core idea is that quantization shouldn't be an afterthought -- the model should train as an INT5/INT6 model from step 1, with full-precision weights serving only as optimizer state.

### Mixed-Precision Quantization

MLP weights get INT5 (max_val=15) and attention weights get INT6 (max_val=31). The rationale: MLP weights are more robust to quantization noise (confirmed empirically -- Run 4 showed attention needs the extra precision), so we spend fewer bits where it matters less. Both bitwidths use int8 storage containers with per-row amax scaling, compressed with zstd level 22.

This saves ~2MB vs uniform INT6, which is the difference between fitting in 16MB and going over budget.

### QAT From Step 1

Every forward pass fake-quantizes weights to their target bitwidth before computing the loss. Gradients flow to the real weights via straight-through estimation (STE) using `.data` swaps -- the compiled model sees fake-quantized weights but the optimizer updates full-precision latent weights.

The key finding: QAT from step 1 dropped quantization degradation from **+0.093 bpb** (post-training quantization, Run 4) to **+0.028 bpb** (Run 5). A 3.3x improvement. The model learns INT5/INT6-friendly weight distributions throughout training, not just at the end.

QAT also acts as an implicit entropy regularizer. Trained weights compress worse than undertrained ones (1.50x vs 1.56x compression ratio for the same model), so pushing weights toward quantization grid points lowers entropy at the source. This matters because zstd-22 already hits 97.6% of the Shannon entropy limit -- the compressor isn't the bottleneck, weight entropy is.

### Architecture

- 11 layers, dim 512, 8 heads, 4 KV heads (GQA), 3x MLP expansion
- 26.5M parameters, 1024 BPE vocabulary, sequence length 1024
- Rotary positional embeddings, U-Net skip connections
- SiLU-gated MLP, tied input/output embeddings
- Muon optimizer (Newton-Schulz) for matrix params, Adam for embeddings/scalars

---

## Run History

| Run | Config | GPUs | Steps | Post-Q bpb | Size | Key Learning |
|-----|--------|------|-------|------------|------|--------------|
| 1 | 9L 2x INT8+zlib | 8 | 4,578 | 1.2921 | 15.8MB | 8 GPU too many for 17M model |
| 2 | 9L 2x INT8+zlib | 4 | 6,104 | **1.2798** | 15.8MB | 4 GPU sweet spot for 17M |
| 3 | 10L 3x INT6+zstd | 4 | 2,861 | 1.3115 | 15.6MB | Slow compile warmup wasted 91s |
| 4 | 10L 3x INT6+zstd | 8 | 6,485 | 1.3434 | 16.3MB | Best pre-Q (1.2500) but late QAT catastrophe |
| 5 | 11L 3x MIX+zstd QAT | 8 | 5,913 | 1.3039 | 15.4MB | QAT from step 1 works |
| 6 | 11L 3x MIX+zstd QAT | 8 | 0 | crash | -- | CUDA graphs incompatible with .data swaps |

### Key Findings

1. **QAT from step 1 >> late QAT.** Run 4 applied QAT in the last 85 steps: +0.093 bpb degradation. Run 5 applied QAT from step 1: +0.028 bpb. The model must train as a quantized model.

2. **Mixed INT5/INT6 saves ~2MB vs uniform INT6.** MLP weights tolerate INT5; attention weights need INT6. Per-tensor max_val routing (`quant_max_val_for(name)`) is simple and effective.

3. **zstd-22 is at the entropy limit.** 97.6% of Shannon theoretical -- can't compress better. The only lever is reducing weight entropy at the source (QAT does this).

4. **Trained weights compress worse.** Run 3 (undertrained, 2861 steps) = 15.6MB. Run 4 (trained, 6485 steps) = 16.3MB. Same model. Higher entropy from more training.

5. **`mode="max-autotune"` enables CUDA graphs** which crash with `.data` swaps and rotary cache reassignment. Use default compile mode.

6. **4 GPU > 8 GPU for 17M model** (all-reduce overhead dominates), **but 8 GPU > 4 GPU for 26.5M model** (enough compute to saturate).

---

## Next Steps

- EMA (0.997 decay) -- competition-proven, ~0.0006 bpb gain
- Weight decay 0.04 on Muon -- improves convergence and compression
- Best-checkpoint selection for quantization
- XSA (Exclusive Self-Attention) -- used by top submissions
- Runtime optimizations (cuDNN SDP, NCCL NVLink tuning) to maximize steps in 600s

---

## Reproducing

```bash
# Requires 8xH100 80GB
torchrun --nproc_per_node=8 train_gpt.py
```

Environment variables: `TRAIN_BATCH_TOKENS=786432`, `MAX_WALLCLOCK_SECONDS=600`

The script downloads data automatically via `cached_challenge_fineweb.py` (requires `data/` directory setup per competition instructions). Training logs and quantized artifacts are saved to the working directory.
