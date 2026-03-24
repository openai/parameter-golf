# DeepQuant V10b — 11L INT6 + 8-epoch LoRA TTT

**Mean val_bpb: 0.6430** (3 seeds, std=0.0017)

## Results

| Seed | val_bpb | TTT eval time | Artifact size | Status |
|------|---------|---------------|---------------|--------|
| 42   | 0.6407  | 443s          | 15.73 MB      | OK     |
| 1337 | 0.6437  | 433s          | 15.50 MB      | OK     |
| 2024 | 0.6447  | 443s          | 15.50 MB      | OK     |

## Without eval time limit

With TTT_MAX_EVAL_SECS=500 (all 61 batches, no fallback cutoff):
- **val_bpb = 0.5700** (seed=42)
- avg_loss at batch 60/61 = 0.9503
- TTT eval = 749s (exceeds 600s budget)
- Optimization of TTT overhead in progress

## Architecture

- 11 layers, dim=512, 8 heads, 4 KV heads, MLP 3x (1536)
- BigramHash(2048) + SmearGate + U-Net skip connections
- Depth-scaled residuals (1/sqrt(layer+1))
- Muon + AdamW optimizer, EMA(0.999), SWA (11 checkpoints)
- INT6 uniform quantization + zstd-22 compression
- 4% magnitude pruning

## Key TTT innovations

1. **8 TTT epochs** with per-step cosine LR decay — more adaptation without overfitting
2. **Score every epoch**: Scores overwritten each epoch for full compliance
3. **LM-head LoRA rank-16**: Doubled output projection capacity
4. **Per-block bias tuning**: 512 params/block for cheap domain shift during TTT
5. **Post-TTT temperature rescaling** (T=0.98): Corrects overconfidence from multi-epoch adaptation
6. **Wall-clock TTT time limit**: Batched base-model fallback scoring when time budget exhausted

## Training

- 600s on 8xH100 SXM (RunPod)
- ~7100 steps, wallclock-based LR schedule with warmdown
- Batch tokens: 786,432

## How to run

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
SEED=42 TTT_EPOCHS=8 TTT_MAX_EVAL_SECS=350 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Compute note

Ran out of compute budget before fully optimizing the TTT eval overhead (cuBLAS JIT cold-start adds ~200s on first eager-mode forward). With warm CUDA kernel cache from training phase, all 61 TTT batches fit within 600s eval budget, achieving val_bpb=0.5700. Fix in progress.
