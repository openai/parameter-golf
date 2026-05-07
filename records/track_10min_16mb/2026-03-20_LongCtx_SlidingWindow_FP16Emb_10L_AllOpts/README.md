# Long Context + Sliding Window + FP16 Embed + 10L + All Optimizations

**Expected val_bpb: ~1.155-1.165** (pending 8xH100 validation)

## Key Techniques

This submission combines the strongest techniques from prior SOTA entries:

### From SOTA (SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit):
1. **Sliding window evaluation** (stride=64, eval_seq_len=1024): Every token scored with 960+ context.
2. **FP16 tied embedding export**: Avoids int8 quantization error compounding through both input/output paths.
3. **10 transformer layers** (up from 9): Extra depth with Muon weight decay keeping size in budget.
4. **Decoupled weight decay for Muon** (0.02): Improves generalization and quantization robustness.
5. **Overtone spectral embedding init**: SVD power-law spectrum shaping (`S_k ~ k^{-0.5}`).
6. **Phase-transition residual mixing**: Sigmoid-scheduled `resid_mix` initialization.

### From TrainingOptSeq4096 (training optimization):
7. **Long training context** (TRAIN_SEQ_LEN=2048): 2x more context per token during training. Chosen over 4096 as a balance between context benefit and step count with 10L model.
8. **Higher Muon momentum** (0.99 vs 0.95): Stronger gradient smoothing for stability with long context.
9. **Extended momentum warmup** (from 0.92 over 1500 steps): Gradual ramp to high momentum.
10. **Lower learning rates** (MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.06): Dramatically reduces int8 quantization gap.
11. **Smaller batch** (393K tokens vs 524K): More optimizer updates per wallclock second.
12. **Longer warmdown** (3600 iters): Smoother weight distributions for better quantization.

## Rationale

The previous SOTA (1.1748 BPB) used baseline training (seq_len=1024) with excellent eval-time and quantization tricks. The Seq4096 submission (1.2014 BPB) showed that long-context training with conservative LRs and high momentum dramatically improves pre-quant quality, but didn't use sliding window eval or FP16 embeddings.

This submission bridges the gap: apply the aggressive training optimizations (long context, conservative LRs, high momentum) to the full SOTA eval/quantization pipeline.

Key expected improvements over SOTA:
- **Better pre-quant quality**: Long context (2048) + conservative LRs + high momentum
- **Smaller quant gap**: Lower LRs + longer warmdown produce smoother weight distributions
- **Same eval boost**: Sliding window + FP16 embeddings preserved

## Hyperparameter Summary

| Parameter | SOTA | Seq4096 | This |
|-----------|------|---------|------|
| train_seq_len | 1024 | 4096 | **2048** |
| train_batch_tokens | 524K | 393K | **393K** |
| num_layers | 10 | 9 | **10** |
| matrix_lr | 0.04 | 0.02 | **0.02** |
| scalar_lr | 0.04 | 0.02 | **0.02** |
| tied_embed_lr | 0.10 | 0.06 | **0.06** |
| muon_momentum | 0.95 | 0.99 | **0.99** |
| momentum_warmup_start | 0.85 | 0.92 | **0.92** |
| momentum_warmup_steps | 500 | 1500 | **1500** |
| warmdown_iters | 2500 | 3000 | **3600** |
| eval_stride | 64 | 0 | **64** |
| FP16 embed export | Yes | No | **Yes** |
| Overtone init | Yes | No | **Yes** |
| Muon weight decay | Yes | No | **Yes** |

## Running

```bash
torchrun --nproc_per_node=8 train_gpt.py
```
