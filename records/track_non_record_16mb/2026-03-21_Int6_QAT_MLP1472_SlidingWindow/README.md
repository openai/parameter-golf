# Int6 QAT MLP1472 SlidingWindow

**val_bpb: 1.1958** | Artifact: 15.75MB | 12,072 steps in 600s on 8xH100

## Approach

Combines five techniques on top of the baseline architecture:

1. **Int6 Quantization-Aware Training (QAT):** STE fake-quantize to [-31,31] during forward pass in the last 20% of training. Reduces int6 quantization gap from ~0.022 to -0.0007 BPB.

2. **Expanded MLP (hidden=1472):** Int6 packing (4 values per 3 bytes) saves 25% vs int8, freeing space for wider MLP. 21.2M total parameters fit in 15.75MB with zstd-22 compression.

3. **Aggressive Warmdown (WARMDOWN_ITERS=20000):** Linear LR decay far exceeding actual steps (~12K), producing smooth weight distributions optimal for quantization.

4. **FP16 Tied Embeddings:** The tied embedding matrix (serving dual input/output roles) is preserved in FP16 during quantization instead of int6, as it is disproportionately sensitive to quantization.

5. **Batched Sliding Window Eval (stride=64):** Every scored token has near-full 1024 context via overlapping windows. Batched with `torch.unfold` for 62s eval time on 8xH100.

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 9 |
| Model dim | 512 |
| Attention heads | 8 (4 KV heads, GQA) |
| MLP hidden | 1472 (custom, ~2.875x) |
| Vocab | 1024 (SentencePiece) |
| Total params | 21,188,680 |
| Tied embeddings | Yes (FP16 preserved) |
| Quantization | Int6 per-row + zstd-22 |

## Training

| Parameter | Value |
|-----------|-------|
| Hardware | 8x H100 80GB SXM |
| Training time | 600s (wallclock cap) |
| Steps completed | 12,072 / 20,000 |
| Step time | 49.7ms avg |
| Batch tokens | 524,288 |
| Sequence length | 1024 |
| Data | fineweb10B_sp1024 (80 shards) |
| Optimizer | Muon (matrix) + Adam (embed/scalar) |
| Matrix LR | 0.06 |
| Embed LR | 0.07 |
| QAT start | 80% of training (step ~9,661) |

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1965 |
| Post-quant val_bpb | **1.1958** |
| Quantization gap | -0.0007 (QAT eliminates it) |
| Val loss | 2.0191 |
| Artifact size | 15,747,508 bytes |
| Code size | 67,029 bytes |

## Key Insights

- **QAT is essential for int6:** Without QAT, int6 PTQ has 0.022 BPP degradation. With STE QAT (last 20% of training), the gap drops to essentially zero.
- **Weight decay conflicts with QAT:** MuonWD during QAT phase corrupts weight magnitudes. Must disable WD when QAT activates.
- **Z-loss stabilization:** Adding `1e-4 * logsumexp²` to the loss prevents logit drift during aggressive training.

## Reproduction

```bash
WARMDOWN_ITERS=20000 VAL_LOSS_EVERY=0 EVAL_STRIDE=64 \
QUANT_BITS=6 QAT_START_FRAC=0.8 MLP_HIDDEN=1472 \
SMEARGATE=0 BIGRAM_HASH_BUCKETS=0 MUON_WD=0 ADAM_WD=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
