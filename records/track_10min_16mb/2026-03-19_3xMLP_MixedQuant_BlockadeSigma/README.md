This record captures a 10-minute 8xH100 submission using 3x MLP expansion with mixed INT8/INT6 quantization.

## Approach

Three changes from the baseline, each independently motivated:

### 1. 3x MLP Expansion (architecture)

Wider MLPs (3x vs 2x baseline) add ~50% more parameters but capture richer token representations. The tradeoff: slower steps (~81ms vs ~72ms) but substantially lower BPB. At 10 minutes, the quality-per-byte gain outweighs fewer total steps.

The 3x MLP pushes INT8 model size to ~17MB (over the 16MB cap), requiring mixed quantization.

### 2. Mixed Quantization: INT8 Attention + INT6 MLP (compression)

Attention weights (Q, K, V, output projections) use INT8 quantization. MLP weights (gate, up, down projections) use INT6. Rationale:

- **Attention is precision-sensitive**: Q/K dot products amplify quantization error through softmax
- **MLP is robust**: Feed-forward layers tolerate lower precision with minimal BPB degradation
- **MLPs are 2/3 of parameters**: INT6 on MLPs saves the most bytes for the least quality cost

Results across quantization methods (same trained weights):

| Method | BPB | Artifact Size | Under 16MB? |
|--------|-----|---------------|-------------|
| INT8 | 1.1723 | 17.05 MB | No |
| **Mixed (INT8 attn + INT6 MLP)** | **1.1812** | **15.37 MB** | **Yes** |
| INT6 | 1.1859 | 14.44 MB | Yes |

Mixed quantization recovers ~50% of the INT8-to-INT6 quality gap while fitting comfortably under the 16MB cap.

### 3. Attention Diversity + Uncertainty-Gated Residuals (training signal)

Two lightweight forward-pass modifications with zero inference overhead:

- **Blockade attention** (strength=0.15): Inter-head suppression inspired by Rydberg blockade dynamics. Heads with overlapping attention patterns suppress each other via soft logit penalties, encouraging representational diversity across the head ensemble.

- **Sigma residual** (strength=0.3): Per-head uncertainty estimate modulates residual stream contribution. Uncertain heads have dampened residual connections, preventing noisy gradients from destabilizing training early in convergence.

Both features cost <1ms/step overhead.

### Other tuning

- **LR=0.03** (vs baseline default): Optimal for 3x MLP in the 10-minute budget
- **Warmdown=2500 iters**: Cosine LR decay in final phase; warmdown acceleration provides the largest BPB gains
- **Sliding window eval at stride=64**: Standard sliding window for more accurate BPB measurement
- **Tied embeddings with LR=0.10**: Higher embedding LR helps with small vocab

## Configuration

```bash
export MODEL_DIM=512 NUM_LAYERS=9 NUM_HEADS=8 NUM_KV_HEADS=4
export MLP_MULT=3
export USE_BLOCKADE=1 BLOCKADE_STRENGTH=0.15
export USE_SIGMA_RESIDUAL=1 SIGMA_RESIDUAL_STRENGTH=0.3
export MATRIX_LR=0.03 SCALAR_LR=0.03
export MUON_WD=0.02
export TIED_EMBED_LR=0.10
export WARMDOWN_ITERS=2500
export EVAL_STRIDE=64 EVAL_SEQ_LEN=0

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- **Training**: 7,441 steps in 600s (80.65 ms/step) on 8xH100 PCIe
- **Pre-quant val_bpb**: 1.2053 (raw FP32 at wallclock stop)
- **INT8 roundtrip**: val_bpb=1.1723 (17.05 MB, over cap)
- **Mixed roundtrip**: val_bpb=1.1812 (15.37 MB, under cap)
- **INT6 roundtrip**: val_bpb=1.1859 (14.44 MB, under cap)
- **Model params**: 21,778,504
- **Peak memory**: 11,255 MiB allocated

## Hardware Note

This run was on 8xH100 **PCIe** (80.65 ms/step). H100 SXM2 would achieve ~1.3x more steps in the same 10 minutes, placing more steps in the warmdown acceleration zone where BPB drops fastest. An extended 15-minute run on the same hardware reached 11,153 steps and 1.1616 BPB (INT8), demonstrating significant remaining headroom from additional steps.

## Included Files

- `train_gpt.py` — Full training script (self-contained, muon_t3 import is optional)
- `train.log` — Training log from 8xH100 PCIe run
- `submission.json` — Leaderboard metadata
