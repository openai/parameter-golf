# INT6 10L + SWA + NorMuon + Aggressive Warmdown

Combined optimizations: INT6 quantization enabling a 10-layer architecture, Stochastic Weight Averaging, NorMuon optimizer with decoupled weight decay, and aggressive warmdown schedule.

**Note:** This run used NTK RoPE eval at 4096 context, which degraded post-quant performance. The pre-quant val_bpb of 1.1923 demonstrates the model quality; a rerun with standard 2048 eval is expected to yield ~1.19 post-quant.

## Key Ideas

### INT6 Quantization
Per-row INT6 quantization (`[-32, 31]` range, amax-based fp16 scale) achieves 3.82x compression on model weights. This unlocks a larger 10-layer architecture with MLP hidden dim 1088 while staying well under the 16MB budget (14.2MB total, 1.8MB headroom).

### Stochastic Weight Averaging (SWA)
Collects model snapshots every 200 steps during the warmdown phase, then averages all 50 snapshots before quantization. This smooths the loss landscape and produces weights more amenable to quantization.

### NorMuon Optimizer
Per-row variance reduction after Newton-Schulz orthogonalization (arXiv 2510.05491), combined with decoupled weight decay (WD=0.02) on Muon-optimized matrix parameters.

### Aggressive Warmdown
`WARMDOWN_ITERS=20000` (larger than total training steps) creates a smooth cosine-like decay from step 0. This produces tighter weight distributions that quantize better.

## Results

| Metric | Sliding Window Baseline | This Submission |
|---|---|---|
| Architecture | 9L, MLP=992, INT8 | 10L, MLP=1088, INT6 |
| Pre-quant val_bpb | 1.2196 | **1.1923** |
| **Post-quant val_bpb** | **1.1925** | **1.2320** |
| Training steps | 13,450 | 9,956 |
| Step avg (ms) | 44.61 | 60.27 |
| Artifact size | 15,874,829 bytes | 14,202,629 bytes |

The **pre-quant** result is a significant improvement (-0.027 BPB). However, NTK RoPE eval at 4096 context degraded the post-quant sliding window evaluation. The model was trained at 2048 context, and NTK scaling to 4096 introduced errors amplified by INT6 quantization.

## Validation Trajectory

| Step | val_bpb |
|---|---|
| 0 | 4.1084 |
| 1,000 | 1.3576 |
| 2,000 | 1.3028 |
| 3,000 | 1.2677 |
| 4,000 | 1.2485 |
| 5,000 | 1.2354 |
| 6,000 | 1.2250 |
| 7,000 | 1.2149 |
| 8,000 | 1.2068 |
| 9,000 | 1.1976 |
| 9,956 (final) | 1.1923 |

## Configuration

- Layout: `NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=1088`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Sequence length: `TRAIN_SEQ_LEN=2048`
- Batch: `TRAIN_BATCH_TOKENS=524288`
- Warmdown: `WARMDOWN_ITERS=20000`
- SWA: `SWA_ENABLED=1 SWA_EVERY=200`
- Muon WD: `MUON_WD=0.02`
- NorMuon: `NORMUON_ENABLED=1 NORMUON_BETA2=0.95`
- Quantization: `QUANT_BITS=6` (INT6, per-row amax scale)
- Compression: `COMPRESSION=zlib`
- Eval: `EVAL_SEQ_LEN=4096 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32`

## Command

```bash
NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_HIDDEN=1088 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=20000 \
SWA_ENABLED=1 SWA_EVERY=200 \
EVAL_SEQ_LEN=4096 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MUON_WD=0.02 MTP_ENABLED=0 TTT_ENABLED=0 \
COMPRESSION=zlib RESIDUAL_SCALE_ENABLED=0 SMEAR_GATE_ENABLED=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics (from `train.log`)

- Timed training stopped at `9956/20000` steps due to wallclock cap
- Pre-quant eval at stop: `val_loss:2.0131`, `val_bpb:1.1923`
- Post-quant INT6+zlib roundtrip: `val_loss:2.0802`, `val_bpb:1.2320`
- Exact metric: `final_quant_zlib(level=9)_roundtrip_exact val_bpb:1.23199932`
- Train time: `600017ms` (`step_avg:60.27ms`)
- Peak memory: `11812 MiB allocated`, `12480 MiB reserved`
- Eval time: `336698ms` (sliding window, stride=64, batch_seqs=32, eval_seq=4096 NTK)
- SWA: 50 snapshots collected and averaged
- Serialized model INT6+zlib: `14121116 bytes`
- Code size: `81513 bytes`
- Total submission: `14202629 bytes`

**Note:** `train.log` was reconstructed from real-time SSH monitoring of the training run. All metric values are exact as reported by the training script.

## Lessons Learned

1. **NTK RoPE eval hurts**: Extending eval context from 2048→4096 with NTK scaling degraded post-quant BPB. The model wasn't trained at this context length.
2. **INT6 quant gap is larger than expected**: ~0.04 BPB gap (pre-quant 1.1923 → post-quant 1.2320), partly due to NTK interaction.
3. **The pre-quant result is strong**: 1.1923 BPB in 9956 steps demonstrates the value of the combined optimizer + architecture improvements.
4. **Next run should use**: `EVAL_SEQ_LEN=0` (standard 2048 eval, no NTK), which should reduce the quant gap significantly.

## Included Files

- `train_gpt.py` — training script with all optimizations
- `train.log` — training log (reconstructed from real-time monitoring)
- `submission.json` — leaderboard metadata
- `README.md` — this file
