# Universal Transformer with Adaptive Computation Time

## Non-record submission

Exploration of depth-recurrent Universal Transformer with Adaptive Computation Time (ACT).

## The Architecture

Instead of stacking N unique transformer blocks, this model uses a smaller set of **shared blocks** that are applied **recurrently for multiple passes**. A learned halting mechanism (ACT) allows the model to decide per-token how much computation to use.

```
Standard Transformer:    Block1 → Block2 → ... → Block9  (9 unique blocks, 9 effective depth)
Universal Transformer:   [Block1 → ... → Block9] × 2 passes  (9 unique, 18 effective depth)
```

### Key components

- **Shared blocks**: Transformer layers reused across passes, increasing effective depth without adding unique parameters
- **Pass embeddings**: Learned vectors injected at each pass so the model can distinguish recursion depth
- **ACT halting head**: A single linear layer predicting per-token halt probability at each pass. Easy tokens halt early, hard tokens get more passes
- **Ponder cost**: A weighted penalty encouraging early halting — later passes cost more, normalized by `max_passes`

### Baseline components retained
GQA (8 heads, 4 KV heads), RoPE, RMSNorm, relu² MLP, Muon optimizer, tied embeddings, logit softcap, int8 quantization + zlib compression.

## Results

### 8×H100 comparison (10-minute cap)

| Config | Layers | Passes | Effective Depth | Steps | ms/step | Compressed Size | val_bpb |
|--------|--------|--------|-----------------|-------|---------|-----------------|---------|
| Naive baseline | 9 unique | 1 | 9 | 13,780 | 43.54 | 15.86 MB | **1.2244** |
| **This submission** | **9 shared** | **2** | **18** | **7,392** | **81.18** | **15.79 MB** | **1.2409** |

Same unique parameters, same disk budget (~15.8 MB), but 18 effective depth vs 9. The UT achieves nearly the same BPB (only 0.0165 worse) despite completing **almost half the training steps** due to compute overhead from recursion.

### Key finding: recursion works (1×H100, 3-minute exploratory runs)

At equal parameter budget, recursion significantly outperforms single-pass:

| Config | Unique Layers | Passes | Disk (int8+zlib) | val_bpb |
|--------|---------------|--------|------------------|---------|
| 6 layers, no recursion | 6 | 1 | ~4.9 MB | 1.9591 |
| **6 layers, 2 passes** | **6** | **2** | **~5.9 MB** | **1.6390** |

Same unique parameters, **0.32 BPB improvement** from a single extra pass. This demonstrates that depth recurrence provides genuine value — the model learns meaningfully different representations on the second pass through the same weights.

### Additional exploratory runs (1×H100, 3-minute runs)

| Config | val_bpb | Notes |
|--------|---------|-------|
| 3 shared × 3 passes, MLP 2× | 2.3003 | Baseline UT config |
| 3 shared × 3 passes, MLP 3× | 2.3288 | Wider MLP didn't help |
| 3 shared × 5 passes | 2.3887 | More passes hurt — compute cost outweighs depth |
| 3 shared × 3 passes, d768 | 2.5006 | Wider model too slow to train |
| 6 unique, no recursion | 1.9591 | Single-pass reference |
| **6 shared × 2 passes** | **1.6390** | **Recursion sweet spot** |

### Why it doesn't beat baseline (yet)

The bottleneck is **step speed, not architecture quality**. Each pass re-runs all shared blocks, doubling forward/backward time (81ms vs 44ms per step). In a time-limited competition, this means ~46% fewer training steps. The architecture is parameter-efficient but compute-hungry.

## What's Next

1. **Hybrid partial recursion**: Use 7+ unique early layers (run once) with only the last 2-3 layers shared and recursed. This preserves most of the step speed while adding extra effective depth only in the refinement layers.

2. **ACT halting optimization**: Tune the ponder cost coefficient and verify the halting head is learning meaningful per-token computation allocation.

3. **Combining with proven techniques**: MLP 3× expansion, warmdown 4000, EMA/SWA weight averaging, better quantization (int6, GPTQ).

## Reproduction

Trained on 8×H100 SXM with the official RunPod Parameter Golf template.

```bash
RUN_ID=ut_act_d512_L9_P2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
NUM_SHARED_LAYERS=9 \
MAX_PASSES=2 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train_log.txt`):
* Timed training stopped at `7392/20000` steps due to the wallclock cap.
* Pre-quant eval at stop: `val_loss:2.0847`, `val_bpb:1.2347`
* Post-quant roundtrip eval: `val_loss:2.0953`, `val_bpb:1.2409`
* Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.24094165`
* Train time: `600094ms` (`step_avg:81.18ms`)
* Peak memory: `19834 MiB allocated`, `20156 MiB reserved`
* Serialized model int8+zlib: `15743230 bytes`
* Code size: `49574 bytes`
* Total submission size int8+zlib: `15792804 bytes`

## File Structure

```
├── README.md            # This file
├── submission.json      # Submission metadata
├── train_gpt.py         # Modified training script with UT + ACT
└── train_log.txt        # Full training log from 8×H100 run
```
