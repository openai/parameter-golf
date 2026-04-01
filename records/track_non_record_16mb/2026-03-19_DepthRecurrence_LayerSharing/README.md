# Depth Recurrence via Layer Sharing

## Approach

Use 3 unique transformer blocks cycled over N virtual layers (ALBERT-style weight sharing). The model has the same forward-pass depth as a standard transformer but 1/3 the unique parameters, freeing massive artifact budget for wider layers or deeper recurrence.

Key changes to `train_gpt.py`:
- `NUM_UNIQUE_LAYERS` env var controls how many distinct blocks are created (0 = baseline behavior)
- Virtual layer → physical block mapping via `layer_map = [i % num_unique for i in range(num_layers)]`
- Encoder/decoder skip connections and `x0` residual blending preserved across shared blocks
- Optimizer (Muon + Adam) automatically handles shared params — gradients accumulate from all applications

## Local Validation (Apple Silicon, mini shards)

Tested on 500K-token subset of FineWeb, 2048 batch, 100 steps. Not comparable to H100 scores — only relative differences matter.

| Config | Unique Params | Virtual Depth | Post-quant BPB | int8+zlib Size |
|--------|--------------|---------------|----------------|----------------|
| **Baseline** (9 unique, 512d) | 17.1M | 9 | 3.157 | ~5.0MB |
| **3 shared, 512d** | **6.0M** | **9** | **3.151** | **~1.6MB** |
| 3 shared, 640d | 8.5M | 12 | 3.174 | 2.4MB |
| 3 shared, 768d | 12.6M | 12 | 3.208 | 3.5MB |

3 shared layers at 512d matches or slightly beats the 9-unique-layer baseline with **1/3 the parameters** and **30% faster training** (173ms/step vs 247ms/step).

## Why It Works

1. Each shared block receives gradient signal from all N/K virtual applications per step — richer updates
2. The `resid_mix` (x0 injection) provides an identity path that prevents representation collapse across recurrences
3. int8+zlib serializes only unique parameters, so 6M params → ~1.6MB instead of 17M → ~5MB
4. With ~14MB of freed artifact budget, the model can go wider (768d+), use MLP 3x expansion, or accommodate larger vocabularies

## Composability with the Meta

This approach stacks cleanly with the dominant competition techniques:
- **Int6 + zstd-22**: Smaller per-param footprint × fewer unique params = even more headroom
- **MLP 3x**: Freed budget funds the wider MLP
- **Sliding window eval**: Orthogonal improvement, no interaction
- **FP16 tied embedding**: Compatible, embedding is not shared across layers
- **Extra recurrence at eval time**: Unique to layer sharing — run more cycles of the shared blocks at test time for free BPB gains

## Suggested H100 Configs

```bash
# Config A: Drop-in replacement (same speed as baseline, 1/3 params)
NUM_UNIQUE_LAYERS=3 torchrun --nproc_per_node=8 train_gpt.py

# Config B: Wider + deeper
NUM_UNIQUE_LAYERS=3 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=2 NUM_LAYERS=12 \
  torchrun --nproc_per_node=8 train_gpt.py

# Config C: Combined with MLP 3x
NUM_UNIQUE_LAYERS=3 MLP_MULT=3 NUM_LAYERS=12 \
  torchrun --nproc_per_node=8 train_gpt.py
```

## Status

- Local experiments complete (Apple Silicon, mini data)
- Awaiting H100 compute for full validation
- Code changes are minimal (~20 lines to `train_gpt.py`)
