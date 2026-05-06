# DG Attention: Depth-Scheduled Differential Payloads for Language Modeling

**val_bpb: 1.1898** (initial submission) → **1.1554** (v4.1, post-quantization) | 8xH100 SXM

Alternative attention mechanism where deep layers transmit inter-token changes instead of absolute content, with a parameter-free linear depth schedule. Named "DG" for **D**esignator/**G**uided.

## Results

DG Attention matches standard attention but does not definitively improve it:

| Variant | val_bpb (post-quant) |
|---------|---------------------|
| DG Attention (v4.1) | 1.1554 |
| Matched Standard | 1.1516 |

The 0.004 BPB gap is within noise. DG does use 33% less VRAM and produces a 6% smaller artifact, but these savings don't translate to better language modeling quality at this scale.

The main contribution is documenting the design trajectory: four iterations, a scale-dependent gate-collapse phenomenon, and variance ratio analysis showing DG induces steeper inter-token redundancy gradients across depth — even when aggregate BPB is unchanged.

See the [full paper](../../paper/dg_attention.pdf) for details.

## Run

```bash
# DG attention
ATTN_VARIANT=dg torchrun --standalone --nproc_per_node=8 train_gpt.py

# Standard attention baseline for comparison
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
