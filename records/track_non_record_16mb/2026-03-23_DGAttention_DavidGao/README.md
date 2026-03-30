# DG Attention: Differential-Gated Attention with Depth-Scheduled Novelty Encoding

**val_bpb: 1.1898** | 2,979 steps @ 201ms/step | 8xH100 SXM, 600s

Novel attention mechanism where deep layers transmit *what's new* about each token instead of raw content. Named "DG" for **D**esignator/**G**radient.

## Run

```bash
# DG attention
ATTN_VARIANT=dg torchrun --standalone --nproc_per_node=8 train_gpt.py

# Standard attention baseline for comparison
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
