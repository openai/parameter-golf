# Depth-Recurrent Transformer with Competitive Recipe

**Author:** Brian Mwai (@brn-mwai)
**Score:** TBD (pending 8xH100 validation)

## Architecture

Two modes, controlled by `DEPTH_RECURRENCE` env var:

### Mode A: Standard (11 unique layers)
- 11 layers, 512 dim, 3x MLP (hidden=1536)
- 8 attention heads, 4 KV heads (GQA)
- ~27M parameters

### Mode B: Depth Recurrence
- 2 prelude + 1 shared (looped 7x) + 2 coda = 5 unique blocks, 11 effective
- Iteration embeddings tell shared block which pass it's on
- Freed parameters can go to wider model (640+ dim)

## Techniques

| Technique | Impact | Source |
|-----------|--------|--------|
| Int6 quantization + zstd-22 | ~30% size savings vs int8+zlib | Competition meta |
| 11 layers, MLP 3x | Funded by int6 savings | PR #162 |
| SmearGate | ~0.005 BPB from token blending | PR #135 |
| BigramHash | Token-pair context embeddings | PR #162 |
| Muon weight decay (0.03) | Quant-friendly weight distributions | PR #179 |
| SWA (last 50%) | Smoother weights, better quant | PR #162 |
| Sliding window eval (stride=64) | ~0.03 BPB free improvement | PR #56 |
| Orthogonal init | Better training dynamics | PR #135 |
| FP16 embedding passthrough | Most quant-sensitive tensor | PR #42 |
| Depth recurrence (optional) | Novel - share blocks, widen model | Original |

## Usage

Standard mode:
```bash
RUN_ID=standard_v1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Depth recurrence mode:
```bash
RUN_ID=depth_recur_v1 \
DEPTH_RECURRENCE=1 \
MODEL_DIM=640 \
NUM_PRELUDE=2 \
NUM_CODA=2 \
RECURRENT_LOOPS=7 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Theoretical Motivation

From MDL (Minimum Description Length) theory: a model with 3 unique layers looped 3 times has lower L(model) than 9 unique layers. The freed description bits go to better L(data|model). Combined with the competitive recipe, this should push BPB below the current frontier.
