# Aweb GDN — Gated DeltaNet + EMA + Warmdown + TTT

## Architecture
- 8 layers: 7 GatedDeltaNet (linear attention, O(n)) + 1 standard Attention
- 384 dim, 6 heads, SiLU activation, 4x MLP expansion
- Unigram frequency bias in lm_head
- Tied embeddings, depth-scaled residuals (1/√(2·layer))

## Enhancements over PR #875
- EMA(0.997) weight averaging
- Cosine warmdown (last 30% of training)
- Per-row int8 quantization + LZMA (vs per-tensor int8 + zip)
- Proper SentencePiece BPB evaluation (vs approximate /3.5)
- Score-First TTT (3 epochs SGD, momentum=0.9)

## Reproduction

```bash
pip install flash-linear-attention==0.4.2 fla-core==0.4.2
TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-28_AwebGDN/train_gpt.py
```

## Author
Daniel Wahnich (@manfromnowhere143)
