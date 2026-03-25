# CTW N-gram Backoff + Legal TTT + Full SOTA Stack

**val_bpb: TBD** (pending 8xH100 evaluation)

## Key Innovation: CTW-Weighted N-gram Mixing

Replaces the heuristic entropy-adaptive alpha (`0.05 + 0.55*sigmoid(2*(H-4))`) used in all existing n-gram submissions with **Context Tree Weighting** -- a provably optimal Bayesian model averaging method (Willems et al., 1995).

Instead of a single scalar alpha, CTW provides:
- **Per-context adaptation**: automatically trusts deeper n-gram matches where they're reliable
- **Per-sequence learning**: the beta ratios track which context depth works best
- **Logistic domain mixing**: combines predictions in log-odds space (PAQ-style) rather than probability space
- **Theoretical guarantee**: redundancy bounded by Gamma_D(S) + |S|*gamma(T/|S|) + 2 bits

## Architecture (inherited from PR #549 SOTA)

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x (1536 hidden), LeakyReLU(0.5)^2
- XSA on last 4 layers, Partial RoPE (16/64)
- LN Scale 1/sqrt(layer+1), Value Embedding at layers 9-10
- SmearGate + BigramHash(1536, dim=128)
- Tied embeddings, logit softcap=30.0

## Training

- Parallel Muon (parameter banking, async RS/AG)
- LR: matrix=0.025, scalar=0.025, embed=0.035
- EMA(0.997), Late QAT (STE int6 when scale < 0.15)
- GPTQ-lite int6 multi-percentile + LZMA

## Eval Pipeline

1. Training (10 min, ~7000 steps on 8xH100)
2. EMA -> Quantize int6 -> Dequantize
3. Sliding window eval (stride=64) -- baseline bpb
4. Legal score-first TTT (SGD lr=0.002, 3 epochs per 32K chunk)
5. **CTW n-gram backoff eval** (orders 2-7, 4M hash buckets, logistic mixing)

## Run Command

```bash
SEED=42 TTT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-25_CTW_Mousse_OptRot/train_gpt.py
```

Built on PR #549 by @sanjeevmadhav. N-gram technique adapted from PR #753 by @newjordan.
CTW mixing based on Willems, Shtarkov & Tjalkens (1995).
