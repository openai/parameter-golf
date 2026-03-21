# What Doesn't Work

## Two Failure Patterns

1. **Throughput cost exceeds quality gain:** In 600s, anything adding >10% step overhead needs >10% per-step improvement. QAT (115ms vs 67ms), NorMuon (110ms), MTP (86ms) all fail this.
2. **Mechanism redundancy:** Stacking techniques that extract the same signal yields diminishing returns. TTT+XSA underperforms XSA alone; EMA without XSA hurts.

## Specific Negative Results

| Technique | Finding | Source |
|-----------|---------|--------|
| **12L at seq2048** | Slower steps (107ms) cancel extra capacity. 1.1541 vs 11L's 1.1326. 12L works at seq1024 (1.1468) | #219, #76 |
| **Late QAT at 12L** | Step overhead costs ~770 steps; overhead-to-gain ratio worsens at 12L's already slower step time | #332 |
| **Int5-MLP at 11L** | Quant penalty (0.029) outweighs savings. But at 10L, int5 funds BigramHash(10240) → SOTA | #236, #180 |
| **Larger vocab + fewer layers** | Vocab 4096 (8L) at 1.1642, SP4096 (9L) at 1.2012 — depth wins over vocab breadth | #123, #200 |
| **SmearGate without OrthoInit** | Hurts BPB by 0.003 | #212 ablation |
| **SWA with bf16 accumulation** | Catastrophic precision loss. Must use fp32. | #212 |
| **MTP (multi-token prediction)** | No BPB improvement (1.1947 vs 1.1929 control) | #212 |
| **Content-based curriculum** | No effect | #212 |
| **EMA without XSA** | 0.023 BPB worse than SWA on #198 base. EMA needs XSA to work. | #201 |
| **EMA decay=0.999** | Too slow to average, hurts BPB. Sweet spot is 0.997. | #287 |
| **cuDNN SDP vs Flash SDP** | cuDNN is 40% faster per op but worse BPB (different accumulation precision) | #281 |
| **Error-guided TTT** | Concentrating on highest-loss tokens doesn't help — they're genuinely unpredictable | #296 |
| **TTT on strong XSA+EMA** | +0.016 worse (1.1280 → 1.1436). SGD disrupts EMA's weight landscape. Helps weak bases though. | #303 vs #317 |
| **INT4 quantization** | Fits 33.5M params but 0.06 BPB quant gap makes it strictly worse than INT6 with fewer params | #281 |
| **FTLE per-row precision** | Uniform int-N beats mixed-row at every bit width — mixing increases entropy, defeats zstd | #316 |
| **Int8 QAT** | 20% step overhead costs ~2000 steps; lost tokens hurt more than closing ~0.007 int8 gap | #145 |
| **Layer sharing at 512d** | Costs 0.09 BPB with no space benefit | #316 |
| **Depth recurrence loop gates** | Initialized at 1/N, pull representation ~67% back toward input → effectively 1.3 loops instead of 3 | #319 |
| **DenseFormer DWA** | +0.003 BPB (hurts) | #328 |
| **NTK-RoPE extrapolation** | +0.06 BPP (hurts significantly) | #328 |
| **Depth recurrence + int6** | Catastrophic failure | #328 |
