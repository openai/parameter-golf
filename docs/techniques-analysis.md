# Techniques Analysis — Parameter Golf Leaderboard

Full breakdown of every technique that has appeared in submissions, with BPB impact and implementation notes.

---

## Technique Lineage Map

```
Naive Baseline (1.2244)
    └── FP16 Embed + LR tuning             → 1.2197  (−0.0047)
    └── 2048 seq length                    → 1.2060  (−0.0184)
    └── 4096 seq length                    → 1.2014  (−0.0230)
    └── Sliding Window Eval                → 1.1925  (−0.0319)
    └── LoRA TTT                           → 1.1928  (−0.0316)
    └── Muon WD + 10L + Overtone Init      → 1.1748  (−0.0496)
    └── Mixed Int6/Int8 + Sliding          → 1.1630  (−0.0614)
    └── Int6 QAT + Zstd-22 + 10L          → 1.1586  (−0.0658)
    └── Ternary quantization               → 1.1570  (−0.0674)
    └── SmearGate + OrthoInit + Muon WD   → 1.1556  (−0.0688)
    └── MLP3x + Int6 QAT                   → 1.1502  (−0.0742)
    └── Int6 + SmearGate + BigramHash      → 1.1458  (−0.0786)
    └── 10L Int5 MLP + BigramHash(10240)   → 1.1428  (−0.0816)
    └── 11L Efficient Partial XSA          → 1.1307  (−0.0937)
    └── 11L XSA4 + EMA + Int6 MLP3x       → 1.1271  (−0.0973)
    └── 11L XSA4 + EMA + PartialRoPE      → 1.1248  (−0.0996)
    └── 11L EMA + GPTQ-lite + WD3500      → 1.1228  (−0.1016)
    └── LeakyReLU² + Legal TTT + ParMuon  → 1.1194  (−0.1050)
    └── AR Self-Gen GPTQ + XSA-all + BH3072 → 1.1147 (−0.1097)  ← CURRENT SOTA
```

---

## Technique-by-Technique Breakdown

### 1. FP16 Tied Embeddings
- **BPB gain**: −0.0047
- **How**: Store embedding table in fp16 instead of bf16/fp32. Cuts embedding size by ~50%, freeing bytes for more parameters.
- **Config**: `TIE_EMBEDDINGS=True` + store in fp16 during export.
- **Introduced by**: Renier Velazco (PR #?)

### 2. Longer Sequence Length (2048 / 4096)
- **BPB gain**: −0.018 to −0.023 (seq 2048 → 4096)
- **How**: Longer context → model sees more dependencies → better compression.
- **Cost**: More memory per step, fewer steps per second. 4096 has diminishing returns vs compute cost.
- **Config**: `TRAIN_SEQ_LEN=2048` or `4096`

### 3. Sliding Window Evaluation
- **BPB gain**: −0.019 (vs fixed eval at same seq_len)
- **How**: Evaluate with a stride smaller than seq_len (e.g. stride=64). Each token benefits from more left context on average.
- **No training change needed**: pure eval trick.
- **Config**: `SLIDING_WINDOW_STRIDE=64`

### 4. LoRA Test-Time Training (TTT)
- **BPB gain**: −0.0316 vs baseline (combined with other tricks)
- **How**: Fine-tune small LoRA adapters on validation tokens during evaluation. Can only use tokens already graded.
- **Caution**: Legal TTT (score-first) is allowed; training on future val tokens is not.
- **PR**: #549, #?

### 5. Muon Weight Decay + 10 Layers
- **BPB gain**: large cumulative step
- **How**: Adding weight decay to Muon (`muon_wd`) regularizes matrix updates. Combined with depth increase to 10 layers.
- **Config**: `NUM_LAYERS=10`, `MUON_MOMENTUM=0.99`

### 6. Mixed Precision Quantization (Int6 + Int8)
- **BPB gain**: −0.0614 cumulative
- **How**: Use int6 for weight matrices (more precision) and int8 for embeddings. Lower quant noise → better BPB.
- **Implementation**: STE (straight-through estimator) for int6 during training.

### 7. SmearGate
- **BPB gain**: contributes ~−0.01 as part of stack
- **How**: A learnable position-mixing gate applied to hidden states. Allows information from neighboring positions to mix without full attention.
- **Introduced by**: @aquariouseworkman (PR #65)
- **Free parameter budget**: tiny (just a scalar gate per layer)

### 8. BigramHash Embedding
- **BPB gain**: ~−0.005 to −0.008 depending on size
- **How**: Hash consecutive token pairs into a fixed-size embedding table. Adds a learned bigram co-occurrence feature at low parameter cost.
- **Sizes tried**: 1536 → 2048 → 3072 (with dim=112)
- **Config**: `BIGRAM_VOCAB_SIZE=3072`, `BIGRAM_DIM=112`
- **Introduced by**: @raahilshah (PR #162)

### 9. MLP 3× Expansion (MLP_MULT=3)
- **BPB gain**: ~−0.005 vs MLP_MULT=2 at same layer count
- **How**: Wider MLP hidden layer = more capacity for pattern storage. With int6 QAT, the extra parameters fit in the budget.
- **Config**: `MLP_MULT=3` (hidden = 512 × 3 = 1536)

### 10. LeakyReLU² Activation
- **BPB gain**: ~−0.002 vs GELU in this setting
- **How**: Replace GELU with `LeakyReLU(0.5)²`. Smooth, always-positive, cheap to compute.
- **Introduced by**: @parinzee (PR #493)

### 11. Orthogonal Initialization (OrthoInit)
- **BPB gain**: small, ~−0.001
- **How**: Initialize weight matrices to be orthogonal. Better gradient flow early in training.

### 12. EMA (Exponential Moving Average) Weights
- **BPB gain**: ~−0.003 vs no averaging
- **How**: Maintain a shadow copy of weights as EMA(0.997). Use shadow for evaluation.
- **Combined with SWA**: Take snapshots every 50 steps and average.
- **Config**: `EMA_DECAY=0.997`, `SWA_EVERY=50`
- **Introduced by**: @newjordan (PR #401)

### 13. Partial RoPE (16/64 dims)
- **BPB gain**: ~−0.002
- **How**: Apply RoPE only to the first 16 out of 64 head dimensions. Remaining dims are position-unaware, allowing absolute position info to leak through.
- **Config**: `ROPE_PARTIAL_DIMS=16`
- **Introduced by**: @jfprincz (PR #315)

### 14. Layerwise LN Scale (1/√(layer+1))
- **BPB gain**: small but consistent
- **How**: Scale LayerNorm output by `1/√(layer+1)` to prevent later layers from dominating.
- **Introduced by**: @jfprincz (PR #315)

### 15. 11 Layers (from 10)
- **BPB gain**: ~−0.003 vs 10 layers at same param budget
- **How**: One more transformer block. Within int6 budget this fits under 16MB.

### 16. Cross-Sequence Attention (XSA)
- **BPB gain**: ~−0.005 (all layers) vs −0.003 (last 4 layers)
- **How**: At certain layers, keys/values are shared across positions via a global token or cross-attention mechanism. Zero additional parameters.
- **Layers**: Initially last 4 → then all 11.
- **Introduced by**: @gowtham0992 (PR #478)

### 17. VE128 (Variable-Expansion layers 9–10)
- **BPB gain**: small
- **How**: Expand intermediate dim only on the last 2 layers (layers 9–10).
- **Introduced by**: @unnir (PR #374)

### 18. U-Net Skip Connections
- **BPB gain**: part of cumulative stack
- **How**: Add residual connections from early layers to late layers (encoder-decoder style).
- **Introduced by**: PR #289

### 19. GPTQ-lite (Diagonal Hessian)
- **BPB gain**: ~−0.005 vs int8 per-row
- **How**: Post-training quantization using approximate second-order info (diagonal of Hessian). Better than per-row int8.
- **Introduced by**: @signalrush (PR #374)

### 20. Full Hessian GPTQ + AR Self-Gen Calibration
- **BPB gain**: ~−0.008 vs GPTQ-lite (at same stack)
- **How**: Full (non-diagonal) Hessian with Cholesky error compensation + column reordering. Calibration data is generated by the model itself (autoregressive sampling), which avoids using any training or val data during quantization.
- **Config**: `GPTQ_CALIB_SEQS=64`, `GPTQ_CALIB_TEMP=0.8`
- **Introduced by**: @abaybektursun (PR #1019, building on #535)

### 21. LZMA Compression (preset=9)
- **BPB gain**: small but free (−0.001 to −0.002 artifact bytes saved)
- **How**: Replace zlib with LZMA at max compression. Costs ~30s of eval time.
- **Introduced by**: @ChaseWNorton (PR #160)

### 22. Parallel Muon + Parameter Banking
- **BPB gain**: systems speedup → more steps in 600s → better BPB
- **How**: Parallelize Newton-Schulz orthogonalization across GPUs. Reduces per-step time by ~10%.
- **Introduced by**: @abaybektursun (PR #399)

### 23. Warmdown Length (1200 → 4000)
- **BPB gain**: ~−0.003
- **How**: Longer cosine decay at end of training gives model more time to settle.
- **Config**: `WARMDOWN_ITERS=4000`
- **Introduced by**: @shikhar1729 (PR #364)

### 24. Ternary Quantization (1.157)
- **BPB gain**: different tradeoff — more params at ternary precision
- **How**: 106M params quantized to {-1, 0, +1}. Uses U-Net + YaRN + FP8 training.
- **Status**: Non-competitive with the int6 GPTQ stack currently.

### 25. Selective ±1 Pruning
- **BPB gain**: ~−0.001 (by reducing artifact size)
- **How**: After GPTQ, prune weights at exactly ±1 quantization level if they cause high reconstruction error. Reduces artifact size slightly.
- **Introduced by**: @saml212 (PR #609)

---

## Open Techniques (Not Yet Implemented)

These are on the README's wish list:

| Technique | Key paper / concept | Difficulty |
|-----------|--------------------|-----------| 
| JEPA | Joint Embedding Predictive Architecture (LeCun) | Very Hard |
| Text diffusion | Masked LM as diffusion model | Hard |
| H-net tokenization | Hierarchical subword tokenization | Hard |
| State-space models | Mamba / S4 within the 16MB budget | Hard |
| E2E TTT | End-to-end test-time training | Medium |
| Super long context eval | 8192+ token eval sequences | Medium |
| Learning adapters on random linear maps | Novel linear compression | Unknown |
| Universal transformer (depth recurrence) | Shared weights across all layers | Medium |
| Megakernels | Custom CUDA kernels for attention/MLP | Very Hard |
