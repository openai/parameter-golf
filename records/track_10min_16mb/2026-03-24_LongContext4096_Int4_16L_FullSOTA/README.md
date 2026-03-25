# LongContext 4096 + Int4 16L + Full SOTA Stack

**val_bpb: TBD** (3-seed mean, post int4+lzma, sliding window stride=64 + TTT)

## Summary

Three untried ideas combined on top of the complete PR #549 SOTA stack:

1. **16 layers** — Int4 nibble-packing (2 weights/byte) fits 16 transformer layers in the same 16MB
   budget as the SOTA's 11. That is a 45% increase in depth.

2. **4096 training context** — Each token sees up to 4,095 causal context tokens during training
   (vs 2,047 in the current SOTA). The existing dynamic NTK in `Rotary` auto-scales
   `rope_base` to ~48,550 for 4096-length sequences (scale=4, rope_dims=16,
   `10000 × 4^(16/14) ≈ 48,550`).

3. **Full SOTA stack** — Every technique from PR #549 is carried forward unchanged:
   11L → 16L, LeakyReLU(0.5)², SmearGate, BigramHash(1536), XSA on last 4 layers,
   Partial RoPE (16/64 dims), LN Scale 1/√(layer+1), Value Embedding (VE128, layers 9-10),
   EMA decay=0.997 + Tight SWA, GPTQ-lite clip search, OrthoInit + muP, Parameter Banking,
   Parallel Muon, Legal Score-First TTT (enabled via `TTT_ENABLED=1`).

## Int4 Quantization

**QAT**: `CastedLinear` applies int4 STE fake-quant (`/7.0`, clamp `[-8, 7]`) during late warmdown
(when LR scale < `late_qat_threshold=0.15`). QAT and export precision are consistent.

**At export**:

- GPTQ-lite: 5 candidate clip percentiles per row, pick min reconstruction MSE
- Range: [-7, 7] (4-bit signed, symmetric)
- Nibble packing: 2 weights per byte, halving raw weight storage vs int8
- Scales: float16, per row (same overhead as int6)
- Compression: lzma level 6 (same as SOTA)
- Artifact: `final_model.int4.ptz`

**Budget estimate for 16 layers, MLP 3×, 512-dim:**
- Per-layer weights: Q(512×512) + K(256×512) + V(256×512) + Out(512×512)
  + MLP-up(1536×512) + MLP-down(512×1536) = 2,359,296 weights
- 16 layers: 37.7M weights → nibble-packed: 18.9MB raw
- After lzma (~1.6× ratio): ~12MB for weights
- Non-weight params (embeddings, bigram, VE, scalars): ~1.5MB
- Code: ~70KB
- **Estimated total: ~13.5–14MB** — well under 16MB

## Code Changes from PR #549 Base

```diff
-    num_layers = int(os.environ.get("NUM_LAYERS", 11))
+    num_layers = int(os.environ.get("NUM_LAYERS", 16))
-    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
-    eval_seq_len  = int(os.environ.get("EVAL_SEQ_LEN",  2048))
+    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
+    eval_seq_len  = int(os.environ.get("EVAL_SEQ_LEN",  4096))
```

Plus: `pack_nibbles`, `unpack_nibbles`, `quantize_int4_per_row`, `mixed_quantize_int4`
functions added; export path uses int4 instead of int6.

## Run Command

```bash
# 3 seeds required for submission
SEED=1337 TTT_ENABLED=1 bash eval/eval.sh
SEED=42   TTT_ENABLED=1 bash eval/eval.sh
SEED=2025 TTT_ENABLED=1 bash eval/eval.sh
```

Or directly:
```bash
SEED=1337 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

*(Pending H100 runs)*

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|-------------|--------------|-----------------|
| 1337 | — | — | — | — |
| 42   | — | — | — | — |
| 2025 | — | — | — | — |
| **Mean** | | | | |
