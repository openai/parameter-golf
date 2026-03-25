# QAT Int4 16L + Full SOTA Stack

**val_bpb: TBD** (3-seed mean, post int4+lzma, sliding window stride=64 + TTT)

## Summary

Full PR #549 SOTA stack with 16 layers via int4 nibble-packing, plus proper QAT:

1. **16 layers** — Int4 nibble-packing (2 weights/byte) fits 16 transformer layers in 16MB.
2. **Int4 QAT on CastedLinear** — When LR scale drops below `late_qat_threshold=0.15` (late warmdown),
   CastedLinear applies int4 STE fake-quant (`/7.0`, clamp `[-8, 7]`). The model hardens to int4
   noise before export. QAT and export are consistent (both int4).
3. **Full SOTA stack** — All PR #549 techniques: LeakyReLU(0.5)², SmearGate, BigramHash(1536),
   XSA-4, Partial RoPE, LN Scale, VE128, EMA+SWA, Parallel Muon, OrthoInit, Legal TTT.

## Int4 Quantization

- GPTQ-lite: 5 candidate clip percentiles per row, pick min reconstruction MSE
- Range: [-7, 7] (4-bit signed, symmetric), nibble-packed (2 weights/byte)
- QAT: `CastedLinear` uses `/7.0` fake-quant (matching export) during late warmdown
- Compression: lzma level 6
- Artifact: `final_model.int4.ptz`

## Run Command

```bash
SEED=1337 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42   TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=2025 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

*(Pending H100 runs)*

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|-------------|--------------|-----------------|
| 1337 | — | — | — | — |
| 42   | — | — | — | — |
| 2025 | — | — | — | — |
| **Mean** | | | | |
