## Standard V6 -- val_bpb=1.1465

Standard FineWeb training (no val-only), tuned specifically to beat the standard SOTA with a mixed-precision export profile and denser SWA.

### Key Metrics

| Metric | Value |
|---|---|
| **Post-quant val_bpb** | **1.14649233** |
| Pre-quant val_bpb | 1.1633 |
| Steps | 7,223 |
| Artifact | 15,930,918 bytes |
| SWA checkpoints | 30 |

### Recipe deltas vs V5

- Disable STE fake quant during training for standard mode (`STE_QAT_ENABLED=0`).
- Mixed quantization at export: int6 for MLP/attention, int8 fallback for others.
- fp16 passthrough expanded to `tok_emb` + `blocks.8.attn.c_k`.
- Muon weight decay increased to `0.04`.
- SWA sampling every `50` steps with start threshold at `scale < 0.5`.
