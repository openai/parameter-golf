# 11L SwiGLU + XSA4 + EMA + U-Net + AdamW TTT + BigramHash(8192) (pending compute)

## Results
- **val_bpb: pending** — awaiting 8xH100 compute credits
- Expected range: ~1.07-1.10 based on architecture

## Approach

Full frontier stack combining SwiGLU activation, U-Net skip connections, XSA4, EMA weight averaging, AdamW TTT, and GPTQ-lite quantization. Built on top of proven techniques from PRs #398, #442, #462.

### Architecture
- 11 transformer layers, 512-dim, 8 heads (8 KV heads)
- **SwiGLU FFN** with Star-ReLU activation (hidden=1792)
- **U-Net skip connections** with learned gating (encoder=5, decoder=6)
- **BigramHash** (8192 buckets, 128 dim) + SmearGate
- **Partial RoPE** (16 dims only)
- **LN Scale** (1/sqrt(layer_idx+1) per block)
- Tied embeddings, logit softcap=30.0
- **XSA4** on last 4 layers

### Training
- Muon optimizer: lr=0.025, momentum=0.99
- AdamW for embeddings/scalars
- Weight decay: 0.04
- Warmdown: 6000 iterations
- **EMA** (decay=0.9985) replacing SWA
- Batch size: 524,288 tokens, seq_len=1024

### Eval-time
- **AdamW TTT** (lr=0.0005, 10 epochs) — legal score-first protocol
- Sliding window eval (stride=64)

### Quantization
- Int6 per-row quantization with GPTQ-lite calibration
- zstd level 22 compression

### Credits
- SwiGLU + U-Net + GEPA architecture: @JoeProAI (PR #462)
- XSA + EMA + Partial RoPE + LN Scale: @felipe-parodi (PR #398)
- AdamW TTT: @sjp611 (PR #442)
- Late QAT: @fbedev (PR #410)
- DDP compile fix: our contribution

## Checklist
- [x] Submission folder in `records/track_10min_16mb/`
- [x] `README.md` with approach description
- [x] `submission.json` with metadata
- [x] `train_gpt.py` (single file, self-contained)
- [ ] Training log (pending compute)
- [ ] Verified BPB score (pending compute)
