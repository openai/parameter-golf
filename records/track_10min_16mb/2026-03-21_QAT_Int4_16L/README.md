# QAT Int4 → 16 Layers

**Expected val_bpb: ~1.13–1.14** 

## Key Insight

Int8 quantization uses 1 byte per weight. Int4 packs 2 values per byte, halving model storage. This allows fitting **16 transformer layers** within the same 16MB budget as the SOTA's 10 layers — a 60% parameter increase.

The catch: int4 has only 15 distinct values (range [-7, 7]) vs 255 for int8, causing higher quantization error. We solve this with **Quantization-Aware Training (QAT)**: during training, a fake-quantize operation (with straight-through estimator) is applied to all matrix weights, so they gradually cluster near int4 grid points and export with minimal quality loss.

## Techniques

1. **16 transformer layers** (vs 10 in SOTA): 60% more parameters in the same 16MB budget.
   - 16 layers × 1.84M params / 2 (int4) = 14.72 MB raw → ~13.4 MB after zlib + 1.05 MB FP16 embed = **~14.45 MB** ✓

2. **Int4 nibble packing**: Two int4 weights packed per byte via bit operations. Values stored in [-7, 7] shifted to [0, 14] for unsigned nibble storage.

3. **Quantization-Aware Training (QAT)**: `FakeQuantize` with straight-through estimator (STE) applied to all `CastedLinear` weight matrices. Activates at 15% of training iterations after initial structure has formed. STE allows gradients to flow through the rounding operation during backward pass.

4. **All SOTA improvements carried forward**:
   - Muon optimizer with decoupled weight decay (WD=0.02)
   - FP16 tied embedding export
   - Overtone spectral embedding init + phase-transition resid_mix init
   - Sliding window eval (stride=64, seq_len=1024) — same eval time as SOTA

5. **Stability adjustments for 16L + QAT**:
   - Lower LR (matrix=0.030): more layers + QAT gradient noise
   - Higher Muon momentum (0.97): smooths noisy QAT gradients
   - Gradient clipping enabled (norm=1.0): prevents spikes at QAT activation

## Architecture

- Vocab: 1024, Dim: 512, Layers: **16**, Heads: 8/4 (GQA), MLP: 2× ReLU²
- Tied embeddings (FP16 export), U-Net skip connections (8 enc + 8 dec)
- ~29.4M parameters, ~14.45 MB artifact (int4+zlib with FP16 embed)

## QAT Implementation Details

```
FakeQuantize STE:
  forward:  w_q = round(clip(w, ±clip_abs) / scale) * scale  (per-row clipping)
  backward: grad_w = grad_out  (identity — straight-through)

Activation schedule:
  step < 15% of iterations:  fake_quant_bits = 0 (normal fp32 weights)
  step ≥ 15% of iterations:  fake_quant_bits = 4 (int4 STE active)

Int4 packing:
  pack: two int8 values in [-7,7] → shift to [0,14] → nibble-pack into uint8
  unpack: split nibbles → shift back to [-7,7] → reconstruct int8
  roundtrip: verified with assertion before export
```

## Run Command

```bash
RUN_ID=qat_int4_16l_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_QAT_Int4_16L/train_gpt.py
```

## Results

*(to be filled after runs)*

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | - | - | - | - |
| 42 | - | - | - | - |
| 7 | - | - | - | - |
| **Mean** | - | - | | |
