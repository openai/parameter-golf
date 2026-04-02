# Trinity Ternary GPT — Parameter Golf Submission

## Summary

A ternary quantization approach inspired by the [Trinity](https://github.com/gHashTag/trinity) ternary computing framework. All large weight matrices use **BitNet b1.58 ternary weights** ({-1, 0, +1}) with **Quantization-Aware Training (QAT)** from step 0, enabling ~73M parameters to fit within the 16MB artifact limit.

## Key Innovations

### From Trinity
- **Absmean ternary quantization** (per-group, group_size=128): `scale = mean(|w|)`, `w_q = round(w/scale).clamp(-1,1)` — adapted from Trinity's `ternary_pipeline.zig`
- **Base-3 ternary packing** (5 trits per byte, 3^5=243<256) — adapted from Trinity's `ternary_packing.zig`
- **Trinity Identity philosophy** (φ²+φ⁻²=3): ternary is the natural base for efficient computing

### Architecture
- **10 layers**, 768 model dim, 8 heads / 4 KV heads (GQA)
- **relu² activation** with **4× MLP expansion** (3072 hidden) — ternary weights are cheap, so we go wide
- **U-Net skip connections** with learned skip weights
- **Partial RoPE** (16/96 dims) — position info only where needed
- **Z-loss regularization** (1e-4) for stable logits with ternary STE

### Training
- **NeoMuon optimizer** (3 Newton-Schulz steps vs standard 5) — faster per-step, more gradient updates
- **No weight decay** — incompatible with ternary STE
- **EMA** (0.997 decay, starts at step 500)
- **Warmdown** 3500 iterations
- **524k batch tokens**, seq_len=1024

### Compression
- Ternary weights: **base-3 packing** (~1.6 bits/param)
- Small params: **FP16**
- Final compression: **LZMA preset=9**
- Also produces standard int8+zlib for comparison

## Parameter Budget

| Component | Params | Storage |
|-----------|--------|---------|
| 10× Attention (QKVO) | ~23.6M ternary | ~5.9MB packed |
| 10× MLP (fc + proj) | ~47.2M ternary | ~11.8MB packed |
| Embeddings | ~786K fp16 | ~1.5MB |
| Norms, scales, skip | ~80K fp32 | ~0.3MB |
| **Total** | **~71.6M** | **~15.2MB (before LZMA)** |

After LZMA compression, the artifact should be well under 16MB since ternary weights have very low entropy.

## Running

```bash
# On 8xH100:
RUN_ID=trinity_ternary \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# On 1xH100 (testing):
RUN_ID=trinity_test \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Lineage

Built on the Parameter Golf baseline with ideas from:
- [Trinity](https://github.com/gHashTag/trinity) — ternary computing framework
- BitNet b1.58 — ternary quantization with absmean scaling
- PR #549 stack — relu², EMA, NeoMuon
- PR #287 — Partial RoPE
