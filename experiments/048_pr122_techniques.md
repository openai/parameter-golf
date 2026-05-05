# Experiment 048: Incorporate PR122 Techniques

## Status: PLANNING

## PR122 Techniques to Adopt (val_bpb=1.1585)

### High Priority (biggest impact)
1. **NorMuon optimizer** — replaces Muon. Adds second momentum + normalization. From NorMuon paper. Simple drop-in replacement.
2. **STE int6 QAT (CastedLinear)** — fake-quantize weights to 6-bit during forward pass, STE backprop. Near-zero quant penalty.
3. **Bit-packing int6** — pack 6-bit weights into actual 6 bits instead of int8. Saves 25% storage → more room for model params.
4. **FA3 (FlashAttention 3)** — ~10ms/step faster. Need to install flash-attn package.

### Medium Priority
5. **SWA** — like our LAWA but different: starts at 50% through warmdown, snapshots every 200 steps. May work better than our LAWA approach.
6. **zstd compression** (level 22) instead of zlib — better compression ratio.
7. **fp16 embedding passthrough** — we already have this.

### Their config differences
- VOCAB_SIZE=2048, NUM_LAYERS=8 (trade layer for vocab) — needs sp2048 tokenizer
- TRAIN_SEQ_LEN=4096
- MLP_MULT=3 (hidden=1536 for vocab=1024, or adjusted for vocab=2048)
- matrix_lr=0.02, muon_momentum=0.99, warmdown=3000
- weight_quantization_bits=6, embed_quantization_bits=16

## Plan
Merge NorMuon + int6 QAT + bit-packing + FA3 into our SwiGLU script.
Keep our vocab=1024, 9 layers, SwiGLU architecture.
Test with and without SWA.

## Script
- PR122 source: experiments/pr122_train_gpt.py
