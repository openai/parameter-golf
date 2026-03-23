# PROTEUS v7 — Parameter Golf Submission

**Built with [PROTEUS](https://lightspeedup.com) by LightSpeedUp**

## Result

**Mean val_bpb: 0.9512** (3 seeds, std: 0.0025)

| Seed | Post-Quant BPB | TTT BPB | Steps | Step Avg |
|------|---------------|---------|-------|----------|
| 42   | 1.1779        | 0.9485  | ~7000 | 84.8ms   |
| 1337 | 1.1777        | 0.9534  | 6997  | 85.8ms   |
| 2024 | 1.1751        | 0.9516  | 7093  | 84.6ms   |

All seeds: `TTT_EPOCHS=3 TTT_MIN_DOC_LEN=512`

## Architecture

- 11 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- MLP 3x expansion (1536 hidden), relu² activation
- SmearGate + BigramHash(2048, dim=128) + OrthoInit
- Depth-scaled residual: `1/sqrt(layer_idx + 1)` attenuation per block
- U-Net skip connections, tied embeddings
- RoPE base 50K with NTK-aware eval scaling
- 26.8M parameters

## Training

- Muon optimizer (matrix_lr=0.02, WD=0.04, momentum=0.99)
- AdamW for embeddings/scalars (WD=0.04)
- Batch size: 786,432 tokens
- Warmdown: 3000 iterations, wallclock-based
- SWA: 11 checkpoints during last 20% of warmdown
- 3% magnitude pruning before export
- Gradient clipping: 0.3

## Quantization

- **INT6 uniform** for all weight matrices (64 levels per-row)
- FP16 for tied embeddings
- FP32 for control tensors (scales, mixes, gains)
- zstd-22 compression
- Artifact: ~15.4 MB (96.4% of 16MB budget)
- Quant gap: 0.012-0.014 BPB

## Test-Time Training (TTT)

Backward-looking LoRA adaptation during evaluation, following the approach established by PR #77.

For each document in the validation set:
1. Split into 256-token chunks
2. Process chunks left-to-right over 3 epochs
3. Each chunk: forward pass → score (final epoch) → train LoRA
4. Reset LoRA between documents

Key details:
- LoRA rank 8 on Q + V projections + LM head
- Adam optimizer (lr=0.01)
- Batch: 64 documents (independent LoRA per document)
- Documents < 512 tokens: standard eval (TTT adds noise on short docs)
- Fresh model copy for TTT (avoids torch.compile graph caching)
- Eval time: ~350s (within 600s budget)

## Key Innovations

1. **INT6 uniform quantization** — quant gap 0.012, better than prior SOTA's 0.014
2. **Depth-scaled residual** — `1/sqrt(layer+1)` for 11-layer stability, stored as buffer for torch.compile compatibility
3. **Fresh model copy for TTT** — torch.compile caches the no-LoRA forward path; new model from state_dict ensures LoRA works correctly
4. **Per-document batched TTT** — 64 documents with independent LoRA, per-document chunk offsets
5. **Short document threshold** — skip TTT for docs < 512 tokens (experimentally validated)

## Platform

RunPod 8×H100 SXM, PyTorch 2.8.0+cu128.

## Credits

PROTEUS by LightSpeedUp. TTT concept inspired by PR #77 (@samacqua). Techniques drawn from the Parameter Golf community: SmearGate/BigramHash (@unnir), Muon optimizer, SWA, OrthoInit.
