# BigramDim160 + 10% Prune + SWA + SmearGate + OrthoInit + Int6+Zstd

**val_bpb: 1.14767** (mean of 2 seeds, sliding window stride=64, post int6+zstd quantization roundtrip)

## Run Command

```bash
# Install dependencies
pip install --break-system-packages zstandard sentencepiece huggingface-hub

# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train + evaluate (default seed=42)
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed beyond `SEED`.

## Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
| 42 | 1.14986 | 15,899,011 | yes |
| 1337 | 1.14548 | 15,812,010 | yes |
| 2024 | -- | -- | not completed (compute budget exhausted) |
| **Mean (2 seeds)** | **1.14767** | | |
| **Std** | **0.00219** | | |

### Note on Statistical Significance

We provide 2 seeds rather than the typical 3. Our third seed (2024) could not be completed due to exhausted compute budget across multiple cloud providers (RunPod, Vast.ai). The two completed seeds show tight variance (std=0.00219), consistent with the inter-run variance observed across other submissions in this range. We believe this is sufficient to characterize the method's performance, and submit this as a non-record entry documenting our approach. We are happy to run the third seed if compute becomes available.

Full training logs were lost when RunPod instances were terminated; the included log excerpts are extracted from our automation session transcripts and contain the verified final evaluation lines.

## Key Techniques

This submission builds on the SOTA foundation (10L + Int5MLP + BigramHash + SmearGate + SWA) with modifications to fit within 16MB reliably across seeds:

1. **BigramHash dim=160** (reduced from 192): The bigram embedding dimension is the primary knob for controlling artifact size. At dim=192, artifacts frequently exceeded 16MB due to seed-dependent compression variance. Dim=160 provides ~200KB of headroom while sacrificing minimal quality (~0.001 bpb).

2. **10% weight pruning** (post-SWA, pre-quantization): After SWA averaging, the smallest 10% of weights are zeroed out. This improves compressibility without meaningfully hurting quality at this scale. The pruning is applied to all non-embedding linear layers.

3. **SWA with start_frac=0.5**: Stochastic Weight Averaging begins at step ~5450 (halfway through the cosine warmdown phase), averaging 23 checkpoints. This was the default from the SOTA entry and works well.

4. **Int6+zstd-22 quantization**: Mixed int5/int6 quantization for MLP layers, int6 for attention, with zstandard compression at level 22.

5. **SmearGate + OrthoInit**: Carried forward from the SOTA entries. SmearGate replaces standard LayerNorm gating; OrthoInit provides better initial weight structure.

6. **GQA (8 heads, 4 KV heads)**: Grouped query attention reduces parameter count while maintaining quality.

## Architecture

- 10 transformer layers
- Model dim: 768
- MLP ratio: 3x
- Attention: GQA with 8 heads, 4 KV heads
- Sequence length: 2048
- Vocab size: 1024 (sp_bpe_1024 tokenizer)
- ~25.8M parameters pre-pruning
- Bigram vocabulary: 10,240 buckets
- Bigram embedding dim: 160

## Training

- Optimizer: Muon (matrix params) + Adam (scalars/embeddings)
- Learning rates: embed_lr=0.03, matrix_lr=0.02, scalar_lr=0.02
- Weight decay: 0.04
- Batch size: 786,432 tokens
- Warmup: 20 steps
- Wallclock cap: 600 seconds
- ~6,500 steps on 8xH100 SXM (~91ms/step avg)

## Observations

- **Artifact size variance is the real challenge.** The same config can produce artifacts differing by >600KB between seeds (observed: 15.97MB vs 16.65MB with dim=192). This forced us to reduce dim to 160 for reliable sub-16MB across all seeds.
- **dim=176 with 10% pruning achieved our best single-run quality (1.1426 bpb)** but produced a 16.69MB artifact that exceeded the limit.
- **4xH100 is not equivalent to 8xH100** for this competition: the 10-minute wallclock cap means 4 GPUs only complete ~2,700 steps vs ~6,500 on 8 GPUs, producing significantly worse results.
