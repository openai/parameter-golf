# Sequential Momentum TTT

**val_bpb = 1.0116** (3-seed mean, std 0.0041; seeds 1337/42/2025: 1.0147, 1.0069, 1.0132)

## Hardware

All reported results from 4x NVIDIA A10G (development hardware), 10K iterations uncapped. On 8xH100 SXM with MAX_WALLCLOCK_SECONDS=600, training will complete ~7200 iterations. Final val_bpb will differ and will be updated after official run.

## Approach

10-layer GQA transformer (d=512, 8 heads, 4 KV heads, MLP 3x, seq 2048, 24.7M params) with XSA4, EMA(0.997), SmearGate, BigramHash(4096), learned activation mixing (relu^2 + leaky_relu(0.5)^2 blend), and mixed int5/int6 quantization compressed via LZMA preset 6.

The key contribution is **Sequential Momentum TTT** -- an EMA of LoRA weights carried across document batches during test-time training. Standard LoRA TTT resets adapters between documents, forcing re-learning of shared patterns. Sequential Momentum TTT initializes each batch's LoRA from a blend of accumulated EMA weights and fresh random weights. Because the EMA averages over per-batch mean LoRA weights (averaging across documents within a batch and then across batches), the result is a low-rank prior capturing recurring distributional patterns rather than document-specific noise -- the same variance-reduction principle that makes SGD with large batch sizes converge to flatter minima. Documents sharing topic or style get a warm start that reduces TTT epochs needed for convergence. The EMA is updated only after scoring, so no future information leaks into past scores.

The asymmetric initialization -- A gets `kaiming + ema_A` while B gets pure `ema_B` -- exists because A is the down-projection (input-side) where fresh random signal helps break symmetry across batch elements, while B is the up-projection (output-side) where stable accumulated knowledge matters more for consistent output adaptation.

Momentum is set to 0.3 (swept over {0.0, 0.1, 0.3, 0.5, 0.7}). Lower values underutilize cross-document signal; higher values overfit to earlier documents. 0.3 gives enough carry-over to help topically similar documents without contaminating dissimilar ones.

## Results

| Seed | val_bpb | Params | Artifact (bytes) | Fits 16MB? |
|------|---------|--------|-------------------|------------|
| 1337 | 1.0147 | 24,730,715 | 10,810,924 | YES |
| 42 | 1.0069 | 24,730,715 | 10,841,556 | YES |
| 2025 | 1.0132 | 24,730,715 | 10,853,748 | YES |
| **Mean** | **1.0116** | -- | **10,835,409** | **std=0.0041** |

## Compression

| Step | Details |
|------|---------|
| Export | Mixed precision: int5 (clip to [-16, 15]) for MLP, int6 (clip to [-32, 31]) for attention |
| Pruning | 3% magnitude pruning (improves compressibility) |
| Compression | LZMA preset 6 |
| Embeddings | FP16 passthrough; control params FP32 passthrough |

## Ablations

Note: this is an engineering progression, not a controlled ablation. Compression format and eval method vary across rows.

| Configuration | val_bpb | Notes |
|---------------|---------|-------|
| Base (no TTT, sliding eval stride=256) | 1.1624 | XSA4+EMA+int6/zstd, 10K iters, 4xA10G |
| + Standard TTT (momentum=0, fixed) | 1.1300 | Single-GPU partial eval |
| **+ Sequential Momentum TTT (momentum=0.3)** | **1.0147** | **Full 4xA10G 3-seed validated** |

## Credits

Builds on techniques from the `openai/parameter-golf` community:

- **XSA4**: arXiv:2603.09078 (Shuangfei Zhai), openai/parameter-golf#287, #414
- **EMA**: openai/parameter-golf#287, #548
- **SmearGate + BigramHash**: openai/parameter-golf#102, #162
- **LoRA TTT framework**: openai/parameter-golf#77 (samacqua), #548 (loqui)
- **Int6 QAT**: openai/parameter-golf#180, #335
- **LeakyReLU(0.5)^2**: openai/parameter-golf#493 (parinzee)
- **Muon optimizer**: modded-nanogpt (Keller Jordan)

## Reproduction

```bash
git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
pip install torch numpy sentencepiece huggingface-hub datasets tqdm
MAX_WALLCLOCK_SECONDS=600 RUN_ID=submission SEED=1337 TTT_ENABLED=1 TTT_MOMENTUM=0.3 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-25_SequentialMomentumTTT/train_gpt.py
```
