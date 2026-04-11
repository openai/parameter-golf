# Non-record: Universal Transformer + Adaptive Density

**val_bpb: 3.2483 (legal, no TTT)** | DGX Spark GB10, 200 steps sp1024, no torch.compile

## Update (April 11, 2026)

An earlier version of this PR reported **val_bpb 1.4390** using multi-epoch TTT on `val_tokens` without score-first discipline. @MatoTeziTanka correctly flagged this as the same illegal pattern that closed PR #1376 and the rest of the Pre-Quant TTT cluster.

The flagged code path (`ttt_adapt(args, base_model, device, val_tokens, ...)` at line 1194) was part of the `train_gpt_kitchen_sink.py` base script and defaulted to enabled. This submission has been updated to disable TTT by default and report honest BPB from a clean run on the DGX Spark.

Thanks to @MatoTeziTanka for the careful review.

## What This PR Is

Implements OpenAI's requested "Universal transformer" research direction from the README. Single shared transformer block looped N times with per-iteration learnable parameters (attn_scale, mlp_scale, resid_mix, iteration_embed). 50% sparse-to-dense curriculum during training.

This is a non-record track research submission. It exists to answer the question: does weight-shared depth recurrence work at the parameter golf scale? The answer is yes, but it plateaus fast and is dominated by mini depth recurrence (repeat 2-3 specific layers) as used in PR #1204 and PR #1334.

## Legal Results (No TTT)

All runs on NVIDIA DGX Spark GB10 (single GPU, 128GB unified memory), sp1024, 200 training steps, SEED=42, no torch.compile.

| Run | Config | Params | val_bpb | ms/step |
|-----|--------|--------|---------|---------|
| UT-1 | 1 block x 6 iters | 4,546,568 | **3.2483** | 707 |
| UT-2 | 1 block x 24 iters | 4,601,864 | 3.2490 | 2,734 |

## Finding

Doubling iterations from 6 to 24 (4x compute per step) produces identical BPB to 3 decimal places. Full weight sharing hits a ceiling almost immediately at this model size. The compute budget is better spent on:

1. **Mini depth recurrence** (repeat 2-3 specific layers) as in PR #1204, which avoids the weight-sharing penalty on the non-repeated layers
2. **More training steps** rather than more iterations per step
3. **Wider models** (per MEGA-2 ablation: d=640 beats 11 layers at d=512)

The 2.87 MB artifact size means there is substantial headroom under the 16 MB limit. A hybrid approach combining partial weight sharing with a larger base model would likely beat the pure-shared approach tested here.

## Reproduction

```bash
pip install sentencepiece brotli
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
VOCAB_SIZE=1024 NUM_ITERS=6 TORCH_COMPILE_DISABLE=1 ITERATIONS=200 TTT_ENABLED=0 \
  python3 records/track_non_record_16mb/2026-03-31_UniversalTransformer_AdaptiveDensity/train_gpt.py
```

## Full Ablation Data

Raw logs and CSV for all 22 runs across 7 architectures (Universal Transformer, Text Diffusion, Random Adapters, JEPA, Mamba SSM, H-Net, Megakernels):

https://gist.github.com/dentity007/324ac35505c27acd18e7ffb468f4fa08

## Hardware Notes

DGX Spark GB10 is approximately 6x slower per step than 8xH100. Absolute BPB values are much higher than competition runs due to the short 200-step training budget. The relative ordering between configurations is what matters here: more iterations does not help, and depth recurrence plateaus quickly.

## Related

- PR #1191 H-Net Dynamic Chunking (non-record, same cluster)
- PR #1192 Fused Triton Megakernels (non-record)
- PR #1194 Text Diffusion (non-record)
- PR #1195 Random Adapter Maps (non-record)
- PR #1196 LLM-JEPA (non-record)
- PR #1197 Mamba SSM Hybrid (non-record)
- PR #1204 msisovic Mini Depth Recurrence (record-track implementation of this idea)
- PR #1334 aryanbhosale Track A with depth recurrence + parallel residuals (1.0897 BPB)
