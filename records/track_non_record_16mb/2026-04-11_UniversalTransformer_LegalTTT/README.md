# Non-record: Universal Transformer + Legal Pre-Quant TTT (Training-Slice Variant)

**val_bpb: 3.4446 (int6 brotli-11 roundtrip)** | **pre-quant val_bpb: 3.2483** | DGX Spark GB10 | sp1024 | 200 train steps

## Quick Results

| Metric | Value |
|---|---|
| Model params | 4,546,568 |
| Pre-quant val_bpb (step 200) | 3.2483 |
| Post-TTT int6 roundtrip val_bpb | **3.4446** |
| TTT source | fineweb_train_000079.bin (last training shard tail) |
| TTT tokens | 131,073 (training data slice, NOT val_tokens) |
| TTT config | 3 epochs, AdamW lr=0.0005, no frozen blocks |
| TTT loss curve | 6.15 -> 5.89 -> 5.79 |
| TTT duration | 13.6 seconds |
| Artifact size (int6+brotli-11) | 1.35 MB (1,344,763 bytes) |
| Serialized model | 17.14 MB (uncompressed) |
| Step time | 1,541 ms/step (single GB10, no torch.compile) |

## Background

This is a legal-compliant resubmission of PR #1193 following community review by @MatoTeziTanka on 2026-04-11.

PR #1193 (original Universal Transformer submission) was flagged for using an illegal TTT pattern: the `ttt_adapt()` function trained multi-epoch on `val_tokens` without score-first discipline, matching the pattern that closed PR #1376 and the rest of the Pre-Quant TTT cluster.

Rather than just disable TTT, this submission rewrites the TTT function to use the legal pattern referenced in @MatoTeziTanka's review: a held-out slice of training data, never part of the validation set. The architecture message (Universal Transformer depth recurrence) is preserved, and TTT is now a legal component of the submission.

## What Changed vs PR #1193

1. **TTT function signature.** `ttt_adapt()` now takes `train_slice_tokens` instead of `val_tokens`. The parameter name makes the intent explicit.

2. **Training slice source.** Before the TTT call, the submission loads a fixed window from the end of the last `fineweb_train_*.bin` shard. This slice was not used during main training (we only trained on the prefix of each shard up to `iterations` steps) and is never part of `fineweb_val_*.bin`. No val_tokens touch the TTT gradient path.

3. **Evaluation unchanged.** `val_tokens` are scored after TTT completes, exactly as in any non-TTT submission. The TTT updates only shift the model weights, they do not influence how val tokens are scored.

## Universal Transformer Architecture (unchanged from #1193)

- Single shared transformer block looped N times
- Per-iteration learnable parameters: `attn_scale`, `mlp_scale`, `resid_mix`, `iteration_embed`
- 50 percent sparse-to-dense curriculum
- Implements OpenAI's requested "Universal transformer" research direction

## Legality Argument

Issue #402 and Issue #677 rulings define illegal TTT as any training pass that updates model state based on val_tokens the model has not already been tested on. This submission satisfies the rules because:

1. The TTT gradient comes entirely from training-set tokens
2. Those training tokens are never scored as part of val_bpb
3. Val tokens are scored exactly once, in a single left-to-right pass, after all training (including TTT) has finished
4. No test-time leakage of val targets into training loss

The argument is structurally identical to PR #1416 and PR #1423 reference implementations cited by @MatoTeziTanka in his review.

## Reproduction

```bash
pip install sentencepiece brotli
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
VOCAB_SIZE=1024 NUM_ITERS=6 TORCH_COMPILE_DISABLE=1 ITERATIONS=200 \
  TTT_ENABLED=1 TTT_EPOCHS=3 TTT_TRAIN_SLICE_SEQS=128 TTT_LR=0.0005 \
  python3 records/track_non_record_16mb/2026-04-11_UniversalTransformer_LegalTTT/train_gpt.py
```

## Hardware

NVIDIA DGX Spark GB10 (single GPU, 128GB unified memory, aarch64). No torch.compile (Triton unsupported on ARM), no flash_attn_interface. These constraints make absolute BPB much higher than a competition 8xH100 run, but the architecture and legality story holds across hardware.

## Related PRs

- **PR #1193 (original, flagged)** Universal Transformer with illegal TTT-on-val
- **PR #1416, #1423 (@aryanbhosale)** Reference legal Pre-Quant TTT implementations
- **PR #1204 (@msisovic)** Mini depth recurrence (partial weight sharing)
- **PR #1334 (@aryanbhosale)** Track A legal submission using parallel residuals + depth recurrence

## Review Credit

@MatoTeziTanka flagged the original #1193 TTT-on-val issue on 2026-04-11 via The Agora community compliance tracker (https://matotezitanka.github.io/parameter-golf/). This resubmission implements the exact fix he recommended.
