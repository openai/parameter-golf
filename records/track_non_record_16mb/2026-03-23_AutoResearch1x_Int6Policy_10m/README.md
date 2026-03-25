This non-record submission focuses on **policy search over mixed quantization categories** on 1xH100 under a fixed 10-minute budget.

## Core idea
Instead of hard-coding one mixed-precision recipe, this script exposes `INT6_CATS` to control which parameter groups are quantized to int6 at export time.

This turns quantization into a simple policy search space:
- `INT6_CATS=attn,mlp,other`
- `INT6_CATS=attn,mlp`
- (plus architecture/MLP-width variants)

## What changed in code
- Added `INT6_CATS` environment variable to the record script and logging line `int6_policy:...`.
- Export path now applies int6 only to categories selected by policy.

## Main run (this folder's submission)
- Hardware: `1x H100 80GB`
- Budget: `600s`
- Policy: `INT6_CATS=attn,mlp`
- Model: `MLP_MULT=2`, `TRAIN_SEQ_LEN=1024`, `TRAIN_BATCH_TOKENS=524288`
- Result: `final_int8_zlib_roundtrip_exact val_bpb=1.52481100`
- Pre-quant at stop: `val_bpb=1.3934`
- Size: `13,144,462 bytes` (under 16MB)

## Policy-sweep observations (same session)
| Run | Policy / Variant | bytes_total | final val_bpb |
|---|---|---:|---:|
| run3 | `attn,mlp,other` + MLP3x + seq2048 | 16,210,551 | 1.4882 |
| run4 | `attn,mlp` + MLP3x + seq1024 | 16,343,293 | 1.4790 |
| run6 (main) | `attn,mlp` + MLP2x + seq1024 | 13,144,462 | 1.5248 |

## Takeaway
On 1xH100 and short budgets, aggressive int6 policies can preserve size but create a large pre/post quantization gap. This is a useful negative result and motivates future policy search with stronger quantization-aware training or selective fp16 passthrough.

Included files:
- `train_gpt.py` (policy-enabled script)
- `train.log` (main run)
- `train_policy_allcats.log` and `train_policy_mlp_attn.log` (extra sweep logs)
- `submission.json`
