# Non-record: SP8192 Past-only Delta Geo Ruler

**val_bpb = 1.08979034** (single seed, quantized sliding-window eval) | **15,995,193 bytes** | 8xH100

This submission adds a small strictly past-only delta ruler on top of the SP8192 + depth recurrence + parallel residual + QK-Gain 5.25 stack. It is submitted as a non-record architecture result: the run improves over my previous local sliding-window bar, but it is a single-seed result and does not clear the current public SOTA threshold.

## Result

| Seed | Exact BPB | Sliding BPB | Artifact bytes |
|------|----------:|------------:|---------------:|
| 42 | 1.10642039 | **1.08979034** | 15,995,193 |

Local comparison:

- Previous local sliding bar: `1.09073075`
- Delta-ruler sliding result: `1.08979034`
- Improvement: `0.00094041` BPB

## Mechanism

The delta geo ruler injects tiny learned corrections into late layers from strictly past hidden states at geometric offsets:

```text
RULER_MODE=delta
RULER_LAYERS=9,10
RULER_OFFSETS=1,3,7,15,31,63,127,255
RULER_INIT_ALPHA=0.005
```

The key constraint is that every correction reads only earlier sequence positions. There is no anchor summary, same-block averaging, n-gram cache, future-token state, or score-after-update adaptation.

## Architecture

- SP8192 tokenizer and dataset variant
- 11 layers, 512 dim, 8 heads / 4 KV heads
- LeakyReLU(0.5)^2 MLP, 4x expansion
- Partial RoPE, tied embeddings, logit softcap
- 3-layer depth recurrence over layers 3,4,5
- Parallel residuals from layer 7
- QK-Gain 5.25
- GPTQ SDClip export, int6 matrices, int8 embeddings, Brotli compression
- No TTT in this run

## Compliance

- Strictly causal ruler: only past offsets are read.
- Full normalized softmax distribution over the tokenizer vocabulary.
- No validation data access during training.
- No score-after-update TTT; `TTT_ENABLED=0`.
- No eval-built cache, no hardcoded byte ratios, no rescoring.
- Artifact is under the decimal 16 MB cap by `4,807` bytes.
- Training run stopped on the 588 second effective wallclock guard.

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 \
DATA_VARIANT=sp8192 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=0 \
RULER_MODE=delta \
RULER_LAYERS=9,10 \
RULER_OFFSETS=1,3,7,15,31,63,127,255 \
RULER_INIT_ALPHA=0.005 \
RULER_NORM=1.0 \
MAX_WALLCLOCK_SECONDS=600 \
SLIDING_WINDOW_ENABLED=1 \
GPTQ_CALIBRATION_BATCHES=64 \
GPTQ_RESERVE_SECONDS=12 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
