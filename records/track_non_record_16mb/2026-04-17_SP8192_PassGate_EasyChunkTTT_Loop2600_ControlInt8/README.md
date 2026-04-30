# Non-Record Submission: SP8192 Pass-Gated Recurrence + Easy-Chunk TTT + Late Loop Onset

**Author:** Artem Buldin ([@Buld1n](https://github.com/Buld1n))  
**Track:** `non_record_16mb`  
**Best single seed:** **1.08065825 bpb**  
**3-seed mean:** **1.08119835 bpb**  
**Artifact size:** **15,983,090 - 15,985,665 bytes**

## What This Is

This is **not** a direct wrapper of the current public top-1 submission. The starting scaffold is the public SP8192 recurrent/legal-TTT line, but this submission adds and studies a different combination of ideas:

1. **Late step-based loop onset** instead of the usual wallclock-based enable point
2. **Pass-gated recurrent attention** inside the looped band
3. **Easy-chunk legal TTT** with lighter adaptation on easy chunks and stronger adaptation on harder chunks
4. **Control-int8 packing** for small scalar/control tensors so the same recipe fits cleanly under 16 MB

This submission is intentionally framed as a **non-record submission** because the 3-seed mean does not beat the current public SOTA mean of `1.0810`, even though the best single seed is competitive.

## 3-Seed Results

| Seed | Steps | Pre-Quant BPP | Sliding BPP | TTT BPP | Artifact |
|------|------:|--------------:|------------:|--------:|---------:|
| 42 | 4836 | 1.08724989 | 1.08272856 | **1.08065825** | 15,983,090 |
| 314 | 4832 | 1.08805657 | 1.08287242 | **1.08135101** | 15,985,665 |
| 999 | 4833 | 1.08834731 | 1.08292244 | **1.08158578** | 15,984,091 |
| **Mean** | **4833.7** | **1.08788459** | **1.08284114** | **1.08119835** | **15,984,282** |
| **Std** | **1.70** | **0.00046** | **0.00008** | **0.00039** | **1060** |

## Unique Findings

### 1. Loop onset matters a lot in this stack

I swept `ENABLE_LOOPING_AT_STEP` while keeping the rest of the recipe fixed:

| Loop Step | Steps | Sliding BPP | TTT BPP | Artifact |
|----------:|------:|------------:|--------:|---------:|
| 1600 | 4499 | 1.08309419 | 1.08111750 | 15,982,187 |
| 2000 | 4628 | 1.08302203 | 1.08105545 | 15,982,936 |
| 2400 | 4765 | 1.08256852 | 1.08068098 | 15,983,901 |
| **2600** | **4836** | **1.08272856** | **1.08065825** | **15,983,090** |
| 2800 | 4893 | 1.08283314 | 1.08084628 | 15,979,922 |
| 3000 | 4961 | 1.08275928 | 1.08086412 | 15,982,265 |

In this pass-gated stack, later onset helps until about `2600`, then starts to hurt again. That gives a useful empirical target for this family of recurrent SP8192 models.

### 2. Pass-gated recurrence is not just the public top-1 recipe

The looped band uses an extra recurrent attention gate (`recur_attn_delta`) so the reused middle blocks are not exact repeats of the ungated path. This keeps the recurrent band more controllable than a plain repeated-block stack.

### 3. Easy-chunk legal TTT

Evaluation uses legal score-first chunk TTT, but with:

- `TTT_EASY_CHUNK_RATIO=0.998`
- `TTT_EASY_CHUNK_EPOCHS=1`
- `TTT_OUTLIER_DROP_FRACTION=0.03`
- `TTT_SCORE_WEIGHT_POWER=0.5`

So easier chunks get less adaptation work, while harder chunks still receive the full update budget.

### 4. Control-int8 packing

To stay under the 16 MB submission limit, the small control tensors are packed as int8 with scales instead of float16 passthrough. This includes tensors such as:

- `attn_scale`
- `mlp_scale`
- `resid_mix`
- `recur_attn_delta`
- `q_gain`
- `skip_weights`
- `skip_gates`

The main GPTQ setup stays the same:

- int6 attention and MLP matrices
- int8 token embeddings
- byte-shuffle + Brotli compression

## Recipe

- SP8192 tokenizer / dataset
- 11 layers, 512 model dim, 8 heads / 4 KV heads
- looped band over layers `3..5`
- `QK_GAIN_INIT=5.0`
- `QK_GAIN_DEPTH_RAMP=0.5`
- `PARALLEL_RESIDUAL_START=6`
- `ENABLE_PARALLEL_RESIDUAL_AT_STEP=0`
- `ENABLE_LOOPING_AT_STEP=2600`
- `RECUR_ATTN_GATE=1`
- `RECUR_ATTN_GATE_SCALE=0.5`
- `TTT_ENABLED=1`
- `TTT_PARAM_MODE=full`
- `TTT_LR=0.005`
- `TTT_EPOCHS=3`

## Setup

```bash
pip install brotli sentencepiece numpy
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

## Reproduction

```bash
SEED=42 \
QK_GAIN_INIT=5.0 \
QK_GAIN_DEPTH_RAMP=0.5 \
PARALLEL_RESIDUAL_START=6 \
ENABLE_PARALLEL_RESIDUAL_AT_STEP=0 \
ENABLE_LOOPING_AT_STEP=2600 \
RECUR_ATTN_GATE=1 \
RECUR_ATTN_GATE_SCALE=0.5 \
TTT_ENABLED=1 \
TTT_PARAM_MODE=full \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
TTT_EASY_CHUNK_RATIO=0.998 \
TTT_EASY_CHUNK_EPOCHS=1 \
TTT_OUTLIER_DROP_FRACTION=0.03 \
TTT_SCORE_WEIGHT_POWER=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Attribution

This submission is built on the public SP8192 recurrent/legal-TTT line and is not presented as a from-scratch architecture. The main upstream ingredients came from:

- `@clarkkev` for the SP8192 + GPTQ + SDClip stack
- `@dexhunter` for the recurrent SP8192 line and legal TTT stack
- `@abaybektursun` for the score-first legal TTT framing
- `@Robby955` and `@msisovic` for the parallel residual direction

The contribution here is the **specific pass-gated, easy-chunk, late-onset, control-int8 variant** and the loop-onset sweep showing that `2600` is the best point among the tested settings.

## Included Files

- `README.md`
- `submission.json`
- `requirements.txt`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`
