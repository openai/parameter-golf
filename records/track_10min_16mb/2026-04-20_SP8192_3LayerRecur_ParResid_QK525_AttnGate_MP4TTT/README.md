# Record Candidate: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + AttnGate + MP4 TTT

**Candidate name:** `2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT`  
**Target BPB:** `<= 1.073`  
**Reference record to beat:** `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` at `1.0810`

This folder implements the requested "safe SP8192 transformer line" candidate directly in `records/`, with code intended to run on an 8xH100 SXM machine such as RunPod.

It keeps the accepted April 9 recipe as the base and adds three changes:

1. **Attention output gate**
2. **4-phase legal score-first SGD TTT**
3. **Hadamard rotation + AWQ-style column scaling + Hessian-tuned GPTQ clip search at export**

The goal is to stack recent improvements without crossing into the higher-risk areas that dominated some April 15-20 PRs.

## Why this candidate

This is not the most radical frontier idea in the repo. It is the highest expected-value candidate if the target is a result that is both strong and likely to survive rules review.

What stays intentionally unchanged from Apr 9:

- SP8192 tokenizer and byte accounting
- 11L x 512d transformer
- 3-layer recurrence
- parallel residuals
- QK gain 5.25
- EMA
- legal score-first eval-time adaptation structure
- sub-16MB quantized artifact target

What changes:

- a tiny per-head attention output gate
- finer-grained legal adaptation inside each 32K TTT chunk
- a stronger but still deterministic export path

## Expected outcome

The intended gain model is:

- **AttnGate:** small but cheap architectural gain
- **MP4 TTT:** the primary expected improvement
- **Export improvements:** protect quality under the size cap rather than create the whole win

Reasonable expectation for this code path is roughly:

- `quantized_sliding_window`: modest improvement vs Apr 9, or near-flat
- `quantized_ttt`: strongest gain path
- artifact size: still below `16,000,000` bytes if the exporter behaves as intended

This implementation is meant as a runnable candidate, not a claimed record. The numbers still need to be established on 8xH100 with 3 seeds.

## What is implemented

### 1. Attention output gate

The attention module now has a learned **per-head output gate**:

- parameter name: `attn_out_gate`
- shape: `(num_heads,)`
- initialization: `0`
- effective multiplier: `1 + attn_out_gate`

So the gate is **zero-initialized in parameter space** but starts as an identity transform in forward pass. That preserves the Apr 9 behavior at step 0 and only lets training move away if the gate helps.

The gate is applied after the attenttion and before heads are flattened back to model dimension.

Why this version:

- preserves the base path at initialization
- tiny parameter cost
- easy to audit
- no artifact-size concern

### 2. 4-phase legal score-first TTT

The old legal TTT path scored a full 32K chunk, then trained on it.

This candidate splits each `TTT_CHUNK_TOKENS=32768` chunk into:

- `TTT_PHASES=4`
- `8192` scored tokens per phase

The phase order is:

1. score phase 0 windows
2. train only on phase 0 tokens
3. score phase 1 windows
4. train only on phase 1 tokens
5. repeat for phases 2 and 3

This keeps the same legal invariant:

- **every scored token is predicted before any update that uses that token**

What changed in code:ion output projec

- `eval_val_ttt()` now partitions each chunk into 4 sub-windows
- scoring stays under `torch.no_grad()`
- SGD updates happen only after that phase has been fully scored
- the last phase of the last chunk is not trained, since there is no future benefit

The script logs:

```text
ttt:start chunks=... phases=4 phase_tokens=8192 ...
```

so the legal shape is visible in the log.

### 3. Hadamard rotation + AWQ-style scaling + Hessian-tuned GPTQ clip search

The export path remains post-training only.

It now does this for large 2D tensors:

1. collect Hessians from calibration batches
2. for each matrix, search over clip sigma candidates
3. optionally apply **AWQ-style column scaling** derived from Hessian diagonal mass
4. optionally apply a deterministic signed **Hadamard block rotation**
5. quantize with GPTQ
6. choose the best candidate by **Hessian-weighted reconstruction error**
7. reject lower-score candidates if raw MSE becomes materially worse than the current best

Important scope decision:

- **rotation and AWQ are only allowed on attention/MLP matrices**
- embeddings stay on the safer int8 GPTQ path with sigma search only

Why:

- the requested candidate was supposed to stay on the safe SP8192 line
- rotating embeddings in the first pass adds unnecessary risk

### MSE guard

The exporter uses a **raw-MSE guard** on top of Hessian-weighted scoring.

Why that exists:

- Hessian-weighted error is the correct GPTQ-style objective for preserving predictive quality
- but it can occasionally prefer a candidate that looks good on the weighted objective while creating ugly unweighted distortion elsewhere
- the MSE guard prevents the search from accepting a numerically pathological candidate just because its weighted score wins

So the logic is:

- primary criterion: lower Hessian-weighted reconstruction error
- safety check: raw MSE must not degrade too much relative to the current best

## What is intentionally not included

This candidate deliberately skips:

- **GatedDeltaNet / FLA**
- **CaseOps / casefold / lowercase tokenizer changes**
- **SLOT**
- **pre-quant TTT as the scored path**

Reason:

- those directions may be powerful
- they also introduce much more rules risk or implementation surface
- this folder is intended to be a serious "safe-line" record attempt

## Run on 8xH100 SXM

Recommended environment:

- 8xH100 SXM
- PyTorch 2.9+ CUDA 12.8-ish
- `torchrun`
- `sentencepiece` and `brotli` installed
- FlashAttention-3 is required for priority runs, using the same `flash_attn_3` wheel path as the April 9 record

### Persistent RunPod setup

If you create a fresh pod each time, the only reliable way to avoid re-installing packages and re-downloading SP8192 is to attach the same RunPod persistent volume or network volume to every pod and mount it at `/workspace`.

This record folder includes a bootstrap script that assumes the persistent mount is `/workspace`:

```bash
cd /workspace
bash /workspace/parameter-golf/records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT/runpod_bootstrap.sh
```

What it does:

- clones or updates the repo under `/workspace/parameter-golf`
- creates a reusable venv under `/workspace/.venvs/parameter-golf-cu128-torch291`
- installs the same `flash_attn_3` wheel family used by the April 9 record
- downloads `sp8192` only if the tokenizer or shard files are missing

After the first pod, a new pod with the same mounted volume only needs:

```bash
source /workspace/.venvs/parameter-golf-cu128-torch291/bin/activate
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT
```

Example RunPod launch:

```bash
# First pod on a fresh persistent volume:
cd /workspace
git clone https://github.com/IanniMuliterno/parameter-golf.git
bash /workspace/parameter-golf/records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT/runpod_bootstrap.sh

# Later pods with the same mounted volume:
source /workspace/.venvs/parameter-golf-cu128-torch291/bin/activate
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT

# The bootstrap uses the April 9 FlashAttention-3 wheel path:
# python -m pip install flash_attn_3 --no-deps --find-links \
#   https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

set -o pipefail
SEED=42 \
RUN_ID=apr20_attngate_mp4ttt_seed42 \
DATA_DIR=/workspace/parameter-golf/data \
VOCAB_SIZE=8192 \
REQUIRE_FLASH_ATTN=1 \
QK_GAIN_INIT=5.25 \
ATTN_OUT_GATE_ENABLED=1 \
TTT_ENABLED=1 \
TTT_PHASES=4 \
TTT_CHUNK_TOKENS=32768 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
ROTATION_AWARE_ENABLED=1 \
ROTATION_BLOCK_SIZE=128 \
AWQ_POWERS=0.0,0.25,0.5 \
MATRIX_CLIP_SIGMAS=12.85,13.5,15.0 \
EMBED_CLIP_SIGMAS=20.0 \
MATRIX_BITS=6 \
EMBED_BITS=8 \
GPTQ_CALIBRATION_BATCHES=64 \
GPTQ_RESERVE_SECONDS=12 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed42.log
```

For the 3-seed record test:

```bash
set -o pipefail
for SEED in 42 314 999; do
  RUN_ID=apr20_attngate_mp4ttt_seed${SEED} \
  SEED=${SEED} \
  DATA_DIR=/workspace/parameter-golf/data \
  VOCAB_SIZE=8192 \
  REQUIRE_FLASH_ATTN=1 \
  QK_GAIN_INIT=5.25 \
  ATTN_OUT_GATE_ENABLED=1 \
  TTT_ENABLED=1 \
  TTT_PHASES=4 \
  TTT_CHUNK_TOKENS=32768 \
  TTT_LR=0.005 \
  TTT_EPOCHS=3 \
  ROTATION_AWARE_ENABLED=1 \
  ROTATION_BLOCK_SIZE=128 \
  AWQ_POWERS=0.0,0.25,0.5 \
  MATRIX_CLIP_SIGMAS=12.85,13.5,15.0 \
  EMBED_CLIP_SIGMAS=20.0 \
  MATRIX_BITS=6 \
  EMBED_BITS=8 \
  GPTQ_CALIBRATION_BATCHES=64 \
  GPTQ_RESERVE_SECONDS=12 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${SEED}.log
done
```

## Useful env vars

### Core candidate knobs

- `ATTN_OUT_GATE_ENABLED=1`
- `TTT_ENABLED=1`
- `TTT_PHASES=4`
- `TTT_CHUNK_TOKENS=32768`
- `TTT_LR=0.005`
- `TTT_EPOCHS=3`

### Export knobs

- `ROTATION_AWARE_ENABLED=1`
- `ROTATION_BLOCK_SIZE=128`
- `AWQ_POWERS=0.0,0.25,0.5`
- `MATRIX_CLIP_SIGMAS=12.85,13.5,15.0`
- `EMBED_CLIP_SIGMAS=20.0`
- `HESSIAN_DAMPING=0.01`
- `GPTQ_ACCEPT_MSE_RATIO=1.05`
- `MATRIX_BITS=6`
- `EMBED_BITS=8`

### Standard record knobs

- `SEED=42|314|999`
- `ITERATIONS=20000`
- `MAX_WALLCLOCK_SECONDS=600`
- `TRAIN_BATCH_TOKENS=786432`
- `VAL_BATCH_TOKENS=524288`

### Attention backend guardrail

- `REQUIRE_FLASH_ATTN=1` makes the run fail before training if FlashAttention is not importable.
- The hyperparameter dump logs `attention_backend`; for priority runs it should be `flash_attn_interface`, `flash_attn.flash_attn_interface`, or `flash_attn`, not `torch_sdpa`.
- If `attention_backend: torch_sdpa`, the run is using PyTorch SDPA fallback, not FlashAttention.

## Metrics to compare

For comparability with the prior baselines, especially the March 25 Colab-comparison workflow, this script keeps the standard logged metrics:

- `pre-quantization post-ema`
- `quantized`
- `quantized_sliding_window`
- `quantized_ttt`
- serialized model bytes
- code bytes
- total quantized submission bytes

The key value for record evaluation is:

- `quantized_ttt val_bpb`

But the supporting numbers matter too:

- `quantized_sliding_window` tells you whether the base model/export improved
- artifact bytes tell you whether the candidate is submission-feasible

## Compliance checklist

This is the intended compliance story. It still needs to be validated from actual 8xH100 logs.

- **Causal eval:** sliding-window scoring remains causal.
- **Normalized distribution:** standard softmax over the full vocab.
- **Score-before-update:** in MP4 TTT, each phase is fully scored before training on that phase.
- **Single scoring pass:** tokens are not rescored after adaptation.
- **No validation use during training:** training code does not access validation data.
- **No tokenizer accounting changes:** still uses the standard SP8192 byte-accounted validation path.
- **No SLOT / no n-gram cache / no ETLB by default:** candidate stays inside the safer rule shape.

## Risk table

| Area | Risk | Why it matters | Current mitigation |
|---|---|---|---|
| MP4 TTT legality | Medium | Multi-phase adaptation is easy to implement incorrectly | Phase-by-phase score then train ordering is explicit in `eval_val_ttt()` |
| Export regression | Medium | Rotation/AWQ can hurt quality if overused | Embeddings excluded from rotation/AWQ; Hessian score plus raw-MSE guard |
| Eval budget | Medium | MP4 TTT can cost more than 1-phase chunk TTT | No extra rescoring; same base chunk size; same SGD family |
| Artifact bytes | Low/Medium | Better quality can still fail the size cap | Still int6 matrices / int8 embeddings / compressed artifact |
| Training regression from gate | Low | Small new parameter can still destabilize | Zero-init delta around identity gate |

## What we would expect from adding LaCT

The LaCT direction from `TTT_done_right.pdf` is not included in this primary candidate, but it is worth thinking about as a follow-up.

The useful idea is:

- keep the base model mostly fixed at eval time
- attach a small **fast-weight learner**
- update that fast learner causally on already-scored text
- let it absorb local style, formatting, repetition, and short-range domain shifts

Relative to this MP4 TTT candidate, I would expect LaCT to behave like this:

- **Best case:** it gives additional gain on top of `quantized_sliding_window`, and possibly on top of legal base-model TTT, by putting adaptation pressure into a small fast state instead of the whole model.
- **Most likely first outcome:** it helps some seeds but increases eval complexity and runtime pressure enough that it needs a dedicated sweep rather than being dropped in blindly.
- **Main tradeoff:** LaCT is attractive because it can be more adaptation-efficient than updating the whole model, but it adds another eval-time mechanism that has to fit inside the 600s budget and be explained clearly.

For this candidate specifically, LaCT is more likely to be useful as:

- a **follow-up branch**
- possibly with `LACT_BASE_TTT=1` style hybrid evaluation

than as part of the first scored record attempt.

Why I would not include it immediately:

- MP4 TTT is already the main eval-time bet
- adding LaCT at the same time would make attribution muddy
- the safe-line candidate should first prove that AttnGate + MP4 TTT + stronger export are enough to move the mean materially

## Files

- [runpod_bootstrap.sh](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT/runpod_bootstrap.sh)
- [train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT/train_gpt.py)
- [requirements.txt](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/records/track_10min_16mb/2026-04-20_SP8192_3LayerRecur_ParResid_QK525_AttnGate_MP4TTT/requirements.txt)

## Recommended first validation order

1. Run seed `42` once and verify:
   `quantized_ttt`, `quantized_sliding_window`, artifact bytes, train/eval runtime.
2. If seed `42` is not clearly below about `1.076`, do not spend 3 seeds yet.
3. If the exporter regresses sliding-window quality, reduce exporter aggressiveness before changing training.
4. Only after a clean seed `42` win, run seeds `314` and `999`.
