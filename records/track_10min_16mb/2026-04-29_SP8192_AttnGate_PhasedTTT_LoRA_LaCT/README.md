# 2026-04-29 — SP8192 · AttnGate · PhasedTTT · LQER · LaCT

**Track:** 10-min / 16 MB  
**Primary eval path:** `quantized_ttt_phased`  
**Hardware target:** 8 × H100 SXM5 (RunPod)  
**Current accepted SOTA to beat:** `2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611` at `1.06108` BPB  

---

## Overview

This record combines three main ideas:

| Component | Lineage | Role |
|---|---|---|
| **GatedAttn + QuantGate** | PR `#1769` | Input-conditioned attention gating with dedicated int8-per-row gate export |
| **Phased TTT + Global SGD** | PR `#1727` | Primary scored eval path: score-first document batching, phased prefix scheduling, and global SGD between phases |
| **Mixed GPTQ + LQER** | PR `#1855` | Exact low-rank quantization error repair on top of mixed GPTQ export |

`LaCT` is still present as an optional additional eval path, but it is not the primary scored mode and is disabled by default.

---

## Main Path

### Base model

- Tokenizer: SP8192 sentencepiece, vocab `8192`
- Layers: `11`
- Model dim: `512`
- Attention: `8` heads / `4` KV heads
- Depth recurrence: loop over layers `3..5`, `NUM_LOOPS=2`
- Parallel residual start: layer `7`
- QK gain init: `5.25`
- EMA decay: `0.9965`

### GatedAttn + QuantGate

The attention block uses the dense GatedAttn mechanism from PR `#1769`:

```python
g = sigmoid(W_g x)
attn_out = g * attention(x)
```

The gate tensor `attn_gate_w` has a dedicated int8-per-row export path (`QuantGate`) so it does not silently fall back to generic GPTQ or fp16 passthrough.

### Phased TTT

The active TTT path is the PR `#1727` style phased controller:

1. Documents are discovered from BOS boundaries.
2. Validation documents are grouped into global batches.
3. Tokens are always scored before any update that could have seen them.
4. A prefix of scored documents is accumulated phase by phase.
5. After each phase boundary, the code runs global SGD on the scored prefix tokens.
6. Scoring then resumes with the updated base model.

Inside each batch, temporary LoRA adapters are still used as the local score-first update target, so the relevant `TTT_LORA_*` knobs still matter. What was removed is the older standalone LoRA-only eval mode.

### Mixed GPTQ + LQER

The export path is the mixed GPTQ + LQER path from PR `#1855`:

- `MLP_CLIP_SIGMAS=12.0`
- `ATTN_CLIP_SIGMAS=13.0`
- `LQER_ENABLED=1`
- `LQER_RANK=4`
- `LQER_TOP_K=3`
- `LQER_FACTOR_BITS=4`
- `LQER_ASYM_ENABLED=1`
- `LQER_ASYM_GROUP=64`

### LaCT

`LaCT` remains available as an optional extra eval path under `LACT_TTT_ENABLED=1`. It is off by default and does not affect the primary scored metric.

---

## No Fallbacks

This record is set up to fail explicitly instead of silently choosing a different TTT path.

- `TTT_LORA_ENABLED=1` is rejected. Standalone LoRA-only eval is no longer supported.
- `TTT_ENABLED=1` with `PHASED_TTT_ENABLED=0` is rejected. The old single-phase TTT path was removed.
- Missing FlashAttention fails explicitly.
- Missing RunPod network volume fails explicitly in endpoint mode.

---

## Default Env Profile

These are the record defaults currently wired into `train_gpt.py`, `run.sh`, and `handler.py`.

### Core model

| Variable | Default |
|---|---|
| `VOCAB_SIZE` | `8192` |
| `NUM_LAYERS` | `11` |
| `MODEL_DIM` | `512` |
| `EMBEDDING_DIM` | `512` |
| `NUM_HEADS` | `8` |
| `NUM_KV_HEADS` | `4` |
| `QK_GAIN_INIT` | `5.25` |
| `NUM_LOOPS` | `2` |
| `LOOP_START` | `3` |
| `LOOP_END` | `5` |
| `ENABLE_LOOPING_AT` | `0.35` |
| `PARALLEL_RESIDUAL_START` | `7` |
| `GATED_ATTN_ENABLED` | `1` |
| `GATED_ATTN_INIT_STD` | `0.01` |
| `GATED_ATTN_QUANT_GATE` | `1` |

### Phased TTT

| Variable | Default | Notes |
|---|---|---|
| `TTT_ENABLED` | `1` | Required for the primary path |
| `PHASED_TTT_ENABLED` | `1` | Required for the primary path |
| `PHASED_TTT_PREFIX_DOCS` | `2000` | Exact PR `#1727` run value |
| `PHASED_TTT_NUM_PHASES` | `4` | Exact PR `#1727` run value |
| `TTT_BATCH_SIZE` | `64` | Document batch size for local LoRA adaptation |
| `TTT_CHUNK_SIZE` | `48` | Chunk size within each document batch |
| `TTT_GRAD_STEPS` | `1` | Local adapter update steps per chunk |
| `TTT_LORA_RANK` | `96` | Local adapter rank |
| `TTT_LORA_LR` | `0.0001` | Local adapter learning rate |
| `TTT_WEIGHT_DECAY` | `0.5` | Local adapter weight decay |
| `TTT_BETA1` | `0.0` | Local adapter Adam beta1 |
| `TTT_BETA2` | `0.999` | Local adapter Adam beta2 |
| `TTT_OPTIMIZER` | `adam` | Local adapter optimizer |
| `TTT_K_LORA` | `1` | Enable K LoRA |
| `TTT_MLP_LORA` | `1` | Enable MLP LoRA |
| `TTT_O_LORA` | `1` | Enable output-proj LoRA |
| `GLOBAL_TTT_LR` | `0.001` | Exact PR `#1727` run value |
| `GLOBAL_TTT_MOMENTUM` | `0.9` | Exact PR `#1727` run value |
| `GLOBAL_TTT_EPOCHS` | `1` | Exact PR `#1727` run value |
| `GLOBAL_TTT_CHUNK_TOKENS` | `32768` | Exact PR `#1727` run value |
| `GLOBAL_TTT_BATCH_SEQS` | `32` | Exact PR `#1727` run value |
| `GLOBAL_TTT_WARMUP_START_LR` | `0.0` | Exact PR `#1727` run value |
| `GLOBAL_TTT_WARMUP_CHUNKS` | `0` | Exact PR `#1727` run value |
| `GLOBAL_TTT_GRAD_CLIP` | `1.0` | Exact PR `#1727` run value |
| `GLOBAL_TTT_RESPECT_DOC_BOUNDARIES` | `1` | Exact PR `#1727` run value |
| `GPTQ_CALIBRATION_BATCHES` | `64` | Full record-profile Hessian calibration count |
| `TTT_LORA_ENABLED` | `0` | Standalone LoRA-only eval is intentionally unsupported |

### LaCT

| Variable | Default |
|---|---|
| `LACT_TTT_ENABLED` | `0` |
| `LACT_FAST_WEIGHT` | `swiglu` |
| `LACT_STATE_DIM` | `128` |
| `LACT_SCALE` | `0.08` |
| `LACT_LR` | `0.02` |
| `LACT_MOMENTUM` | `0.9` |
| `LACT_EPOCHS` | `1` |
| `LACT_CHUNK_TOKENS` | `32768` |
| `LACT_UPDATE` | `muon` |
| `LACT_BASE_TTT` | `1` |
| `LACT_BATCH_SEQS` | `16` |
| `LACT_GRAD_CLIP` | `1.0` |
| `LACT_INIT_STD` | `0.02` |
| `LACT_NORMALIZE` | `1` |

### Export / compression

| Variable | Default |
|---|---|
| `ARTIFACT_TARGET_BYTES` | `16000000` |
| `MATRIX_BITS` | `6` |
| `EMBED_BITS` | `8` |
| `MATRIX_CLIP_SIGMAS` | `12.85` |
| `EMBED_CLIP_SIGMAS` | `20.0` |
| `MLP_CLIP_SIGMAS` | `12.0` |
| `ATTN_CLIP_SIGMAS` | `13.0` |
| `LQER_ENABLED` | `1` |
| `LQER_RANK` | `4` |
| `LQER_TOP_K` | `3` |
| `LQER_FACTOR_BITS` | `4` |
| `LQER_ASYM_ENABLED` | `1` |
| `LQER_ASYM_GROUP` | `64` |

---

## Score-First Compliance

The primary path is legal score-first TTT:

1. Tokens are scored before any update that could have used them.
2. Local adapter updates happen only after chunk scoring.
3. Global SGD updates happen only after the scored prefix for a phase has been fully collected.
4. Future phases see the update; already scored tokens do not get rescored under the updated state.

LaCT follows the same ordering when enabled.

---

## Eval Sequence

After training and serialization, the script runs:

1. `pre-quantization post-ema`
2. `quantized`
3. `quantized_sliding_window` when `SLIDING_WINDOW_ENABLED=1`
4. `quantized_ttt_phased` when `TTT_ENABLED=1` and `PHASED_TTT_ENABLED=1`
5. `quantized_lact_ttt` when `LACT_TTT_ENABLED=1`

The primary scored path is `quantized_ttt_phased`.

---

## RunPod Reproduction

### Pod setup

Use Python `3.10` for this record path. The `flash_attn_3` wheel family used here is expected to work with the same CUDA / Torch stack as the April 9 record and may not behave correctly under other interpreter builds.

```bash
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev 

git clone https://github.com/IanniMuliterno/parameter-golf.git
cd /parameter-golf
git checkout apr29-phasedttt-runpod
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
python -m pip install -r records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT/requirements.txt
python -m pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

### Verify Python and FlashAttention

```bash
cd /parameter-golf/records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT

which python
python -c "import sys; print(sys.version); print(sys.executable)"
python -c "import pkgutil; print([m.name for m in pkgutil.iter_modules() if 'flash' in m.name])"
python - <<'PY'
import sys
sys.path.insert(0, ".")
import flash_attn_interface as fai
print("flash_attn_func:", hasattr(fai, "flash_attn_func"))
print("FlashAttnFunc:", hasattr(fai, "FlashAttnFunc"))
print("_flash_attn_forward:", hasattr(fai, "_flash_attn_forward"))
PY
```

Expected:

- `python` resolves to `/parameter-golf/.venv/bin/python`
- the flash module list includes `flash_attn_interface`
- all three final checks print `True`

This wheel family may expose a top-level `flash_attn_interface` module rather than a `flash_attn` package. That is fine for this record; the local wrapper is expected to handle that layout explicitly.

### Dataset setup

Use the matched FineWeb repo and clear any stale manifest before downloading:

```bash
cd /parameter-golf
rm -f data/manifest.json
export MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf
python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 8
```

If the dataset repo requires auth:

```bash
export HF_TOKEN=<your_token>
```

### 1×H100 smoke test

For a manual smoke test on fewer than `8` GPUs, run `train_gpt.py` directly. Do not use `handler.py`, and do not use `run.sh` for the smoke test because the endpoint path requires exactly `8` GPUs and `run.sh` is tuned for the full run.

```bash
cd /parameter-golf/records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT
export REPO_ROOT=/parameter-golf
source /parameter-golf/.venv/bin/activate

SEED=42 \
DATA_DIR="$REPO_ROOT/data/" \
MAX_WALLCLOCK_SECONDS=180 \
GPTQ_RESERVE_SECONDS=20 \
ITERATIONS=20 \
EMA_DECAY=0.0 \
TRAIN_BATCH_TOKENS=131072 \
VAL_BATCH_TOKENS=131072 \
TRAIN_LOG_EVERY=5 \
VAL_LOSS_EVERY=0 \
SLIDING_WINDOW_ENABLED=0 \
VAL_DOC_FRACTION=0.01 \
PHASED_TTT_PREFIX_DOCS=64 \
PHASED_TTT_NUM_PHASES=2 \
TTT_BATCH_SIZE=8 \
TTT_CHUNK_SIZE=16 \
GLOBAL_TTT_BATCH_SEQS=8 \
GPTQ_CALIBRATION_BATCHES=1 \
LACT_TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee smoke_1xh100.log
```

Notes:

- `EMA_DECAY=0.0` avoids misleading post-EMA degradation on a very short smoke run.
- The export path is fixed to mixed GPTQ + LQER. If the packed artifact exceeds `ARTIFACT_TARGET_BYTES`, the run fails explicitly.
- This smoke run is for codepath validation only. It is not representative of the final competition metric.

### Full manual 8×H100 run

For the actual full run on an 8-GPU pod, `run.sh` is the intended launcher. It now pins the same profile documented above, including `LQER_RANK=4`, `LQER_TOP_K=3`, and `GPTQ_CALIBRATION_BATCHES=64`:

```bash
cd /parameter-golf/records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT
source /parameter-golf/.venv/bin/activate
export MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf

SEED=42  bash run.sh 2>&1 | tee logs/seed42.log
SEED=314 bash run.sh 2>&1 | tee logs/seed314.log
SEED=999 bash run.sh 2>&1 | tee logs/seed999.log

grep "quantized_ttt_phased" logs/seed*.log | grep "bpb"
```

Competition note: the historical PR `#1727` run used seeds `42 / 0 / 1234`. The current competition requirement is `42 / 314 / 999`, which is what you should run now.

### Useful overrides

```bash
# Reduce phase count for ablation
PHASED_TTT_NUM_PHASES=3 bash run.sh

# Global SGD ablation
GLOBAL_TTT_LR=0.0005 bash run.sh

# Disable sliding-window eval
SLIDING_WINDOW_ENABLED=0 bash run.sh

# Enable LaCT extra eval
LACT_TTT_ENABLED=1 bash run.sh

# Attention-gate ablation
GATED_ATTN_ENABLED=0 bash run.sh
```

---

## RunPod `@Endpoint`

The endpoint path is built for one seed per request on 8 GPUs.

### Build

```bash
docker build \
  -f records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT/Dockerfile \
  -t parameter-golf:apr29-attngate-phasedttt .
```

### Endpoint configuration

- GPU type: H100 SXM
- GPUs per worker: `8`
- Timeout: `1400` seconds
- Network volume mounted at `/runpod-volume`

The handler persists data under:

- `/runpod-volume/parameter-golf/data`
- `/runpod-volume/parameter-golf/endpoint_results/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT`

### Request pattern

Run one seed per request and repeat for `42`, `314`, `999`.

```json
{
  "input": {
    "seed": 42,
    "run_group_id": "apr29-phasedttt-001",
    "prepare_dataset_if_missing": true
  }
}
```

### Streamed outputs

`handler.py` uses aggregated streaming output, so each job provides:

- live stream logs
- aggregated response text
- `combined_response_log.txt`
- `logs/<run_id>.txt`
- `artifacts/final_model.int6.ptz`
- `summary.json`
- `summary_index.jsonl`

The handler parses `quantized_ttt_phased` as the primary metric and compares it against the latest valid record.

---

## Files

| File | Description |
|---|---|
| `train_gpt.py` | Training, quantization, phased TTT, LaCT |
| `handler.py` | RunPod endpoint wrapper |
| `Dockerfile` | Pre-baked RunPod image |
| `run.sh` | Default one-seed launch script |
| `flash_attn_interface.py` | Strict FlashAttention wrapper |
| `requirements.txt` | Python dependencies |

---

## References

- PR `#1727`: phased TTT + global SGD controller
- PR `#1769`: dense GatedAttn + QuantGate
- PR `#1855`: exact LQER export path
- Zhang et al., *Test-Time Training Done Right*, arXiv:2505.23884
