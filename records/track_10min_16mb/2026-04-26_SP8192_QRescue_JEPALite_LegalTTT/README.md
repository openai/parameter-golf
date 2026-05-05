# Submission: SP8192 QRescue + JEPA-Lite + LQER + Pergroup/lrzip + Legal TTT

**val_bpb = 1.08064386** (3-seed mean, std 0.00096256) | **15.70 MB max artifact** | 8xH100 80GB HBM3

## 3-Seed Results

| Seed | Quantized BPB | Sliding BPB | TTT BPB | Artifact bytes | Train seconds | Eval seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.09831799 | 1.08152899 | **1.07971401** | 15,693,775 | 588.075 | 503.000 |
| 314 | 1.09973854 | 1.08312592 | **1.08163610** | 15,696,850 | 588.067 | 512.571 |
| 999 | 1.09876592 | 1.08208204 | **1.08058146** | 15,695,674 | 588.030 | 495.416 |
| **Mean** | **1.09894082** | **1.08224565** | **1.08064386** | **15,695,433** | **588.057** | **503.662** |
| **Std** | | | **0.00096256** | | | |

## Base Lineage

This submission builds on the `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` record lineage:
SP8192, 11 layers x 512d x 8 heads / 4 KV heads, MLP 4x, LeakyReLU(0.5)^2, partial RoPE 16/64, layerwise LN scale, tied embeddings, logit softcap 30.0, depth recurrence layers 3-5, parallel residuals from layer 7, QK-Gain 5.25, MuonEq-R, EMA 0.9965, GPTQ SDClip int6 matrices / int8 embeddings, and legal score-first TTT.

## What Changed

- `COMPRESSOR=pergroup` groups quantized tensors by role and compresses each group with the system `lrzip` binary using `lrzip -z -L 9` when available, falling back to Brotli.
- Pergroup roundtrip is lossless and logged with `roundtrip_ok: True`.
- `LQER_ENABLED=1` with `LQER_BUDGET_BYTES=140000`, `LQER_MAX_RANK=4`, and targets `loop_mlp_proj,late_mlp_proj,attn_proj`.
- `JEPA_LITE_ENABLED=1` is training-side only; the predictor is removed before serialization and is not present in the eval model.
- `TTT_LORA_ENABLED=0` uses the legal legacy full-model TTT path: `chunkwise_score_first_full_sgd`.
- TTT remains score-first and no-rescore, with explicit protocol logs.
- QRescue/Hessian layer-group SDClip multipliers are enabled for GPTQ threshold selection.

## What Did Not Change

No tokenizer changes.  
No dataset changes.  
No validation data during training.  
No retokenization.  
No SLOT.  
No ETLB.  
No n-gram cache.  
No score-after-update TTT.

The dataset/tokenizer path remains the SP8192 cached FineWeb flow:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192
```

## Quantization And Compression

The final artifacts use GPTQ int6 for attention/MLP matrices, int8 token embeddings, and LQER rank-4 residuals on selected projection matrices. The pergroup compressor logs:

```text
pergroup_model_bytes
code_bytes
artifact_bytes
compressor_used
roundtrip_ok
```

All three submitted runs used `compressor_used: pergroup-lrzip`, `roundtrip_ok: True`, and stayed below the decimal 16,000,000 byte cap. `lrzip` is a system dependency rather than a Python package; the reproduction command below installs it with `apt-get install -y lrzip`.

## Legal TTT Compliance

Submitted runs use:

```text
TTT protocol: chunkwise_score_first_full_sgd
TTT reset_policy: cumulative_full_model
TTT phase_boundaries: none
TTT score_before_update: true
TTT no_rescore: true
```

For each chunk, all scored windows are evaluated before any SGD update is applied. No token is re-scored after an update. TTT trains only on already-scored validation tokens and uses the normal full-vocabulary softmax distribution.

## Reproduction

From this record folder on an 8xH100 pod:

```bash
pip install -r requirements.txt
apt-get update && apt-get install -y lrzip

./run_seed.sh 42
./run_seed.sh 314
./run_seed.sh 999
```

Or:

```bash
./run_3seeds.sh
```

Equivalent direct command:

```bash
SEED=42 \
JEPA_LITE_ENABLED=1 \
COMPRESSOR=pergroup \
LQER_ENABLED=1 \
LQER_BUDGET_BYTES=140000 \
LQER_MAX_RANK=4 \
LQER_TARGETS=loop_mlp_proj,late_mlp_proj,attn_proj \
TTT_LORA_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Environment

The submitted logs were produced with:

```text
PyTorch 2.9.1+cu128
NVIDIA driver 580.126.09
8x NVIDIA H100 80GB HBM3
FlashAttention 3 / flash_attn_interface
```

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `requirements.txt`
- `run_seed.sh`
- `run_3seeds.sh`
- `smoke_1xh100.sh`
- `preflight_env.sh`
- `parse_run_logs.py`
- `update_submission_json.py`
- `validate_submission_artifacts.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`

## Attribution

Base record and techniques follow the 2026-04-09 SP8192 legal TTT submission and its cited PR lineage: SP8192/GPTQ/SDClip/MuonEq-R, depth recurrence, parallel residuals, legal score-first TTT, QK gain tuning, and hyperparameter tuning. Pergroup compression and LQER are based on later public Parameter Golf PR lineage.
