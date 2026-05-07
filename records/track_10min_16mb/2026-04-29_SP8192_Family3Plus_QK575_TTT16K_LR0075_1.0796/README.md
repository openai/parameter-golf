# SP8192 Family3+ QK5.75 TTT16K LR0.0075

**val_bpb = 1.07955293** on one 8xH100 SXM run.

This folder packages the best local Family3+ result from 2026-04-29 for submission review.

Important caveat: this run beats the public README leaderboard BPB score of `1.0810`, but the record-submission guide also asks for a `0.005`-nat improvement and enough independent logs for `p < 0.01`. This folder contains one completed run, so it should be treated as a strong record-candidate package unless additional seeds are added.

## Result

| Metric | Value |
| --- | ---: |
| legal score-first TTT exact `val_bpb` | **1.07955293** |
| legal score-first TTT exact `val_loss` | `2.78859734` |
| quantized sliding-window `val_bpb` | `1.08103157` |
| pre-quant post-EMA `val_bpb` | `1.08651206` |
| train stop | `4598/20000` |
| train wallclock | `588040ms` |
| TTT eval wallclock | `575974ms` |
| total submission bytes | `15,989,120` |
| cap margin | `10,880 bytes` |

## Technique

Base family:

- `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`

Core stack:

- SP8192 tokenizer
- 11 layers, 512 model dimension, 8 attention heads, 4 KV heads
- 3-layer depth recurrence over layers 3-5
- parallel residuals from layer 7 onward
- QK gain initialization raised to `5.75`
- MuonEq-R / AdamW training stack
- EMA decay `0.9965`
- GPTQ SDClip int6 matrices and int8 token embeddings
- Brotli compression
- legal score-first TTT

Candidate-specific changes versus the April-9 public SOTA family:

- `QK_GAIN_INIT=5.75`
- `TTT_CHUNK_TOKENS=16384`
- `TTT_LR=0.0075`
- `TTT_EPOCHS=3`

## Compliance Notes

- Training used `8x NVIDIA H100 80GB HBM3`.
- Training finished under 600 seconds.
- Legal TTT evaluation finished under 600 seconds.
- Artifact size was below the `16,000,000` byte cap.
- TTT is score-first: each chunk is scored before update.
- No SLOT, no pre-quant validation TTT, no ETLB, no n-gram cache.
- No tokenizer or dataset changes were made.

## Reproduction

Environment used:

- RunPod image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- PyTorch: `2.9.1+cu128`
- FlashAttention-3: `flash_attn_interface`

Install runtime extras:

```bash
python3 -m pip install -q --break-system-packages brotli sentencepiece huggingface_hub datasets tqdm
python3 -m pip install -q --break-system-packages flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

Prepare data from repo root:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

Run from repo root:

```bash
ALLOW_SDP_FALLBACK=0 \
COMPRESSOR=brotli \
EMA_DECAY=0.9965 \
ENABLE_LOOPING_AT=0.35 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
LOOP_END=5 \
LOOP_START=3 \
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
MATRIX_LR=0.022 \
MAX_WALLCLOCK_SECONDS=600 \
MUON_WD=0.095 \
PARALLEL_RESIDUAL_START=7 \
QK_GAIN_INIT=5.75 \
REQUIRE_TORCH_PREFIX=2.9.1 \
TRAIN_SEQ_LEN=2048 \
TTT_CHUNK_TOKENS=16384 \
TTT_ENABLED=1 \
TTT_EPOCHS=3 \
TTT_LR=0.0075 \
VOCAB_SIZE=8192 \
WARMDOWN_FRAC=0.72 \
RUN_ID=family3plus_qk575_ttt16k_lr0075 \
DATA_DIR="$PWD/data" \
TOKENIZER_PATH="$PWD/data/tokenizers/fineweb_8192_bpe.model" \
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-29_SP8192_Family3Plus_QK575_TTT16K_LR0075_1.0796/train_gpt.py
```

## Included Files

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- `runpod_remote.log`

`train.log` is the exact training log recovered from the completed RunPod run.

## Attribution

- `@clarkkev` - SP8192 + GPTQ SDClip base stack
- `@dexhunter` - depth recurrence and legal TTT stack
- `@Robby955`, `@msisovic` - parallel residuals
- `@abaybektursun` - score-first TTT precedent
- `@X-Abhishek-X` - April-9 optimizer/hyperparameter tuning
- `@PrzemyslaV88` - Family3+ QK/TTT push and run packaging
