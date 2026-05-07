# Record: SP8192 + QK Depth Ramp + Parallel Residuals from Layer 6 + Legal TTT

**val_bpb = 1.0809** (seed 42) | **15,993,776 bytes** | 8xH100 SXM, PyTorch 2.9.1+cu128

This folder captures the best run we obtained with the `qkramp05_par0` recipe after moving it onto a fresh CUDA 12.8 / PyTorch 2.9.1 / `flash_attn_3` image. This is a single-seed systems rerun, not a 3-seed new-SOTA claim. The model recipe is the same QK-depth-ramp + early parallel-residual + legal score-first TTT stack we had already stabilized; the gain came from recovering the intended runtime path and training throughput.

## Seed 42 Result

| Metric | Value |
|------|------:|
| Pre-quantization post-EMA BPB | 1.08755571 |
| Quantized BPB | 1.09986947 |
| Sliding-window BPB | 1.08306921 |
| **TTT BPB** | **1.08088517** |
| Artifact bytes | 15,993,776 |
| Code bytes | 16,936 |
| Training steps | 4,661 |
| Train time | 588,092 ms |
| TTT eval time | 337,814 ms |

## What Changed

1. **QK depth ramp** — `QK_GAIN_INIT=5.0` with `QK_GAIN_DEPTH_RAMP=0.5`, giving a 5.0 -> 5.5 schedule across depth instead of a flat QK gain.
2. **Earlier parallel residuals** — parallel residual lanes start at layer 6 (`PARALLEL_RESIDUAL_START=6`) and are enabled from step 0.
3. **Legal score-first TTT** — chunked SGD on already-scored tokens only, matching the recent SP8192 legal-TTT line.
4. **Runtime stack correction** — same training script, but rerun on `torch 2.9.1+cu128` with `flash_attn_3` instead of the older `torch 2.4.1+cu124` image that forced us into a slower effective path.

## Why This Rerun Matters

Our earlier stable-container run of the same recipe (`qkramp05_par0_s42`) finished at `1.08797291` with only `3774` steps in the 588s train budget. On the updated runtime stack, the exact same modeling recipe reached `4661` steps and closed almost the entire gap to the current top seed-42 result.

| Metric | Older stabilized image | New runtime image |
|------|-----------------------:|------------------:|
| Training steps | 3,774 | **4,661** |
| Sliding-window BPB | 1.08958727 | **1.08306921** |
| TTT BPB | 1.08797291 | **1.08088517** |
| Artifact bytes | 15,991,437 | 15,993,776 |

This is a systems optimization submission in spirit: the main gain came from restoring throughput and the intended attention backend, not from introducing a new architecture after the slower run.

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), tied embeddings, layerwise LN scale, logit softcap 30. The model uses SP8192 tokenization, looped layers 3-5, skip-gated U-Net connections, and parallel attention/MLP residual lanes starting from layer 6.

## Training

The run used the SP8192 + GPTQ SDClip + MuonEq-R stack with EMA decay `0.9965`, Muon weight decay `0.095`, matrix LR `0.022`, and a `0.72` warmdown fraction. Training stopped on the wallclock cap at `4661` steps in `588.092s` on 8xH100 80GB SXM.

Throughput profile from the actual run:

- `500` steps: `7.87M tok/s`
- `1000` steps: `7.88M tok/s`
- `2000` steps: `7.89M tok/s`
- `4500` steps: `6.27M tok/s`

The recurrence-enabled portion still slows the later steps, but the updated environment recovered enough budget to add `887` extra optimizer steps versus the older container.

## Quantization

Full-Hessian GPTQ with SDClip was used for int6 attention/MLP matrices and int8 token embeddings. The quantized model plus code snapshot compresses to `15,993,776` bytes, leaving `6,224` bytes of margin under the 16MB cap.

## TTT

Evaluation uses legal score-first TTT:

- chunk size `32768`
- SGD LR `0.005`
- momentum `0.9`
- 3 epochs per chunk
- score all tokens in a chunk before any weight update

The TTT stage improved the sliding-window score from `1.08306921` to `1.08088517`.

## Compliance

- Training completed under the 600s limit
- Final artifact is under 16,000,000 bytes
- Evaluation is causal and uses a full-vocabulary softmax
- TTT is score-first and single-pass over the validation stream
- No SLOT
- No pre-quant TTT on validation data
- No ETLB
- No n-gram cache
- No tokenizer or dataset changes

## Reproduction

```bash
python3 -m venv .venv291
source .venv291/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
pip install brotli sentencepiece huggingface_hub tqdm

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

RUN_ID=qkramp05_par0_s42_t291 SEED=42 QK_GAIN_INIT=5.0 QK_GAIN_DEPTH_RAMP=0.5 \
PARALLEL_RESIDUAL_START=6 ENABLE_PARALLEL_RESIDUAL_AT=0 TTT_ENABLED=1 \
TTT_LR=0.005 TTT_EPOCHS=3 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@clarkkev** — SP8192 + GPTQ embeddings + SDClip + MuonEq-R base stack
- **@dexhunter** — depth recurrence and legal score-first TTT on SP8192
- **@Robby955** and **@msisovic** — parallel residuals
- **@bigbag** — QK-gain line that this depth-ramp variant builds on
- **@X-Abhishek-X** — tuned SP8192 hyperparameters used in the base stack

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
