# Late STE QAT + Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Overtone + SWA + SGD TTT

## Score

**Measured (seed=1337, single run):** `val_bpb = 1.16292025` · `val_loss = 1.96353693` (after int6+zstd roundtrip + sliding-window eval; see `train.log`).

Trained on **8×H100 SXM** with a **600s** wallclock cap (`step=5464`). **Total submission size `15,948,643` bytes** (~15.95 MB decimal), **below** the 16,000,000-byte limit — **int6 + zstd-22** artifact plus UTF-8 `train_gpt.py` bytes (`64,426`).

> *Note:* The template-style multi-seed table below is **not** part of this folder’s logs; only **seed 1337** is recorded here. Re-run with other `SEED` values if you want a proper mean/std.

## Approach

Stacked techniques on a **9-layer, 512-dim** GPT-style model, plus **late STE QAT**, **Overtone-style init**, and optional **full-model SGD TTT** (this script defaults to SGD TTT on, LoRA TTT off).

### 1. Per-row int6 quantization + zstd-22

MLP and attention weight matrices are quantized to int6 (roughly `[-32, 31]`) with **per-row scaling**. Tied embeddings stay in a higher-precision path where it matters; the implementation follows the repo’s mixed-quant rules. After `torch.save` of the quantized payload, the blob is compressed with **zstd level 22** (`zstandard`), which is typically a few percent smaller than **zlib-9** on the same bytes — enough here to land **under** the decimal 16MB cap when zlib did not.

### 2. 3× MLP expansion

Hidden FFN width **1536** (3×) instead of 2× **1024**, paid for in the budget by int6 + strong compression.

### 3. SmearGate

A learned gate blending each token’s embedding with the **previous** token’s embedding for cheap bigram-like signal at the embedding layer (on the order of **~512** extra parameters in the usual setup).

### 4. BigramHash embedding

A **4096**-bucket table (e.g. dim **128**, projected to model width) keyed by adjacent token pairs via a small hash of `(prev, curr)`. Adds on the order of **~0.5M** parameters and complements SmearGate with an **additive** bigram path.

### 5. Orthogonal init (+ muP-style scaling)

Large matrices initialized orthogonal where applicable; readouts scaled with depth-aware factors consistent with muP-style training in this codebase.

### 6. Muon + AdamW, weight decay

**Muon** on matrix blocks with tuned **weight decay** and momentum schedule; scalar/embedding groups use **AdamW** with their own WD. This run uses **`muon_weight_decay=0.038`**, **`matrix_lr=0.025`** (see env overrides in `train_gpt.py`).

### 7. Stochastic weight averaging (SWA)

SWA accumulates weights over the **last fraction** of training (default **`swa_start_frac=0.5`**) every **`swa_every`** steps (default **`200`** in this script). The logged run averaged **5** checkpoints before quantization.

### 8. Late STE QAT (last ~15% of wallclock)

**Fake-quant (STE)** for int6 is only enabled after **`qat_start_frac≈0.85`** of the wallclock budget, with **`qat_lr_factor=0.5`** on the affected optimizer groups when QAT turns on — so Muon is not fighting quant noise for the whole run.

### 9. Full-model SGD test-time training (optional)

A short **SGD** pass on the validation stream (**not** LoRA) to adapt all weights, including gates and bigram paths LoRA often misses. Controlled by **`SGD_TTT_ENABLED`** / **`TTT_LORA_ENABLED`**.

## Main Hyperparameters

| Parameter | Value (this script / logged run) |
|-----------|----------------------------------|
| num_layers | 9 |
| model_dim | 512 |
| mlp_mult | 3.0 (hidden=1536) |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| warmdown_iters | 3000 |
| matrix_lr | 0.025 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| muon_momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| muon_weight_decay | 0.038 |
| weight_decay (AdamW scalars) | 0.01 |
| grad_clip_norm | 0.3 |
| eval_stride | 64 |
| swa_every | 200 |
| swa_start_frac | 0.5 |
| qat_start_frac | 0.85 |
| qat_lr_factor | 0.5 |
| bigram hash buckets | 4096 |
| bigram dim | 128 |
| compressor | **zstd (level 22)** |
| SGD TTT | LR `3e-4`, momentum `0.95` (when enabled) |

## Key metrics (this snapshot)

| Item | Value |
|------|--------|
| **val_bpb** | **1.16292025** (`final_int8_zstd_roundtrip_exact`) |
| **val_loss** | **1.96353693** |
| Wallclock cap | 600s |
| Steps completed | 5464 |
| Model params (logged) | ~22.37M |
| **bytes_total** | **15,948,643** (under 16MB cap) |
| **bytes_code** | **64,426** |
| int6+zstd blob (logged) | 15,884,217 bytes |

## Reproducibility

**Logged run** (seed **1337**):

| Seed | val_loss | val_bpb |
|------|----------|---------|
| 1337 | 1.96353693 | 1.16292025 |

For multiple seeds, re-launch with e.g. `SEED=42`, `SEED=7`, etc. Byte totals and BPB can shift slightly across machines due to GPU non-determinism.

## Evaluation pipeline (order)

1. Train until the 600s cap (late QAT only in the tail).
2. Apply SWA checkpoint average.
3. Quantize to int6 + **zstd-22** → `final_model.int8.ptz`.
4. Decompress, dequantize, **sliding-window eval** (`eval_stride=64`).
5. If enabled: **SGD TTT**, then final metrics.

## How to reproduce

Install **zstandard** and cache FineWeb (`sp1024`) from the repo root; set **`HF_TOKEN`** if downloads require it.

```bash
pip install zstandard
export HF_TOKEN="your_token"   # if needed
python3 data/cached_challenge_fineweb.py --variant sp1024
```

```bash
cd /path/to/parameter-golf

RUN_ID=late_qat_sgd_ttt_zstd \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
EVAL_STRIDE=64 \
SGD_TTT_ENABLED=1 \
TTT_LORA_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 \
  old/20/03/26-zstandard/train_gpt.py
```

## Files in this folder

| File | Purpose |
|------|---------|
| `train_gpt.py` | Training + zstd artifact |
| `train.log` | Log for the run above |
| `submission.json` | Summary JSON for the challenge |
