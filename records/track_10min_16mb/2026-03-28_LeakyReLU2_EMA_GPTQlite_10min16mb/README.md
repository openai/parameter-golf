## LeakyReLU(0.5)² + 11L EMA + GPTQ-lite (10 min / 16 MB track)

**Intent:** Respectable **non-record** style submission on [`records/track_10min_16mb/`](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb): same train/eval/artifact rules as other 10-minute records ([main README](https://github.com/openai/parameter-golf/blob/main/README.md)), with a **small, documented** architecture delta.

### Delta vs parent

| | [2026-03-22 GPTQ-lite record](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233) | This folder |
|---|--------|--------|
| MLP activation | `relu(fc(x))` then `square()` on pre-activation | **`F.leaky_relu(..., negative_slope=0.5)`** then **`square()`** (same width, no extra params) |
| Everything else | EMA + tight SWA + late QAT + GPTQ-lite int6 + partial RoPE + LN scale + XSA4 + VE128 + SmearGate + BigramHash | **Unchanged** (copied `train_gpt.py` then one-line MLP edit) |
| TTT | No | **No** |

### Before opening a PR — fill these in from your logs

1. **`submission.json`**: set `val_loss`, `val_bpb`, `bytes_total`, `date`, and finalize `blurb` from your best seed or from the mean (say which in this README). `author` / `github_id` are preset for [tejas-goyal](https://github.com/tejas-goyal); adjust `author` if you want a different display name.
2. **Three logs**: `train_seed1337.log`, `train_seed42.log`, and a third seed (e.g. `train_seed2024.log`) — full `torchrun` stdout/stderr ([submission process](https://github.com/openai/parameter-golf/blob/main/README.md)).
3. Confirm **train ≤ 10 min** and **eval ≤ 10 min** on **8×H100**, and **code + compressed model ≤ 16,000,000 bytes** (decimal MB; [FAQ](https://github.com/openai/parameter-golf/blob/main/README.md)).

### Hardware and data

- **Fork:** [tejas-goyal/parameter-golf](https://github.com/tejas-goyal/parameter-golf) (sync from [openai/parameter-golf](https://github.com/openai/parameter-golf) as needed). On RunPod, clone **your branch** with this submission, for example:
  `git clone https://github.com/tejas-goyal/parameter-golf.git && cd parameter-golf && git checkout <branch-with-this-folder>`
- **GPUs:** 8× H100 (SXM if possible), matching the record track.
- **Data:** Official FineWeb `sp1024` export — see [Getting Started](https://github.com/openai/parameter-golf/blob/main/README.md):

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### Run (from repo root or pod `/workspace/parameter-golf`)

```bash
cd records/track_10min_16mb/2026-03-28_LeakyReLU2_EMA_GPTQlite_10min16mb

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=600
export EVAL_STRIDE=64
export ITERATIONS=9000
# Parent defaults in train_gpt.py already match warmdown 3500, GPTQ-lite, XSA4, RoPE partial, etc.
# Set seed and capture logs:
export SEED=1337
export RUN_ID="leaky_gptq_1337"
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed1337.log
```

Repeat with **`SEED=42`** and **`SEED=2024`** (and matching `RUN_ID`) into `train_seed42.log` / `train_seed2024.log`.

### Results (fill after runs)

| Seed | Steps | val_loss | val_bpb (sliding s64) | bytes_total |
|------|-------|----------|------------------------|-------------|
| 1337 | | | | |
| 42 | | | | |
| 2024 | | | | |
| **Mean** | — | — | — | — |

### Included files

- `train_gpt.py` — training, GPTQ-lite int6 export, sliding eval
- `submission.json` — **update before PR**
- `README.md` — this file
- `train_seed*.log` — **add after runs**
