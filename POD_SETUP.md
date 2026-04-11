# RunPod Setup & Training Sequence

Template: `runpod/parameter-golf:latest` — 8×H100 SXM, $21.52/hr on-demand

---

## 1. First-time pod setup

### Install flash-attn (FA2)
The template does not include flash-attn. Install it once per pod:
```bash
pip install flash-attn --no-cache-dir --no-build-isolation
```
Takes ~10 minutes to compile. `--no-build-isolation` is required so the build can find the
already-installed torch. `--no-cache-dir` avoids a cross-device link error on this filesystem.

Verify:
```bash
python3 -c "from flash_attn import flash_attn_func; print('FA2 available')"
```

### Install zstandard
```bash
pip install zstandard
```

---

## 2. Clone the repo

The `/workspace/parameter-golf` directory from the template is not a git repo. Remove it and clone:
```bash
cd /workspace && rm -rf parameter-golf
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf
git checkout 11l-xsa-ema-ttt   # current active branch
```

---

## 3. Download dataset and tokenizer

Only needed once per pod (persists in `/workspace/parameter-golf/data/`):
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This downloads ~8B tokens of FineWeb shards + the sp1024 tokenizer from HuggingFace. Takes a few minutes.

Verify:
```bash
ls data/datasets/fineweb10B_sp1024/
ls data/tokenizers/
```

---

## 4. Testing workflow — always follow this sequence

**Never stack multiple new features in a single full run. Test one at a time.**

### Step 1 — establish the Tier 2 baseline (~3 min, ~$1)

Run the current best config with all schedule-dependent features off:
```bash
git pull
unset MLP_HIDDEN QUANT_BITS RUN_ID SEED
TIER2_MODE=1 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
```
Record **val_bpb at step 2000**. This is your baseline for the current session.
Expected: `*** TIER2_MODE ***` banner in startup log.

### Step 2 — Tier 2 test of a new feature (~3 min, ~$1)

Enable exactly ONE new feature:
```bash
TIER2_MODE=1 XSA_LAST_N=0 torchrun ...   # test: disable XSA
TIER2_MODE=1 SMEAR_GATE=0 torchrun ...   # test: disable SmearGate
TIER2_MODE=1 NUM_LAYERS=9 torchrun ...   # test: fewer layers
```
If val_bpb@step2000 is worse than baseline → skip the feature.
If better → proceed to Tier 3.

### Step 3 — full 10-minute run (Tier 3, ~$3.60)

Only run after Tier 2 shows improvement:
```bash
unset MLP_HIDDEN QUANT_BITS RUN_ID SEED
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
```
Startup log should confirm:
- `*** TIER2_MODE ***` NOT present
- `ntk_rope:enabled train_seq_len:1024 eval_seq_len:2048`
- `xsa_last_n:4 active_layers:[7, 8, 9, 10]`
- `ema:initialized decay=0.997`
- `qat:True (activates when 480s elapsed; guarantees 120s of QAT)`
- step_avg ~50ms (NTK-RoPE at seq_len=1024), targeting ~10000+ steps

### What TIER2_MODE cannot test

These features only matter at the end of training — skip them in Tier 2:
| Feature | Why Tier 2 can't test it | How to test |
|---------|--------------------------|-------------|
| EMA | Benefits from averaging converged weights, not early-stage weights | Full run only |
| TTT | Needs a well-trained model to adapt | Full run only |
| SWA | Requires many checkpoint samples across long warmdown | Full run only |
| QAT | Disabled in TIER2_MODE automatically | Full run only |

---

## 5. IMPORTANT — always unset env vars before any run

```bash
unset MLP_HIDDEN QUANT_BITS RUN_ID SEED
echo "MLP_HIDDEN=${MLP_HIDDEN} QUANT_BITS=${QUANT_BITS}"  # should be empty
```

MLP_HIDDEN inherited from a prior run will silently bloat the model from ~26.8M to ~31.2M params,
causing artifact > 16MB. This has happened before.

---

## Notes
- PyTorch version on template: `2.9.1+cu128`
- `flash_attn_interface` (FA3) is NOT available — use FA2 (`flash_attn`)
- FA3 install fails due to cross-device link error on this pod filesystem
- Data path: `./data/datasets/fineweb10B_sp1024/`
- Tokenizer path: `./data/tokenizers/fineweb_1024_bpe.model`
- Logs saved to `./logs/<RUN_ID>.txt`
- At NTK-RoPE seq_len=1024: expect ~50ms/step, ~10000 steps in 600s
- At seq_len=2048 (no NTK): expect ~85-92ms/step, ~6500-7000 steps in 600s
