# How to Complete This Submission

This folder is a **scaffold** — you need to run the experiments, collect logs, fill in the results, and then open a PR to the upstream `openai/parameter-golf` repo.

## Step 1: Launch RunPod 8×H100 Pod

1. Go to [RunPod GPU Cloud](https://console.runpod.io/deploy)
2. Use the official Parameter Golf template: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
3. Select **8×H100 SXM** (required for leaderboard submissions)
4. Enable SSH terminal access → Deploy → SSH in

## Step 2: Setup

```bash
cd /workspace
git clone https://github.com/Vickyrrrrrr/parameter-golf.git
cd parameter-golf

# Download SP8192 dataset
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# Install dependencies
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

## Step 3: Run All 3 Seeds

Run each seed. Each run takes ~10 minutes. Save the output logs!

```bash
# SEED 42
SEED=42 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/train_gpt.py \
  2>&1 | tee train_seed42.log

# SEED 314
SEED=314 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/train_gpt.py \
  2>&1 | tee train_seed314.log

# SEED 999
SEED=999 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/train_gpt.py \
  2>&1 | tee train_seed999.log
```

## Step 4: Read the Final BPB

At the end of each log, find the line like:
```
final_int8_zlib_roundtrip | val_bpb=X.XXXX | artifact_bytes=XXXXXXX
```

You need mean val_bpb < **1.0760** (beats 1.0810 - 0.005 threshold).

## Step 5: Update submission files

Fill in `submission.json` and `README.md` with the real values.
Copy the 3 log files into this folder:
```bash
cp train_seed42.log records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/
cp train_seed314.log records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/
cp train_seed999.log records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/
```

## Step 6: Commit & Open PR

```bash
git add records/track_10min_16mb/2026-04-14_SP8192_QK55_4LayerRecur_ParResid_LegalTTT/
git commit -m "Record attempt: SP8192 + QK-Gain 5.5 + 4-Layer Recurrence"
git push origin attempt/qk-gain-5.5-deeper-recurrence
```

Then open a PR from your fork to `openai/parameter-golf` main.

## ⚠️ Important: Do NOT open the PR until logs are real!

The upstream repo has strict requirements — fabricated logs will be disqualified.
Only submit after you have real 3-seed results showing val_bpb < 1.0760.
