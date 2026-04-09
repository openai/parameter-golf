# Validation guide

This submission ships **without** validated `train_seed*.log` files because the
synthesis was authored on a Mac without 8xH100 access. The code is syntactically
verified (`python3 -m py_compile train_gpt.py` clean) and is a 186-line minimal
patch over PR #1487's `train_gpt.py`.

To convert this from "non-record pending" to a record claim, someone with
8xH100 SXM access needs to run 3 seeds and post the logs.

## Cost estimate

| Item | Cost |
|---|---|
| 1× 8xH100 SXM hour on RunPod (community / spot) | $20-25 |
| 3 seeds × ~13 min wall = ~40 min compute | ~$15-20 |
| 1× iteration in case of OOM / config tune | ~$5 |
| **Total realistic** | **$15-30** |

If you applied for and received an **OpenAI Parameter Golf compute grant** via
the form on the README, the cost is $0.

## Step-by-step

### 1. Spin up a RunPod 8xH100 SXM pod

Use the official template: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
(linked from the parameter-golf README). Make sure SSH terminal access is
enabled.

### 2. Clone and install on the pod

```bash
cd /workspace
git clone https://github.com/owizdom/parameter-golf
cd parameter-golf
git checkout synthesis-valgptq-stackedttt    # this branch
pip install brotli sentencepiece kernels
pip install flash_attn_3 --no-deps --find-links \
  https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

All other Python deps are in the RunPod template image.

### 3. Download the SP8192 dataset

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192
```

This downloads the full validation split + 80 training shards (8B tokens, ~16
GB on disk). Takes ~5 min on RunPod's network.

### 4. Run the 3-seed sweep

```bash
cd records/track_10min_16mb/2026-04-09_PreQuantTTT11_ValCalibGPTQ_LegalEvalTTT_Synthesis
chmod +x run.sh
./run.sh
```

Wallclock budget per seed:
- Training (5161-5174 steps, hits 600s wallclock cap): 590 s
- Pre-Quant AdamW TTT (11 epochs): ~190 s
- Val-Calibrated GPTQ (Hessian collection on val): ~10 s
- Final int6 sliding window eval: ~80 s
- Eval-Time Legal Score-First TTT (2 epochs, 32K chunks): ~250 s
- **Total per seed: ~19 min**
- **Total for 3 seeds: ~60 min**

### 5. Read the results

After all 3 seeds complete, `run.sh` prints a summary block:

```
============ FINAL VAL_BPB BY SEED ============
--- seed 42 ---
val_calib_gptq:collected n_batches_per_rank=... global_batches=... layers=66
post-prequant-ttt val_loss:... val_bpb:1.04...   # FP weights know val
final_int6_sliding_window val_loss:... val_bpb:1.05...   # post-quant baseline
final_int6_ttt val_loss:... val_bpb:1.05...      # post-quant + eval-time TTT (FINAL)
...
```

The number to report as the submission's `val_bpb` is the **mean of
`final_int6_ttt` across the 3 seeds** (this is what PR #1493 does and is the
score the eval scoring uses).

### 6. Interpret the result

| Mean `val_bpb` (3 seeds) | Verdict |
|---|---|
| ≤ 1.0550 | **NEW SOTA RECORD.** Update `submission.json` `val_bpb`, post the 3 seed logs, file as a record submission. |
| 1.0551 - 1.0599 | **Strong non-record**. Beats PR #1487 but doesn't clear the 0.005-nat threshold. Still publishable as a meaningful synthesis improvement; iterate. |
| 1.0600 - 1.0699 | **Marginal**. Synthesis works but tuning needs work. Try `TTT_EPOCHS=3`, `PREQUANT_TTT_EPOCHS=12`, or revert one of the freeze knobs. |
| ≥ 1.0700 | **Regression**. Most likely the val-calib GPTQ is overfitting or the freeze=0 destabilizes pre-quant TTT. Revert via `GPTQ_CALIB_SOURCE=train` and `PREQUANT_TTT_FREEZE_BLOCKS=1`. |

### 7. Update the submission

If the result clears the SOTA bar:

```bash
# Edit submission.json: set val_bpb to your mean, set val_bpb_pending_compute to false,
# add per-seed numbers, set bytes_total to the artifact size from the logs.

# Rename the folder to reflect the actual val_bpb (matches naming convention of PR #1487):
cd records/track_10min_16mb
mv 2026-04-09_PreQuantTTT11_ValCalibGPTQ_LegalEvalTTT_Synthesis \
   2026-04-09_PreQuantTTT11_ValCalibGPTQ_LegalEvalTTT_${VAL_BPB}

git add . && git commit -m "Validate synthesis: val_bpb=${VAL_BPB} (3-seed mean)"
git push
# The PR will auto-update with the new commit
```

Then comment on the PR with the validated numbers.

## Failure modes & fallbacks

| Symptom | Likely cause | Fallback |
|---|---|---|
| `final_int6_ttt > final_int6_sliding_window` | Eval-time TTT is destabilizing | Reduce `TTT_LR=0.003` or `TTT_EPOCHS=1` |
| `post-prequant-ttt > 1.05` | freeze=0 + 11 epochs over-trained | `PREQUANT_TTT_FREEZE_BLOCKS=1`, `PREQUANT_TTT_EPOCHS=10` (PR #1487 baseline) |
| Eval clock exceeds 600s | TTT 2 epochs too slow | `TTT_EPOCHS=1` or `TTT_CHUNK_TOKENS=65536` |
| OOM during val-calib GPTQ | Hessian batch too large | `GPTQ_CALIBRATION_BATCHES=32` |
| Val-calib makes things worse | Distribution shift overfit | `GPTQ_CALIB_SOURCE=train` (reverts to PR #1487 path) |

The fallbacks are independent — you can revert any single component without
touching the others.
