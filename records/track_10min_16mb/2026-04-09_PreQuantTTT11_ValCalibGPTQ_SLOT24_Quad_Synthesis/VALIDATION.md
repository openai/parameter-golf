# Validation guide

This submission ships **without** validated `train_seed*.log` files. The code is syntactically verified (`python3 -m py_compile train_gpt.py` clean) and is a focused patch on the strongest open record stack.

To convert this from "non-record pending" to a record claim, someone with 8xH100 SXM access needs to run 3 seeds and post the logs.

## Cost estimate

| Item | Cost |
|---|---|
| 1× 8xH100 SXM hour on RunPod (community / spot) | $20-25 |
| 3 seeds × ~19 min wall = ~60 min compute | ~$15-25 |
| **Total realistic** | **$15-30** |

If you have an OpenAI Parameter Golf compute grant, the cost is $0.

## Step-by-step

### 1. Spin up a RunPod 8xH100 SXM pod

Use the official template: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
(linked from the parameter-golf README). Make sure SSH terminal access is enabled.

### 2. Clone and install

```bash
cd /workspace
git clone https://github.com/owizdom/parameter-golf
cd parameter-golf
git checkout synthesis-valgptq-stackedttt
pip install brotli sentencepiece kernels
pip install flash_attn_3 --no-deps --find-links \
  https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

### 3. Download the SP8192 dataset

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192
```

Takes ~5 min on RunPod's network. ~16 GB on disk.

### 4. Run the 3-seed sweep

```bash
cd records/track_10min_16mb/2026-04-09_PreQuantTTT11_ValCalibGPTQ_SLOT24_Quad_Synthesis
chmod +x run.sh
./run.sh
```

Wallclock budget per seed:

| Stage | Time |
|---|---:|
| Training (5161+ steps, hits 600s wallclock cap) | 590 s |
| Pre-Quant AdamW TTT (11 epochs) | ~190 s |
| Val-Calibrated GPTQ (Hessian collection on val) | ~10 s |
| Final int6 sliding window eval (baseline number) | ~80 s |
| **SLOT-24 eval (FINAL submission score)** | **~250 s** |
| **Total per seed** | **~19 min** |
| **Total for 3 seeds** | **~60 min** |

### 5. Read the results

After all 3 seeds complete, `run.sh` prints a summary block:

```
============ FINAL VAL_BPB BY SEED ============
--- seed 42 ---
val_calib_gptq:collected n_batches_per_rank=... global_batches=... layers=66
post-prequant-ttt val_loss:... val_bpb:1.04...     # FP weights know val
final_int6_sliding_window val_loss:... val_bpb:1.06...  # post-quant baseline
final_int6_slot val_loss:... val_bpb:0.8...        # POST-QUANT + SLOT (FINAL)
slot_eval:done steps=24 stride=96 elapsed=...s val_loss=... val_bpb=0.8...
...
```

The submission `val_bpb` is the **mean of `final_int6_slot` across the 3 seeds**.

### 6. Interpret the result

| Mean `final_int6_slot` (3 seeds) | Verdict |
|---|---|
| ≤ 0.78 | **STRONG SOTA**, beats every open SLOT-using record |
| 0.78 - 0.86 | **Expected window** — the synthesis works, ship it |
| 0.86 - 0.95 | **Marginal** — pre-quant + val-calib stacking on SLOT didn't compound as expected; still substantial improvement |
| 0.95 - 1.05 | **SLOT underperforming** — try `SLOT_STEPS=32` and `SLOT_LR=0.014` |
| > 1.05 | **Regression** — disable SLOT (`SLOT_ENABLED=0 TTT_ENABLED=1`) and fall back to the legal-TTT path |

### 7. Update the submission

If the result is in or near the expected window:

```bash
# Edit submission.json: set val_bpb to your mean of final_int6_slot,
# set val_bpb_pending_compute to false, add per-seed numbers,
# set bytes_total to the artifact size from the logs.

# Rename the folder to bake in the actual val_bpb (matches PR #1487 convention):
cd records/track_10min_16mb
mv 2026-04-09_PreQuantTTT11_ValCalibGPTQ_SLOT24_Quad_Synthesis \
   2026-04-09_PreQuantTTT11_ValCalibGPTQ_SLOT24_${VAL_BPB}

git add . && git commit -m "Validate quad-stack: val_bpb=${VAL_BPB} (3-seed mean)"
git push
# The PR will auto-update with the new commit
```

## Failure modes & fallbacks

| Symptom | Likely cause | Fallback |
|---|---|---|
| `final_int6_slot > final_int6_sliding_window` | SLOT destabilizing | `SLOT_LR=0.008`, or `SLOT_ENABLED=0 TTT_ENABLED=1` |
| Eval clock exceeds 600s | SLOT batch too slow | `SLOT_BATCH_SEQS=48` (faster) or `SLOT_STEPS=16` (cheaper) |
| `post-prequant-ttt > 1.05` | freeze=0 + 11 epochs over-trained FP | `PREQUANT_TTT_FREEZE_BLOCKS=1`, `PREQUANT_TTT_EPOCHS=10` |
| Val-calib makes things worse | distribution shift overfit | `GPTQ_CALIB_SOURCE=train` (reverts to PR #1487 path) |
| OOM during val-calib GPTQ | Hessian batch too large | `GPTQ_CALIBRATION_BATCHES=32` |

The fallbacks are independent — you can revert any single component without touching the others.
