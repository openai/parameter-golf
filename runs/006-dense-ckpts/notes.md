# Spec 006 — execution notes

**Run dir:** `runs/006-dense-ckpts/` (this dir) + JP volume `jlxvxeiol4:/workspace/runs/006-dense-ckpts/`
**Pod:** `p1cuz1a3ntvxaw` (8×H100 SXM, AP-JP-1, $23.92/hr)
**Code:** `exp/dense-ckpts-grad-logging @ a671090`
**Date:** 2026-04-20

## Result

Training completed natively at step 4550/4550. Watcher auto-killed the pod when `ckpt_final_post_ema_step4550.pt` landed and log confirmed, before the post-training GPTQ quantization started. All 49 checkpoints saved + full train.log with per-step loss + grad norms.

### Key numbers at final step

| metric | value |
|---|---|
| final train_loss (step 4550) | 2.8240 |
| final val_loss (step 4550) | 2.8088 |
| final val_bpb (step 4550, base weights) | 1.0875 |
| pre-quantization post-EMA val_bpb | **1.08639** |
| training wallclock | 11.8 min |
| tok/s | 5.04M |
| peak memory | 39 GB / 80 GB |

### Schedule events fired as predicted

- warmdown_start at step **1275** (predicted ~1274 ✓)
- pre_recurrence / recurrence activation at step **1593** (predicted ~1593 ✓)

Step-based schedule confirmed to shift these events relative to spec 000's wallclock-based positions (which were 1048 and 1378).

## Why AP-JP-1 not US-NE-1

US-NE-1 had no 8×H100 capacity for several hours on 2026-04-20 (retry attempts all failed with "no instances available"). AP-JP-1 was probed successfully (1×H100 probe landed first try, then the full 8×H100 run pod landed immediately). Data was seeded onto the JP volume via hybrid HF + rsync.

**Data provenance note:** spec 006's training data on JP was seeded via:
- **80 of 128 train shards + val + tokenizer**: `hf download sproos/parameter-golf-tokenizers` → bit-identical to NA (SHA-verified on shards 000, 040, 079, val, tokenizer)
- **48 of 128 train shards (080–127)**: rsync from NA pod → JP pod (integrity via rsync's built-in checksums; spot-verified on shards 100, 127 — note SHA reference for these post-rsync shards is the rsync result itself, no separate NA baseline captured for these two specific files)

All 129 shard files present on JP volume. File count matches NA exactly.

## Configuration used

```
BIGRAM_VOCAB_SIZE=0
QK_GAIN_INIT=5.25
TTT_ENABLED=1                # set per spec, never reached (killed pre-eval)
SEED=42
ITERATIONS=4550
MAX_WALLCLOCK_SECONDS=0      # disables wallclock cap → step-based frac
TRAIN_LOG_EVERY=5            # ~910 log rows
VAL_LOSS_EVERY=100           # 45 val samples
CKPT_STEPS=100,200,…,4500    # 45 explicit ckpts
CKPT_DIR=/workspace/runs/006-dense-ckpts/checkpoints
```

## Code patch notes (exp/dense-ckpts-grad-logging)

Purely observational. 11-line diff to `train_gpt_sota.py`:
- Captured `torch.nn.utils.clip_grad_norm_` return value (pre-clip total grad norm).
- Added per-block grad-norm computation: `torch.sqrt(sum(p.grad.pow(2).sum() for p in block.parameters()))` for each block in `base_model.blocks`.
- Logged both at every `should_log_train` milestone (every 5 steps with TRAIN_LOG_EVERY=5).

**Spec said `base_model.transformer.h`; actual attribute is `base_model.blocks` (verified at `train_gpt_sota.py:483`).** Used the correct attribute. No control flow changes.

## Log sample

```
4550/4550 train_loss: 2.8240 train_time: 11.8m tok/s: 5039876 grad_norm: 0.0610 per_layer_gn: [0.0143,0.0136,0.0135,0.0211,0.0212,0.0208,0.0145,0.0147,0.0145,0.0195,0.0230]
4550/4550 val_loss: 2.8088 val_bpb: 1.0875
peak memory allocated: 39045 MiB reserved: 39124 MiB
Checkpoint saved: /workspace/runs/006-dense-ckpts/checkpoints/ckpt_final_pre_ema_step4550.pt (313.6 MB)
ema:applying EMA weights
Checkpoint saved: /workspace/runs/006-dense-ckpts/checkpoints/ckpt_final_post_ema_step4550.pt (313.6 MB)
pre-quantization post-ema val_loss:2.80577455 val_bpb:1.08639319 eval_time:6337ms
```

## Artifacts summary

**Local (this dir):**
- `train.log` — full log, 1132 lines (~910 training milestones + 46 val milestones + schedule events + ckpt saves)
- `checkpoints.md` — pointer file enumerating all 49 ckpts on JP volume
- `notes.md` — this file

**On JP volume (`jlxvxeiol4`):**
- `/workspace/runs/006-dense-ckpts/checkpoints/` — 49 × ~313 MB (15 GB total)
- `/workspace/runs/006-dense-ckpts/train.log`

## Cost accounting

| item | cost |
|---|---|
| Phase 0 (earlier in day) NA audit pod for volume cleanup | ~$0.30 |
| Phase 1 parallel seed (NA audit + JP seed, ~15 min combined) | ~$1.20 |
| Phase 2 preflight + Phase 3 training (~15 min 8×H100) | ~$6.00 |
| Phase 4 handback (brief pod restart for artifact pull, ~2 min 8×H100) | ~$0.80 |
| **Total spec 006 today** | **~$8.30** |
| Plus sunk cost on earlier botched rsync attempts | **~$5.00** |
| **Project spec 006 spend** | **~$13.30** |

## Kill protocol result

File-exists poll + log grep both triggered at 20:32:53 UTC. Discord ping posted. `runpodctl pod stop p1cuz1a3ntvxaw` executed. GPTQ Hessian collection (post-training eval stage) had just started; cleanly terminated before any quant work completed — exactly what the spec wanted.

## Handback to research

- `val_bpb 1.08639` pre-quant post-EMA is in the expected range for a 4550-step step-based training (spec 000 fraction-based landed at 1.09289; step-based schedule shifts events slightly, which changes the final bpb by ~0.006).
- Step-based schedule worked as predicted: schedule events landed at step 1275 and 1593, matching the fraction math (0.28 × 4550 = 1274, 0.35 × 4550 = 1592.5).
- **49 checkpoints × full train.log** = the dense dataset research requested. Ready for spec 007 (weight-delta analysis on dense ckpts) or any of the other analyses listed in the spec's hypothesis section.
- Research owns the evaluation writeup + experiments.md row per CLAUDE.md — execution did not touch those.
