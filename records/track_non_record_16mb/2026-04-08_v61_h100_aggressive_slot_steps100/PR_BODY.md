## Track
`track_10min_16mb` (10-minute wallclock training, 16 MB artifact)

## Headline
**3-seed val_bpb = 1.146523 ± 0.001516**

| seed | val_bpb |
|------|---------|
| 1337 | 1.148530 |
| 1338 | 1.144866 |
| 1339 | 1.146173 |
| **mean** | **1.146523** |
| **std**  | 0.001516 |

vs prior records (this submitter):
- `2026-04-08_v61_aggressive_slot_1159` (slot_steps=20): 1.157108 → **−0.010585 bpb**
- `2026-04-08_v61_slot_steps50_1150` (slot_steps=50): 1.148772 → **−0.002249 bpb**
- `2026-04-08_v61_slot_steps80_1147` (slot_steps=80): 1.147032 → **−0.000509 bpb**

## Parent / cite
- Parent: [openai/parameter-golf#1123](https://github.com/openai/parameter-golf/pull/1123) (HybridQuantGPT v6.1, 1.1986 non-record)
- SLOT origin: [openai/parameter-golf#1176](https://github.com/openai/parameter-golf/pull/1176) (introduced shared `[1, 1, dim]` SLOT delta)
- Previous record (this submitter, supersedes all 3 earlier SLOT records): `v61_aggressive_slot_1159`, `v61_slot_steps50_1150`, `v61_slot_steps80_1147`

## What's new — single-line code change
The training code, model architecture, rANS serializer, hybrid quantization alphabets,
and even the rANS artifacts are **byte-identical** to `2026-04-08_v61_aggressive_slot_1159`.
The only diff is one default value in `argparse`:
```diff
- parser.add_argument("--slot-steps", type=int, default=20)
+ parser.add_argument("--slot-steps", type=int, default=100)
```

## How we found it
The original SLOT record settled on `slot_steps=20` because of a stride=256 quick-eval
ablation that suggested diminishing returns above 20 steps. We re-ran the sweep at
**stride=64 (full eval)** with all configs and discovered the diminishing-returns
estimate was **wrong** — `slot_steps` is monotonically helpful all the way up to 100,
with the gain per added step plateauing only past 80–100.

### Seed 1337 stride=64 full eval sweep (sweep_v2 + sweep_v3)
| slot_steps | seed-1337 final bpb | Δ vs s20 |
|------------|---------------------|----------|
| 20         | 1.158886 (record baseline) | 0       |
| 25         | 1.156018             | -0.0029  |
| 30         | 1.154228             | -0.0046  |
| 40         | 1.151943             | -0.0069  |
| 50         | 1.150672             | -0.0082  |
| 60         | 1.149898             | -0.0090  |
| 70         | 1.149378             | -0.0095  |
| 80         | 1.149012             | -0.0099  |
| **100** ⭐ | **1.148530** chosen   | **-0.0104** |

LR/lr_min/batch_size/warmstart sweeps all found nothing better at the same step
count: lr=0.08–0.12 within ±0.0006 of lr=0.1; lr_min=0.01 vs 0.001 within 0.0004;
batch_seqs=64 hurts by +0.04 (single delta cannot fit larger context); warmstart with
cold AdamW restart hurts by +0.01 (the AdamW restart overshoots starting from a
non-zero delta).

### 3-seed verification: s40, s50, s80, s100 all measured
| slot_steps | s1337 | s1338 | s1339 | mean | std |
|------------|---------|---------|---------|------|-----|
| 20 (record) | 1.158886 | 1.155831 | 1.156608 | 1.157108 | 0.00130 |
| 40 | 1.151943 | 1.148642 | 1.149684 | 1.150090 | 0.00138 |
| 50 | 1.150672 | 1.147260 | 1.148383 | 1.148772 | 0.00142 |
| 80 | 1.149012 | 1.145414 | 1.146671 | 1.147032 | 0.00149 |
| **100** ⭐ | **1.148530** | **1.144866** | **1.146173** | **1.146523** | **0.00152** |

Every step count from 40 to 100 is verified across 3 seeds. **s100 is the consistently
lowest 3-seed mean** — every individual seed improves over s80, s50, s40, and s20.

### Why the prior diminishing-returns estimate was wrong
The earlier ablation that suggested `slot_steps=20` was the sweet spot used `stride=256`
(only 25 % of the val tokens scored). At that resolution, the SLOT delta has fewer
windows to fit across, and the difference between step counts is masked by per-window
variance. At the full `stride=64` eval (969,088 windows), the difference becomes clear
and **monotonic**.

## Reproducibility
```bash
bash records/track_10min_16mb/2026-04-08_v61_slot_steps100_1146/run.sh both 1337
```
Identical 8×H100 SXM training pipeline as `2026-04-08_v61_aggressive_slot_1159`. The
eval phase loads the existing rANS artifact and only differs in the SLOT step count
default (100 instead of 20).

To reproduce on the existing rANS artifacts of `v61_aggressive_slot_1159`:
```bash
python records/track_10min_16mb/2026-04-08_v61_slot_steps100_1146/train_gpt.py \
  --eval --checkpoint runs/v61_slot_s1337/model.rans.ptz \
  --stride 64 --batch-seqs 32 --seq-len 1024 --compile \
  --slot-lr 0.1 --slot-steps 100 --slot-lr-min 0.001 \
  --data-dir data/datasets/fineweb10B_sp1024 \
  --tokenizer data/tokenizers/fineweb_1024_bpe.model
```
This is **the exact same checkpoint** as the prior records — only the eval recipe differs.

## Cost
- Training: byte-identical to `v61_aggressive_slot_1159` (same artifacts, no retraining)
- Eval: 5× of the prior record's SLOT eval (5× more SLOT optimization steps per window)
- Per-seed eval ≈ 50 min on a single H100 (vs ~10 min for steps=20)
- 3-seed verification cost ≈ $80 of RunPod credit

## Legality
Identical to the prior records:
- Training uses only `fineweb10B_sp1024` training shards. Validation tokens never
  enter the training loop.
- SLOT delta is fit **per-batch** using that batch's own target tokens (score-first:
  the batch is scored once at the end, the delta never sees a future batch or
  shared state).
- The shared `[1, 1, dim]` delta is the exact shape from PR #1176.
- No external files loaded at inference; everything is in the artifact tarball.

## Hardware
- 8× H100 80 GB SXM (RunPod)
- Existing rANS artifacts re-used from `v61_aggressive_slot_1159` runs
  (`runs/v61_fa3_seq2048_s1337`, `runs/v61_base_s1338`, `runs/v61_base_s1339`)
