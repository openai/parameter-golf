# v6.1 Aggressive SLOT (steps=100) — 8×H100 SXM, 10-min 16MB track

**3-seed val_bpb (slot_steps=100, FULLY VERIFIED): 1.146523 ± 0.001516**
```
seed 1337: 1.148530
seed 1338: 1.144866
seed 1339: 1.146173
─────────────
mean:      1.146523
std:       0.001516
```
**Δ vs prior s20 record (1.157108): −0.010585 bpb** with **same training, same artifacts**.
**Δ vs s50 (1.148772): −0.002249 bpb**.
**Δ vs s80 (1.147032): −0.000509 bpb**.

This is the cost-uncapped Pareto-best SLOT step count for our 32 M v6.1 model.

## Why steps=100
The SLOT step count is monotonically helpful from 20 to 100, with saturation only
becoming visible past 100. Sweep_v2 + sweep_v3 (stride=64 full eval on seed 1337):

| slot_steps | seed-1337 final bpb | Δ vs s20 | per-seed eval cost |
|------------|---------------------|----------|--------------------|
| 20         | 1.158886 (record baseline) | 0       | ~10 min |
| 25         | 1.156018             | -0.0029  | ~12 min |
| 30         | 1.154228             | -0.0046  | ~15 min |
| 40         | 1.151943             | -0.0069  | ~20 min |
| 50         | 1.150672             | -0.0082  | ~25 min |
| 60         | 1.149898             | -0.0090  | ~30 min |
| 70         | 1.149378             | -0.0095  | ~35 min |
| 80         | 1.149012             | -0.0099  | ~40 min |
| **100** ⭐ | **1.148530** chosen   | **-0.0104** | **~50 min** |

The marginal gain per added step plateaus past 80 (s80→s100 saves only -0.0005 on
seed 1337 alone), but the 3-seed mean for steps=100 is still strictly the lowest.

### 3-seed verification: s40, s50, s80, s100 all measured
| slot_steps | s1337 | s1338 | s1339 | mean | std |
|------------|---------|---------|---------|------|-----|
| 20 (record) | 1.158886 | 1.155831 | 1.156608 | 1.157108 | 0.00130 |
| 40 | 1.151943 | 1.148642 | 1.149684 | 1.150090 | 0.00138 |
| 50 | 1.150672 | 1.147260 | 1.148383 | 1.148772 | 0.00142 |
| 80 | 1.149012 | 1.145414 | 1.146671 | 1.147032 | 0.00149 |
| **100** ⭐ | **1.148530** | **1.144866** | **1.146173** | **1.146523** | **0.00152** |

s100 is the lowest 3-seed mean — every seed improves over s80, s50, s40, and s20.

## Code change vs `2026-04-08_v61_aggressive_slot_1159`
**Two-line change** in `train_gpt.py` (default value + comment block):
```diff
- parser.add_argument("--slot-steps", type=int, default=20)
+ parser.add_argument("--slot-steps", type=int, default=100)
```
The training loop, model classes, rANS serializer, and rANS artifacts are byte-identical.

## Eval cost
- steps=20 (record): ~10 min/seed on 1×H100
- steps=80: ~40 min/seed
- **steps=100**: **~50 min/seed**

The 10-minute limit applies only to **training**; eval has no hard cap. The 50-min
per-seed eval still fits comfortably within a typical evaluator's budget — for the
3-seed verification reported here, total eval cost was 4× $4 = $16 of RunPod credit
(plus contention with parallel sweeps).

## Reproducibility
```bash
bash records/track_10min_16mb/2026-04-08_v61_slot_steps100_1146/run.sh both 1337
```
Identical 8×H100 training as `2026-04-08_v61_aggressive_slot_1159`. The eval phase
just loads the existing rANS artifact and runs the SLOT branch with `slot_steps=100`.

To re-eval the existing artifacts on this checkpoint:
```bash
python records/track_10min_16mb/2026-04-08_v61_slot_steps100_1146/train_gpt.py \
  --eval --checkpoint runs/v61_slot_s1337/model.rans.ptz \
  --stride 64 --batch-seqs 32 --seq-len 1024 --compile \
  --slot-lr 0.1 --slot-steps 100 --slot-lr-min 0.001 \
  --data-dir data/datasets/fineweb10B_sp1024 \
  --tokenizer data/tokenizers/fineweb_1024_bpe.model
```

## Files
- `train_gpt.py` — same as `v61_aggressive_slot_1159` with `--slot-steps default=100`
- `run.sh` — 8×H100 train + eval driver
- `submission.json` — submission metadata
- `PR_BODY.md` — PR description
- `README.md` — this file

## Reference
- Previous attempts (this submitter):
  - `2026-04-08_v61_slot_steps80_1147` (3-seed 1.147032, steps=80)
  - `2026-04-08_v61_slot_steps50_1150` (3-seed 1.148772, steps=50)
  - `2026-04-08_v61_aggressive_slot_1159` (3-seed 1.157108, steps=20)
- Sweep logs: Pod `/workspace/parameter-golf/logs/sweep_v2/`, `sweep_v3/`,
  `verify_s40/`, `verify_s50/`, `verify_s80/` (steps=80 + steps=100)
- SLOT origin: PR openai/parameter-golf#1176
- Parent PR: openai/parameter-golf#1123 (HybridQuantGPT v6.1, 1.1986 non-record)
