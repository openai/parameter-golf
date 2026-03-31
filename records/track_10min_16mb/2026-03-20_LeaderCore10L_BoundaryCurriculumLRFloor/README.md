This directory contains the next size-safe leader-core attempt, rooted in the valid `2026-03-20_LeaderCore10L_ValidEval_TempOnly_Int8Search` line and aimed at importing the most portable modded-nanogpt lessons without spending artifact bytes.

Primary file:
- `train_gpt.py`

What changed in this branch:
- stateful train loader sequence length so the curriculum hits the real runtime control point
- boundary-seeking short-sequence curriculum: default `512 -> 1024`
- synchronized compile refresh when returning to full context
- LR floor during warmdown: default `0.15 * base_lr`

Why this is the next attempt:
- the 11-layer line is materially over the 16MB cap, so the next serious shot should stay on the strongest valid 10-layer artifact path
- this branch spends the expected savings in training time rather than in extra parameters
- all added knobs are size-free and easy to ablate cleanly

Recommended ablation batch:
- `base`
- `no_curriculum`
- `no_lrfloor`
- `floor010`
- `switch035`
- `boundary_off`

RunPod launcher:
```bash
bash /workspace/parameter-golf/launch_leadercore_curriculum_runpod.sh base
```

For timing-parity runs, first stage data/tokenizer to local disk on the pod:
```bash
bash /workspace/parameter-golf/setup_local_parity_data_runpod.sh
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_leadercore_curriculum_runpod.sh base
```

Variant names map to these env changes:
- `base`: `SHORT_SEQ_LEN=512`, `CURRICULUM_SWITCH_FRACTION=0.30`, `CURRICULUM_BOUNDARY_ALIGN=1`, `LR_FLOOR_SCALE=0.15`
- `no_curriculum`: disable the short-sequence phase
- `no_lrfloor`: decay all the way to zero as before
- `floor010`: keep the curriculum but lower the warmdown floor to `0.10 * base_lr`
- `switch035`: keep short context for the first `35%` of wallclock instead of `30%`
- `boundary_off`: keep the short-sequence curriculum but disable boundary seeking

Data root modes:
- `workspace`: use `/workspace/parameter-golf/data/...`
- `tmp`: use `/tmp/parameter-golf-data/...` after local staging
