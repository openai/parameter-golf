# Experiment Log

This file is a short record of what has already been tried in this repo so we do not waste time repeating the same ideas without a clear change in approach.

## Current best result

- Branch: `submission/sota-attempt`
- Record folder: `records/track_10min_16mb/2026-03-22_11L_MixedInt56_QAT_TTT_1.1466`
- Best result so far: `1.14657797 val_bpb`
- Artifact size: `14,706,424 bytes`
- Setting: `8xH100`, about `605s` train and `340s` eval
- Important note: this run used PyTorch SDPA, not FA3, so training throughput was worse than expected

## What has already been tried

### Baseline reproduction

- Reproduced the public baseline path and used it as the starting point.
- Baseline leaderboard score in this repo: `1.2244 val_bpb`.
- No need to keep re-running the baseline unless checking infra or correctness.

### 11-layer 512-dim Transformer line

- Main working direction is an `11L`, `512d`, tied-embedding Transformer.
- Includes U-Net-style skip connections, partial RoPE, EMA, LN scale, XSA on last layers, relu-squared MLP, SmearGate, ortho+muP init, and Muon.

### Mixed low-bit export

- Mixed quantization is already in use and works better than a simple uniform export in this line:
  - MLP weights: int5
  - Attention weights: int6
  - Embeddings: int8
- Magnitude pruning before export is also already tried.
- Outcome: artifact dropped to `14.7MB` with useful headroom left.

### QAT

- QAT was already added and fixed.
- The earlier issue was that late QAT was effectively dead code under `torch.compile`.
- Current code now supports:
  - `QAT_ENABLED=1`
  - `QAT_START_STEP`
  - `QAT_START_FRAC`
- What failed before: QAT only kicked in at the very end and got effectively `1` training step.
- The next sensible step is tuning earlier QAT, not proposing QAT from scratch again.

### Test-time training

- Post-quantization TTT is already implemented and clearly helps.
- Current setup:
  - SGD
  - 3 epochs
  - `lr=0.002`
  - `momentum=0.9`
  - first 2 blocks frozen
- Outcome:
  - post-quant roundtrip: about `1.1697`
  - post-TTT sliding eval: about `1.1466`

### Small byte-efficient additions

- Already tried in the current best run:
  - BigramHash increased to `10240`
  - `64` memory tokens
  - backout connection
  - per-head temperature
- These are already part of the current recipe, so they are not new ideas unless the mechanism changes.

### Eval stride

- Sliding-window eval with stride `32` was tested against stride `64`.
- In this run, it made essentially no difference: both landed at `1.1466`.

## Operational lessons

### Remote failures were infra, not model bugs

- A previous RunPod failure happened because the repo was not cloned on the pod.
- Another source of confusion was running from the wrong path.
- Before debugging model code on remote, verify:
  - the repo is cloned
  - the correct branch is checked out
  - the target file exists

### Attention backend matters a lot

- The strong run used SDPA and got about `110ms/step`.
- FA3 is expected to be materially faster and is a high-priority next run.
- Current code supports `ATTN_BACKEND=auto|fa3|fa2|sdpa`.
- If FA3 is required, use `ATTN_BACKEND=fa3` so the run fails fast instead of silently falling back.

## Things that likely deserve the next experiments

- Earlier QAT that runs for a meaningful chunk of training
- FA3 on Hopper to buy more steps under the 10-minute budget
- Spending the remaining artifact headroom on a robust capacity increase:
  - likely a 12th layer
  - or a parameter-shared / recurrent refinement step
- Tighter TTT tuning instead of inventing a completely new eval trick

## Papers already tied to this line

- QQQ: `https://arxiv.org/abs/2406.09904`
- BitNet b1.58: `https://arxiv.org/abs/2402.17764`
- TTT: `https://arxiv.org/abs/2407.04620`
- FlashAttention-3: `https://tridao.me/blog/2024/flash3/`

## Read this first

- `README.md`
- `records/track_10min_16mb/2026-03-22_11L_MixedInt56_QAT_TTT_1.1466/README.md`
- `records/track_10min_16mb/2026-03-22_11L_MixedInt56_QAT_TTT_1.1466/train_gpt.py`
