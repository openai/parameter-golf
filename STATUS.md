# SP8192 Rebase — Sprint Status

Date: 2026-04-20, ET
Author: Tanish Gudise (CMU '27)

## Base

Rebased onto openai/parameter-golf PR #1394 (Kevin Clark, SP8192, 1.08563 BPB, 5-seed mean).
SP8192 = SentencePiece BPE vocab 8192 + GPTQ embedding quantization + SDClip + Layer looping (Loop45x2) + MuonEq-R + EMA.

All upstream commits through PR #1511 included (branch tracks upstream/main at 75700cb).

Source: `records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt_human.py` (1408 lines).
Working file: `train_gpt_sp8192_opt.py` (1537 lines after levers).

## Levers ported (in `train_gpt_sp8192_opt.py`)

### Lever 1: QUANT_ONLY_CHECKPOINT (commit b4f0b51)
Env var: `QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt`

Skips training entirely. Loads the specified checkpoint, then runs the full GPTQ + SDClip quantization + eval pipeline. Enables ~15-min calibration sweep iterations vs 6-8hr full retrains on 1×H100. Critical for rapid hyperparameter iteration on calibration knobs without burning GPU-hours.

### Lever 2: CALIB_SPLIT_BY_MODULE (commit 1047be9)
Env vars: `CALIB_SPLIT_BY_MODULE=1`, `CALIB_ATTN_BATCHES=N`, `CALIB_MLP_BATCHES=M`

Independent Hessian collection pass counts for attention vs MLP layers during GPTQ calibration. Allows attention layers (with long-range dependencies) to use more calibration data than MLP layers (locally structured). Novel mechanism motivated by the hypothesis that attention Hessians benefit from more diverse activation patterns. Prior sweep on PR #1019 base showed saturated signal under GPTQ SDClip — testing on SP8192 base where the attention regime (vocab 8192, q_gain=4.0) is fundamentally different.

### Lever 3: QK_GAIN_INIT_SCHEDULE (commit 52b0a80)
Env var: `QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5"` (11 values = num_layers)

Per-layer initialization of the q_gain parameter (per-head learned attention scaling). SP8192 base uses uniform init=4.0 across all 11 layers. bigbag (PR #1493, 1.0810 BPB) found 5.25 optimal uniformly. This lever sweeps a gradient schedule — lower early layers, peak at mid/deep, taper at output — motivated by the intuition that earlier layers should attend more broadly (lower gain = softer attention) while deeper layers benefit from sharper focus. The schedule is learned from init; q_gain remains a trainable parameter.

## Key architectural notes

- SP8192 uses `CastedLinear` for all projection matrices — GPTQ calibration hooks (`collect_hessians`, `collect_hessians_split_by_module`) attach to `CastedLinear` modules, compatible with prior calib_sweep approach.
- `classify_param` logic identical to `_classify_param` in `train_gpt_calib_sweep.py` — no renaming needed for the split calibration lever.
- SP8192 uses SDClip GPTQ (`row_std * clip_sigmas / clip_range` threshold) rather than percentile search — preserved as-is. Mixed-regime lever wraps around this quantization path.
- Layer looping (`num_loops=2`, `loop_start=4`, `loop_end=5`) creates 17 virtual layers from 11 physical. `QK_GAIN_INIT_SCHEDULE` takes 11 values (physical layers). Looped layers 4-5 share q_gain weights.

## New imports / dependencies (vs PR #1019 base)

All were already present in `train_gpt_calib_sweep.py`:
- `sentencepiece` — SP8192 tokenizer
- `flash_attn_interface` (Flash Attention 3, Hopper-only) — already on pod
- `brotli` — compression (vs lzma in PR #1019 base)

No new pip dependencies added.

## Overnight run — launch pending

Launch playbook: `LAUNCH_COMMANDS.md`
Tanish will execute manually from SSH before sleep tonight.

Planned config:
```
QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5"
SEED=42
DATA_DIR=/workspace/parameter-golf/data
python train_gpt_sp8192_opt.py
```

Expected runtime: ~6-8 hours on 1×H100 at 3$/hr (~$20-25).
Target BPB: beat 1.08563 (SP8192 base). Stretch: approach 1.0810 (bigbag SOTA).

## SOTA context (not porting tonight)

bigbag PR #1493 (1.0810 BPB) = SP8192 + 3L recurrence (layers 3-5, 17 virtual from 11 physical) + Parallel residuals (layers 7+, GPT-J style) + QK-Gain 5.25 (uniform, monotonic improvement 4.0→5.0→5.25) + Legal TTT (score-first SGD, 3 epochs cosine, causal chunks).

Our q_gain schedule sweeps from 2.0 to 4.5 — lower than bigbag's 5.25. If overnight confirms monotonic improvement from schedule, the next experiment is pushing upper values toward 5.25.

## Prior work

See branch `tanish-calibration-sweep` and `README_SWEEP_RESULTS.md` for the calibration-sensitivity null finding that motivated this rebase.
BPB gap closed: 1.1147 (our Mar 25 base) → 1.0856 (SP8192 base) = -0.0291 just from rebasing.
