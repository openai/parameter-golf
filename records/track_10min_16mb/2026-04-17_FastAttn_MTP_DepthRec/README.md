# FastAttn + MTP + Depth-Recurrence (10 min, 16 MB)

A pragmatic fork of the proven 43ms/step baseline with three targeted upgrades
that are known to improve per-step quality without hurting throughput.

## Why this plan

The baseline (9L × 512d, pure SDPA GQA attention) hits **43 ms/step** and
**val_bpb = 1.2244** in 10 minutes. The previous `GatedMixer` attempt ran at
220 ms/step — 5x slower — which cannot reach competitive scores regardless of
tuning. This record throws out exotic mixers and instead layers three
well-established techniques on the fast baseline.

## Upgrades vs. baseline

| # | Upgrade | Effect |
|---|---------|--------|
| 1 | **Depth recurrence** — run block stack `NUM_REPS=2` times, weights shared | +100% compute per step, 0 extra params. Works well under a parameter-size cap. |
| 2 | **Multi-token prediction (MTP)** — aux CE loss on token `t+2` via a tiny projection | +3-5% BPB reduction at same tokens (DeepSeek-V3, MoC). Disabled at eval. |
| 3 | **Wider model (576 vs 512), fewer layers (7 vs 9)** | Same param budget, more attention heads see the data, better GQA layout (8/2). |

Everything else (Muon, SDPA/FlashAttn, U-Net skips, tied embeddings, int8+zlib
GPTQ, logit softcap) is inherited verbatim from the baseline.

## Architecture

- **Layout**: 7 layers × 576 dim, 8 heads / 2 KV heads, MLP×2 (ReLU²).
- **Effective depth**: 14 (7 physical × 2 reps).
- **Vocab**: 1024 SentencePiece BPE, tied embed/unembed.
- **Context**: 1024 tokens.
- **Recurrence gate**: per-rep learned vector, init 0 → training learns to lean
  on recurrence gradually. At init the model behaves identically to the
  baseline U-Net.
- **MTP**: single `CastedLinear(d,d)` → RMSNorm → tied head, predicting
  `target[t+1]` from hidden at `t`. Aux loss weight 0.3, training only.

## Training recipe

- **Steps**: 12000 target (wall-clock cap at 590s usually stops earlier).
- **Batch**: 524288 tokens/step, seq_len 1024.
- **Schedule**: 30-step warmup, flat, 1500-step linear cooldown.
- **Optimizers**: Muon (matrices, lr=0.04), Adam (embeddings lr=0.05, scalars
  lr=0.04).
- **No mid-training validation** — eats the wall-clock budget. Final eval runs
  once post-training over full validation split.

## Expected outcome

- **Target: `val_bpb` ≈ 1.08–1.15**, a meaningful jump from baseline 1.22.
- Reaching 1.02 likely requires 2-3 more iterations on top (span corruption,
  4x recurrence, maybe distillation). This is a solid foundation.

## Reproduce

```bash
# one-time
pip install brotli sentencepiece -q
python3 data/cached_challenge_fineweb.py --variant sp1024

# full 8xH100 run
SEED=42 bash records/track_10min_16mb/2026-04-17_FastAttn_MTP_DepthRec/run_leaderboard_8xh100.sh 2>&1 | tee train_seed42.log

# smoke (1 GPU, 2 min)
bash records/track_10min_16mb/2026-04-17_FastAttn_MTP_DepthRec/run_smoke_1gpu.sh
```

## Files

- `train_gpt.py` — forked from root baseline + depth-recurrence + MTP
- `run_leaderboard_8xh100.sh` — production launcher
- `run_smoke_1gpu.sh` — sanity check
- `submission.json` — leaderboard metadata (val_bpb filled after run)
