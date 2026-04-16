# Record: Block Attention Residuals + Tuned TTT

## Status

This folder is configured for a full **8xH100** run with legal score-first TTT and single-pass sliding evaluation.

Current `submission.json` values are placeholders from prior tuning; after your run, update this README and `submission.json` using the exact metrics emitted by the new log.

## What This Run Adds

Built on PR #549 stack (LeakyReLU^2 + legal TTT + parallel Muon), with:

- Block Attention Residuals (arXiv:2603.15031)
- XSA enabled on all 11 layers
- Tuned TTT (`TTT_LR=0.003`, `TTT_FREEZE_BLOCKS=1`)
- BigramHash disabled
- SmearGate disabled

## Architecture / Training Defaults

- 11L, 512d, 8 heads, 4 KV heads
- MLP 3x with `LeakyReLU(0.5)^2`
- Partial RoPE: 16 dims
- LN scale: `1/sqrt(layer+1)`
- VE128 on layers 9-10
- EMA + SWA
- Late QAT threshold: 0.15
- Quantization: mixed int6 + LZMA
- Iterations: 7000
- Warmdown: 3500

## AttnRes Details

- `ATTN_RES_ENABLED=1`
- `ATTN_RES_NUM_BLOCKS=2`
- `ATTN_RES_MIX=0.25`
- `ATTN_RES_TEMPERATURE=1.1`

Depth sources are detached snapshots at block boundaries; each layer learns a depth query and attends over the source bank.

## Legal TTT / Compliance Notes

The script uses score-first chunked adaptation:

1. Score chunk segments under `torch.inference_mode()`.
2. Train only on already-scored chunk tokens.

Single-pass scoring is implemented with an explicit non-overlapping segment scheduler, so each target token is scored exactly once for both sliding eval and TTT scoring.

No n-gram cache / ETLB / SLOT logic is used in this record script.

## Reproduction (8xH100)

Run from the repo root (recommended, keeps default relative data paths valid):

```bash
cd /home/brao/Desktop/parameter-golf
RUN_ID=attnres_b2_x11_tttlr003_seed1337_h100x8 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-15_AttnRes_BlockDepthAttention_TunedTTT/train_gpt.py
```

If your RunPod path layout differs, set these explicitly before launch:

```bash
export DATA_PATH=/your/repo/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/your/repo/data/tokenizers/fineweb_1024_bpe.model
```

Recommended 3-seed batch for significance:

```bash
cd /home/brao/Desktop/parameter-golf
for S in 42 314 1337; do
  RUN_ID=attnres_b2_x11_tttlr003_h100x8_s${S} SEED=${S} \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-15_AttnRes_BlockDepthAttention_TunedTTT/train_gpt.py
done
```

The script writes:

- `logs/${RUN_ID}.txt`
- `final_model.pt`
- `final_model.int6.ptz`

all directly into this record folder (independent of launch CWD).

## Required Post-Run Updates

After your run finishes, update:

1. `submission.json`
- `val_bpb` = exact value from `legal_ttt_exact` (or your chosen official metric)
- `blurb` with final confirmed settings

2. This README
- Fill a results table from log lines:
  - `step:... val_loss... val_bpb...`
  - `final_int6_roundtrip_exact`
  - `final_int6_sliding_window_exact`
  - `legal_ttt_exact`
  - `Total submission size int6+lzma`
- Add run logs for all required seeds/significance

## Lineage

- PR #549 (@abaybektursun): LeakyReLU^2 + legal TTT + parallel Muon
- PR #414 (@signalrush): 11L/512d/8H/4KV architecture
- PR #399 (@abaybektursun): parameter banking + parallel Muon optimizer
- PR #461 (@Christopher-Lee-McClendon): TTT framework lineage
- arXiv:2603.15031: Block Attention Residuals

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
