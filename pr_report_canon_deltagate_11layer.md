# Record: 11L CANON-AC(last5)+DeltaGate Report (Humble Record Attempt)

## Summary

This run builds on the **current leaderboard-aligned stack** (official + pending-validated direction) and focuses on a scoped CANON placement with CANON delta gate.

Best observed result in this sweep:

- `final_int6_sliding_window_exact val_bpb: 1.12961770`

Compared to my previous PR [#312](https://github.com/openai/parameter-golf/pull/312):

- `1.16682362 -> 1.12961770` (large improvement)

## Quick Comparison (vs #312)


| Run                                                                | Setup                    | Steps Before Wallclock Stop  | Final sliding-window val_bpb | Submission size (int6+zstd) |
| ------------------------------------------------------------------ | ------------------------ | ---------------------------- | ---------------------------- | --------------------------- |
| Previous [#312](https://github.com/openai/parameter-golf/pull/312) | ACD (all) + SWA          | 7210 (batch size is default) | `1.16682362`                 | `13,267,347` bytes          |
| This work (seed 1337)                                              | AC(last5)+delta+tightSWA | `6278`                       | `1.12961770`                 | `15,581,348` bytes          |
| This work (seed 1336)                                              | AC(last5)+delta+tightSWA | `6243`                       | `1.1303`                     | `15,505,544` bytes          |
| This work (seed 1335)                                              | AC(last5)+delta+tightSWA | `6252`                       | `1.12970337`                 | `15,579,865` bytes          |


## What Was Reused From Current Leaderboard (not unofficial-only additions)

This run intentionally reuses patterns already common in official/pending leaderboard entries, to check the possibility of Canon layers.:

- 11L / 512-dim / GQA (8 heads, 4 KV heads), MLP 3x
- BigramHash + SmearGate
- XSA on last 4 layers (`XSA_LAST_N=4`)
- Partial RoPE (`ROPE_DIMS=16`) + LN Scale
- Late QAT
- WD 0.04, Tight SWA schedule
- Sliding-window eval (`stride=64`)

## Main Configuration (this report)

- `CANON_SET=AC`
- `CANON_LAST_N=5`
- `CANON_DELTA_GATE=1`
- `SWA_ENABLED=1`, `TIGHT_SWA=1`, `TIGHT_SWA_EVERY=50`, `TIGHT_SWA_START_LRMUL=0.2`, `TIGHT_SWA_MAX_CHECKPOINTS=12`
- `TRAIN_BATCH_TOKENS=786432`, wallclock-capped run (`MAX_WALLCLOCK_SECONDS=600`)

## Definitions (for this report)

- `Delta` (in `AC(last5)+delta`) means **CANON delta gate**:
  - `CANON_DELTA_GATE=1`
  - each CANON branch output is scaled by a learnable sigmoid gate before residual add.
- `Last 4` means **XSA is enabled only on the last 4 transformer blocks**:
  - `XSA_LAST_N=4`
- `XSA learnable gate` means an extra learnable scalar that mixes normal attention output and XSA output:
  - `y <- y + sigmoid(g) * (y_xsa - y)`
  - controlled by `XSA_LEARNABLE_GATE` and `XSA_GATE_INIT`

## Final Run Command (renamed RUN_ID)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
env \
  RUN_ID=frontier_canon_ac_k3_8gpu_final_report_seed1336 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 SEED=1336 \
  NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0 \
  TRAIN_SEQ_LEN=2048 \
  EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
  ITERATIONS=7000 WARMUP_STEPS=20 WARMDOWN_ITERS=3000 MAX_WALLCLOCK_SECONDS=600 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  MUON_WEIGHT_DECAY=0.04 ADAM_WEIGHT_DECAY=0.04 \
  EMA_ENABLED=0 \
  SWA_ENABLED=1 TIGHT_SWA=1 TIGHT_SWA_EVERY=50 TIGHT_SWA_START_LRMUL=0.2 TIGHT_SWA_MAX_CHECKPOINTS=12 \
  XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
  LATE_QAT=1 QAT_THRESHOLD=0.1 \
  INT6_CATEGORIES=mlp,attn TRAIN_BATCH_TOKENS=786432 GRAD_CLIP_NORM=0.3 \
  CANON_SET=AC CANON_KERNEL=3 CANON_RESIDUAL=1 CANON_ACTIVATION=0 CANON_BIAS=0 \
  CANON_FIRST_N=0 CANON_LAST_N=5 CANON_DELTA_GATE=1 CANON_DELTA_GATE_INIT=-4.0 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

### Seed-level excerpts

- Seed `1337`:
  - `step:6278/7000 val_loss:1.9339 val_bpb:1.1454`
  - `final_int6_sliding_window_exact val_loss:1.90730712 val_bpb:1.12961770`
  - `Total submission size int6+zstd: 15581348 bytes`
- Seed `1335`:
  - `step:6252/7000 val_loss:1.9349 val_bpb:1.1460`
  - `final_int6_sliding_window_exact val_loss:1.90745178 val_bpb:1.12970337`
  - `Total submission size int6+zstd: 15579865 bytes`
- Seed `1336`:
  - `step:6243/7000 val_loss:1.9365 val_bpb:1.1469`
  - `final_int6_sliding_window_exact val_bpb: 1.1303`
  - `Total submission size int6+zstd: 15505544 bytes`

### Wallclock / speed notes

- AC(last5)+delta runs stopped around ~6250-6280 steps due to 600s wallclock cap.
- No-canon run reached `6930` steps under the same cap (faster, but lower quality).

## Ablations (sliding-window val_bpb)

- Full CANON `ACD`: `1.14083538`
- CANON `AC` (broad): `1.13218808`
- CANON `AC` (first 4 layers): `1.1314`
- No CANON: `1.13587538` -- it was faster, but it doesn't have a better bpb. 
- CANON `AC(last5)+delta`: best observed `1.1296`
- XSA learnable gate (`XSA_LEARNABLE_GATE=1`): not helpful here (`~1.131`)

## Comparison vs Previous PR

Previous: [#312](https://github.com/openai/parameter-golf/pull/312)

- `final_int6_sliding_window_exact val_bpb: 1.16682362`

Current best in this report:

- `final_int6_sliding_window_exact val_bpb: 1.12961770`

Approx improvement:

- `Δ bpb = -0.03720592`
- `Δ nats ≈ 0.0258` (using `bpb * ln(2)` conversion)

## Significance Note

Against official SOTA context (`1.1428 BPB`), this run clears the `>=0.005 nat` improvement margin by a comfortable amount in point estimate.
For formal `p < 0.01` reporting, include the completed 3-seed list (1335/1336/1337) and test output in PR comments.

## Humble Notes

- This is an incremental engineering result built on existing leaderboard-proven ideas plus scoped CANON placement and gating.
- The strongest gain seems to come from the interaction of AC(last5), CANON delta gate, and tight SWA under the same compute budget.

