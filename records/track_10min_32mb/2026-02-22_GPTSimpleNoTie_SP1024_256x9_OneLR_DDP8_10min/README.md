This record captures a tuned `train_gpt_simple_no_tied_embeddings.py` baseline on the matched `SP-1024` dataset using one global learning rate for all parameter groups, selected via LR sweeps and then validated on `8x H100`.

Important metric note:
- This record's score is the trainer's direct final `val_bpb` on the saved `.pt` model (`metric=final_val_bpb` in `submission.json`).
- It is **not** an `int8+zlib` roundtrip score like the current top leaderboard rows.

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
DATA_PATH=data/matched_10B/datasets/fineweb10B_sp1024 \
FIXED_TOKENIZER_PATH=data/matched_10B/tokenizers/fineweb_1024_bpe.model \
ENABLE_VAL_BPB=1 \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=256 \
NUM_HEADS=4 \
MLP_MULT=4 \
GLOBAL_LR=0.0077063153221854225 \
BETA2=0.95 \
WARMUP_STEPS=20 \
NUM_SCHEDULED_ITERATIONS=25860 \
VAL_LOSS_EVERY=0 \
VAL_TOKENS=1048576 \
VAL_BATCH_SIZE=524288 \
FINAL_MODEL_PATH=/tmp/no_tie_sp1024_d256_l9_h4_m4_ddp8_10min_lr7706_final_20260222T021002.pt \
torchrun --standalone --nproc_per_node=8 train_gpt_simple_no_tied_embeddings.py
```

LR tuning summary (one LR for everything):
- Dense 1x proxy sweep (`24` LRs, `700` steps each) best LR: `0.0077063153221854225`
- 8x DDP confirmation sweep (`6` LRs, `400` steps each) best LR: `0.0077063153221854225`
- 8x confirmation winner (`400` steps): `val_loss=3.0468`

Key metrics (from `train.log`):
- Final eval: `val_loss:2.2789`, `val_bpb:1.3319`
- Train time: `596608ms` (`step_avg:23.07ms`)
- Peak memory: `6337 MiB allocated`, `6634 MiB reserved`
- Serialized model size: `29912636 bytes`
- Code size: `35150 bytes`
- Total submission size: `29947786 bytes`

Init sanity checks (logged at startup):
- `tie_embeddings:False`
- `_zero_init` modules all-zero at init
- `lm_head` starts at exactly zero (`lm_head_abs_mean=0`)
- `tok_emb` and attention/MLP weights are non-zero as expected

Saved submission artifact:
- `final_model.pt` (copied from remote `/tmp`)
- SHA256: `2d80eb95e9ab8edb43ae48cbb0ae17b31b4ecaee6f318b4fd7cdfe0738e9fe43`

Included files:
- `train_gpt_simple_no_tied_embeddings.py` (code snapshot used)
- `train.log` (full training log from `speedruna1`)
- `params.env` (explicit env/config for the recorded run)
- `submission.json` (leaderboard metadata)
- `final_model.pt` (saved submission file)
