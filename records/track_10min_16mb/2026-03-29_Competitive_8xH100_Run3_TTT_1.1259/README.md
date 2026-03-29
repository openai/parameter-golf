This record captures our best competitive run (`run_3`) on 8xH100 via Modal.com.

Trainer: `train_gpt_competitive.py` (SOTA-derived architecture and training stack).

Configuration highlights:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0`
- Attention/modeling: GQA, XSA on last 4 layers, Partial RoPE (`ROPE_DIMS=16`), SmearGate, BigramHash, ValueEmbedding
- Optimization: Muon + AdamW, EMA, late QAT enable threshold `0.15`
- Quantization: mixed int6/int8 with `int6+lzma` artifact
- Eval: sliding-window scoring (`stride=64`) + legal score-first TTT
- Infrastructure: 8xH100 SXM on Modal.com, 600s wallclock cap

Command used:
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=run_3 \
DATA_PATH=/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 \
VALUE_RESIDUAL=0 \
COOLDOWN_SHAPE=linear \
torchrun --standalone --nproc_per_node=8 train_gpt_competitive.py
```

Key metrics (from `train.log`):
- Timed training stopped at `5779/20000` due to wallclock cap.
- Post-EMA eval: `val_loss:1.9308`, `val_bpb:1.1436`
- Final int6 sliding-window eval: `val_loss:1.90493315`, `val_bpb:1.12821170`
- Legal score-first TTT exact metric: `legal_ttt_exact val_loss:1.90099177 val_bpb:1.12587738`
- Train time: `600090ms` (`step_avg:103.84ms`)
- Peak memory: `22060 MiB allocated`, `22128 MiB reserved`
- Serialized model int6+lzma: `15,852,844 bytes`
- Code size: `90,684 bytes`
- Total submission size int6+lzma: `15,943,528 bytes` (under 16,000,000 cap)

Included files:
- `train_gpt.py` (code snapshot used for this run)
- `train.log` (exact run log)
- `submission.json` (leaderboard metadata)
