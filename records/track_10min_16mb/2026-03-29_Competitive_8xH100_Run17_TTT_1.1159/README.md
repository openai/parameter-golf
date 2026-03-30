This record captures our competitive run (`run_17`) on 8xH100 via Modal.com.

Trainer: `train_gpt_v3.py` (parameter-banking stack with legal score-first TTT).

Configuration highlights:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0`
- Attention/modeling: GQA, XSA on last 4 layers, Partial RoPE (`ROPE_DIMS=16`), SmearGate, BigramHash(1536), TrigramHash(1024), ValueEmbedding, ValueResidual
- Optimization: Parallel Muon + AdamW, EMA, SWA, late QAT enable threshold `0.15`
- Quantization: `int6+lzma` artifact
- Eval: sliding-window scoring (`stride=64`) + legal score-first TTT (`TTT_LR=0.0025`, `TTT_EPOCHS=6`, `TTT_FREEZE_BLOCKS=0`)
- Infrastructure: 8xH100 SXM on Modal.com, 900s wallclock cap

Key metrics (from `train.log`):
- Timed training stopped at `7692/9000` due to wallclock cap.
- Final int6 sliding-window exact metric: `final_int6_sliding_window_exact val_loss:1.89330345 val_bpb:1.12132391`
- Legal score-first TTT exact metric: `legal_ttt_exact val_loss:1.88884778 val_bpb:1.11868501`
- Train time: `900072ms` (`step_avg:116.99ms`)
- Peak memory: `23863 MiB allocated`, `23978 MiB reserved`
- Serialized model int6+lzma: `15,893,952 bytes`
- Code size: `91,881 bytes`
- Total submission size int6+lzma: `15,985,833 bytes` (under 16,000,000 cap)

Included files:
- `train_gpt.py` (code snapshot used for this run)
- `train.log` (exact run log excerpt)
- `submission.json` (leaderboard metadata)
