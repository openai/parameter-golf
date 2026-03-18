This record captures the `DepthRecurrentQAT` submission.

Trainer changes in this snapshot:
- Quantization-Aware Training (QAT) via fake-quantize with straight-through estimator (STE) in CastedLinear, so the model learns to produce quantization-friendly weights during training rather than degrading post-hoc
- Byte grouping compression replaces torch.save for serialization: INT8 weights with zlib compression to stay under the 16MB artifact cap
- Depth recurrence: 3 shared transformer blocks looped 3 times for 9 effective layers, dramatically reducing unique parameter count while preserving representational depth
- Per-iteration LoRA deltas (rank 4) applied to shared blocks, giving each loop iteration unique low-rank adaptations via a separate Adam optimizer
- Model widening: dim=768 (up from 512) to reallocate the parameter budget freed by weight sharing into wider representations
- LAWA (Latest-Weight Averaging) checkpoint averaging enabled by default for free BPB improvement at negligible cost
- NTK-aware RoPE scaling applied post-training for extended 2048-token evaluation without degradation

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_SHARED_BLOCKS=3 NUM_LOOPS=3 LORA_RANK=4 MODEL_DIM=768 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Evaluation sequence length: `EVAL_SEQ_LEN=2048`
- LAWA enabled: `LAWA_ENABLED=1 LAWA_INTERVAL=100`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command (track-relevant params):
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=depth_recurrent_qat_8gpu \
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (PLACEHOLDER -- to be filled after H100 runs):
- Pre-quant eval at stop: `val_loss:PLACEHOLDER`, `val_bpb:PLACEHOLDER`
- Post-quant roundtrip eval: `val_loss:PLACEHOLDER`, `val_bpb:PLACEHOLDER`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:PLACEHOLDER`
- Train time: `PLACEHOLDER` (`step_avg:PLACEHOLDER`)
- Peak memory: `PLACEHOLDER`
- Serialized model int8+zlib: `PLACEHOLDER bytes`
- Code size: `PLACEHOLDER bytes`
- Total submission size int8+zlib: `PLACEHOLDER bytes`
- Training steps completed: `PLACEHOLDER`

Baseline comparison:
- Baseline (NaiveBaseline) val_bpb: 1.2244 (post-quant roundtrip)
- Baseline artifact: 15,863,489 bytes

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log -- to be added after run)
- `submission.json` (leaderboard metadata)
