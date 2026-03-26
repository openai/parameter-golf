This non-record submission captures the strongest branch found so far while iterating on the 10-minute / 16MB track: a dense 10-layer model with 2048-token train/eval context and sliding-window evaluation at stride 64.

The main outcome of the ablation cycle was negative but useful: several intuitive changes regressed relative to the base dense recipe. In particular, recurrent/shared-depth experiments underperformed badly, 4096-token context regressed in the tested setup, lower Muon matrix LR regressed, longer warmdown regressed, and the first attempt at LoRA test-time training was unstable. The dense 2048/64 branch remained the best.

Configuration of the best completed run:
- Base trainer: copied from `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`
- Train / eval context: `TRAIN_SEQ_LEN=2048`, `EVAL_SEQ_LEN=2048`
- Sliding eval: `EVAL_STRIDE=64`
- Tokenizer: provided SP-1024 tokenizer
- Export: int8 + zlib, under the 16,000,000 byte cap
- TTT: present in code but disabled by default (`TTT_LORA_RANK=0`) after regressions

Best completed result so far:
- `final_int8_zlib_roundtrip_exact val_loss:2.18285341 val_bpb:1.29280874`
- Total submission size int8+zlib: `13039699` bytes

Completed ablations:

| Run | Key Change | Final val_bpb |
|-----|------------|---------------|
| Best dense 2048 | `TRAIN_SEQ_LEN=2048`, `EVAL_SEQ_LEN=2048`, `EVAL_STRIDE=64` | **1.29280874** |
| Lower matrix LR | `MATRIX_LR=0.035` | 1.30365282 |
| Longer warmdown | `WARMDOWN_ITERS=3200` | 1.31444131 |
| Longer context | `TRAIN_SEQ_LEN=4096`, `EVAL_SEQ_LEN=4096` | 1.32191594 |
| LoRA TTT attempt | same dense base + TTT | roundtrip 1.29586995, TTT 1.7587 |

Ongoing work:
- An 8xH100 run of the best dense 2048/64 recipe is in progress. Early validation on that run reached `val_bpb:1.2484` by step 6000 at ~52 ms/step, suggesting the dense branch is the right direction for the main record attempt.

Files included:
- `train_gpt.py`: code snapshot used for the dense branch
- `train.log`: summarized log excerpts from the completed experiments so far
- `submission.json`: metadata for this non-record in-progress submission
