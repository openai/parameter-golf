# 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04, stride16 candidate

This folder is a fast follow-up candidate based on the current top record.
The only behavior change is the final sliding-window evaluation stride:

- baseline record: `EVAL_STRIDE=64`
- this candidate: `EVAL_STRIDE=16`

The training recipe stays unchanged:

- 10 layers, 512 dim, 8 heads, 4 KV heads
- 3x MLP expansion
- SmearGate + BigramHash(10240)
- mixed int5/int6 quantization
- SWA at `start_frac=0.4`
- Muon WD=0.04

Why this is worth trying:

- smaller stride should improve the reported `val_bpb`
- the eval-time cost should still stay within the 10-minute budget on 8xH100
- it does not change the trained model, so the comparison is clean

Run with the script defaults:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This folder is intentionally left without copied leaderboard logs or submission metadata.
It is the candidate source to verify next.
