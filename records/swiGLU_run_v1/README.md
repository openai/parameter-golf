# SwiGLU Parameter Golf Submission

## Summary
- Replaced MLP ReLU^2 with SwiGLU
- Reduced MLP width (MLP_MULT=1) to stay under size limit
- Trained for 2000 iterations

## Results
- val_loss: 2.4848
- val_bpb: 1.4716
- Model size: ~13.6MB (int8 + zlib)

## Notes
- Trained on 1 shard (not full dataset)
- Development run (not optimized for leaderboard yet)
