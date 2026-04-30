This record captures `11L DepthRec PolarNS SWA`. Non-record submission on the 10min / 16MB track.

## Summary

A 28.5M-param 11-layer transformer trained for 600s on 8×H100 SXM, serialized to an int6 + zstd-22 artifact totaling 15,999,891 bytes (109 bytes under the 16MB cap). Pre-int6 `val_bpb` at the wallclock cap is `1.1444`. The post-int6 sliding-window eval didn't complete on this run due to a pod interruption right after the artifact was written; a 3-seed run with proper sliding measurement is planned as a follow-up.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4`
- Tied embeddings, partial RoPE (16 / 64 dims), layerwise LN scale
- BigramHash (3072 buckets, dim=112)
- Depth recurrence: blocks 4 and 5 reuse the MLP of block 3, each pass gated by a learned scalar
- XSA on the last 4 layers
- Parallel residuals from layer 7 onward
- int6 per-row quantization on MLP and attention 2D weights, tied embedding stays fp
- zstd-22 serialization

## Training

- Muon for matrices (Newton-Schulz with Polar Express coefficients + AOL preconditioning, 5 iters); Adam for scalars and embeddings
- `TIED_EMBED_LR=0.035 MATRIX_LR=0.025 SCALAR_LR=0.025`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048`
- Late QAT kicks in at scale 0.15
- SWA starts at scale 0.2 and averages every 50 steps; the final serialized weights are a blend of EMA and SWA
- `MAX_WALLCLOCK_SECONDS=600`, seed 1337

## Command

```bash
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
cd records/track_10min_16mb/2026-04-15_11L_DepthRec_PolarNS_SWA
torchrun --standalone --nproc-per-node=8 train_gpt.py
```

## Key metrics

| step | val_loss | val_bpb |
|-----:|---------:|--------:|
|    0 | 6.9288   | 4.1036  |
| 2000 | 2.1428   | 1.2691  |
| 4000 | 2.0641   | 1.2225  |
| 6000 | 1.9881   | 1.1775  |
| 7000 | 1.9374   | 1.1474  |
| 7171 | 1.9323   | 1.1444  |

- Training stopped at 7171 / 20000 steps against the wallclock cap (`step_avg:83.68ms`)
- Peak memory: 18,204 MiB allocated, 19,866 MiB reserved
- Artifact: 15,968,114 bytes (int6 + zstd-22)
- Code: 31,777 bytes
- Total: 15,999,891 bytes

## Approach

The stack is a combination of several published ideas on top of the public baseline. Depth recurrence lets 11 physical MLPs cover 13 attention positions at zero parameter cost, with a learned scalar per reused pass so the model can weigh the repeated MLP differently from the first pass. XSA on the last 4 layers and parallel residuals from layer 7 onward take some compute pressure off the deep blocks. Inside Muon, Polar Express coefficients and AOL preconditioning replace the classic Newton-Schulz triplet, which keeps the orthogonalization well-conditioned in 5 iterations. SWA averages late-training checkpoints once the warmdown schedule is below a fraction threshold, and the final serialized weights are a blend of EMA and SWA.

The byte budget was the tight constraint: the int6 state dict for this config compresses to ~16.2 MB under the standard lzma-9 path, which is over the cap. Switching the serialization path brought it under 16 MB with room left over for a minified training script.

## Caveats

- Single seed (1337), so no statistical significance claim over the current SOTA yet. Submitting as non-record for iteration signal.
- `val_bpb` above is pre-int6; the post-int6 sliding-window number was not measured on this run. Will report once the 3-seed follow-up lands.
