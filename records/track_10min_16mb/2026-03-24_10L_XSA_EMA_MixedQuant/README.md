# Non-Record Submission: 11L Frontier MixedQuant Trigram

This is a non-record submission documenting a newbie experiment on an **8×H100** cluster to test the scaling of 11 layers with TrigramHash and BigramHash embeddings entirely without magnitude pruning.

The final score (**1.3434 val_bpb**) did not reach baseline levels, and the unpruned Int6/Int8 weight distribution fundamentally failed the artifact limit, generating a payload of 19.3 MB. 

## Summary of Changes (cumulative)

1. **11 Layers** — scaling up depth to push network capacity.
2. **TrigramHash Embedding** — alongside BigramHash to capture triplet context.
3. **U-Net Skip Gates** — sigmoid gating connecting encoder/decoder segments.
4. **Star-ReLU** — quadratic activation scaling.
5. **No Pruning** — exact 0.0 clamping was removed to preserve absolute model density.

## Key Discovery: Pruning is Mandatory for 16MB

The 11-layer architecture uses 27.19M parameters. Because we explicitly disabled the `3%` magnitude pruning strategy (which artificially injects massive zero-entropy spans), PyTorch's `zlib` compression completely failed to squash the raw 6-bit randomness below 16MB. The dense, unpruned weight distributions rigidly maxed out the information entropy, resulting in a **19.36 MB** artifact size.

## Full Experiment Log

| Exp | Description | val_bpb | Artifact (MB) | Status |
|-----|-------------|---------|---------------|--------|
| 001 | 11L MixedQuant Trigram (Unpruned) | 1.3434 | 19.36 | discard (>16MB) |

## Configuration

```
VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 
```

Key metrics:
- `val_bpb` (post-quant): **1.34344213**
- Artifact size: **19,362,024 bytes** (> 16MB Limit)
- Model params: 27,191,386
- Steps completed: 4,301
- Peak memory: 20,890 MiB
- GPU: 8xH100 SXM, 600s wallclock

## Included Files

- `train_gpt.py` — code snapshot of the configuration
- `train.log` — full training telemetry
- `submission.json` — leaderboard metadata