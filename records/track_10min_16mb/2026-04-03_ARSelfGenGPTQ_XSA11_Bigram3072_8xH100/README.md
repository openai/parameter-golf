# Non-record: AR Self-Gen GPTQ + XSA-11 + BigramHash 3072x112 (8xH100)

**3-seed mean val_bpb: 1.1156** (std 0.0004) | **max artifact: 15,856,186 bytes** | 8xH100 SXM | no TTT

This submission keeps a strict legal path:
- single-pass causal evaluation
- no two-pass/full-rescore logic
- no tokenizer/dataset changes
- no TTT/SLOT/ngram scoring path

## Results

| Seed | Steps | Final val_bpb | Artifact bytes |
|---|---:|---:|---:|
| 42 | 6,759 | 1.11505464 | 15,852,402 |
| 1337 | 6,765 | 1.11613083 | 15,856,186 |
| 2025 | 6,767 | 1.11546497 | 15,847,666 |
| **Mean** |  | **1.11555015** |  |

## Configuration

- 11 layers, model dim 512, GQA (8 heads / 4 KV heads)
- XSA on all 11 layers (`XSA_LAST_N=11`)
- BigramHash: `3072 x 112`
- AR self-generated GPTQ calibration (`64 x 2048`, temperature 0.8)
- Warmdown: 4000
- Late QAT enabled
- Compression: int6 + lzma preset 9
- Submission target budget: 15.9MB

## Command

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 TARGET_MB=15.90 \
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

This run improves our earlier legal internal checkpoints, but does not exceed the current top by the required 0.005-nat margin for a record claim.
