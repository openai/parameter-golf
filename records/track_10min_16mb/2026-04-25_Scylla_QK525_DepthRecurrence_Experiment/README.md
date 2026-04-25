# Record: Scylla + QK-Gain 5.25 + 3-Layer Depth Recurrence + GPTQ

**val_bpb = 0.94166052** (3-seed mean, std 0.000665) | **~15.86 MB** | 8xH100 SXM | No TTT

This run targets the current Scylla-era record baseline of `0.9485` bpb. It improves by `0.00683948` bpb while staying under the 16,000,000 byte artifact cap for all three seeds.

## 3-Seed Results

| Seed | Steps | Post-EMA BPB | Roundtrip BPB | Sliding BPB | Artifact bytes |
|------|-------|--------------|---------------|-------------|----------------|
| 1337 | 5,249 | 0.9585 | 0.96107371 | **0.94106817** | 15,860,133 |
| 42 | 5,260 | 0.9596 | 0.96234529 | **0.94238042** | 15,868,157 |
| 2025 | 5,254 | 0.9588 | 0.96145544 | **0.94153297** | 15,854,517 |
| **Mean** | **5,254** | **0.9590** | **0.96162481** | **0.94166052** | **15,860,936** |

Max artifact size: `15,868,157` bytes, leaving `131,843` bytes of margin.

## Key Techniques

1. **Scylla tokenizer and data path** - Uses the Scylla candidate tokenizer with the correct HF tokenizer metadata file.
2. **XSA-all + FA3** - Keeps Scylla's strong attention/eval path and GPTQ quantization stack.
3. **QK gain 5.25** - Transfers the high query/key gain that worked in later SP8192 records.
4. **3-layer depth recurrence** - Reuses layers 3-5 as virtual layers after 35% of training, with zero additional weight matrices.
5. **Bigram dimension reduced to 40** - Creates enough compressed-artifact headroom for recurrence while retaining most of the quality gain.
6. **Full GPTQ int6 + LZMA artifact** - GPTQ calibration uses training data only; the submission artifact is self-contained.

## Architecture

11 physical transformer layers, 512 model dimension, 8 attention heads, 4 KV heads, train sequence length 2048, tied embeddings, XSA active on all 11 layers, and Scylla candidate vocab size 998.

Depth recurrence is activated at `training_frac >= 0.35`:

```text
encoder: [0, 1, 2, 3, 4, 5, 3, 4]
decoder: [5, 3, 4, 5, 6, 7, 8, 9, 10]
```

This repeats the layer group 3-5 without increasing serialized model parameter count. Skip weights are sized to the virtual encoder/decoder depth.

## Training Configuration

```bash
for SEED in 1337 42 2025; do
  TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_pg \
  TMPDIR=/tmp \
  RUN_ID=scylla_qk525_loop3to5_bg40_seed${SEED} \
  SEED=$SEED \
  ITERATIONS=9000 \
  VAL_LOSS_EVERY=0 \
  DATA_PATH=/workspace/pg/data/datasets/fineweb10B_scylla \
  TOKENIZER_PATH=/workspace/pg/data/tokenizer/candidate.vocab \
  TOKENIZER_META_PATH=/workspace/pg/data/tokenizer/candidate.meta.npz \
  VOCAB_SIZE=998 \
  XSA_LAST_N=11 \
  USE_GPTQ=1 \
  GPTQ_RESERVE_MS=9000 \
  TTT_ENABLED=0 \
  BIGRAM_VOCAB_SIZE=2816 \
  BIGRAM_DIM=40 \
  QK_GAIN_INIT=5.25 \
  NUM_LOOPS=2 \
  LOOP_START=3 \
  LOOP_END=5 \
  ENABLE_LOOPING_AT=0.35 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

Training stopped by wallclock at about 591.1 seconds for each seed. Peak memory was about 35 GB per GPU during training on the clean runs.

## Compliance

- No TTT or eval-time adaptation.
- No SLOT, n-gram cache, ETLB, or validation-derived logit bias.
- `VAL_LOSS_EVERY=0`; validation is only used after training stops for scoring/diagnostics.
- GPTQ calibration uses training batches only.
- All submission artifacts are below 16,000,000 bytes.
- Training stays below 600 seconds on 8xH100 SXM.
- Final sliding-window eval stays below the 600 second eval budget.
- The clean runs used byte-validated Scylla dataset shards after pod migration/data repair.

## Included Files

- `train_gpt.py` - Training, quantization, and evaluation script.
- `train_seed1337.log` - Clean seed 1337 run log.
- `train_seed42.log` - Clean seed 42 run log.
- `train_seed2025.log` - Clean seed 2025 run log.
- `submission.json` - Leaderboard metadata.
