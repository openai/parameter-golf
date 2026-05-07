# Non-Record: 11L 3x MLP Seq2048 — val_bpb 1.1791 (8xH100 SXM)

Architecture scaling from the naive baseline: deeper (11 layers), wider MLP (3x), and longer context (seq 2048). Post-quant val_bpb **1.1791** vs baseline **1.2244** (-3.7%). Artifact is 24.5MB — over the 16MB limit because the baseline script only uses int8+zlib. Int6 QAT + GPTQ + LZMA would bring this under 16MB.

## what changed

**11 layers** (from 9): two more transformer blocks. The extra depth gives the model more capacity to learn token relationships. Costs ~50% more params but the BPB gain is substantial.

**3x MLP** (from 2x): hidden dim 1536 instead of 1024. Wider MLPs improve representation capacity, especially with tied embeddings where the output head shares the embedding matrix.

**seq_len 2048** (from 1024): doubled context window. Longer sequences mean the model sees more context during training, which helps with longer-range dependencies and improves the BPB calculation.

**warmdown 2000** (from 1200): longer warmdown schedule for better final convergence under the 600s wallclock cap.

**batch tokens 786432** (from 524288): larger batch for more stable gradients with the bigger model.

## config

```
VOCAB_SIZE=1024  NUM_LAYERS=11  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=3  TIE_EMBEDDINGS=1  TRAIN_SEQ_LEN=2048  WARMDOWN_ITERS=2000
TRAIN_BATCH_TOKENS=786432  MAX_WALLCLOCK_SECONDS=600
```

## run command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=improved_8xH100_v2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=11 MLP_MULT=3 TRAIN_SEQ_LEN=2048 \
WARMDOWN_ITERS=2000 TRAIN_BATCH_TOKENS=786432 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Note: used `NCCL_IB_DISABLE=1` due to RunPod secure cloud networking. This slows step time slightly (~107ms vs ~44ms baseline).

## results

8xH100 SXM (RunPod secure cloud, AP-IN-1):

| seed | steps | ms/step | val_loss | val_bpb | artifact (int8+zlib) |
|------|-------|---------|----------|---------|---------------------|
| 1337 | 5,611 | 107.0 | 1.9908 | 1.1791 | 24,456,914 bytes |

Pre-quant eval at step 5000: `val_loss:2.0022`, `val_bpb:1.1858`
Post-quant roundtrip: `val_loss:1.9908`, `val_bpb:1.1791`

Training progression:

| step | train_loss | val_bpb | time |
|------|-----------|---------|------|
| 200 | 2.460 | — | 21s |
| 1,000 | 2.264 | 1.3133 | 107s |
| 2,000 | 2.149 | 1.2528 | 215s |
| 3,000 | 2.114 | 1.2277 | 321s |
| 4,000 | 1.957 | 1.2078 | 428s |
| 5,000 | 2.014 | 1.1858 | 535s |

Peak memory: 24,326 MiB allocated, 24,870 MiB reserved.

## why the artifact is over 16MB

The baseline `train_gpt.py` uses int8 quantization + zlib compression, which gives ~3.9x compression. At 26.5M params, the compressed artifact is 24.5MB. The SOTA submissions solve this with:
- int6 quantization (6 bits vs 8 bits per weight)
- QAT during warmdown to reduce quantization error
- GPTQ with Hessian-aware error compensation
- LZMA/zstd compression instead of zlib

These techniques are the clear next step — they'd bring this model under 16MB while maintaining or improving the BPB.

## next steps

1. Integrate int6 QAT from the SOTA stack (late-stage STE quantization)
2. Add GPTQ with self-generated calibration data
3. Switch to LZMA compression
4. Add BigramHash embeddings and XSA for further BPB gains

## files

- `train_gpt.py` — baseline script snapshot (no modifications to the code itself, only env var config)
- `train.log` — 8xH100 training log (seed 1337)
- `train_full.log` — full verbose training output
- `submission.json`
