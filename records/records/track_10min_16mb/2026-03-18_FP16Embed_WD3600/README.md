Kept the tied embedding in fp16 instead of quantizing it to int8, and tuned the LR schedule. Turns out the embedding is by far the most sensitive tensor to quantize — it's pulling double duty as the output head, so every bit of precision matters.

## what changed

**fp16 embedding passthrough**: one-line change in the quantization function. Instead of int8-quantizing `tok_emb.weight`, I pass it through as fp16. This drops the post-quant BPB degradation from ~0.007 to basically nothing (~0.0005). The tradeoff is ~500KB extra in the artifact, so I shrank the MLP hidden from 1024 to 992 to stay under 16MB.

**warmdown + LR**: bumped `WARMDOWN_ITERS` from 1200 to 3600 and `MATRIX_LR` from 0.04 to 0.06. The default schedule assumes way more steps than you actually get in 10 minutes, so a longer warmdown and higher LR help the model converge properly.

## config

```
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_HIDDEN=992  TIE_EMBEDDINGS=1  WARMDOWN_ITERS=3600  MATRIX_LR=0.06
```

## run command

```bash
RUN_ID=fp16embed \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_HIDDEN=992 \
WARMDOWN_ITERS=3600 \
MATRIX_LR=0.06 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Note: don't set `NCCL_IB_DISABLE=1` — it tanks step throughput on pods with IB/NVLink (~60ms vs ~44ms per step).

## results

8xH100 SXM (RunPod secure cloud):

| seed | steps | val_loss | val_bpb | artifact size |
|------|-------|----------|---------|---------------|
| 1337 | 13,692 | 2.0595 | 1.2197 | 15.90MB |
| 42   | 13,722 | 2.0600 | 1.2201 | 15.90MB |

Pre-quant vs post-quant gap: ~0.0005 BPB (baseline gap is ~0.007).

Improvement over baseline: ~0.013 nats.

Also ran 3 seeds on 8xH200 SXM (all consistent, 1.2163-1.2179 BPB).

## things I tried that didn't work

- **SwiGLU**: better per-step quality but 45% slower on 8-GPU, so fewer total steps. Net negative.
- **depth recurrence** (looping layers): promising idea but needs way more steps than 10 min allows.
- **QAT**: tried both full-training and late-stage. The overhead per step wasn't worth the small quant gap reduction.
- **lzma compression**: actually compresses worse than zlib for int8 weight data.
- **higher embed LR** (0.08 vs 0.05): hurt convergence.

## files

- `train_gpt.py` — modified training script
- `train.log` — 8xH100 log (seed 1337)
- `train_seed42.log` — 8xH100 log (seed 42)
- `submission.json`
