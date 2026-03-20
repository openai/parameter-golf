# SmearGate + OrthoInit + Int6 STE QAT + MLP 3x + Bigram Hash + Sliding Window

All-in-one competitive submission stacking every proven technique from the Parameter Golf leaderboard, plus orthogonal init and bigram hash embeddings.

## Techniques

### Architecture

- **10-layer, 512-dim transformer** with 8 attention heads (4 KV heads via GQA), tied embeddings, U-Net skip connections — one extra layer funded by int6 savings
- **MLP 3x expansion** (hidden=1536): enabled by int6 quantization freeing ~4MB of artifact budget
- **SmearGate**: per-dimension sigmoid gate (init ≈ 0.95) blending each token with the previous token's embedding, injecting bigram context at the embedding layer
- **Bigram hash embedding**: 4096-bucket hash table (dim=128 projected to 512) mapping consecutive token pairs to learned embeddings via `(prev * 92821 + cur) % 4096`
- **Orthogonal weight init**: all non-zero-init weight matrices initialized with `nn.init.orthogonal_()` for uniform gradient flow from step 1

### Training

- **STE int6 quantization-aware training**: every forward pass fake-quantizes 2D weight matrices to int6 [-31, 31] via straight-through estimator — the model learns to be robust to its own post-training quantization
- **Muon optimizer** with decoupled weight decay (0.01), momentum warmed from 0.92 → 0.99 over 1500 steps
- **Tuned LR**: tied_embed_lr=0.030, matrix_lr=0.020, scalar_lr=0.020
- **Warmdown 3000 steps** for better convergence under the 10-minute cap

### Evaluation

- **Sliding window eval** (stride=64): overlapping 1024-token windows where each scored token gets 960+ tokens of prior context. Zero artifact cost.

### Compression

- **Int6 per-row quantization** for all 2D weight matrices ([-31, 31] range, per-row scale in fp16)
- **fp16 passthrough** for tied embedding (most quant-sensitive tensor)
- **zstd-22** compression (or zlib-9 fallback)

## Config

```
VOCAB_SIZE=1024  NUM_LAYERS=10  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=3  TIE_EMBEDDINGS=1  USE_SMEARGATE=1
BIGRAM_HASH_BUCKETS=4096  BIGRAM_HASH_DIM=128
USE_INT6_QAT=1  EVAL_STRIDE=64
MATRIX_LR=0.020  SCALAR_LR=0.020  TIED_EMBED_LR=0.030
MUON_MOMENTUM=0.99  MUON_MOMENTUM_WARMUP_START=0.92  MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_WEIGHT_DECAY=0.01  WARMDOWN_ITERS=3000
TRAIN_SEQ_LEN=1024  TRAIN_BATCH_TOKENS=524288
MAX_WALLCLOCK_SECONDS=600
```

## Run command

```bash
RUN_ID=allinone \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

**Pending 8xH100 run.** Expected val_bpb based on technique stacking:

| Technique | Estimated BPB impact |
|-----------|---------------------|
| Baseline | 1.2244 |
| + sliding window stride=64 | ~-0.034 |
| + int6 + MLP 3x | ~-0.019 |
| + fp16 embed | ~-0.007 |
| + Muon momentum 0.99 + LR | ~-0.005 |
| + SmearGate + bigram hash | ~-0.005 |
| + STE QAT (near-zero quant gap) | ~-0.007 |
| + orthogonal init | ~-0.003 |
| **Expected** | **~1.145** |

## Artifact size analysis

Int6 + zstd-22 frees massive headroom. At 10L 512d MLP3x: **24.7M params, ~10.4MB artifact, 5.6MB headroom**.

| Config | Params | Artifact | Headroom |
|--------|--------|----------|----------|
| 9L 512d MLP3x | 22.4M | 9.5MB | 6.5MB |
| **10L 512d MLP3x** | **24.7M** | **10.4MB** | **5.6MB** |
| 12L 512d MLP3x | 29.5M | 12.3MB | 3.7MB |
| 10L 640d MLP3x | 38.2M | 15.6MB | 0.4MB |

Default is 10L for safe training time; push to 12L or 640d if step throughput allows.

## Verified locally

- Python syntax OK, 1242 lines (under 1500 cap)
- CPU forward pass, forward_logits, int6 quantization roundtrip all verified
- Int6 roundtrip loss gap: ~0.0001 on test model

## Files

- `train_gpt.py` — full training + eval script
- `submission.json` — leaderboard metadata (to be updated post-run)
- `README.md` — this file
