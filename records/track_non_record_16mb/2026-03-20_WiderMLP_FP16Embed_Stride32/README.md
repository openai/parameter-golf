## FP16 Tied Embedding + MLP_MULT=3 (Non-Record Submission)

10-layer GPT with MLP_MULT=3 (wider feedforward), fp16 tied embedding export, mixed int6/int8 quantization on layers 2–8, and stride-32 sliding window evaluation. Trained on 8xH100 SXM for 600s (8566 steps).

**Note:** This is a non-record submission. The artifact is 18.97MB (over the 16MB cap) and the evaluation took ~17 minutes (over the 10-minute eval limit). The val_bpb=1.1601 is reported for reference.

### Results

| Metric | Value |
|--------|-------|
| `val_bpb` (post-quantization, sliding window stride=32) | **1.1601** |
| Artifact size (int8+zlib compressed model + code) | 18,970,656 bytes |
| Training steps | 8566 / 20000 (wallclock cap hit) |
| Training time | 599,932ms |
| Eval time | 1,020,868ms (~17 min) |

### What Worked / What Exceeded Limits

**fp16 tied embedding** — The tied embedding matrix is serialized as fp16 passthrough instead of int8. This eliminates the quantization gap for the most precision-sensitive tensor at a cost of ~1MB.

**MLP_MULT=3** — Wider feedforward (hidden=1536 vs 1024) added model capacity and improved bpb but pushed the artifact to 18.97MB, over the 16MB cap.

**Stride-32 sliding window** — Each token evaluated with up to 4064 tokens of preceding context (vs 4032 at stride=64). This gave better bpb but increased eval time to 17 minutes, over the 10-minute eval limit.

**Mixed int6/int8 quantization** — Layers 2–8 use int6 (step=4), layers 0/1/9 use int8 per-row. Chosen to maximize zlib compression on middle layers.

### Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied embeddings: `TIE_EMBEDDINGS=1` with fp16 passthrough for `tok_emb.weight`
- LRs: `MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03`
- Muon: `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500`
- Batching: `TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=4096`
- Warmdown: `WARMDOWN_ITERS=3000`
- Eval: `EVAL_STRIDE=32 EVAL_BATCH_SEQS=64`
- Quant: `INT4_LAYERS=2,3,4,5,6,7,8 INT4_STEP=4`
- Hardware: 8×H100 80GB HBM3 (RunPod), PyTorch 2.4.1+cu124

### Training Curve

| Step | val_bpb | train_time |
|------|---------|------------|
| 0 | 4.1087 | 0ms |
| 1000 | 1.3437 | 78,087ms |
| 2000 | 1.2848 | 161,519ms |
| 3000 | 1.2474 | 228,485ms |
| 4000 | 1.2279 | 295,455ms |
| 5000 | 1.2155 | 362,227ms |
| 6000 | 1.2016 | 428,927ms |
| 7000 | 1.1865 | 495,544ms |
| 8000 | 1.1733 | 562,262ms |
| 8566 (final) | 1.1679 | 599,932ms |
| **Post-quant roundtrip (stride-32)** | **1.1601** | — |

### Run Command (8xH100)

```bash
TOKENIZER_PATH=$(find ~/.cache/huggingface -name 'fineweb_1024_bpe.model' | head -1) \
torchrun --nproc_per_node=8 train_gpt.py
```
