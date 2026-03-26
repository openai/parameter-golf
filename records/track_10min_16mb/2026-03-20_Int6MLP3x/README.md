# Int6 + MLP 3x + Sliding Window

## Score
**val_bpb = 1.1574** (sliding window eval, stride=256)

## Approach

Int6 post-training quantization compresses weight matrices to 6-bit precision (±31 range with per-row scaling), freeing enough artifact space to triple the MLP hidden dimension from 1024 to 1536. The result: 21.8M effective parameters in a 15.98MB artifact.

### Quantization Strategy
- **Weight matrices**: int6 per-row quantization (6-bit range, stored as int8 bytes)
- **Tied embedding** (`tok_emb.weight`): kept in fp16 — this tensor serves as both input lookup and output projection, making it disproportionately sensitive to quantization
- **Late-K passthrough**: last 2 layers' `c_k.weight` kept in fp16 — late-layer key projections need precision for fine-grained token discrimination
- **Control tensors** (scales, mixes, gains): fp16

### Architecture
- 9 layers, dim=512, 8 heads, 4 KV heads
- **MLP hidden=1536** (3x expansion, up from default 1024)
- ReLU-squared activation (unchanged)

### Training
- `TRAIN_SEQ_LEN=2048` with `TRAIN_BATCH_TOKENS=786432`
- `MATRIX_LR=0.02`, `MUON_MOMENTUM=0.99`, `GRAD_CLIP_NORM=0.3`
- `WARMDOWN_ITERS=3000`

### Evaluation
- Sliding window with `EVAL_STRIDE=256` at `EVAL_SEQ_LEN=2048`
- Every scored token sees 1792 context tokens
- Eval time: 240 seconds on 8xH100

### Results

| Eval Method | val_bpb |
|------------|---------|
| Non-overlapping | 1.1792 |
| Sliding stride=256 | **1.1574** |

**15.98MB** artifact. **600s** training + **240s** eval on 8xH100 SXM.

### Multi-Seed Validation
| Seed | val_bpb | val_loss |
|------|---------|----------|
| 1337 | 1.1574 | 1.9543 |
| 1338 | 1.1576 | 1.9546 |
| 1339 | 1.1576 | 1.9546 |
| **Mean** | **1.1575** | **1.9545** |
| Std | 0.0001 | 0.0002 |

Improvement over baseline: 0.118 nats (p << 0.01).

## Reproduction

```bash
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 MATRIX_LR=0.02 SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
MLP_HIDDEN=1536 EVAL_SEQ_LEN=2048 EVAL_STRIDE=256 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

8xH100 SXM (RunPod, 83ms/step with MLP 3x).
