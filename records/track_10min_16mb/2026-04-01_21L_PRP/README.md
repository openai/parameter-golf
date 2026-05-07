# 21L + PRP SwiGLU + Shared Attention + sp8192 + 2048 context length

While this run demonstrates end-to-end results, the current int8+zlib artifact (17.7 MB) exceeds the 16 MB track limit. Furthermore, due to local hardware constraints (Nvidia RTX 3080), the model processed only 2.5–5% of the total target tokens required for a full convergence run (defined as a <10-minute training window on an 8xH100 cluster)

This variant moves away from the default 1k-vocab, 1024-context, 9-layer starter and spends the budget on a deeper 8k-vocab model with parameter sharing, PRP-based MLPs, and a longer-horizon LR schedule.

### Changes from Baseline

**1. 8K tokenizer + longer context**

The script switches from `fineweb10B_sp1024` to `fineweb10B_sp8192`, uses the 8k SentencePiece tokenizer, and doubles context length to `TRAIN_SEQ_LEN=2048`. The default model shape becomes 21 layers at width 512 with 8 query heads and 2 KV heads.

**2. PRP SwiGLU MLP instead of the baseline relu^2 MLP**

The dense MLP is replaced with a SwiGLU block built from 'ParametrizedRandomProjection' (https://arxiv.org/pdf/2512.13480). Each PRP layer keeps a fixed random projection buffer and only trains low-dimensional controls: input scaling, output scaling, output bias, and a rank-32 LoRA update. The MLP projects once to `2 * hidden`, then splits into gate and value halves. A higher MLP multiplier is used: 4 instead of 2.

**3. Pairwise shared attention in the transformer body**

Layer 1 and the final layer remain unique. Interior layers share attention modules in adjacent pairs: `(2,3)`, `(4,5)`, `(6,7)`, and so on. Each block still keeps its own norms, residual mixing, scales, and MLP, so depth increases without paying for 21 fully independent attention stacks.

In the logged run, that produces 21 transformer blocks but only 12 unique attention modules.

**4. Separate optimizer path for PRP vector controls**

The script adds `unique_named_parameters` and splits PRP vector parameters (`.alpha`, `.weight`, `.bias`) into their own Adam group with `PRP_LR`. Matrix-shaped parameters still use Muon, while the remaining scalar and control tensors stay on Adam.

**5. Three-phase cosine LR schedule**

The default warmdown schedule is replaced with a longer-run schedule:
- cosine warmup from `LR_INIT_SCALE` to 1.0
- cosine flash drop from 1.0 to `LR_MAIN_SCALE`
- powered cosine tail from `LR_MAIN_SCALE` to `LR_MIN_SCALE`, shaped by `LR_GAMMA`

This will be adjusted for the full schedule run.

**6. Shared-aware int8 export**

The post-training export remains per-row int8 for 2D tensors and fp16/fp32 passthrough for small or control tensors, but it now deduplicates repeated storage before counting payload bytes. That matters once attention modules or embeddings are shared or tied.

### Default Configuration in This Record

```bash
DATA_PATH=./data/datasets/fineweb10B_sp8192
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model

VOCAB_SIZE=8192
NUM_LAYERS=21
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=2
MLP_MULT=4
TIE_EMBEDDINGS=1

TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=65536
VAL_BATCH_SIZE=65536
VAL_FRAC=0.25

ITERATIONS=4000
WARMUP_STEPS=2
LR_MAIN_SCALE=0.5
LR_MIN_SCALE=0.05
LR_INIT_SCALE=0.0
LR_GAMMA=0.8

TIED_EMBED_LR=0.03
MATRIX_LR=0.03
SCALAR_LR=0.03
PRP_LR=0.06
MUON_MOMENTUM=0.95
```

Notes:
- `LR_WARMUP_STEPS` defaults to `int(0.01 * ITERATIONS)`.
- `LR_DROP_STEPS` defaults to `int(0.2 * ITERATIONS)`.
- The file lives under the 10-minute track, but the current defaults are closer to a longer research run than a finalized 10-minute submission.

### Actual Results

| Metric | Value |
|--------|-------|
| Final pre-quant val_bpb | 1.3527 |
| Final pre-quant val_loss | 3.4657 |
| int8+zlib roundtrip val_bpb | 1.3540 |
| int8+zlib roundtrip val_loss | 3.4690 |
| Quantization gap | +0.0013 BPB |
| Total params | 17,171,040 |
| Peak memory allocated | 7,329 MiB |
| Serialized model | 50,783,797 bytes |
| int8+zlib artifact | 17,603,010 bytes |
| Total submission size (artifact + code) | 17,658,504 bytes |


### Validation Trajectory

| Step | val_loss | val_bpb |
|------|----------|---------|
| 0 | 9.0102 | 3.5169 |
| 400 | 4.2170 | 1.6460 |
| 800 | 3.9027 | 1.5233 |
| 1200 | 3.7963 | 1.4818 |
| 1600 | 3.7088 | 1.4476 |
| 2000 | 3.6501 | 1.4247 |
| 2400 | 3.6022 | 1.4060 |
| 2800 | 3.5515 | 1.3862 |
| 3200 | 3.5092 | 1.3697 |
| 3600 | 3.4804 | 1.3585 |
| 4000 | 3.4657 | 1.3527 |

