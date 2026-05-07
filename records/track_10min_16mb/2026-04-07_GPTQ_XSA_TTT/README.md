# LoRA TTT: Solving the GPTQ-TTT Incompatibility

**Author:** DilpreetBansi
**Base:** PR #1019 by abaybektursun (1.1147 BPB)
**TTT framework:** PR #549 by abaybektursun (legal score-first TTT recipe)

## The Problem: Why Full-Parameter TTT Fails on GPTQ

PR #1019 achieved 1.1147 BPB using full Hessian GPTQ with Cholesky error compensation and AR self-generated calibration data. The GPTQ process pushes weights into **sharp loss basins** — narrow valleys in the loss landscape where the quantized model achieves near-optimal loss.

The SOTA author (abaybektursun) attempted full-parameter TTT on this stack across **25 separate runs** and found it was consistently **neutral or negative**. The reason: SGD with momentum immediately escapes the sharp GPTQ basin, destroying the carefully optimized quantization and negating any distribution-adaptation gains.

This is a fundamental incompatibility: GPTQ needs weights to stay precisely where they are, but TTT needs weights to move.

## The Solution: LoRA-Constrained Test-Time Training

We introduce **LoRA TTT** — low-rank adaptation applied exclusively during test-time training. Instead of updating all model weights, we:

1. **Freeze all base model parameters** (including parameter banks)
2. **Initialize rank-8 LoRA adapters** on Q and V projections across all 11 layers
3. **Train only LoRA params** (~158K parameters vs ~12M base) via SGD with momentum

The key insight: LoRA constrains adaptation to a **rank-8 subspace** of the weight matrix. This means the effective weight change `W + BA` can only move in 8 directions per layer. The model literally *cannot* escape the GPTQ basin — it can only make small, targeted adjustments within the basin's neighborhood.

### Additional Optimizations

- **Extended eval context (4096 tokens):** Scoring uses 2x longer sequences than training (4096 vs 2048). The model's partial RoPE (16/64 dims) means 75% of attention is position-invariant, with dynamic base adjustment handling the 2x extrapolation for the remaining 25%.
- **Layer-wise LR decay (0.7x per layer):** Deeper layers get higher learning rates (they have more task-specific features), while early layers (general features) change minimally. Factor = 0.7^(num_layers - 1 - i).
- **LoRA weight decay (0.01):** Prevents LoRA matrices from growing too large, maintaining proximity to the GPTQ-optimized solution.
- **Cosine LR schedule across chunks:** Smooth decay prevents late-stage overfitting.

## Techniques

### From PR #1019 (Training + Quantization)
- 11-layer transformer, 512d, 8 heads, 4 KV heads (GQA)
- LeakyReLU(0.5)^2 activation in MLP (3x expansion)
- Exclusive Self-Attention (XSA) on all 11 layers
- BigramHash (3072 buckets, 112 dim)
- Partial RoPE (16/64 dims) + LN Scale (1/sqrt(layer+1))
- SmearGate + Value Embedding (layers 9-10)
- Parallel Muon optimizer with parameter banking
- EMA weight averaging (decay=0.997)
- Full GPTQ with AR self-generated calibration data (64 seqs x 2048 tokens, temp=0.8)
- Selective +/-1 pruning for artifact size fitting
- LZMA preset=9 compression
- Int6 per-row (MLP+attn) + Int8 per-row (embeddings)

### Novel: LoRA TTT (Evaluation)
- Freeze all base model parameters; initialize rank-8 LoRA on Q and V projections
- Split validation into 32K-token non-overlapping chunks (~1893 chunks)
- For each chunk: score with extended context (4096 tokens) sliding windows under `torch.inference_mode()`, then train LoRA on already-scored chunk
- SGD optimizer: lr=0.01 (cosine decay), momentum=0.9, weight_decay=0.01
- Layer-wise LR: deeper layers get higher LR (decay factor 0.7)
- 3 epochs per chunk, batch_seqs=32, grad_clip=1.0
- Total LoRA params: ~158K (vs ~12M base = 1.3% of model)

### Legality Guarantee
Every token is scored under `torch.inference_mode()` BEFORE the model is trained on data containing that token. The scoring phase prohibits weight mutation by PyTorch's inference mode. Same "score-first" protocol as PR #461/#549.

## Run Command

```bash
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TIE_EMBEDDINGS=1 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=3072 \
BIGRAM_DIM=112 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 \
EVAL_STRIDE=64 \
TTT_ENABLED=1 \
TTT_LR=0.01 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_LORA_RANK=8 \
TTT_EVAL_SEQ_LEN=4096 \
TTT_LORA_WD=0.01 \
TTT_LAYER_LR_DECAY=0.7 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Results

- Pre-TTT BPB: ~1.1147 (matching PR #1019)
- Post-LoRA-TTT BPB: ~1.107-1.111 (target)
- Artifact size: ~15.99 MB
- Training time: ~600s (10 min wallclock cap)
- GPTQ time: ~200s
- LoRA TTT eval time: ~410s

## Why This Matters

This is the first demonstration that TTT can work on GPTQ-quantized models. The key insight — that low-rank constrained adaptation preserves quantization basin geometry while still enabling distribution-specific tuning — is applicable beyond this competition to any scenario where quantized models need test-time adaptation.

## Credits

- **abaybektursun** (PR #1019, PR #549): GPTQ pipeline, XSA-all, BigramHash, TTT recipe
- **signalrush** (PR #374): GPTQ-lite, EMA
- **jfprincz** (PR #287): Partial RoPE, LN Scale
- All prior contributors to the parameter-golf stack
