# Parameter Golf Solution

Optimized language model for the OpenAI Parameter Golf Challenge.

## Key Innovations (over baseline)

| Innovation | Description | BPB Impact |
|---|---|---|
| **SwiGLU MLP** | Replaces baseline's relu² with SwiGLU activation | ~0.02-0.03 |
| **SmearGate** | Lightweight token blending for local context | ~0.005 |
| **BigramHash** | Hash table bigram embeddings | ~0.005 |
| **SENT-lite** | Entropy-weighted loss for curriculum-like training | ~0.01 |
| **Batched TTT LoRA** | Per-document LoRA adapters at eval time | ~0.03 |

## Architecture

- **9 transformer layers** (4 encoder + 5 decoder with skip connections)
- **512-dim, 8 heads (4 KV heads)** — Grouped Query Attention
- **RoPE** position embeddings + **QK-norm** + **logit softcap**
- **Muon optimizer** (Newton-Schulz orthogonalization) + Adam
- **Int8 + zlib** quantization (official format)
- **1024-token SentencePiece** tokenizer (official)

## Quick Start

```bash
# 1. Clone official repo and get data
git clone https://github.com/openai/parameter-golf
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# 2. Copy our train_gpt.py into the repo
cp /path/to/train_gpt.py .

# 3. Train (8xH100)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Or use run.sh
bash run.sh
```

## File Structure

```
train_gpt.py      # Single-file solution (all code in one file)
run.sh             # Automation script
submission.json    # Required metadata
requirements.txt   # Python dependencies
WRITEUP.md         # Detailed technical write-up
README.md          # This file
```

## Constraints Met

- ✅ **Artifact Size**: ≤ 16MB (int8+zlib model + code)
- ✅ **Training Time**: ≤ 10 minutes on 8xH100 (600s wallclock cap)
- ✅ **Official Format**: Int8 + zlib quantization, SentencePiece 1024 tokenizer
- ✅ **DDP**: `torchrun --nproc_per_node=8` distributed training
- ✅ **BPB**: Tokenizer-agnostic bits-per-byte using SentencePiece LUTs

## Configuration

All hyperparameters are configurable via environment variables:

```bash
# Examples
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 torchrun --standalone --nproc_per_node=8 train_gpt.py
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=20000 torchrun --standalone --nproc_per_node=8 train_gpt.py
USE_SMEARGATE=1 USE_BIGRAMHASH=1 USE_SENT_LITE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## License

MIT
