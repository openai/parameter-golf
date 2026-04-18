# SP8192 + 11L MLP4x + Partial RoPE (16/64) + GPTQ SDClip + SGD TTT

**val_bpb: 1.0820** (3-seed mean, std 0.00085) | **Size: 15.86MB** | **Hardware: 8xH100 SXM** | **Train: 600s** | **Eval: ~370s**

## 3-Seed Results

| Seed | Steps | ms/step | Raw BPB | Int8 BPB | TTT BPB | Artifact |
|------|-------|---------|---------|----------|---------|----------|
| 42   | 6666  | 90.01   | 1.1206  | 1.1214   | 1.0820  | 15.78MB  |
| 314  | 6659  | 90.11   | 1.1212  | 1.1221   | 1.0829  | 15.86MB  |
| 999  | 6667  | 90.00   | 1.1196  | 1.1205   | 1.0812  | 15.85MB  |
| **Mean** | | |  |  | **1.0820** | |

## Key Innovation: Partial RoPE

Rotate only the first 16 of 64 head dimensions with RoPE, leaving the remaining 48 dims unrotated. This separates positional encoding from content representation, giving the attention mechanism cleaner signals for both tasks.

- Controlled via `ROPE_DIMS=16` environment variable (default: full head_dim)
- Consistently improves val_bpb at every training checkpoint vs full RoPE baseline
- Gap widens during warmdown phase (steps 5000-7000): up to -0.02 BPB advantage
- Concept from PR #287 (jfprincz), applied here to the latest SOTA stack

## Architecture

- **Tokenizer**: SentencePiece BPE 8192 vocab
- **Model**: 11 layers, 512 dim, 8 heads (4 KV), 64 head dim, 4x MLP (2048 intermediate)
- **Attention**: GQA with QK-Gain 5.25, Partial RoPE (16/64 dims)
- **Parameters**: 35.9M (float32)
- **Tie embeddings**: Yes

## Training

- **Optimizer**: Muon (matrices) + Adam (scalars/embeddings)
- **Batch**: 524,288 tokens (8x grad accum on 1xGPU, 1x on 8xGPU)
- **Sequence length**: 2048
- **Warmup**: 20 steps (compile warmup, weights restored)
- **Warmdown**: wallclock-proportional
- **EMA**: decay 0.9965
- **Max wallclock**: 600s

## Quantization

- **GPTQ+SDClip** for attention and MLP weights (int6, k=12.85)
- **SDClip** for tok_emb (int8, k=15.0) — no GPTQ Hessian (dimension mismatch for embeddings)
- **Brotli-11** compression with byte shuffling
- Final artifact: ~15.8MB (under 16MB limit)

## Test-Time Training

- **SGD all-weights TTT** (lr=0.005, momentum=0.9)
- **Chunk size**: 2048, score cap: 2048
- **Docs**: 50,000 (sharded across 8 GPUs)
- **Eval time**: ~370s on 8xH100

## Run Command

```bash
SEED=42 ROPE_DIMS=16 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Comparison vs Previous Record (PR #1493)

| Metric | Previous (full RoPE) | This (Partial RoPE 16/64) | Delta |
|--------|---------------------|---------------------------|-------|
| 3-seed mean | 1.0825 | **1.0820** | -0.0005 |
| Best seed | 1.0813 | **1.0812** | -0.0001 |

## Included Files

- `README.md` — This file
- `submission.json` — Structured metadata
- `train_gpt.py` — Complete training script
- `train_seed42.log` — Seed 42 training log
- `train_seed314.log` — Seed 314 training log
- `train_seed999.log` — Seed 999 training log
