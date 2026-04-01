Wider SwiGLU model (dim=576) with Muon weight decay, FP16 embedding export, and sliding window evaluation. Architecture was discovered through 111 automated experiments on a single RTX 3090 before scaling to H100.

## what changed

**wider model (dim=576, 7 layers)**: increasing width from 512 to 576 gave a much larger improvement than adding layers. SwiGLU MLP with mult=2 keeps it under 16MB. 16.9M params, ~13.2MB artifact.

**Muon weight decay 0.02**: decoupled weight decay on Muon optimizer params. Improves both generalization (-0.002 BPB) and quantization robustness (smaller weights compress better, saving ~0.5MB artifact). No weight decay was used in the baseline.

**FP16 embedding passthrough**: keep tok_emb.weight in fp16 instead of int8. With tied embeddings, int8 errors compound through both the input lookup and output projection paths.

**sliding window evaluation (stride=64)**: instead of scoring non-overlapping 1024-token chunks, slide with stride=64 so every token gets ~960 tokens of context. This is purely an eval-time optimization. Eval completes in ~8 minutes on a single H100.

**wallclock-based warmdown at 60%**: with weight decay, longer warmdown (60% vs stock 10-15%) gives better convergence. Proven through ablation on 3090.

**other proven wins**: RoPE base 50K, beta2=0.99, batch 262K with LR 0.03.

## config

```
NUM_LAYERS=7  MODEL_DIM=576  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=2 (SwiGLU, hidden=768)  TIE_EMBEDDINGS=1
ROPE_BASE=50000  LOGIT_SOFTCAP=30  QK_GAIN_INIT=1.5
TRAIN_BATCH_TOKENS=262144  MATRIX_LR=0.03  SCALAR_LR=0.03
TIED_EMBED_LR=0.04  MUON_WEIGHT_DECAY=0.02  WARMDOWN_FRAC=0.6
BETA2=0.99  SEED=1337
```

## run command

```bash
RUN_ID=submission \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Note: ran on 4xH100 with grad_accum=2. On 8xH100 with grad_accum=1, expect ~22K steps and ~1.19 BPB.

## results

4xH100 SXM (RunPod secure cloud):

| seed | steps | ms/step | val_bpb (standard) | val_bpb (sliding window) | artifact |
|------|-------|---------|-------------------|--------------------------|----------|
| 1337 | 10,169 | 59.0 | 1.2441 | **1.2093** | 13.24MB |
| 42   | 10,293 | 58.4 | 1.2432 | **1.2086** | 13.23MB |
| 7    | 10,310 | 58.2 | 1.2439 | **1.2092** | 13.23MB |
| **mean** | | | 1.2437 | **1.2091** | |

Standard eval quant degradation: ~0.001 BPB (FP16 embed helps).
Sliding window improvement: ~0.035 BPB over standard eval.

## methodology

Architecture was found through 111 automated experiments on a single RTX 3090 (5-minute training budget each). Key findings:
- Width > depth: dim=576 with 7 layers beat dim=512 with 9 layers
- Weight decay is free improvement + smaller artifacts
- SwiGLU beats ReLU^2 by ~0.004 BPB at same param count
- RoPE base 50K beats 10K by ~0.001 BPB
- Warmdown fraction 0.6 optimal with weight decay (0.4 without)
