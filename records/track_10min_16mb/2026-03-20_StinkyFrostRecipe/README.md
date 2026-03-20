# The Stinky Frost Recipe

## Score: val_bpb = 1.1725

Trained on 8xH100 SXM in 600 seconds. 15.58MB artifact (int6+zlib).

## Techniques

### 1. Int6 Quantization with Early QAT (25%)
Per-row int6 quantization on all weight matrices except embeddings. QAT with STE activates at 25% of training (~step 5000), giving ~6000 steps of quantization-aware training. Weights learn distributions that survive int6 rounding.

### 2. FP16 Tied Embeddings
Token embeddings kept in float16 instead of int6 during serialization. With 1024 vocab tokens, the embedding table is the model's entire interface to language — int6 quantization (64 discrete levels) destroys token distinguishability. FP16 preserves embedding quality at a cost of ~600KB, worth ~0.1 BPB.

### 3. MLP Hidden = 1344
Custom MLP hidden dimension (2.625× expansion) instead of standard 2× or 3×. Sized to maximize model capacity while fitting FP16 embeddings under the 16MB limit.

### 4. SmearGate
Learned per-dimension gate blending each token's embedding with the previous token's embedding. 512 parameters. Provides cheap bigram context before attention.

### 5. BigramHash Embedding
4096-bucket hash table (dim=128, projected to 512) mapping consecutive token pairs to learned embeddings. Gives the model direct bigram context before layer 0. ~590K parameters.

### 6. Orthogonal Weight Initialization
All linear layers ≥64×64 initialized with `orthogonal_(gain=1.0)`, except zero-init output projections. Accelerates early convergence and synergizes with Muon optimizer.

### 7. Muon Weight Decay
Decoupled weight decay (WD=0.01) on Muon optimizer for matrix parameters. Regularizes weight magnitudes for better int6 quantization.

### 8. Sliding Window Eval (stride=64)
Evaluation with overlapping windows at stride=64. Each scored token has nearly full context, eliminating cold-start degradation.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 9 |
| model_dim | 512 |
| num_heads | 8 |
| num_kv_heads | 4 |
| mlp_hidden | 1344 |
| tie_embeddings | True |
| quant_bits | 6 |
| qat_start_frac | 0.25 |
| eval_stride | 64 |
| muon_wd | 0.01 |
| fp16_embed | True |
| smear_gate | True |
| bigram_buckets | 4096 |
| bigram_dim | 128 |
| ortho_init | True |
| train_seq_len | 1024 |
| train_batch_tokens | 524288 |
| iterations | 20000 (wallclock-capped) |
| max_wallclock_seconds | 600 |

## Key Metrics

| Metric | Value |
|--------|-------|
| Steps completed | 11134 |
| Pre-quant val_bpb | 1.2022 |
| Int6+zlib quant val_bpb | **1.1725** |
| Artifact size | 15,575,922 bytes |
| Peak memory | 12,565 MiB |
| Step avg | 53.87 ms |

## Reproduction

```bash
QUANT_BITS=6 QAT_START_FRAC=0.25 EVAL_STRIDE=64 MUON_WD=0.01 FP16_EMBED=1 SMEAR_GATE=1 BIGRAM_HASH=1 ORTHO_INIT=1 MLP_HIDDEN=1344 RUN_ID=stinky_frost NCCL_IB_DISABLE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
