# First JEPA Submission: 1.4447 BPB — Joint Embedding Predictive Architecture for LLMs

LLM-JEPA Auxiliary Loss + Random Token Masking Views + Cosine Similarity + Asymmetric Stop-Gradient + 11L Transformer + int6 GPTQ + sliding window eval

**val_bpb: 1.4447 (seed=42)** | 10.05 MB artifact | 8×H100 SXM, 555s training + 74s eval

## Results (seed=42, 8×H100 SXM)

| Metric | Value |
|--------|-------|
| Sliding BPB | 1.4447 |
| val_bpb | 1.4447 |
| RT bpb | 1.4447 |
| Steps | 1,790 |
| ms/step | 310.09 |
| Training time | 555s |
| Artifact | 10,054,691 bytes (10.05 MB) |
| Parameters | 26,993,788 (same as baseline) |

## Method

Training loss: **L = L_NTP + λ × L_JEPA**

Following LLM-JEPA (Balestriero et al., arXiv:2509.14252), we add a JEPA auxiliary loss that trains the model to predict invariant embeddings across different corruptions of the same sequence.

### View Creation
- **View 1**: input sequence with 15% of tokens replaced by random tokens (mask A)
- **View 2**: same sequence with 15% of tokens replaced by random tokens (mask B)
- Different random masks create different "views" of the same underlying content

### JEPA Loss
1. Each view processed through the full model independently
2. Mean-pooled hidden states serve as sequence embeddings
3. `L_JEPA = 1 - cosine_similarity(emb_view1, emb_view2.detach())`
4. `detach()` on target creates asymmetric objective (prevents collapse without negative pairs)

### Architecture
- Standard 11L causal Transformer (unchanged — JEPA is a training objective, not architecture)
- λ = 1.0 (JEPA loss weight)
- 3 forward passes per step: 1 for NTP + 2 for JEPA views

## Key Techniques

| Technique | Impact | Notes |
|-----------|--------|-------|
| Random token masking views | Creates meaningful view pairs | 15% corruption rate per view |
| Mean pooling | Robust embedding | More stable than last-token for corrupted inputs |
| Stop-gradient on target | Prevents collapse | Asymmetric design from BYOL/I-JEPA |
| Cosine similarity | Scale-invariant | Matches LLM-JEPA paper's recommended metric |

## Why JEPA for Language Models

Standard NTP trains token-level predictions. JEPA adds a complementary embedding-space signal: the model must learn representations where semantically equivalent sequences (same content, different corruptions) map to similar embeddings. The LLM-JEPA paper shows 5-15% downstream improvement across Llama, Gemma, OpenELM, and OLMo families.

## Command

```bash
JEPA_ENABLED=1 \
JEPA_LAMBDA=1.0 \
SEED=42 \
python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] Artifact ≤16,000,000 bytes (10,054,691)
- [x] Training ≤600s on 8×H100 SXM (555s)
- [x] Eval ≤600s (74s)
- [x] No validation data during training
- [x] No network calls during evaluation
- [x] No external compute
- [x] Reproducible from `train_gpt.py`

## References

- LLM-JEPA: [arXiv:2509.14252](https://arxiv.org/abs/2509.14252) (Balestriero et al., 2025)
- I-JEPA: [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) (Assran et al., 2023)

## Included Files

- `train_gpt.py` — full training script
- `train_seed42.txt` — training log
- `submission.json` — metadata
