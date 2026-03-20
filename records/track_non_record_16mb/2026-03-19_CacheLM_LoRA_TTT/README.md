# Cache LM + LoRA TTT

Non-record submission exploring two eval-time techniques with zero artifact cost.

## Method

**Training** uses v8192 7-layer GQA + MTP (k=2, alpha=0.3), tied embeddings, 2000 steps on 1xA100 (no torch.compile). Standard architecture from the baseline, just with larger vocabulary.

**Evaluation** combines two techniques:

### 1. LoRA Test-Time Training (adapted from PR #77)

Per-document LoRA adaptation during eval:
- Find document boundaries via BOS tokens, reset all LoRA params between documents
- Rank-8 LoRA on `lm_head`, `c_q`, `c_v` in all transformer blocks
- Split each document into overlapping 256-token chunks within 1024-token context windows
- For each chunk: **score first** (accumulate BPB), **then** one Adam step (lr=0.01, betas=0.9/0.95) on LoRA params
- Batch 64 documents, sorted by length for efficiency

### 2. Unigram Cache LM (novel contribution, Grave et al. 2017)

Per-document token frequency cache interpolated with model output:
- Maintain decayed unigram counts of tokens seen so far in each document
- At each position: `p_final = (1-lambda) * p_model + lambda * p_cache`
- Loss computed as `-log(p_final(target))` using `torch.logaddexp` for numerical stability
- Cache resets between documents (no cross-document leakage)
- Exploits web text burstiness: repeated entities, jargon, URLs get probability boost
- Default: lambda=0.02, decay=0.98

The cache LM is the novel contribution. While LoRA TTT (PR #77) and sliding window eval are established techniques, no prior submission applies cache language model interpolation. Cache LMs have shown 10-30% perplexity improvements on web text (Grave et al. 2017), and the technique is orthogonal to both LoRA TTT and model architecture improvements.

## Results

Evaluated on `dyneval_base` model (v8192 7L MTP, 2000 steps on 1xA100):

| Eval Mode | val_bpb | Delta vs baseline |
|-----------|---------|-------------------|
| Standard eval | 1.2756 | -- |
| Sliding window (stride=128) | 1.2562 | -0.019 |
| + LoRA TTT | 1.2529 | -0.003 additional |
| + LoRA TTT + Cache LM (λ=0.02) | 1.2544 | +0.002 (cache hurts) |

### Key finding: Cache LM is a negative result on FineWeb

The unigram cache with λ=0.02 **hurts** BPB by +0.0015 on top of LoRA TTT. This makes sense: FineWeb is diverse web text where document-level token repetition is low compared to WikiText/PTB where cache LMs were originally validated (Grave et al. 2017 showed 12-31% perplexity reduction on those datasets). Web documents are shorter and more varied — the cache doesn't accumulate enough signal before the document ends.

LoRA TTT alone gives a clean -0.003 BPB improvement, matching PR #77's ablation exactly.

Note: trained at 1/10th budget (2000 steps on 1xA100 vs 20K steps on 8xH100). With full training budget, expect significantly better base model quality.

## Env vars

```
LORA_TTT=1              # Enable LoRA TTT
LORA_TTT_RANK=8         # LoRA rank
LORA_TTT_LR=0.01        # Adam learning rate
LORA_TTT_CHUNK_SIZE=256  # Chunk size for strided eval
LORA_TTT_BATCH_SIZE=64   # Documents per batch
CACHE_LM=1              # Enable cache LM
CACHE_LM_LAMBDA=0.02    # Interpolation weight
CACHE_LM_DECAY=0.98     # Exponential decay on cache counts
```

## Command

```bash
# Training (1xA100, no compile):
VOCAB_SIZE=8192 NUM_LAYERS=7 MTP_K=2 MTP_ALPHA=0.3 \
ITERATIONS=2000 TRAIN_ONLY=1 TORCHDYNAMO_DISABLE=1 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Eval:
EVAL_SEQ_LEN=1024 EVAL_STRIDE=128 LORA_TTT=1 CACHE_LM=1 \
python3 eval_only.py
```

## Included files

- `train_gpt.py` — training + eval script (1500 lines)
- `train_v0.txt` — training log (2000 steps, 1xA100)
- `submission.json`
