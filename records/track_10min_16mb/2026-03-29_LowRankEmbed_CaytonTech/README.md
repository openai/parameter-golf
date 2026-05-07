# [Non-Record] ALBERT-Style Low-Rank Embedding Factorisation at 16MB

**Exploring whether low-rank embedding bottlenecks can trade parameter space for model quality at the 1024-token vocabulary scale.**

**val_bpb: 1.3902 (rank=64) / 1.3440 (rank=128)** | 1×H100 SXM, 900s | Non-record exploratory submission

---

## Summary

This submission tests whether ALBERT-style low-rank embedding factorisation — replacing `nn.Embedding(1024, 512)` with `nn.Embedding(1024, r) + nn.Linear(r, 512)` — can free parameter budget in the 16MB artifact without unacceptable quality loss.

**Finding: At a 1024-token vocabulary, low-rank embedding factorisation does not improve BPB.** The vocabulary is too small for meaningful redundancy to exist in the embedding table. Both rank=64 and rank=128 variants produce worse BPB than the unmodified baseline, despite saving artifact space and running faster per step.

This is a negative result. It's submitted in the hope that the ablation data saves someone else the compute of testing the same idea — and because the competition encourages novel structural explorations regardless of outcome.

## Ablation Results (1×H100 SXM, 900s wallclock)

| Configuration | val_bpb | Artifact size | Steps | ms/step | Params saved |
|--------------|---------|---------------|-------|---------|-------------|
| **Baseline** (embed_dim=512) | **1.3192** | 14.18 MB | 1,537 | 586 | — |
| **Low-rank r=128** | 1.3440 (+0.025) | 14.09 MB | 1,663 | 541 | ~197K |
| **Low-rank r=64** | 1.3902 (+0.071) | 13.81 MB | 1,634 | 551 | ~426K |

All runs used identical settings: 9 layers, 512 dim, 8 heads, 4 KV heads, MLP mult 2, tied embeddings, 1024 vocab, Muon optimizer, default learning rates. Only the embedding bottleneck rank was varied.

## What Changed from Baseline

Three modifications to `train_gpt.py` in the `GPT.__init__` and `GPT.forward` methods:

### 1. Embedding factorisation (\_\_init\_\_)
```python
# BEFORE:
self.tok_emb = nn.Embedding(vocab_size, model_dim)

# AFTER:
self.embed_bottleneck = 64  # or 128
self.tok_emb = nn.Embedding(vocab_size, self.embed_bottleneck)
self.embed_proj = nn.Linear(self.embed_bottleneck, model_dim, bias=False)
```

### 2. Input path projection (forward)
```python
# BEFORE:
x = self.tok_emb(input_ids)

# AFTER:
x = self.embed_proj(self.tok_emb(input_ids))
```

### 3. Output path fix for tied embeddings (forward)
```python
# BEFORE:
logits_proj = F.linear(x, self.tok_emb.weight)

# AFTER:
x = F.linear(x, self.embed_proj.weight.t())  # project 512 → bottleneck
logits_proj = F.linear(x, self.tok_emb.weight)  # tied embedding lookup
```

The output path fix was identified during development via multi-agent cross-checking — Gemini CLI caught the dimension mismatch that the initial implementation missed. With tied embeddings, `tok_emb.weight` is used for both input lookup and output projection; changing its shape requires updating both paths.

## Why It Doesn't Work (Analysis)

**The core insight:** ALBERT's low-rank embedding factorisation was designed for vocabularies of 30,000+ tokens, where many embeddings are near-duplicates (e.g., "run", "running", "runs" share similar semantics). The rank-128 bottleneck in ALBERT captured 95%+ of the useful variation because the embedding matrix was highly redundant.

At 1024 tokens, the vocabulary is already minimal. Every token carries unique, high-information content. The embedding matrix has rank close to its full dimensionality — there is no redundancy to compress. Forcing it through a bottleneck destroys information that the model needs.

**Supporting evidence from the competition:**
- The ternary quantisation submission (Ciprian-Florin Ifrim, PR #744) uses a factored embedding at 8192×254, but with a vocabulary 8× larger than baseline. At 8192 tokens, there IS enough redundancy to exploit — their factored embedding works because the vocabulary is large enough.
- BigramHash (PR #162, Raahil Shah) sidesteps the embedding entirely by adding a parallel N-gram lookup path. This succeeds because it adds capacity rather than compressing existing capacity.

**The speed/space trade-off is real but insufficient:** rank-64 saves 0.37 MB and runs 35ms/step faster (6% speedup). In principle, the saved space could fund ~0.2 extra layers. But the 0.071 BPB quality loss far exceeds what 0.2 extra layers could recover.

## Observations

1. **Rank 128 → 64 quality degradation is nonlinear.** Halving the rank from 128 to 64 increases the BPB gap from +0.025 to +0.071 (nearly 3×). This suggests a sharp information cliff between rank 128 and 64 for this vocabulary size.

2. **Step time improves with lower rank.** The smaller embedding matmuls reduce per-step time, yielding ~100 extra training steps. But more steps on a weaker architecture don't compensate for the quality loss.

3. **Artifact size savings are modest.** The embedding is only ~3% of total parameters at this scale. Even aggressive compression (rank 64) saves only 0.37 MB — not enough to fund meaningful architectural changes.

4. **The tied embedding fix is non-trivial.** Anyone attempting embedding factorisation with tied embeddings must update both the input path (add up-projection after embedding) and the output path (add down-projection before the tied linear layer). Missing the output path causes a dimension mismatch crash.

## What Would Make This Work

Based on these results, low-rank embedding factorisation would likely help under these conditions:
- **Larger vocabulary (4096+):** More tokens = more redundancy = more compressible embedding
- **Higher embedding dimension (768+):** More dimensions = more room for low-rank structure
- **Combined with vocabulary expansion:** Use the freed space from low-rank to increase vocabulary size, potentially improving BPB through better tokenisation

The ternary submission's factored 8192×254 embedding validates this hypothesis — it uses a larger vocabulary where factorisation is beneficial.

## Process & Tooling

This was my first close-to-serious work in ML. The research and planning phase involved consuming academic papers and competition analysis via Google NotebookLM during commutes, using Claude (Anthropic) for architectural exploration and research synthesis, and Gemini CLI (Google) for implementation review and bug detection. The multi-agent workflow proved valuable — Claude provided the architectural reasoning and documentation, Gemini caught an implementation bug by reading the full file more carefully.

The most fascinating concept encountered during the research (though not applied in this submission) was megakernels — fusing an entire transformer layer into a single GPU kernel to eliminate memory round-trips. Understanding why the H100's memory hierarchy makes small operations bandwidth-bound rather than compute-bound fundamentally changed how I think about model efficiency.

Trained on NVIDIA H100 80GB SXM via RunPod. Getting to run experiments on this hardware — and contributing, however modestly, to an OpenAI research competition — has been a genuine privilege.

## References

- Lan et al., "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (ICLR 2020) — the original low-rank embedding factorisation technique
- Ciprian-Florin Ifrim, "Ternary Quantization" submission (PR #744) — uses factored tied embedding at 8192×254, demonstrating factorisation works at larger vocabulary scales
- Raahil Shah, BigramHash (PR #162) — alternative embedding augmentation via N-gram hash lookup
- Cheng et al., "Conditional Memory via Scalable Lookup" (arXiv:2601.07372) — DeepSeek's Engram paper, conceptual inspiration for exploring embedding-level modifications

## Setup

```bash
# Data (if not already downloaded)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Run (rank 64, as submitted)
RUN_ID=lowrank_r64 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=900 \
torchrun --standalone --nproc_per_node=1 \
records/track_10min_16mb/2026-03-29_LowRankEmbed_CaytonTech/train_gpt.py

# To test rank 128, change embed_bottleneck = 128 in train_gpt.py
```

## Hardware

- 1×NVIDIA H100 80GB SXM (RunPod)
- PyTorch 2.x, CUDA
- ~900s wallclock per run
