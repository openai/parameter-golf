# Parameter Golf V1 — Tokenizer Overlay V3

A reproducible overlay for the OpenAI **Parameter Golf** challenge focused on a single question:

> Can a custom 1024-token vocabulary + boundary-aware dynamic programming tokenizer outperform SentencePiece on compression—and if so, why doesn’t that automatically translate into better training performance?

---

## 🔗 Quick Navigation

- [Thesis](#thesis)
- [Core Finding](#core-finding-of-the-project)
- [Technical Contribution](#technical-contribution)
- [Experiment Arc](#experiment-arc)
- [Repository Map](#repository-map)
- [Minimal Reproduction](#minimal-reproduction)
- [Tokenizer Discovery Process](#quick-snippet-how-the-custom-tokenizer-was-discovered)
- [Why V3 Was Frozen](#why-the-v3-tokenizer-was-frozen)
- [Next Steps](#where-the-work-should-go-next)

---

## Thesis

This project builds and validates a custom tokenizer pipeline on top of the OpenAI Parameter Golf baseline.

### Key Result

- **Compression win over SentencePiece (full validation)**
- **Does NOT translate to better final training score**

This separates:

- ✅ Representation (tokenizer)
- ❌ Optimization (training dynamics)

That distinction is the entire point of this repo.

---

## Core Finding of the Project

The tokenizer problem was solved *enough* to expose the real bottleneck.

### What worked

- Custom tokenizer reduces token count on full validation
- DP segmentation improves boundary handling
- Pipeline is reproducible end-to-end

### What didn’t

- Final BPB still worse than SentencePiece under time constraint
- Tokenizer gains ≠ training gains

---

## Technical Contribution

### 1. Frozen Vocabulary

📄 [`overlay/vocab_best_v3.jsonl`](overlay/vocab_best_v3.jsonl)

Includes:

- byte fallback tokens
- subwords
- phrase tokens
- space-aware tokens
- structural tokens

---

### 2. DP Tokenizer

Implemented in:

📄 [`overlay/export_v3_dataset.py`](overlay/export_v3_dataset.py)

Objective prioritizes:

- fewer tokens
- fewer fallback tokens
- fewer fallback runs
- cleaner boundaries

---

### 3. Full Validation Comparator

📄 [`overlay/compare_full_val_compression.py`](overlay/compare_full_val_compression.py)

This is the **ground truth layer**.

- compares SP vs custom across 50k docs
- outputs token deltas + diagnostics
- prevents false positives

---

### 4. Tokenizer-Aware Training

📄 [`overlay/train_v3_token_class_grad.py`](overlay/train_v3_token_class_grad.py)

Adds:

- custom tokenizer support
- token-class gradient scaling
- tokenizer-aware BPB calculation

---

## Experiment Arc

### Phase 1 — Baseline replication
- Reproduce OpenAI baseline exactly

### Phase 2 — Vocabulary construction
- n-grams → curation → structured tokens

### Phase 3 — DP segmentation
- replace greedy tokenizer

### Phase 4 — Full validation loop
- evaluate → diagnose → refine

### Phase 5 — Freeze (V3)
- diminishing returns reached
- tokenizer no longer bottleneck

---

## Repository Map

### Core Overlay

- [`overlay/export_v3_dataset.py`](overlay/export_v3_dataset.py)
- [`overlay/compare_full_val_compression.py`](overlay/compare_full_val_compression.py)
- [`overlay/train_v3_token_class_grad.py`](overlay/train_v3_token_class_grad.py)
- [`overlay/vocab_best_v3.jsonl`](overlay/vocab_best_v3.jsonl)

---

### Documentation

- [`docs/`](docs/)
- [`INDEX.md`](INDEX.md)
- [`PR_Submission.md`](PR_Submission.md)

---

### Supporting Material

- `artifacts/`
- `supplemental/`

---

## Minimal Reproduction

### 1. Clone baseline

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

---

### 2. Download dataset

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

---

### 3. Import overlay

```bash
git clone https://github.com/Stuckertks09/parameter-golf-v3.git
rsync -av --exclude='.git' parameter-golf-v3/ parameter-golf/
```

---

### 4. Export dataset

```bash
python3 overlay/export_v3_dataset.py   --docs-jsonl data/docs_selected.jsonl   --vocab-jsonl overlay/vocab_best_v3.jsonl   --output-dir data/datasets/fineweb10B_customdp1024_v3   --num-val-docs 50000   --workers 16   --batch-docs 512   --max-shards 81
```

---

### 5. Compare compression

```bash
python3 overlay/compare_full_val_compression.py   --docs-jsonl data/docs_selected.jsonl   --num-val-docs 50000   --sp-model data/tokenizers/fineweb_1024_bpe.model   --vocab-jsonl overlay/vocab_best_v3.jsonl   --output-dir analysis
```

---

### 6. Train baseline

```bash
DATA_PATH=data/datasets/fineweb10B_sp1024 TOKENIZER_KIND=sentencepiece TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=8 scripts/70_training/train_gpt.py
```

---

### 7. Train custom

```bash
DATA_PATH=data/datasets/fineweb10B_customdp1024_v3 TOKENIZER_KIND=custom_jsonl TOKENIZER_PATH=overlay/vocab_best_v3.jsonl VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=8 overlay/train_v3_token_class_grad.py
```

---

## Quick Snippet: How the Custom Tokenizer Was Discovered

1. Start with baseline
2. Build vocab (n-grams + structure)
3. Replace greedy → DP tokenizer
4. Evaluate full validation
5. Analyze worst docs
6. Iterate
7. Freeze when gains plateau

---

## Why the V3 Tokenizer Was Frozen

- Compression win achieved
- Further tweaks unstable
- Gains stopped transferring to training

➡️ Tokenizer no longer highest-leverage area

---

## Where the Work Should Go Next

- embedding initialization
- phrase token learning dynamics
- early-step optimization speed
- architecture tweaks for compressed tokens

---

## Bottom Line

This repo proves:

- tokenizer improvement ≠ model improvement
- compression alone does not win the challenge
- the real problem is now **learning efficiency**

