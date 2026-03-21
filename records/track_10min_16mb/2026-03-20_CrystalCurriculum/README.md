# Crystal Curriculum — TF-IDF Data Distillation for Parameter Golf

**val_bpb: TBD** (pending H100 run)

## Key Technique: Crystallizer

Instead of training on random FineWeb sequences, we add a lightweight **TF-IDF curriculum learning** layer inspired by the [Kona RS Crystallizer](https://github.com/beebytez/kona-rs) — a BitTorrent-style information retrieval system.

### How It Works

1. **Crystallizer** maintains running unigram statistics (document frequency) across training
2. Each training step, the data loader **oversamples 4 candidate batches** from the token stream
3. Each candidate is scored by **TF-IDF information density** — sequences with rare, diverse tokens score higher
4. The **densest candidate** is selected for training
5. Curriculum **decays to uniform** at 70% of training time (warmdown phase uses random data)

**Cost**: ~2ms per batch scoring (vs ~57ms/step training) — negligible overhead.

**Why it works**: Small models have limited capacity. By prioritizing information-dense sequences — those with diverse vocabulary, unusual constructions, technical content — the model learns more per step. This is the same principle behind curriculum learning and data pruning, implemented as an inline TF-IDF scorer.

## Architecture Changes (vs Baseline)

| Change | Baseline | Ours |
|--------|----------|------|
| Layers | 9 | 10 |
| Muon weight decay | 0 | 0.02 |
| Data selection | Sequential | TF-IDF oversampled (4x) |
| Curriculum schedule | N/A | Active 70%, decay to uniform |

## Lineage

The Crystallizer design comes from Kona RS, a Rust-based search engine that uses TF-IDF embeddings for information retrieval. The 5-pass distillation process from that system inspired the idea of scoring training data by information density. This is the first application of that technique to language model pre-training.

## How to Run

```bash
# Single GPU (testing)
RUN_ID=crystal_test \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=10 \
CRYSTAL_OVERSAMPLE=4 \
CRYSTAL_WARMUP_FRAC=0.7 \
MUON_WEIGHT_DECAY=0.02 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (submission)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Environment Variables

| Env Var | Default | Description |
|---------|---------|-------------|
| `CRYSTAL_OVERSAMPLE` | 4 | Number of candidate batches to score per step |
| `CRYSTAL_WARMUP_FRAC` | 0.7 | Fraction of training with curriculum active |
| `MUON_WEIGHT_DECAY` | 0.02 | Decoupled weight decay for Muon optimizer |
| `NUM_LAYERS` | 10 | Transformer layers (up from baseline 9) |
