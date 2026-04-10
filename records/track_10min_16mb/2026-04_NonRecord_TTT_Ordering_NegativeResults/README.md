# Non-Record: TTT Chunk Ordering Does Not Improve BPB

**val_bpb: 1.11961** (non-record)
**Author:** Joel Pfeiffer (@jpfeiffe)
**Base:** Fork of PR #549 (LeakyReLU(0.5)², BigramHash 1536, legal score-first TTT)

## Question

Does reordering validation chunks by document similarity improve test-time training (TTT) score-first BPB?

## Answer

**No.** With correct #549-style TTT semantics (all-GPU-per-chunk, per-chunk cosine LR, full-sequence loss), chunk ordering produces no measurable improvement over plain sequential order.

## Baseline

We reproduced #549's score-first TTT path locally with matched semantics:
- All 8 GPUs collaborate on every chunk (no per-rank sharding)
- Per-chunk cosine LR decay (same LR for all 3 epochs on a chunk)
- Full-sequence cross-entropy loss for TTT training
- Skip training on final chunk
- fp32 master params for TTT weight updates (implementation detail)

| Metric | Value |
|---|---|
| Static full sliding-window BPB | 1.12218 |
| Score-first BPB | **1.11961** |
| Post-TTT full sliding-window BPB | **1.11910** |
| TTT delta (full SW) | -0.00308 |

## Ordering experiments

| Configuration | score-first BPB | Δ vs baseline |
|---|---|---|
| Sequential order (baseline) | **1.11961** | — |
| Document-embedding global ordering | 1.11962 | +0.00001 |
| Sharded sequential (no all-GPU sync) | 1.12117 | +0.00156 |
| + clustered (majority-overlap k=8) | 1.12134 | +0.00173 |
| + microcluster bin-pack (k=32) | 1.12150 | +0.00189 |
| + global ordered contiguous shards | 1.12149 | +0.00188 |

**Global ordering:** No improvement. Identical BPB to sequential.
**Clustering/sharding:** All variants worse. Splitting the adaptation stream across independent ranks hurts more than any ordering benefit.

## What we tested

1. **Document-level clustering** (k=8, k=32) with majority-overlap chunk ownership
2. **Microcluster bin-packing** for workload balance across GPUs
3. **Global ordered stream** split into contiguous shards
4. **Nearest-neighbor document ordering** within clusters
5. **All-GPU-per-chunk** vs sharded-per-rank execution
6. **Alignment gating** — skip TTT updates when gradient misaligns with running direction

## Embedding sanity checks

We verified our document embeddings (layer-5 weighted-pool, normalized) capture real semantic structure:

| Anchor | Closest by embedding | Cosine |
|---|---|---|
| AP sports wire: "HOUSTON (AP) — Don Kelly had an RBI single..." | High school football article: "Acclimatization prepares teams for football..." | 0.77 |
| Knitting blog: "I decided to knit a Christmas stocking..." | Personal newspaper story: "I think it was Wednesday afternoon when I first found out about the holiday..." | 0.83 |
| Political satire: "TSA has been in the news a lot lately..." | Military/political analysis: "Wretchard has an important post about retired Gen. Barry McCaffrey..." | 0.80 |

Pairwise cosine (1000 docs): mean=0.53, std=0.12, range -0.04 to 0.89. Documents are well-separated.

## Evidence for ordering signal (in isolation)

Despite the negative end-to-end result, controlled experiments show real transfer signal:

- **Pairwise transfer:** Embedding NN pairs produce **1.69×** higher TTT transfer than random pairs
- **Multi-step chains:** NN chains show **1.56×** lift over random across 4 sequential steps
- **Gradient alignment:** Embedding similarity correlates 0.59 with gradient cosine

The signal exists but doesn't translate to BPB at scale.

## Why ordering doesn't help

Our alignment-gating experiment provides the answer: **always updating beats every gating policy, including the oracle.** Even with perfect knowledge of gradient alignment, skipping misaligned updates is worse than updating on everything. The compounding benefit of sequential adaptation dominates over ordering effects.

## Architecture

| Component | Value |
|---|---|
| Activation | LeakyReLU(0.5)² |
| BigramHash | 1536 |
| Quantization | INT6 per-row, zstd-22 |
| TTT | Score-first, SGD lr=0.002, momentum=0.9, 3 epochs/chunk |
| TTT precision | fp32 master params |
| Execution | All-GPU-per-chunk (8×H100 SXM collaborative) |

## Run command

Requires 8×H100 SXM. Download FineWeb data first:
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024

# Baseline (sequential, all-GPU cooperative)
TTT_ALL_GPU_PER_CHUNK=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Global ordering experiment
TTT_ALL_GPU_PER_CHUNK=1 TTT_GLOBAL_ORDER=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Clustered sharded (independent ranks)
TTT_DOC_CLUSTER=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Supplementary logs

See `logs/` for train.log files from all experimental variants.
