# Non-Record: TTT Ordering Experiments — Negative Results

**val_bpb: 1.11961** (non-record)
**Author:** Joel Pfeiffer (@jpfeiffe)
**Base:** Fork of PR #549 (LeakyReLU(0.5)², BigramHash 1536, legal score-first TTT)

## Change

**fp32 master params for TTT updates.** PR #549 accumulates TTT weight updates in bf16. We clone trainable params to fp32 master weights before TTT and copy back after each SGD step, eliminating precision loss during adaptation.

## Architecture

| Component | Value |
|---|---|
| Activation | LeakyReLU(0.5)² |
| BigramHash | 1536 |
| Quantization | INT6 per-row, zstd-22 |
| TTT | Score-first, SGD lr=0.002, momentum=0.9, 3 epochs/chunk |
| TTT precision | **fp32 master params** (our change) |
| Execution | All-GPU-per-chunk (8×H100 SXM collaborative) |

## Results

| Configuration | score-first BPB | post-TTT full SW BPB |
|---|---|---|
| #549 reproduction + fp32 fix (sequential) | **1.11961** | **1.11910** |
| + document-embedding global ordering | 1.11962 | 1.11910 |
| Sharded sequential (no all-GPU sync) | 1.12117 | — |
| + clustered (majority-overlap k=8) | 1.12134 | — |
| + microcluster bin-pack (k=32) | 1.12150 | — |
| + global ordered contiguous shards | 1.12149 | — |

## What we tested

Systematic investigation of whether **chunk ordering** affects TTT score-first evaluation. We tested:

1. **Document-level clustering** (k=8, k=32) with various ownership rules
2. **Majority-overlap chunk ownership** vs midpoint heuristic
3. **Microcluster bin-packing** for workload balance
4. **Global ordered stream** with contiguous shards
5. **Nearest-neighbor document ordering** within clusters
6. **All-GPU-per-chunk** vs sharded-per-rank execution

## Key findings

1. **fp32 master params help.** Our TTT delta (-0.0031 on full SW) exceeds #549's (-0.0023).
2. **All-GPU-per-chunk execution matters.** Sharded TTT (1.12117) is significantly worse than all-GPU collaborative (1.11961).
3. **Chunk ordering does not help.** Document-embedding-based ordering produces identical BPP to sequential order.
4. **Clustering hurts.** All clustered variants are worse than sequential.

## Embedding sanity checks

We verified the document embeddings (layer-5 weighted-pool, normalized) capture real semantic structure:

**Nearest-neighbor examples:**

| Anchor | Closest by embedding | Cosine |
|---|---|---|
| AP sports wire: "HOUSTON (AP) — Don Kelly had an RBI single..." | High school football article: "Acclimatization prepares teams for football..." | 0.77 |
| Knitting blog: "I decided to knit a Christmas stocking..." | Personal newspaper story: "I think it was Wednesday afternoon when I first found out about the holiday..." | 0.83 |
| Political satire: "TSA has been in the news a lot lately..." | Military/political analysis: "Wretchard has an important, fascinating post about retired Gen. Barry McCaffrey..." | 0.80 |

The embeddings correctly group sports with sports, personal narrative with personal narrative, and political commentary with political commentary.

**Pairwise cosine distribution (1000 docs):**
- Mean: 0.53, Std: 0.12, Range: -0.04 to 0.89
- Documents are well-separated — not the 0.95+ saturation we initially saw with token unigram cosines.

## Evidence for ordering signal (in isolation)

Despite the negative end-to-end result, we found real transfer signal in controlled experiments:

- **Pairwise transfer:** Embedding NN pairs produce **1.69×** higher TTT transfer than random pairs (100% of NN pairs improved vs 94% random)
- **Multi-step chains:** NN chains show **1.56×** lift over random across 4 sequential TTT steps, with growing deltas
- **Gradient alignment:** Embedding similarity correlates 0.59 with gradient cosine; predicted gradient sketches find **3.1×** better gradient-aligned neighbors

The signal exists but doesn't translate to BPP improvement at scale. The all-GPU collaborative execution with plain sequential order already captures the full adaptation benefit.

## Why ordering doesn't help

Our alignment-gating experiment provides the answer: **always updating beats every gating policy, including the oracle.** Even with perfect knowledge of gradient alignment, skipping misaligned updates is worse than updating on everything. The compounding benefit of sequential adaptation dominates over ordering effects.

## Run command

```bash
JOB=compete GPU_COUNT=8 JOB_PHASE=eval \
  ARTIFACT_TAG=h100/compete-20260330-040640 \
  TTT_ALL_GPU_PER_CHUNK=1 TTT_DOC_CLUSTER=0 \
  bash scripts/launch_h100.sh --confirm
```

## Supplementary logs

See `logs/` for train.log files from all experimental variants.
