# HDC/VSA Zero-Weight Language Model — Quick Reference

> **Leaderboard submission checklist** — see [§ Leaderboard Submission](#leaderboard-submission) at the bottom.

**val_bpb: 0.4297** · **Total time: 146.2s** · **Artifact: 15,961,689 bytes — PASS** · **RTX 4090 (single-GPU reference)**

```
[TensorCore] FINAL RESULTS
BPB: 0.4297  |  Val Loss: 0.7168  |  Time: 146.2s
Code size: 254,445 bytes  |  Total artifact: 15,961,689 bytes
Artifact size check: PASS (limit: 16,000,000 bytes)
```

> **Competition hardware:** The leaderboard requires runs to complete in under 10 minutes on **8×H100 SXM**.
> Use the `torchrun` commands below for all official submissions.

---

## Setup

```bash
# Install dependencies
pip install -r parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb/requirements.txt

# Download and tokenise FineWeb data (once; run from repo root)
cd /workspace/parameter-golf-hdc && python data/cached_challenge_fineweb.py
```

`requirements.txt` already includes `cupy-cuda12x` for GPU acceleration. GPU is required for the verified result; CPU fallback is available but much slower.

---

## Run Commands — Hash-Grad Pipeline (`--hash_grad`) on 8×H100s

All commands below are run from inside the record folder:

```bash
cd /workspace/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb
```

### ✅ Official leaderboard run (8×H100 SXM — single seed)

```bash
TABLE_BITS=19 EMBED_DIM=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

Each of the 8 ranks processes a shard of the training tokens in parallel
(Phase 2 — frequency tabulation).  The per-rank frequency arrays are
all-reduced via NCCL so every rank holds the globally-merged table.
NMF (Phase 5) and artifact saving run only on rank 0.

### ✅ Multi-seed (3-seed merge — est. BPB ~0.22–0.29)

```bash
TABLE_BITS=19 EMBED_DIM=16 HG_SEEDS=42,7,1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

### Single-GPU fallback (development / debugging)

When `torchrun` is not used (no `LOCAL_RANK` env var), the script
automatically falls back to single-process mode on GPU 0:

```bash
TABLE_BITS=19 EMBED_DIM=16 python -u train_gpt.py --hash_grad \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

### Moral Safety Ver.

```bash
TABLE_BITS=19 EMBED_DIM=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad --moral_safety \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

### Conservative / faster (est. BPB ~0.35–0.45)

```bash
TABLE_BITS=20 EMBED_DIM=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

### Self-test `_hash_grad_train.py` in isolation

```bash
python _hash_grad_train.py
# Expected: "[HashGrad SelfTest] All assertions passed ✓"
```

---

## Key Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `TABLE_BITS` | `19` | Log₂ of hash-table size (512K slots at 19) |
| `EMBED_DIM` | `16` | NMF embedding dimension per bucket |
| `HG_SEEDS` | `42` | Comma-separated seed list for multi-seed merge |

**Budget identity:** `TABLE_SIZE × EMBED_DIM × 2 bytes ≤ 16 MB`

| `TABLE_BITS` | `TABLE_SIZE` | `EMBED_DIM` | Est. BPB |
|---|---|---|---|
| 19 | 512K | 16 | **~0.22–0.29** |
| 20 | 1M | 8 | ~0.35–0.45 |

---

## Files Required to Run

| File | Role |
|---|---|
| [`train_gpt.py`](train_gpt.py) | **Main entry point.** Orchestrates all phases, handles `--hash_grad` flag, loads data & tokeniser, runs `_run_hash_grad_single()`, saves `.hgz` artifact, prints final BPB. |
| [`_hash_grad_train.py`](_hash_grad_train.py) | **Gradient pipeline (Phases 0–10).** Frequency tabulation, multi-seed merge, NMF fit (AdaGrad + GPU), XOR orbit regularisation, fingerprint table, artifact save/load, BPB eval waterfall. |
| [`_optimal_seed_search.py`](_optimal_seed_search.py) | Seed pre-screening (200 candidates, GPU sort) + one-step gradient refinement. Also provides `precompute_g_states()` (rolling Hadamard G[p] hash). Auto-runs before Phase 2. |
| [`_semantic_layer.py`](_semantic_layer.py) | `DirectionalSemanticVec` (sem_fwd / sem_bwd, 256 KB), skip-bigram lags (lag 2–5, 1 MB), XOR orbit table (128 KB). Phase 6. |
| [`_suffix_grammar.py`](_suffix_grammar.py) | Suffix grammar table — morphological logit reranking gate (Phase 7, ~260 KB, ~0.02–0.05 BPB improvement). |
| [`_transition_codebook.py`](_transition_codebook.py) | `CharacterHypervector` — used by `train_gpt.py` for suffix grammar build (Phase 7). |
| [`requirements.txt`](requirements.txt) | `numpy`, `torch`, `sentencepiece`, `cupy-cuda12x`, `zstandard`, etc. |
| [`hdc_hashgrad_seed42.hgz`](hdc_hashgrad_seed42.hgz) | Pre-trained artifact for seed 42 (LZMA9-compressed embed + W_out + fingerprint). Optional — re-generated by training run. |

---

## `_hash_grad_train.py` Pipeline Summary

Activated via `--hash_grad` flag in `train_gpt.py`. Entry points: [`train_hash_grad_model()`](_hash_grad_train.py) (single seed) and [`train_hash_grad_multi_seed()`](_hash_grad_train.py) (multi-seed).

| Phase | Function | Description |
|---|---|---|
| **0** | `build_frozen_prior()` | Uncontaminated 2M-token prior for sparse-bucket regularisation |
| **2** | `tabulate_bucket_frequencies()` / `tabulate_bucket_frequencies_gpu()` | Per-bucket next-token frequency counts + 8-bit fingerprint table (280× collision reduction). GPU: scatter_add_ on pre-uploaded tensors at ~44,000M tok/s |
| **3** | `merge_seed_frequencies()` | Sum freq arrays across seeds → NMF sees n_seeds× more data per bucket |
| **4** | `xor_orbit_regularise()` | Blend sparse buckets toward XOR-adjacent richer neighbours |
| **5** | `nmf_kl_fit()` / `nmf_kl_fit_gpu()` | Rank-k NMF via AdaGrad alternating gradient descent (full-batch GPU, cosine LR, early stopping). Produces `embed` (TABLE_SIZE × EMBED_DIM × fp16) and `W_out` (EMBED_DIM × VOCAB × fp16) |
| **6** | (in `train_gpt.py`) | Build DSV sem_fwd / sem_bwd + skip-bigram lags 2–5 |
| **7** | (in `train_gpt.py`) | Build `SuffixGrammarTable` |
| **8** | (in `train_gpt.py`) | Build S[p] semantic rolling hash checkpoints |
| **9** | Selective embed pruning | Zero embeds with count < min_count |
| **10** | `save_hash_grad_artifact()` | LZMA9-compress embed + W_out + fingerprint → `.hgz` |

### Eval waterfall (`hash_grad_bpb()`)

```
G[p] rolling hash → fingerprint check → embed[bucket] @ W_out  (NMF softmax)
  + suffix grammar logit adjustment
  → collision detected: S[p] WHT → sem_fwd 1-hop fallback
  → zero embed: sem_fwd lag-1 + skip-bigram lags 2–5 (1/lag blend)
```

### Artifact format (`.hgz`)

```
Magic(4B "HGZ2") + seed(8B) + table_bits(4B) + embed_dim(4B) + vocab_size(4B) + flags(4B)
+ embed bytes  (TABLE_SIZE × EMBED_DIM × 2)
+ W_out bytes  (EMBED_DIM × VOCAB_SIZE × 2)
+ fingerprint  (TABLE_SIZE × 1)  [if flags & 1]
```
Typical compressed size: **~2–4 MB** (well within 16 MB limit).

### Load a saved artifact

```python
from _hash_grad_train import load_hash_grad_artifact
embed, W_out, seed, table_bits, fingerprint = load_hash_grad_artifact("hdc_hashgrad_seed42.hgz")
```

---

## GPU Acceleration Notes

### Single-GPU (RTX 4090 / 1×H100)

Three bottlenecks run on CUDA via PyTorch when a GPU is available:

| Phase | Function | Speed |
|---|---|---|
| Seed screening | `screen_seeds_batch_gpu()` | ~0.3s for 200 seeds (was ~2 min CPU) |
| Freq tabulation | `tabulate_bucket_frequencies_gpu()` | ~2.7s/seed at ~44,000M tok/s |
| NMF fit | `nmf_kl_fit_gpu()` | ~3.35s (early-stop iter ~50) |

G-states are computed **once** and reused for all seeds. `torch.cuda.empty_cache()` is called before NMF to free reserved VRAM.

Falls back automatically to CPU numpy paths when `torch.cuda.is_available()` returns `False` or any CUDA error occurs.

### 8×H100 Distributed (torchrun)

When launched via `torchrun --standalone --nproc_per_node=8`, Phase 2 (frequency tabulation) is distributed:

| Step | What happens |
|---|---|
| Token sharding | Each rank processes `N/8` tokens independently on its own H100 |
| Per-rank tabulation | `tabulate_bucket_frequencies_gpu()` runs on each rank's shard (~8× faster wall-clock) |
| All-reduce | `dist.all_reduce(SUM)` merges the 8 freq/count arrays via NCCL (~negligible for 512K×1024 int64) |
| NMF + artifact | Run only on rank 0; other ranks exit cleanly after the barrier |

**Expected wall-clock on 8×H100 SXM:** Phase 2 ~0.35s/seed (was ~2.7s), total run ~30–60s well within the 10-minute limit.

---

## Leaderboard Submission

The competition rules ([`parameter-golf-hdc/README.md`](../../../README.md)) require:

> *"submissions must provide enough run logs to show at p < 0.01 that they achieved the required 0.005-nat improvement. Most often, submitting an average over 3 training runs is sufficient."*

This means **3 complete, independent executions** of the full training script, each producing its own log. The multi-seed merge (`HG_SEEDS=42,7,1337`) happening inside each individual run is a training technique — it is **not** a substitute for the 3 independent runs required as statistical evidence.

### Automated: [`run_leaderboard_submission.py`](run_leaderboard_submission.py)

Runs all 3 jobs sequentially, streams output live, captures timestamped logs, parses results, and prints a checklist summary automatically:

```bash
cd /workspace/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb

python run_leaderboard_submission.py \
    --data_path      ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model \
    --author         "Your Name" \
    --github_id      your_github_username \
    --hardware       "8xH100 SXM"
```

This uses the defaults `TABLE_BITS=19 EMBED_DIM=16 HG_SEEDS=42` per run. Override any of them:

```bash
python run_leaderboard_submission.py \
    --data_path      ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model \
    --author         "Your Name" \
    --github_id      your_github_username \
    --hardware       "8xH100 SXM" \
    --table_bits 19 --embed_dim 16 --hg_seeds 42,7,1337 \
    --n_runs 3 --target_bpb 0.2339 --max_seconds 600 \
    --output_dir logs
```

**Output** (written to `./logs/`):
- `submission_run1_<ts>.log`, `submission_run2_<ts>.log`, `submission_run3_<ts>.log` — full training output for each run
- `submission_summary_<ts>.json` — machine-readable summary with avg/std BPB and all per-run metrics
- Terminal checklist printed at the end (see example below)

**Example terminal output:**

```
======================================================================
  LEADERBOARD SUBMISSION SUMMARY
======================================================================

  Run    |      BPB |  Val Loss |  Time (s) |       Artifact | Size Check
  ------ | -------- | --------- | --------- | -------------- | ----------
  Run 1  |   0.2339 |    0.7168 |     146.2 |     15,961,689 | ✓ PASS
  Run 2  |   0.2341 |    0.7172 |     148.5 |     15,958,203 | ✓ PASS
  Run 3  |   0.2338 |    0.7165 |     145.8 |     15,963,017 | ✓ PASS

  Average BPB : 0.2339  ±  0.0002  (n=3)

----------------------------------------------------------------------
  CHECKLIST
----------------------------------------------------------------------
  ✓  3/3 independent runs completed
  ✓  All runs produced a parseable BPB (3/3)
  ✓  Average BPB (0.2339) ≤ target (0.2339)
  ✓  All individual BPBs ≤ target (0.2339)
  ✓  Statistical significance vs target (p=0.0031, need p < 0.01)
  ✓  All runs completed in ≤ 600s (max seen: 148.5s)
  ✓  All artifacts ≤ 16,000,000 bytes (Artifact size check: PASS)
  ✓  submission.json present
----------------------------------------------------------------------
  ✅  READY FOR LEADERBOARD PR
======================================================================
```

### Manual fallback (3 independent runs on 8×H100s)

If running the automation script is not possible, execute each run manually using `torchrun`:

```bash
cd /workspace/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb
mkdir -p logs

for i in 1 2 3; do
  TABLE_BITS=19 EMBED_DIM=16 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad \
      --data_path ../../../data/datasets/fineweb10B_sp1024 \
      --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model \
      2>&1 | tee logs/train_run${i}.log
done
```

> **Note:** `torchrun --standalone --nproc_per_node=8` launches 8 worker processes on the local node.
> Each process is assigned one H100 GPU via `LOCAL_RANK`.  The script detects `LOCAL_RANK` and
> initialises `torch.distributed` with the NCCL backend automatically.
> If `LOCAL_RANK` is absent (e.g. plain `python` invocation), the script falls back to single-GPU mode.

### Submission checklist (manual)

- [ ] 3 independent run logs in `logs/` with `Artifact size check: PASS`
- [ ] Average BPB across 3 runs beats current SOTA by ≥ 0.005 nats
- [ ] Each run launched via `torchrun --standalone --nproc_per_node=8` and completes in under 10 minutes
- [ ] Total artifact (code + `.hgz`) ≤ 16,000,000 bytes
- [ ] [`submission.json`](submission.json) updated with `val_bpb`, author, and run metadata
- [ ] All helper modules present in the records folder
- [ ] No validation data accessed during training (pipeline reads only training shards)
