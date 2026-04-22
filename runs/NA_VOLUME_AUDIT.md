# NA-1 volume (`hvpdph5i3g`) audit — 2026-04-20

Read on pod `29ct5351ipszln` (RTX PRO 6000, $1.89/hr). Volume lives at `/workspace/` on NA-1.

## Top-level layout (~99 GB used)

| Path | Size | Keep? |
|---|---|---|
| `/workspace/parameter-golf/data/docs_selected.jsonl` | **45 GB** | ❌ can delete (see below) |
| `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/` | 24 GB | ✅ keep (our live training data) |
| `/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/` | 16 GB | ❌ can delete (old tokenizer variant, not used on any current spec) |
| `/workspace/runs/` | 14 GB | mixed — see breakdown |
| `/workspace/parameter-golf/` (code + misc) | ~100 MB | ✅ keep |

**Sum of deletable: ~77 GB** (docs_selected.jsonl + sp1024 + misc run junk below).

## What is `docs_selected.jsonl`?

**Not training data.** Training only reads pre-tokenized `.bin` shards in `data/datasets/*/`. This 45 GB JSONL is the **raw text source** used once to produce those shards.

From `manifest.json`:
- 15,368,808 FineWeb documents
- 48.17 GB raw text
- sha256 stamped — canonical snapshot of the challenge's fineweb subset
- "paused snapshot of the 50B shuffled train stream, selection_seed=1337"

**Only needed if:** you want to **re-tokenize** the raw text with a new vocab (e.g., produce an sp4096 variant). For running any training with existing sp1024/sp8192 data, it's inert.

**Recommendation:** DELETE. If we ever want to re-tokenize, we can re-download from `willdepueoai/parameter-golf` on HF (the manifest records the exact source and sha256). ~45 GB back.

## Checkpoints breakdown

### `/workspace/runs/000-sota-replication/checkpoints/` — 2.7 GB, 9 files, each 300 MB
Our baseline run's checkpoints. All ~same size (model + optim + ema state).

| File | Keep? | Notes |
|---|---|---|
| `ckpt_event_step455.pt` | ✅ keep | small, edge reference |
| `ckpt_event_step1137.pt` | ❌ delete | mid-training, redundant once spec 006 lays down every-100-step ckpts |
| `ckpt_event_step1500.pt` | ❌ delete | same |
| `ckpt_event_step2275.pt` | ❌ delete | same |
| `ckpt_event_step3412.pt` | ❌ delete | same |
| `ckpt_warmdown_start_step1048.pt` | ✅ keep | event-boundary marker (wallclock-based schedule) |
| `ckpt_pre_recurrence_step1378.pt` | ✅ keep | event-boundary marker |
| `ckpt_final_pre_ema_step3849.pt` | ✅ **never delete** | baseline final, used for hotstart |
| `ckpt_final_post_ema_step3849.pt` | ✅ **never delete** | baseline final, EMA-applied |

**Free: ~1.2 GB** from deleting the 4 mid-event ckpts.

### `/workspace/runs/004b-qk6-full/checkpoints/` — 2.7 GB, 9 files
Spec 004b was killed (QK=6.0 tied with QK=5.25). No future plan uses these. User already flagged delete.

**Recommendation:** DELETE ENTIRE DIRECTORY. **Free: ~2.7 GB.**

### `/workspace/runs/001-hessian-sdclip/` — 326 MB
Spec 001 killed; Hessians cache (~80 MB) + .ptz artifacts (~50 MB each × 6 λ). No live reuse.

**Recommendation:** DELETE. **Free: ~326 MB.**

### `/workspace/runs/002-swa-plus-ema-1h-c0/` — 326 MB
Spec 002 killed. Same story as 001.

**Recommendation:** DELETE. **Free: ~326 MB.**

### `/workspace/runs/002-swa-plus-ema-8h-c0/`, `003-bigram-hash-screen/`, `004-qk-gain-extension/`, `004c-qk6-verify/`, `005-weight-delta/`, `smoke-hotstart/` — all <5 MB each
Small artifacts (logs, JSON, scripts). **Keep — ~10 MB total.**

### `/workspace/runs/smoke-test/` (2 GB) + `test1/` (2.4 GB) + `test2/` (3.2 GB)
Pre-spec-000 test runs from initial pod bring-up. Stale.

**Recommendation:** DELETE ALL THREE. **Free: ~7.6 GB.**

## Data-side cleanup

| Path | Size | Recommendation |
|---|---|---|
| `data/docs_selected.jsonl` | 45 GB | **DELETE** (not training data; re-downloadable from HF `willdepueoai/parameter-golf`) |
| `data/datasets/fineweb10B_sp1024/` | 16 GB | **DELETE** (no current spec uses sp1024 — all active work on sp8192) |
| `data/datasets/fineweb10B_sp8192/` | 24 GB | **KEEP** — our active training data (129 shards) |
| `data/tokenizers/fineweb_1024_bpe.model` | 254 KB | delete if removing sp1024 data |
| `data/tokenizers/fineweb_8192_bpe.model` | 371 KB | **KEEP** |
| `data/manifest.json` | 2.5 KB | **KEEP** |

## Total if we clean aggressively

```
docs_selected.jsonl            45 GB
data/datasets/fineweb10B_sp1024 16 GB
runs/004b-qk6-full             2.7 GB
runs/smoke-test + test1 + test2 7.6 GB
runs/001-hessian-sdclip        326 MB
runs/002-swa-plus-ema-1h-c0    326 MB
runs/000/ckpt_event_step[1137,1500,2275,3412].pt  1.2 GB
────────────────────────────────────
total freed                    ~73 GB
```

After cleanup, NA-1 volume usage would drop from ~99 GB to ~26 GB (just the essentials: sp8192 data + tokenizer + repo + 5 essential ckpts + current small artifacts).

## Suggested delete commands (NOT YET EXECUTED)

```bash
# Big wins
rm /workspace/parameter-golf/data/docs_selected.jsonl       # 45 GB
rm /workspace/parameter-golf/data/docs_selected.source_manifest.json 2>/dev/null || true
rm -rf /workspace/parameter-golf/data/datasets/fineweb10B_sp1024     # 16 GB
rm /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.*      # tokenizer companion

# Stale runs
rm -rf /workspace/runs/004b-qk6-full                         # 2.7 GB
rm -rf /workspace/runs/smoke-test /workspace/runs/test1 /workspace/runs/test2   # 7.6 GB
rm -rf /workspace/runs/001-hessian-sdclip /workspace/runs/002-swa-plus-ema-1h-c0 # 650 MB

# Mid-training ckpts redundant with spec 006's dense ckpts
rm /workspace/runs/000-sota-replication/checkpoints/ckpt_event_step1137.pt
rm /workspace/runs/000-sota-replication/checkpoints/ckpt_event_step1500.pt
rm /workspace/runs/000-sota-replication/checkpoints/ckpt_event_step2275.pt
rm /workspace/runs/000-sota-replication/checkpoints/ckpt_event_step3412.pt
```

## Kept for reference

- All spec evaluations / summaries / notes (small).
- The 5 essential spec-000 ckpts.
- Everything under parameter-golf/ that isn't data/docs_selected.jsonl or the sp1024 dataset.
