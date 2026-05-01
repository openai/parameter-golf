# Record candidate: PR #1855 stack + MP3 marker-pair fusion + alias smear boundary

**val_bpb: TBD** (3-seed mean on runpod; 1-seed DGX reference 1.06042) | size: TBD (3-seed runpod; expected ~15.9 MB based on PR #1855 baseline) | 8×H100 SXM, 600 s wallclock | TTT (phased)

3-seed verification on runpod (SEEDS=42, 0, 1234) — see `train_seed*.log` after the run.

## Base record extended by this submission

We extend [PR #1855's record candidate](https://github.com/openai/parameter-golf/pull/1855)
(11L XSA + LQER asym int4 + SparseAttnGate + BOS-fixed SmearGate + Polar-Express
Newton-Schulz Muon + fused softcapped CE Triton kernel + GPTQ int6 + int7 embed +
per-row int8 attn-gate + per-group lrzip+brotli compression + phased TTT eval +
9-hparam stack, 3-seed mean **1.06108**) with two additions on top:

1. **MP3 marker-pair fusion** (vocabulary surgery) — the three 2-grams
   `[▁, TITLE]` / `[▁, ALLCAPS]` / `[▁, CAPNEXT]` are fused into single alias
   donor tokens (donors 8/9/10 from byte-fallback IDs that occur 0× in the
   CaseOps corpus). The downstream **word X is preserved**.

2. **Alias smear boundary** — at positions immediately following an alias
   token, SmearGate's previous-position contribution is fully disabled
   (`ALIAS_PREV_SMEAR_SCALE=0.0`). Regular non-alias positions are unchanged.

## The single conceptual rule

```
[regular → regular]: SmearGate normal (PR #1855 default behaviour)
[regular → alias  ]: SmearGate normal at the alias position itself
[alias   → next   ]: SmearGate fully OFF at the position immediately after an alias
```

The intuition: an alias donor token already encodes the dense `(▁ + marker)`
signal in a single position. Letting SmearGate then leak that signal forward
into the following word would over-inject the same information twice. We
treat each alias position as a smear boundary — vocab surgery condenses the
information; the boundary keeps it from spreading.

## Code-level patch

The diff against PR #1855's `train_gpt.py` is **+62 / −2 lines**, organised
in five hunks: an `__init__` flag + `alias_is_alias` buffer (bool over vocab),
a per-position multiplicative mask inside `_forward_hidden`, the same mask
inside the TTT path `forward_ttt`, an `alias_map.json` load + buffer
re-registration in `deserialize` (eval side), and a load + warm-init for the
alias rows in `train_model`. Default `ALIAS_PREV_SMEAR_SCALE=0.0` is set in
`run_3seed.sh` (the code default of 0.25 is overridden so the boundary rule
is unambiguous).

## Component contributions (1-seed DGX ablations)

Each row reports the `val_bpb` impact of the listed change (PR #1855 stack
uses phased TTT eval). PR #1855 author measurements are on runpod (3-seed
mean); ours are 1-seed on DGX, so the absolute numbers are not directly
comparable across rows but the deltas within a row are honest.

| Component | Comparison | Δ val_bpb |
|---|---|---|
| **MP3 marker-pair fusion** | PR #1855 + MP3 + SmearGate as-is (scale=1.0) vs PR #1855 unmodified (DGX same env) | **−3.40 mbpb** |
| **Alias smear boundary (scale=0)** | PR #1855 + MP3 + scale=0.0 vs scale=1.0 | −0.01 mbpb (within noise) |

The boundary rule is mostly a *story / robustness* contribution rather than a
decisive bpb win on this stack — it keeps the model agnostic to the choice
of an arbitrary dampening constant. The bulk of the val_bpb improvement
comes from the MP3 vocab surgery.

## Architecture

| # | Component | Setting | Source |
|---|-----------|---------|--------|
| 1 | **MP3 marker-pair fusion** | 3 alias donor tokens for `[▁,TITLE]` / `[▁,ALLCAPS]` / `[▁,CAPNEXT]`; warm-init `0.4·E[▁]+0.6·E[marker]`, norm-matched | **this work** |
| 2 | **Alias smear boundary (`ALIAS_PREV_SMEAR_SCALE=0.0`)** | SmearGate's previous-position contribution disabled at the position immediately following an alias token; regular positions unchanged | **this work** |
| 3 | **PR #1855 stack** | full architecture + optimizer + quantization + compression + phased TTT eval inherited unchanged: 11L XSA + LQER asym int4 + SparseAttnGate + BOS-fixed SmearGate + Polar-Express MuonEqR + fused softcapped CE Triton kernel + GPTQ int6 + int7 embed + per-row int8 attn-gate + per-group lrzip+brotli + 9-hparam stack + CaseOps tokenizer | [#1855](https://github.com/openai/parameter-golf/pull/1855) |

## MP3 marker-pair fusion (vocabulary surgery)

Three patterns active in MP3, performed by `prepare_marker_pair_v3.py` on the
already-tokenized CaseOps shards:

```
[▁, TITLE]    -> ALIAS_SP_TITLE    (donor=8)
[▁, ALLCAPS]  -> ALIAS_SP_ALLCAPS  (donor=9)
[▁, CAPNEXT]  -> ALIAS_SP_CAPNEXT  (donor=10)
```

Donor IDs come from byte-fallback tokens that occur **0 times** in the
CaseOps corpus (verified by full-corpus token-count audit). They become
alias rows after warm-init + training; their original byte-fallback meaning
is unused in CaseOps text.

Token saving: **8.47 %** (~14.97 B → ~13.71 B train tokens; ~48 M → ~44 M val
tokens, canonical 50 K val docs). Bytes lossless: val sidecar sum unchanged.

### Static warm-init for alias rows

Each alias row is initialised as a weighted, norm-matched composite of its
constituent marker rows so the alias rows do not start at random:

```
E[alias_donor] = 0.4·E[▁] + 0.6·E[marker], renormalized to ‖E[marker]‖
```

Hparams: `MARKER_PAIR_W_SPACE=0.4`, `MARKER_PAIR_W_TITLE=0.6`,
`MARKER_PAIR_NORM_TARGET=title`.

## Why this works

**Token saving (training-side win)**: replacing the three `[▁, MARKER]`
2-grams with single alias donor tokens reduces the train stream by **8.47 %**
(~14.97 B → ~13.71 B train tokens; ~48 M → ~44 M val tokens, canonical 50 K
val docs). With the same 600 s wallclock budget, the model effectively sees
more *unique* documents per training step. Bytes lossless: the val sidecar
byte sum is unchanged (BPB is computed on canonical pre-transform UTF-8
byte counts), so this is a free token-side compression — not a metric trick.

**NLL distance breakdown (eval-side win)**: per-position NLL bucketed by
distance from the most recent alias position (measured on a swttt-eval
stack at int6 level — the mechanism generalizes to phased TTT, only the
absolute mbpb numbers may shift):

| distance | MP1 (single TITLE pair) | **MP3 (3 marker pairs, this work)** |
|---|---|---|
| d=1 (next word) | +0.159 mbpb | **−0.017** ✓ (preserved) |
| d=2             | +1.096        | +0.262 |
| d=3             | +0.301        | −0.003 |
| d=4–8           | −0.435        | **−1.199** ✓✓ |
| d=9–32          | −0.808        | **−1.687** ✓✓✓ |
| d=33+           | −0.100        | −0.247 |
| alias positions | −0.16         | +1.430 (TITLE alias dominates) |
| **GRAND TOTAL** (int6 level) | +0.05 | **−1.46** mbpb |

Insights:
- **d=1 is preserved** because word X is unchanged in the alias scheme —
  full-word-fusion variants paid heavy cost at d=1 (1.07 ‒ 2.78 mbpb) by
  absorbing the word into the alias.
- **d=4+ shows compounding wins** — shorter sequences let attention model
  long-range structure better.
- Alias positions themselves show a +1.43 mbpb cost (TITLE alias dominates);
  this is more than offset by the non-alias gains.

## Files

| File | Purpose |
|---|---|
| `train_gpt.py` | PR #1855 training script (~3.8 k lines) plus the 5-hunk MP3 patch (alias buffer + dampening mask in `_forward_hidden` and `forward_ttt` + `deserialize`/`train_model` setup). |
| `prepare_caseops_data.py` | Multiprocess CaseOps tokenizer (parallel via `multiprocessing.Pool`). |
| `prepare_marker_pair_v3.py` | Vocab-surgery script: fuses `[▁, MARKER]` 2-grams into alias donor tokens; rewrites shard headers correctly. |
| `download_docs.py` | Minimal HF Hub wrapper: fetches `docs_selected.jsonl` from `willdepueoai/parameter-golf`. |
| `lossless_caps.py` | Bijective lowercase + private-use-area sentinel pre-encoding (CaseOps infrastructure). |
| `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` | SentencePiece model used by CaseOps (~367 KB). |
| `alias_map.json` | MP3 alias map (donor ↔ marker mapping). |
| `requirements.txt` | Python deps + system-level lrzip note. |
| `run_3seed.sh` | 3-seed runner. Outputs `train_seed{42,0,1234}.log`. |
| `train_seed*.log` | Per-seed run logs (created by `run_3seed.sh`). |

## Pipeline

If you have **already prepared the MP3 dataset** for an earlier
CaseOps/MP3 submission, you can skip steps 0–2 and go straight to step 3,
just pointing `DATA_PATH` at the existing
`<...>/fineweb10B_sp8192_caseops_marker_pair_v3` directory.

### 0. System deps

```bash
apt-get install -y lrzip
pip install -r requirements.txt
pip install --no-deps flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

### 1a. Download raw FineWeb-10B docs (one-time, ~45 GiB on disk)

```bash
python3 download_docs.py
# writes ./data/datasets/docs_selected.jsonl
#        ./data/datasets/docs_selected.source_manifest.json
```

### 1b. CaseOps tokenization (one-time)

```bash
python3 prepare_caseops_data.py \
    --docs ./data/datasets/docs_selected.jsonl \
    --out  ./data \
    --sp   tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
# writes ./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
```

### 2. Vocabulary surgery (MP3)

```bash
python3 prepare_marker_pair_v3.py
# writes ./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3/
```

### 3. 3-seed training (~3 × 10 min wallclock + per-seed eval)

```bash
bash run_3seed.sh
# writes train_seed{42,0,1234}.log in this dir
```

### Single-seed reference command

The PR #1855 9-hparam stack must be passed explicitly (the values here
mirror the PR #1855 reproduce command); the only additions are
`MARKER_PAIR_*` and `ALIAS_PREV_SMEAR_SCALE=0.0`:

```bash
SEED=42 \
CASEOPS_ENABLED=1 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3 \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
MARKER_PAIR_MODE=1 \
MARKER_PAIR_W_SPACE=0.4 MARKER_PAIR_W_TITLE=0.6 \
ALIAS_PREV_SMEAR_SCALE=0.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`run_3seed.sh` wraps this command and iterates over the seed list.

## Hardware / environment

- 8 × H100 80 GB SXM (NVLink), 600 s wallclock budget
- PyTorch 2.9.1+cu128, CUDA 12.8, FlashAttention 3 (Hopper kernels)
- `lrzip` system binary (used by `COMPRESSOR=pergroup`)

## Submission status

| Seed | val_bpb (phased TTT) | val_loss | size (bytes) | step_avg | steps |
|---|---|---|---|---|---|
| 42   | TBD | TBD | TBD | TBD | TBD |
| 0    | TBD | TBD | TBD | TBD | TBD |
| 1234 | TBD | TBD | TBD | TBD | TBD |
| **Mean** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

(1-seed DGX reference with the same code: seed 42 → val_bpb=1.06042.)
