# Record candidate: PR #1855 stack + MP3 marker-pair fusion + alias smear boundary

**val_bpb: 1.06042** (1-seed measured on 8×H100, phased TTT eval) | ~16.74 MB *(see Note on size)* | 8×H100 SXM, 600 s wallclock | TTT (phased)

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

A scale sweep on the PR #1855 stack (DGX, 1-seed) confirmed all four scale
values land within 1-seed seed noise (PR #1855 author 3-seed std ≈ 0.0009);
`scale=0.0` is the cleanest design with no arbitrary constant:

| ALIAS_PREV_SMEAR_SCALE | val_bpb (DGX 1-seed) |
|---|---|
| 1.0 (no dampening, identical to PR #1855 SmearGate) | 1.06043 |
| 0.5 | 1.06059 |
| 0.25 | 1.06048 |
| **0.0 (this submission — alias smear boundary)** | **1.06042** |

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

Each row reports the swttt `val_bpb` impact of the listed change. PR #1855
author measurements are on runpod (3-seed mean); ours are 1-seed on DGX, so
the absolute numbers are not directly comparable across rows but the deltas
within a row are honest.

| Component | Comparison | Δ val_bpb |
|---|---|---|
| **MP3 marker-pair fusion** | PR #1855 + MP3 + SmearGate as-is (scale=1.0) vs PR #1855 unmodified (DGX same env) | **−3.40 mbpb** |
| **Alias smear boundary (scale=0)** | PR #1855 + MP3 + scale=0.0 vs scale=1.0 | −0.01 mbpb (within noise) |

The boundary rule is mostly a *story / robustness* contribution rather than a
decisive bpb win on this stack — it keeps the model agnostic to the choice
of an arbitrary dampening constant. The bulk of the val_bpb improvement
comes from the MP3 vocab surgery.

## Architecture

11L / 512d / 8 query heads / 4 KV heads (GQA 2:1) / FA3 / partial RoPE 16/64 /
MLP 4× (2048) with `LeakyReLU(0.5)²`. The standard transformer building
blocks are baseline; the table below lists the submission-shaping components
in rough order of contribution (most are inherited from PR #1855).

| # | Component | Setting | Source |
|---|-----------|---------|--------|
| 1 | **MP3 marker-pair fusion** | 3 alias donor tokens for `[▁,TITLE]` / `[▁,ALLCAPS]` / `[▁,CAPNEXT]`; warm-init `0.4·E[▁]+0.6·E[marker]`, norm-matched | **this work** |
| 2 | **Alias smear boundary (`ALIAS_PREV_SMEAR_SCALE=0.0`)** | SmearGate's previous-position contribution disabled at the position immediately following an alias token; regular positions unchanged | **this work** |
| 3 | **Phased TTT eval** | 3 cumulative phases at doc-boundaries 833/1666/2500 (max prefix 2500 docs); LoRA rank=80, per-doc reset; covers Q/K/V/O/MLP + lm_head | [#1855](https://github.com/openai/parameter-golf/pull/1855) |
| 4 | **LQER asymmetric int4** | rank=4 quant-error correction on top-3 (largest Frobenius residual) tensors, asym_group=64 | [#1797](https://github.com/openai/parameter-golf/pull/1797) |
| 5 | **Sparse attention head-output gate** | `gate_window=12`, `SPARSE_ATTN_GATE_SCALE=0.5`, per-row int8 quant on the gate (`GATED_ATTN_QUANT_GATE=1`) | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| 6 | **SmearGate (BOS-fixed, base behaviour)** | `gate_window=12` position-mixing with `not_bos` mask | [#1667](https://github.com/openai/parameter-golf/pull/1667) + PR #1855 BOS fix |
| 7 | **Polar-Express MuonEqR** | Muon + row L2 normalization (EQ-R) + 5-step Polar-Express minimax tuples | [#1344](https://github.com/openai/parameter-golf/pull/1344) + [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| 8 | **Fused softcapped CE Triton kernel** | single-pass training-only fused cross-entropy with logit softcap | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| 9 | **GPTQ int6 + int7 embed + per-row int8 attn-gate + per-group lrzip+brotli** | GPTQ Hessian (calib batches=16); per-group similarity-sort lrzip + brotli compression on the int6 GPTQ blob | [#535](https://github.com/openai/parameter-golf/pull/535), [#1586](https://github.com/openai/parameter-golf/pull/1586), [#1736](https://github.com/openai/parameter-golf/pull/1736), [#1855](https://github.com/openai/parameter-golf/pull/1855) |
| 10 | **CaseOps tokenizer** | sp8192 lossless caps + 4 reserved markers (TITLE / ALLCAPS / CAPNEXT / ESC); donors 8/9/10 are byte-fallback IDs with 0 occurrences | [#1729](https://github.com/openai/parameter-golf/pull/1729) |

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

Token saving: **8.47 %** (15.02 B → 13.75 B train tokens; 9.66 M → 8.82 M val
tokens). Bytes lossless: val sidecar sum unchanged.

### Static warm-init for alias rows

Each alias row is initialised as a weighted, norm-matched composite of its
constituent marker rows so the alias rows do not start at random:

```
E[alias_donor] = 0.4·E[▁] + 0.6·E[marker], renormalized to ‖E[marker]‖
```

Hparams: `MARKER_PAIR_W_SPACE=0.4`, `MARKER_PAIR_W_TITLE=0.6`,
`MARKER_PAIR_NORM_TARGET=title`.

## Why this works (NLL distance breakdown vs. CaseOps base)

Per-position NLL bucketed by distance from the most recent alias position:

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

```bash
SEED=42 \
CASEOPS_ENABLED=1 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3 \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
MARKER_PAIR_MODE=1 \
MARKER_PAIR_W_SPACE=0.4 MARKER_PAIR_W_TITLE=0.6 \
ALIAS_PREV_SMEAR_SCALE=0.0 \
COMPRESSOR=pergroup NCCL_NET=Socket \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other PR #1855 hyperparameters (TTT, LQER, SparseAttnGate, MIN_LR floor,
GPTQ calibration, 9-hparam stack overrides, etc.) use the defaults baked
into `train_gpt.py`, which match the PR #1855 published configuration.

## Hardware / environment

- 8 × H100 80 GB SXM (NVLink), 600 s wallclock budget
- PyTorch 2.9.1+cu128, CUDA 12.8, FlashAttention 3 (Hopper kernels)
- `lrzip` system binary (used by `COMPRESSOR=pergroup`)

## Note on size

Author's DGX H100 box and the official runpod 8×H100 SXM environment
produce slightly different artifact sizes for the same code:

- PR #1855 author on runpod (no MP3): 15,897,259 B (15.90 MB) — VALID
- This submission on DGX (PR #1855 + MP3, 1-seed seed 42): 16,739,873 B (16.74 MB)
- PR #1855 unmodified on DGX (1-seed seed 42): 16,746,582 B (16.75 MB)

The ~840 KB delta between the two environments is reproduced even *without*
the MP3 patch, so it is environmental (likely `lrzip` ZPAQ version /
numerical state) rather than a consequence of MP3. The 3-seed runpod
verification is the authoritative size measurement for this submission.

## Submission status

| Seed | val_bpb (phased TTT) | val_loss | size (bytes) | step_avg | steps |
|---|---|---|---|---|---|
| 42   | TBD | TBD | TBD | TBD | TBD |
| 0    | TBD | TBD | TBD | TBD | TBD |
| 1234 | TBD | TBD | TBD | TBD | TBD |
| **Mean** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

(1-seed DGX reference with the same code: seed 42 → val_bpb=1.06042,
size=16,739,873 bytes; size is over budget on DGX, see *Note on size* above.)
