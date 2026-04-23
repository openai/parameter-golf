# Record: SP8192 + CaseOps + Gated Attention + Quant Gate + Loop4-5 + Phased TTT — val_bpb 1.06549

**val_bpb: 1.06549** (3-seed mean, std=0.00070) | **val_loss: 2.33168 nats/token** (std=0.00152) | **~15.98 MB** | 8×H100 SXM | Phased TTT

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, phased TTT, 10-min train / 10-min eval budgets)

### Core table (phased TTT)

| Seed | Steps  | Pre-TTT BPB | Post-TTT BPB | TTT gain | TTT time | Artifact (bytes) |
|------|-------:|------------:|-------------:|---------:|---------:|-----------------:|
| 42   | 4854   | 1.07847     | 1.06610      | -0.01237 | 396.9s   | 15,978,834       |
| 0    | 4843   | 1.07719     | 1.06473      | -0.01247 | 399.3s   | 15,971,476       |
| 1234 | 4847   | 1.07811     | 1.06563      | -0.01248 | 395.5s   | 15,975,050       |
| **Mean** | **4848** | **1.07792** | **1.06549** | **-0.01244** | **397.2s** | **15,975,120** |
| **Std**  |          | 0.00066     | **0.00070** |          | 1.9s     | 3,698            |

### Supplemental diagnostics

| Seed | Post-EMA BPB (pre-quant) | Quantized BPB (no TTT) | Sliding/TTT BPB | val_loss (nats) | Train time | Eval time |
|------|-------------------------:|-----------------------:|----------------:|----------------:|-----------:|----------:|
| 42   | 1.06907                  | 1.07847                | 1.06610         | 2.33302         | 596.18s    | 396.9s    |
| 0    | 1.06779                  | 1.07719                | 1.06473         | 2.33002         | 596.17s    | 399.3s    |
| 1234 | 1.06872                  | 1.07811                | 1.06563         | 2.33199         | 596.08s    | 395.5s    |

All three seeds clear both 600s budgets (train + eval) and the 16,000,000-byte decimal artifact cap. 3-seed std is 0.00070 BPB ≈ 0.00181 nats, well under the 0.005-nat significance floor.

## Key Innovation — CaseOps tokenizer

CaseOps (`lossless_caps_caseops_v1`) is a **bijective**, character-level text transform applied before SentencePiece training. It removes English capitalization from the body of the text and records it as four operator tokens that become part of the BPE vocabulary as SentencePiece `user_defined_symbols`:

- `TITLE`  — next word is TitleCase
- `ALLCAPS` — next word (or region) is UPPERCASE
- `CAPNEXT` — next letter is capitalized
- `ESC` — escape for a literal operator-looking sequence

Because the transform is fully invertible (`decode_lossless_caps_v2(encode_lossless_caps_v2(s)) == s` for all strings), **no information is lost**. The SP model sees lowercase-normalized text, so the BPE merges allocate vocabulary around content instead of around case-duplicated variants ("the"/"The"/"THE" collapse to one surface form with operator prefixes). This reclaims ~0.005-0.006 nats per token on FineWeb.

**BPB is scored on ORIGINAL pre-transform UTF-8 bytes**, not on the transformed representation. The training pipeline emits per-token byte sidecar shards (`fineweb_val_bytes_XXXXXX.bin`, uint16 parallel to the val token shards) that record the canonical byte cost of each target position; eval sums those to get true bytes. This sidesteps the "bytes-per-token shift" concern: the score is on the same FineWeb text, just with a different tokenization front end.

```python
# Transform (character-level, bijective):
text   = "The quick brown FOX."
encode = encode_lossless_caps_v2(text)
# → "<TITLE>the quick brown <ALLCAPS>fox."
assert decode_lossless_caps_v2(encode) == text
```

## Changes from PR #1530 / PR #1626 baseline

| Component | PR #1530 | This submission |
|-----------|---------:|----------------:|
| Tokenizer | SP8192 FineWeb BPE | SP8192 FineWeb BPE + CaseOps operator tokens |
| BPB accounting | uniform piece.encode() | per-token byte sidecar (original bytes) |
| Attention out-gate | — | **learned `gate` scalar per head** (init_std=0.005) |
| Attention quant gate | — | **quant-time gate scaling** (~40 KB artifact savings) |
| Depth recurrence | — | Loop4-5 (layers 4-5 run twice) |
| TTT | multi-phase SGD score-first | multi-phase SGD score-first (kept) |
| Clip sigmas | (MLP=12, ATTN=13) | (MLP=12, ATTN=13) |
| Embed bits | 7 | 7 |

Net: **-0.00644 BPB / -0.01665 nats vs PR #1626 (1.07193)** ≈ **3.3× the 0.005-nat record bar**.

## Rule compliance

- **Artifact ≤ 16,000,000 bytes DECIMAL**: all 3 seeds ≤ 15,978,834 bytes (21+ KB headroom).
- **train_time ≤ 600s**: all 3 seeds 596.1-596.2s.
- **total_eval_time ≤ 600s**: all 3 seeds 395.5-399.3s.
- **Score-first TTT**: phased TTT snapshots the pre-update score on each chunk BEFORE the LoRA adapter step (per-doc LoRA reset via `reusable_lora.reset()`), satisfying Issue #1017 Condition 3.
- **BPB on original bytes**: per-token byte sidecar encodes the canonical UTF-8 byte count of each val position; transformed text is only the tokenization front end.
- **Reversibility**: `decode_lossless_caps_v2(encode_lossless_caps_v2(x)) == x` checked by the bijectivity test (see `tools/test_caseops_bijectivity.py` in the author's working tree; the transform is also verifiable in-repo via `lossless_caps.py`).
- **No val data in training**: training uses only `fineweb_train_*.bin` shards.
- **No external network during eval**: self-contained; tokenizer + transform ship with the submission.

## Requirements

```bash
# PyTorch 2.9.1+cu128 (or compatible) + Flash Attention 3 for Hopper:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn-interface sentencepiece triton numpy
# Python ≥ 3.12 (minified f-strings use PEP 701 nested same-type quotes).
```

## Data setup (run ONCE)

The submission ships with the trained CaseOps SentencePiece model (`tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`) and the bijective transform module (`lossless_caps.py`). Train/val shards and the byte sidecar are rebuilt from the canonical FineWeb-10B doc stream produced by `data/download_hf_docs_and_tokenize.py` in the repo root:

```bash
# 1. Ensure docs_selected.jsonl exists (standard setup step for the repo).
python3 ../../data/download_hf_docs_and_tokenize.py  # or point to existing file

# 2. Build CaseOps-transformed shards + val byte sidecar.
python3 prepare_caseops_data.py \
    --docs ./fineweb10B_raw/docs_selected.jsonl \
    --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Output layout (what `train_gpt.py` expects with `CASEOPS_ENABLED=1`):

```
data/datasets/fineweb10B_sp8192_caseops/datasets/
  tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
  datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
    fineweb_train_000000.bin
    fineweb_train_000001.bin
    ...
    fineweb_val_000000.bin
    fineweb_val_bytes_000000.bin
```

### Reproduction sanity check (run after step 2)

Each shard must contain `BOS_ID=1` at the start of every document — `train_gpt.py`'s phased TTT eval path (`_find_docs`) requires it. Quick check on the first val shard:

```python
python3 -c "
import numpy as np
d = np.fromfile('data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin', dtype=np.uint16)
# First 256 int32 header slots = 512 uint16 slots; tokens start after.
tokens = d[512:]
bos_count = int((tokens == 1).sum())
print(f'BOS markers in val shard: {bos_count}  (must be > 0)')
assert bos_count > 0, 'prepare_caseops_data.py is broken — re-run with BOS prepend'
"
```

If `bos_count == 0`, the prep script is out of date — pull the latest `prepare_caseops_data.py` from this folder (the SP tokenizer reserves IDs 0–7 for special + CaseOps operator tokens, so the prep script must explicitly prepend `BOS_ID=1` to each doc; the eval path's `_find_docs` has no fallback for missing BOS markers).

## Run command (3-seed reproduction)

```bash
for SEED in 42 0 1234; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 \
  GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
  GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

## Lineage

- Builds on **PR #1530** (samacqua) SP8192 + Loop4-5 + parallel residuals + phased TTT stack.
- Borrows **PR #1626** (ours) multi-phase SGD phased TTT schedule (3 phases on the first 2000 val docs).
- Adopts **CaseOps** reversible case preprocessing + per-token byte sidecar BPB accounting from **PR #1729** (romeerp), which established that bijective text preprocessing that preserves byte-level BPB is rule-compliant.
- Adds the learned `gated_attn` out-gate (init_std=0.005) + quant-gate scaling (`GATED_ATTN_QUANT_GATE=1`) which recovers the ~15-40 KB of artifact overhead introduced by the new control tokens and sidecar path, keeping all three seeds under the 16 MB decimal cap.

## Credits

- @samacqua — PR #1530 base stack.
- @romeerp — PR #1729 CaseOps concept + byte sidecar accounting.
- @bigbag — PR #1493 merged SOTA (1.0810).
- @MarioPaerle — PR #1667 AttnOutGate pattern.

## Included files

- `train_gpt.py` — main training script (131,887 bytes pre-minify).
- `submission.json` — metadata.
- `README.md` — this file.
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — 3-seed run logs.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SentencePiece model (366.5 KB).
- `lossless_caps.py` — bijective CaseOps transform (used by `prepare_caseops_data.py`).
- `prepare_caseops_data.py` — one-time data prep script that tokenizes FineWeb via CaseOps + emits the per-token byte sidecar.
