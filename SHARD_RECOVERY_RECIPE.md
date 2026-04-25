# Parameter Golf — `fineweb10B_sp8192nb` Shard Recovery Recipe (CORRECTED 2026-04-20)

**Date compiled:** 2026-04-20 (supersedes the earlier version of this file)
**Purpose:** Exact recipe to recreate the validated banked environment so the 1.1803 BPB result is reproducible before the 2026-04-30 submission deadline.
**Status:** Recipe recovered from sprint transcript. Reproduction path is a git clone + one tokenization pass with `--reuse-sp-model`.

---

## Retraction notice

An earlier version of this file claimed the banked shards and tokenizer came from `sysrekt/parameter-golf`. **That is wrong.** The sysrekt download happened on the 2026-04-15 pod, which was later wiped. The banked 1.1803 run was trained on shards and a tokenizer that were **built locally** on Unwindology pods on 2026-04-16. This version of the recipe reflects the actual sprint history as reconstructed from `AI_Export_2026-04-17T02-53-02.md` lines 2510-19390.

---

## TL;DR

The banked inputs were produced in two stages on 2026-04-16:

1. **L40S pod `b587cee1e818`, 13:30-14:45 UTC.** SentencePiece BPE tokenizer trained from scratch on `docs_selected.jsonl` via `data/download_hf_docs_and_tokenize.py` with a custom `tokenizer_specs.json`. Output: `fineweb_8192_bpe.model` + 128 train shards. File renamed post-build to `fineweb_8192nb_bpe.model` to match the champion script's hardcoded path. "nb" has **no technical meaning** in our build — it's a naming convention chosen to match the sysrekt-style suffix that the champion script expects.

2. **A40 pod `47e4c7173b05`, ~18:01 local.** The tokenizer `.model` + `.vocab` were force-added and pushed to `github.com/Unwindology/parameter-golf` as commit **`2083fd9`** on main. The 25GB of shards were NOT pushed (too large).

3. **H100 SXM spot pod, ~18:19 local (banked run).** Pod cloned `github.com/Unwindology/parameter-golf` (bringing the tokenizer with it), then re-sharded `docs_selected.jsonl` using `--reuse-sp-model 8192=data/tokenizers/fineweb_8192nb_bpe.model` — this SKIPS BPE training, reuses the already-built tokenizer, and regenerates 128 byte-identical train shards + 1 val shard on the H100. The banked 1.1803 run consumed these.

**Reproducibility implication.** The tokenizer is authoritative and recoverable from GitHub commit `2083fd9`. Shards are deterministic from (tokenizer, docs_selected.jsonl, script version), so re-running `--reuse-sp-model` on any pod should produce byte-identical shards and byte-identical val token count (40,541,184).

---

## 1. What "nb" actually means

**Nothing technically — it is a naming convention Unwindology borrowed from sysrekt's file layout.**

The sprint transcript confirms this directly at lines 12494-12495:

> "the spec format I'm seeing only shows name, dataset_suffix, vocab_size. It doesn't show a 'nb' flag anywhere. The 'nb' in sp8192nb was something you specifically added yesterday."

And at lines 12533-12545 the assistant wrote the custom `tokenizer_specs.json`:

```json
{
  "tokenizers": [
    {
      "name": "sp_bpe_8192",
      "dataset_suffix": "sp8192nb",
      "vocab_size": 8192
    }
  ]
}
```

The `dataset_suffix: sp8192nb` was chosen solely so the output directory would match the champion script's hardcoded `fineweb_8192nb_bpe.model` path. It does not flip a SentencePiece training flag. The tokenizer was built with the script's default SentencePiece options: `byte_fallback=True`, `add_bos=False`, `add_eos=False`, `add_dummy_prefix=False`.

(Earlier analysis speculated "nb" meant "no-boundary" or "no-byte-fallback." Those guesses are wrong for Unwindology's build. Sysrekt may have used the suffix to mean something specific in their own pipeline, but Unwindology's use of it is purely cosmetic path-matching.)

---

## 2. Exact generation history

### 2026-04-15 — sysrekt initial pull (NOT the banked source)

Pod `55f242843a9b`. Transcript lines 2518-2729. Downloaded 1 train shard (`fineweb_train_000000.bin`) + 1 val shard + `fineweb_8192nb_bpe.model` + `.vocab` directly from `sysrekt/parameter-golf` via `hf_hub_download`. Runs labeled `train_shards:1, tokens:40203264`. Used only for proof-of-concept L40S/A100 runs. Pod wiped before the banked run.

### 2026-04-16 13:30-14:45 UTC — local tokenizer + shard build

Pod `b587cee1e818` (L40S, $6.90/hr). Transcript lines 12603-13014.

Build command that actually succeeded (line 12652):

```bash
cd /workspace/parameter-golf && HF_HOME=/workspace/.hf_cache nohup python3 \
  data/download_hf_docs_and_tokenize.py \
  --output-root /workspace/parameter-golf/data \
  --skip-byte \
  > /workspace/tokenize_sp8192.log 2>&1 &
```

The `HF_HOME=` inline override was required because the container `/root/.cache` (40GB) was too small for the 48GB `docs_selected.jsonl`; the first attempt failed with `RuntimeError: Data processing error: File reconstruction error: Internal Writer Error: Background writer channel closed` (transcript lines 12611-12631).

BPE training consumed ~15 minutes (12.4M sentences → 37.4M tokens → 8192 merges). Sharding consumed ~60 more minutes (transcript line 12991: "Full tokenization took ~75 minutes end-to-end").

Outputs on disk at 14:45 UTC:
- `/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model` (363KB)
- `/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.vocab` (111KB)
- `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192nb/fineweb_train_000000.bin` through `fineweb_train_000127.bin` (128 × 200MB = ~25GB)
- `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192nb/fineweb_val_000000.bin` (81MB)

Rename step (transcript lines 12995-13013):

```bash
mv /workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
   /workspace/parameter-golf/data/tokenizers/fineweb_8192nb_bpe.model
mv /workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.vocab \
   /workspace/parameter-golf/data/tokenizers/fineweb_8192nb_bpe.vocab
```

This rename exists purely so the champion script's hardcoded `TOKENIZER_PATH` default of `./data/tokenizers/fineweb_8192nb_bpe.model` resolves. There is nothing technically `nb` about the file — the bytes are identical before and after the `mv`.

### 2026-04-16 ~18:01 local — tokenizer committed to GitHub

Pod `47e4c7173b05` (A40, transcript lines 19311-19391). Quota-constrained on the workspace volume, so cloned the repo to container disk `/root/parameter-golf`, copied the tokenizer in, force-added, pushed:

```bash
mkdir -p data/tokenizers
cp /root/fineweb_8192nb_bpe.model data/tokenizers/
cp /root/fineweb_8192nb_bpe.vocab data/tokenizers/
git add -f data/tokenizers/fineweb_8192nb_bpe.model data/tokenizers/fineweb_8192nb_bpe.vocab
git commit -m "Add SP8192nb tokenizer - skip BPE training on future pods"
git push origin main
```

Push output (transcript lines 19378-19391):

```
[main 2083fd9] Add SP8192nb tokenizer - skip BPE training on future pods
 2 files changed, 8192 insertions(+)
 create mode 100644 data/tokenizers/fineweb_8192nb_bpe.model
 create mode 100644 data/tokenizers/fineweb_8192nb_bpe.vocab
To https://github.com/Unwindology/parameter-golf.git
   ab0cc40..2083fd9  main -> main
```

**This commit `2083fd9` is authoritative for the banked tokenizer.** The md5 `02df22232f3de8bbf668245160fc77bf` is the SHA-identifiable content at this commit.

### 2026-04-16 ~18:19 local — H100 spot pod shards + banked run

Pod startup command (transcript line 19570): auto-clones `github.com/Unwindology/parameter-golf` (bringing tokenizer with it), then runs the sharding pipeline with `--reuse-sp-model` to skip BPE retraining:

```bash
cd /workspace/parameter-golf && HF_HOME=/workspace/.hf_cache python3 \
  data/download_hf_docs_and_tokenize.py \
  --output-root data \
  --skip-byte \
  --reuse-sp-model 8192=data/tokenizers/fineweb_8192nb_bpe.model \
  > /workspace/tokenize.log 2>&1 &
```

`--reuse-sp-model` is the key flag. It tells the script to skip the non-deterministic BPE training step and use the already-trained tokenizer directly. Because tokenization of a fixed corpus with a fixed tokenizer by a fixed script is deterministic, this reproduces the 128 train shards + 1 val shard byte-identically (up to document iteration order, which the script fixes).

Banked `train_seed1337.log` header confirms this is what the champion run consumed:

```
tokenizer_path=./data/tokenizers/fineweb_8192nb_bpe.model
train_loader:dataset:fineweb10B_sp8192nb train_shards:128
val_loader:shards pattern=./data/datasets/fineweb10B_sp8192nb/fineweb_val_*.bin tokens:40541184
```

128 shards and 40,541,184 val tokens are the Unwindology local-build signature. Sysrekt's own published sp8192nb shards produce different counts (40,203,264 val tokens on the 2026-04-15 runs).

---

## 3. Why tonight's reproduction diverged +0.07 BPB

Tonight's attempt in `AI_Export_2026-04-21T01-02-27.md` used `ln -sfn fineweb10B_sp8192 fineweb10B_sp8192nb`, pointing the nb path at a directory that was **not** the banked nb shards. Those shards were tokenized with a different tokenizer (the `_sp8192` non-nb variant from the contest's `cached_challenge_fineweb.py --variant sp8192` path), and the champion script then loaded `fineweb_8192nb_bpe.model` as the BPB tokenizer. So the training loop saw token IDs produced by tokenizer-A while the BPB byte-length table was built from tokenizer-B — a silent mismatch that produces a systematic per-checkpoint shift.

Log-header fingerprint comparison:

|                             | Banked 2026-04-16       | Tonight 2026-04-20      |
|-----------------------------|-------------------------|-------------------------|
| `tokenizer_path`            | `fineweb_8192nb_bpe.model` | `fineweb_8192nb_bpe.model` |
| `train_loader:dataset`      | `fineweb10B_sp8192nb`   | `fineweb10B_sp8192` (symlinked) |
| `val_loader:tokens`         | 40,541,184              | 40,540,160              |
| `val_bpb step 0` (random)   | 3.4876                  | 3.6708                  |
| `val_bpb step 1000`         | 1.3201                  | 1.3919                  |
| `val_bpb step 4000`         | 1.2362                  | 1.3048                  |

The step-0 gap with random weights is 0.1832 BPB. No training has happened yet, so the only possible sources are (a) different validation tokens or (b) different byte-length normalization. The symlink swap causes both simultaneously.

---

## 4. Recovery procedure (exact)

This reproduces the banked environment on any fresh 8xH100 SXM pod using only `github.com/Unwindology/parameter-golf` and the public `willdepueoai/parameter-golf` docs file.

### Step 1 — clone Unwindology repo (brings tokenizer with it)

```bash
cd /workspace
git clone https://github.com/Unwindology/parameter-golf.git
cd parameter-golf
git checkout 2083fd9   # or any later commit that preserves data/tokenizers/fineweb_8192nb_bpe.model
ls -lh data/tokenizers/fineweb_8192nb_bpe.model
md5sum data/tokenizers/fineweb_8192nb_bpe.model
# Expected: 02df22232f3de8bbf668245160fc77bf
```

If the md5 matches, the banked tokenizer is recovered byte-identical. If it doesn't, someone rewrote that blob in the repo history and we need to look at older commits.

### Step 2 — re-shard using `--reuse-sp-model`

```bash
pip install huggingface_hub sentencepiece datasets tqdm --break-system-packages -q
cd /workspace/parameter-golf
mkdir -p data

HF_HOME=/workspace/.hf_cache python3 \
  data/download_hf_docs_and_tokenize.py \
  --output-root data \
  --skip-byte \
  --reuse-sp-model 8192=data/tokenizers/fineweb_8192nb_bpe.model \
  > /workspace/tokenize.log 2>&1 &

tail -f /workspace/tokenize.log
```

Expected wallclock: 30-40 minutes on 224 vCPU H100 pod (no BPE training phase; pure tokenization + shard writing).

### Step 3 — verify shards match banked

```bash
ls /workspace/parameter-golf/data/datasets/fineweb10B_sp8192nb/*.bin | wc -l
# Expected: 129 (128 train + 1 val)

md5sum /workspace/parameter-golf/data/datasets/fineweb10B_sp8192nb/fineweb_val_000000.bin
# Should be consistent across reproduction runs on the same script version.
```

### Step 4 — rerun banked command unchanged

```bash
cd /workspace/parameter-golf
SEED=1337 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
  VOCAB_SIZE=8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192nb_bpe.model \
  DATA_PATH=./data/datasets/fineweb10B_sp8192nb \
  WARMDOWN_ITERS=1200 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee /workspace/logs/V2_repro_local_rebuild_seed1337_8xH100_600s.log
```

Watch for:
- Header line `train_loader:dataset:fineweb10B_sp8192nb train_shards:128`
- `val_loader ... tokens:40541184`
- `step:0/20000 val_bpb:3.4876` (banked step-0 fingerprint)

If all three match, the input layer is identical to banked. Any divergence from there is trajectory drift (stochastic training order, dropout RNG, nondeterministic CUDA kernels) and should be within ±0.003 BPB of 1.18034.

---

## 5. Known irreducible non-determinism

**The tokenizer md5 `02df22232f3de8bbf668245160fc77bf` is NOT reproducible from scratch** — SentencePiece BPE training has tie-breaking nondeterminism across runs. The `--reuse-sp-model` recipe above sidesteps this by always using the committed tokenizer from GitHub rather than retraining. If GitHub commit `2083fd9` ever becomes unavailable, the tokenizer cannot be rebuilt byte-for-byte, only an equivalent that should train to within ±0.010 BPB of banked.

Mitigation: mirror the tokenizer file to at least one secondary location (S3, Zenodo, or another GitHub repo) before the submission deadline. Record its md5 in `submission.json` under an `environment.tokenizer_md5` key so a verifier has a fixed reference.

---

## 6. What to fix in the submission artifacts

1. **`Parameter_Golf_Submission_DRAFT_2026-04-17/README.md`** — the Reproduction block is wrong. It currently instructs the reader to run `MATCHED_FINEWEB_REPO_ID=willdepueoai/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192` and then symlink `_sp8192` to `_sp8192nb`. The `willdepueoai/parameter-golf` repo does not publish an sp8192 variant (transcript line 41 confirms this failed), and the symlink produces the +0.07 BPB divergence documented above. Replace the Reproduction block with the three-step procedure from §4.

2. **`submission.json`** — add an `environment` block:

```json
"environment": {
  "repo": "https://github.com/Unwindology/parameter-golf",
  "commit": "2083fd9",
  "tokenizer_md5": "02df22232f3de8bbf668245160fc77bf",
  "source_corpus_repo": "willdepueoai/parameter-golf",
  "source_corpus_file": "docs_selected.jsonl",
  "shard_count_train": 128,
  "val_tokens": 40541184,
  "build_command": "python3 data/download_hf_docs_and_tokenize.py --output-root data --skip-byte --reuse-sp-model 8192=data/tokenizers/fineweb_8192nb_bpe.model"
}
```

3. **Add a `environment/` directory** to the submission draft containing:
   - `shard_checksums.txt` (md5 of every `.bin` file produced by §4 Step 2)
   - `tokenizer_checksum.txt` (md5 of `fineweb_8192nb_bpe.model`)
   - `docs_selected_checksum.txt` (md5 of the HF-downloaded `docs_selected.jsonl`)

A verifier that follows §4 and the checksums in the `environment/` directory has a deterministic bit-for-bit path to the banked inputs, except for the tokenizer, which is byte-authoritative via GitHub commit `2083fd9`.

---

## 7. Sources used to assemble this recipe

- `AI_Export_2026-04-17T02-53-02.md` (sprint transcript):
  - Lines 2510-2736: 2026-04-15 sysrekt download (initial testbed, later wiped).
  - Lines 12100-13014: 2026-04-16 local tokenizer + shard build on L40S pod `b587cee1e818`.
  - Lines 19311-19391: 2026-04-16 tokenizer push to GitHub commit `2083fd9` on A40 pod `47e4c7173b05`.
  - Lines 19502-19606: 2026-04-16 H100 spot pod spin-up and `--reuse-sp-model` re-sharding.
- `Parameter_Golf_Submission_DRAFT_2026-04-17/train_seed1337.log` — banked log header confirming `train_shards:128, tokens:40541184, step 0 val_bpb=3.4876`.
- `uploads/AI_Export_2026-04-21T01-02-27.md` — tonight's reproduction showing symlink-driven divergence fingerprint.
- `uploads/train_breadcrumb_recur_ema_stochdepth.py` — `build_sentencepiece_luts` confirms BPB byte-length table is built from the tokenizer passed via `TOKENIZER_PATH`, not from shard metadata, so tokenizer/shard mismatch is silent at load time.
