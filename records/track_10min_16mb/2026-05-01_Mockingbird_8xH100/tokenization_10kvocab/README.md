# 10k vocab tooling — reviewer verification

Everything needed to inspect and reproduce the SP10240 CaseOps tokenization stack that mockingbird trained on. This subdir is appendix material — `train_gpt.py` and the seed logs in the parent directory remain the canonical submission.

## Layout

```
tokenization_10kvocab/
├── README.md                                    (this file)
├── tokenizer/
│   ├── fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model    ← actual tokenizer mockingbird used
│   ├── fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.vocab
│   ├── fineweb_10240_bpe.model                  ← base SP10240 BPE (no CaseOps reserves) for reference
│   ├── fineweb_10240_bpe.vocab
│   └── tokenizer_specs_sp10240.json             ← BPE training spec (skip_docs, vocab_size, etc.)
├── build/
│   ├── run_sp10240_build.sh                     ← one-command rebuild from FineWeb 10B docs
│   ├── run_sp10240_upload.sh                    ← HF upload helper (used to publish the dataset)
│   └── sp10240_build.log                        ← BPE training log from the actual build
├── caseops/
│   ├── lossless_caps.py                         ← CaseOps codec module (4 reserved operators)
│   ├── prepare_sp10240_caseops_data.py          ← end-to-end CaseOps tokenizer + dataset prep
│   ├── build_sp10240_caseops_local.sh           ← local rebuild driver
│   ├── upload_sp10240_caseops_to_hf.sh          ← HF upload driver
│   ├── download_sp10240_first80_from_hf.sh      ← partial-shard download (first 80)
│   ├── download_sp10240_full124_from_hf.sh      ← full 124-shard download
│   └── stream_pr1855_caseops_to_pod.sh          ← pod streaming helper used in this lane
└── notes/
    ├── 2026-04-30_10k_caseops_hf_lane.md        ← derivation note: how this lane was built
    └── 2026-04-30_claude_sp10240_bytefit_plan.md ← the byte-fit reasoning (why MLP3.75 not MLP4)
```

## Tokenizer

**Vocab size:** 10,240
**Variant:** SP10240 lossless-caps CaseOps with 4 reserved operator codepoints

The CaseOps-active tokenizer is `fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model`. It is derived from the same trainer spec that PR #1855 used for SP8192:

- BPE, byte fallback enabled
- split-digits enabled
- `nmt_nfkc` normalization
- no dummy prefix
- pad / bos / eos / unk ids = 0 / 1 / 2 / 3
- hard vocab limit disabled
- reserved ids: U+E001=4, U+E002=5, U+E003=6, U+E004=7 (the four CaseOps operators)
- training corpus: FineWeb 10B docs `[50000, end)` (val docs `[0, 50000)` excluded — `tokenizer_skip_docs=50000` in `tokenizer_specs_sp10240.json`)

The standard `fineweb_10240_bpe.model` is included alongside as a reference — it is the same BPE training run **without** CaseOps reserved operators (those four codepoints map to `<unk>` id 3). Useful for diff inspection of the embedding-table cost of reserving the four ops.

## CaseOps codec

`caseops/lossless_caps.py` is the encode/decode module. The four operators are inserted at preprocessing time to record case information losslessly so the BPE doesn't need to allocate vocab to capitalization. At eval time, decode reverses the operators to reconstruct the original text.

The `prepare_sp10240_caseops_data.py` script trains the CaseOps tokenizer when no compatible model is found and tokenizes FineWeb 10B end-to-end into the dataset shards. It is the single source of truth for how mockingbird's training data was produced.

## Dataset

The full preprocessed dataset (124 train shards + 1 val shard, ~5 GB) is published publicly on Hugging Face — too large to commit to git:

**https://huggingface.co/datasets/Frosty40/10k_golfer**

Reviewers can pull it with either of:

```bash
bash caseops/download_sp10240_full124_from_hf.sh    # all shards
bash caseops/download_sp10240_first80_from_hf.sh    # first 80 only — enough to repro the run
```

Both scripts use the standard HF CLI and require `huggingface_hub>=1.8.0`.

## Reproducing the tokenizer + dataset from scratch

If you don't trust the HF artifacts and want to rebuild:

```bash
# 1. Build the standard SP10240 BPE tokenizer (no CaseOps)
bash build/run_sp10240_build.sh

# 2. Re-train the lossless-caps CaseOps variant + tokenize FineWeb 10B end-to-end
bash caseops/build_sp10240_caseops_local.sh
```

Output lands at `data/datasets/fineweb10B_sp10240_caseops/...` matching the paths the training scripts expect.

## Why this is in a non-record PR

A non-record submission is the right venue to land the 10k vocab tooling: it gives reviewers full access to the tokenizer, the CaseOps codec, the build/upload scripts, and the derivation notes — even though mockingbird's BPB does not beat PR #1855. The same machinery applied to SP8192 would produce a near-record SmearGate-class run; we're documenting the SP10240 cost on otherwise-identical compression / phased-TTT machinery.

## Provenance

- Tokenizer file `fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model` size 401,915 B; the byte-identical copy used by all three mockingbird seeds is at `legs/2026-05-01_pr1855_sp10240_caseops_mlp375_late045_seed{0,1}_8x/tokenizers/` and `evidence/pod_pulls/8x_10320714983_20260501_sp10240_mlp375_late045_clean_submission_candidate/...` on the source repo.
- CaseOps module `lossless_caps.py` is the seed-42 lane copy; seeds 0 and 1 used byte-identical copies of the same module.
- Build log `sp10240_build.log` is the actual SentencePiece trainer output from the build that produced the standard SP10240 BPE.
