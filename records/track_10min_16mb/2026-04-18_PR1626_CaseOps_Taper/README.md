# Record: CaseOps Tokenizer + Mild WD Taper

**val_bpb: 1.06780** (3-seed mean, std 0.00037) | **2.33674 nats** | **~15.94 MB** | 8xH100 SXM, ~596s train + ~488s TTT eval

This record builds directly on PR #1626's legal multi-phase TTT stack and adds two changes:

1. A lossless case-operations tokenizer/data export, hosted publicly at [romeerp/parameter-golf-caseops-v1](https://huggingface.co/datasets/romeerp/parameter-golf-caseops-v1)
2. A mild late Muon weight-decay taper (`WD_TAPER_START_FRAC=0.70`, `WD_TAPER_FINAL_MULT=0.50`)

The tokenizer/data are not checked into this PR; they are downloaded from the Hugging Face dataset above with the included `cached_challenge_fineweb.py`.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, Phased TTT)

| Seed | Steps | Pre-Quant BPB | Quantized BPB | **Post-TTT BPB** | Artifact |
|------|-------|---------------|---------------|------------------|----------|
| 0 | 4,921 | 1.07032992 | 1.08152131 | **1.06805820** | 15,932,307 |
| 42 | 4,866 | 1.07065549 | 1.08171495 | **1.06806595** | 15,935,802 |
| 1234 | 4,870 | 1.06971629 | 1.08036614 | **1.06727867** | 15,943,106 |
| **Mean** | | **1.07023390** | **1.08120080** | **1.06780094** | **15,937,072** |

## Supplemental Diagnostics

| Seed | Pre-Quant BPB | Quantized BPB | Post-TTT BPB | val_loss (nats) | Code size | Total | Train time | Eval time |
|------|---------------|---------------|--------------|-----------------|-----------|-------|------------|-----------|
| 0 | 1.07032992 | 1.08152131 | 1.06805820 | 2.33730724 | 28,320 | 15,932,307 | 596.1s | 488.8s |
| 42 | 1.07065549 | 1.08171495 | 1.06806595 | 2.33732420 | 30,985 | 15,935,802 | 596.1s | 482.0s |
| 1234 | 1.06971629 | 1.08036614 | 1.06727867 | 2.33560135 | 30,985 | 15,943,106 | 596.1s | 494.6s |

## Tokenizer: Lossless Case-Ops


The tokenizer uses a lossless text transform, `lossless_caps_caseops_v1`, that factorizes text into:

- a lowercase lexical stream
- a tiny reserved capitalization side-channel

Reserved control symbols:

- `TITLE`
- `ALLCAPS`
- `CAPNEXT`
- `ESC`

Behavior over maximal ASCII alphabetic runs:

- lowercase words stay lowercase
- `TitleCase` becomes `TITLE + lowercase(word)`
- `ALLCAPS` becomes `ALLCAPS + lowercase(word)`
- mixed-case words use sparse `CAPNEXT` markers
- control symbols themselves are escaped losslessly with `ESC`

Examples:

- `The NASA Launch` -> `TITLE the ALLCAPS nasa TITLE launch`
- `iPhone OpenAI` -> `i CAPNEXT phone TITLE open CAPNEXT a CAPNEXT i`

The point is to remove redundant case variation from the main lexical token stream without losing any information. At evaluation time, BPB is still charged against the original raw UTF-8 bytes, not the transformed stream.

## Why This Is Still Real BPB

The exporter writes validation byte sidecars:

- `fineweb_val_000000.bin`
- `fineweb_val_bytes_000000.bin`

The trainer then loads the byte sidecar directly and reports:

- `val_bpb:byte_sidecar:enabled`

So scoring is done against exact original-byte counts rather than tokenized/transformed length. This preserves a true byte-level objective even though the tokenizer uses a reversible preprocessing transform.

## Main Idea

The core intuition is that standard `sp8192` still makes the model represent a lot of casing variation directly in the lexical stream. By transforming capital tokens to sentinel+lowercase, we free up vocabulary for more useful tokens, and possibly may provide some sort of inductive bias helping the model learn capitalization as a rule. The intuition behind tapered weight decay is that the purpose of a high weight decay in this challenge is to make weights more compressible by reducing entropy. While this is necessary at the beginning of training, near the end of training weights tend to be more settled, and therefore unlikely to spike and become outliers, so reducing the weight decay in favor of a better optimization may provide a benefit.

This submission keeps the legal PR #1626 architecture and phased-TTT evaluation path, but swaps in the lossless case-ops tokenizer/data export above. On top of that, it adds a mild late taper on Muon weight decay:

- full Muon WD until 70% of training
- then linearly taper to 50% of the base WD by the end

This combination improves pretrained BPB and quantized phased-TTT BPB while staying under the 16 MB artifact cap.

## Changes from PR #1626

| Change | Source | Effect |
|--------|--------|--------|
| CaseOps tokenizer + exported dataset | **Novel (this work)** | cleaner lexical stream, exact byte-sidecar eval |
| Validation byte-sidecar BPB accounting | **Novel (this work)** | exact raw-byte metric with transformed tokenizer |
| Mild late Muon WD taper (`0.70 -> 0.50`) | This work | small but consistent BPB win |
| Public HF dataset/tokenizer download path | This work | reproducible on fresh pods |

## Rule Compliance

- **Causal:** all scoring remains autoregressive / causal.
- **Normalized:** scoring uses standard cross-entropy over the full vocabulary.
- **Score-before-update:** phased TTT remains PR #1626 style legal score-first TTT.
- **Single pass:** no rescoring of validation tokens.
- **No validation during training:** training uses only train shards.
- **Full validation split:** the full exported validation split is scored.
- **Byte accounting:** BPB is computed from the validation byte sidecar, which exactly matches the raw UTF-8 byte total of the exported docs.

## Public Artifacts

- Dataset + tokenizer: [romeerp/parameter-golf-caseops-v1](https://huggingface.co/datasets/romeerp/parameter-golf-caseops-v1)

The HF dataset repo contains:

- the caseops tokenizer model / vocab
- the exported train shards
- the exported validation shard
- the validation byte-sidecar shard
- `manifest.json`

## Requirements

Python >= 3.12. Flash Attention 3 (Hopper) required.

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install -r requirements.txt
```

## Run Instructions

Run the commands in this section from the record directory:

```bash
cd records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper
```

Prepare the public Hugging Face tokenizer + dataset on a fresh pod:

```bash
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python3 cached_challenge_fineweb.py \
  --variant sp8192_lossless_caps_caseops_v1_reserved \
  --train-shards 80
```

This downloads both:

- the exported caseops dataset shards
- the caseops SentencePiece tokenizer artifact

from [romeerp/parameter-golf-caseops-v1](https://huggingface.co/datasets/romeerp/parameter-golf-caseops-v1).

From this record directory, train + quantize + phased eval for one seed:

```bash
NCCL_NET=Socket \
SEED=0 \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
DATASETS_DIR=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > train_seed0.log 2>&1
```

The submission script itself contains the intended defaults, including:

- `PHASED_TTT_ENABLED=1`
- `PHASED_TTT_NUM_PHASES=3`
- `EMBED_BITS=7`
- `EMBED_CLIP_SIGMAS=15.0`
- `MLP_CLIP_SIGMAS=12.0`
- `WD_TAPER_START_FRAC=0.70`
- `WD_TAPER_FINAL_MULT=0.50`

## Rebuilding the Tokenizer / Dataset

The PR includes the actual Python sources used to create the tokenizer and exported dataset:

- `download_hf_docs_and_tokenize.py`
- `cached_challenge_fineweb.py`
- `lossless_caps.py`
- `tokenizer_specs_export_caseops_v1_reserved_only.json`

Tokenizer export spec:

```json
{
  "tokenizers": [
    {
      "name": "sp_bpe_8192_lossless_caps_caseops_v1_reserved",
      "dataset_suffix": "sp8192_lossless_caps_caseops_v1_reserved",
      "vocab_size": 8192,
      "text_transform": "lossless_caps_caseops_v1",
      "reserve_text_transform_controls": true,
      "model_prefix": "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved"
    }
  ]
}
```

To rebuild the HF artifacts from public docs instead of downloading the prebuilt dataset:

```bash
python3 download_hf_docs_and_tokenize.py \
  --repo-id willdepueoai/parameter-golf \
  --remote-root datasets \
  --output-root data/caseops_export_rebuilt \
  --tokenizer-config tokenizer_specs_export_caseops_v1_reserved_only.json \
  --max-train-shards 80
```

That program imports the lossless transform implementation from `lossless_caps.py`, trains the SentencePiece model with the case-ops transform, exports the `80` train shards, exports the validation shard, and writes the validation byte-sidecar needed for exact BPB scoring.

## Included Files

- `train_gpt.py`
- `requirements.txt`
- `README.md`
- `cached_challenge_fineweb.py`
- `download_hf_docs_and_tokenize.py`
- `lossless_caps.py`
- `tokenizer_specs_export_caseops_v1_reserved_only.json`
- `train_seed0.log`
- `train_seed42.log`
- `train_seed1234.log`
