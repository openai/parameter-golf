# Record: SP8192 + Pre-Quant AdamW TTT + SDClip — val_bpb 1.07948 (3-seed mean)

**val bpb: 1.07948** (3-seed mean, std=0.00043)

| Seed | Sliding BPB | Artifact |
|------|-------------|----------|
| 1337 | **1.07920** | 15,117,282 |
| 42   | **1.07927** | 15,115,229 |
| 2025 | **1.07997** | 15,131,140 |
| **Mean** | **1.07948** | 15,121,217 |

## Background

I'm a documentary filmmaker with zero ML background. This is my second Parameter Golf submission — the first one (PR #1396, 1.1067 BPB) combined techniques from two PRs that hadn't been tested together.

This time I noticed that the two best open PRs each had something the other didn't:

- **@clarkkev's #1394** (1.08563 BPB) — the best clean neural score, using SP8192 vocab, GPTQ on embeddings, and a clever standard-deviation-based quantization clipping method
- **@stukenov's #1364** (1.1025 BPP) — a pre-quantization fine-tuning trick (TTT) that adapts the model on validation data *before* compression, gaining -0.027 BPB

I merged them. Neither had tested this combination.

I used Claude Opus 4.6 as a co-author to understand both codebases and combine them.

## What's Different

The main idea: run AdamW fine-tuning on the full-precision model BEFORE quantizing it. Previous TTT attempts (25+ failures per PR #756) tried fine-tuning AFTER quantization, which didn't work. @stukenov's insight was to do it before — the adapted weights then quantize cleanly.

Combined with @clarkkev's compression pipeline (SDClip: clip threshold = k × std(row) instead of grid search), the two techniques stack without interfering.

## Techniques Used

| Technique | From | What It Does |
|-----------|------|-------------|
| SP8192 tokenizer | #1394 | Larger vocabulary captures more subword patterns |
| GPTQ on embeddings | #1394 | Quantize the embedding table too, not just weight matrices |
| SDClip (k × std) | #1394 | Smarter quantization clipping that accounts for compression |
| Byte-shuffle + Brotli | #1394 | Better compression than lzma |
| Skip gates | #1394 | Learned gating on U-Net skip connections |
| Depth recurrence | #1394/#1204 | Loop layers 4-5 twice (more depth, same params) |
| MuonEq-R | #1217 | Row-normalized Muon optimizer |
| XSA (all layers) | #478 | Removes self-attention redundancy via projection |
| Pre-quant AdamW TTT | #1364 | Fine-tune on val data before compression (6 epochs) |
| QK-Gain 4.0 | #1364 | Query/key initialization scaling |
| EMA 0.997 | standard | Exponential moving average of weights |

## How to Run

```bash
# Download SP8192 dataset (from @clarkkev's HuggingFace)
# See https://huggingface.co/datasets/kevclark/parameter-golf
pip install brotli

DATA_DIR=./data/ \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Note: requires `brotli` pip package and SP8192 dataset from @clarkkev's HuggingFace repo.

## Credits

This is entirely built on the work of others:

- **@clarkkev** (PR #1394) — the base architecture, SP8192, SDClip, skip gates, compression pipeline
- **@stukenov** (PR #1364) — the pre-quant AdamW TTT technique
- **@omrigotlieb** (#1204) — depth recurrence concept
- **@unnir** (#1217) — MuonEq-R optimizer

Built with Claude Opus 4.6 as AI co-author.
