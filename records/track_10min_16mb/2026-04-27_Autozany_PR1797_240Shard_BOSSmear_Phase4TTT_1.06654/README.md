# Record Candidate: PR #1797 Lineage + 240-Shard Train Limit + BOS-Masked SmearGate + Phase-4 TTT

**val_bpb: 1.06654079** (3-seed mean, std 0.00122292) | **val_loss: 2.29152064 nats/token** | **max artifact: 15,950,966 bytes** | 8xH100 SXM | 599.6s train / 317.0s eval mean

This candidate is a conservative submission hardening of the strongest legal open lineage we reproduced locally: PR #1787 / PR #1797 CaseOps + SparseAttnGate + SmearGate + asymmetric LQER + legal score-before-update phased TTT.

The key reproducibility changes versus the upstream PR #1797 folder are `TRAIN_SHARD_LIMIT=240` and a BOS-masked SmearGate. Our GPU run had the full 10k-document validation shard but intentionally trained from the first 240 canonical CaseOps train shards. The code now applies that limit before rank sharding, so the same training subset is used even if the verifier has generated more train shards. SmearGate masks the previous-token residual term wherever the current token is the BOS document marker, preventing cross-document carry in both normal evaluation and TTT.

## Result

| Seed | Steps | Train shards | Pre-quant BPB | Quantized BPB | Post-TTT BPB | Artifact bytes | Train time | TTT eval time |
|------|------:|-------------:|--------------:|--------------:|-------------:|---------------:|-----------:|--------------:|
| 42 | 5047 | 240 | 1.06944908 | 1.07891097 | 1.06642781 | 15,950,222 | 599.600s | 356.7s |
| 314 | 5036 | 240 | 1.06882706 | 1.07790190 | 1.06537828 | 15,950,966 | 599.579s | 322.1s |
| 1234 | 5026 | 240 | 1.07098567 | 1.08042471 | 1.06781627 | 15,950,455 | 599.652s | 272.1s |
| **Mean** |  |  | **1.06975394** | **1.07907919** | **1.06654079** | **15,950,548** | **599.610s** | **317.0s** |
| **Std** |  |  | 0.00111113 | 0.00126979 | 0.00122292 | 381 | 0.038s | 42.5s |

The clean package uses phase-4 TTT. A prior pre-fix TTT sweep on the same 240-shard lineage found phase 4 slightly better than phase 3; after adding the BOS SmearGate mask, we reran the clean package with phase 4:

| TTT setting | BPB |
|-------------|----:|
| Pre-fix `TTT_LORA_LR=0.00005`, `PHASED_TTT_NUM_PHASES=3` | 1.06672217 |
| Clean BOS-masked `TTT_LORA_LR=0.00005`, `PHASED_TTT_NUM_PHASES=4` | **1.06642781** |

## Mechanism Stack

| Component | Role |
|-----------|------|
| CaseOps lossless tokenizer transform | Reduces capitalization fragmentation while scoring original UTF-8 bytes through the byte sidecar. |
| SparseAttnGate + PolarNS + MIN_LR + fused CE | PR #1787 / #1797 base stack. |
| BOS-masked SmearGate | Causal 1-token residual lookback gate with the previous-token term masked at BOS document boundaries. |
| LQER asymmetric rank-4 | Post-GPTQ low-rank recovery on top quantization-error tensors. |
| Phased TTT | Legal score-before-update per-document LoRA adaptation. |
| `TRAIN_SHARD_LIMIT=240` | Reproducibly pins the canonical train subset used by the measured run. |
| `TTT_LORA_LR=0.00005`, `PHASED_TTT_NUM_PHASES=4` | Best measured TTT settings for the 240-shard checkpoint. |

## Data Setup

Build the CaseOps train/validation shards from the canonical FineWeb doc stream as in PR #1797:

```bash
python3 ../../data/download_hf_docs_and_tokenize.py
python3 prepare_caseops_data.py \
  --docs ./fineweb10B_raw/docs_selected.jsonl \
  --out ./data/datasets/fineweb10B_sp8192_caseops/datasets \
  --sp ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

The record expects:

```text
data/datasets/fineweb10B_sp8192_caseops/datasets/
  tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
  datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
    fineweb_train_000000.bin
    ...
    fineweb_val_000000.bin
    fineweb_val_bytes_000000.bin
```

`TRAIN_SHARD_LIMIT=240` is applied in `train_gpt.py`; extra generated train shards are ignored for this candidate. The full validation shard is still used.

## Run Command

```bash
for SEED in 42 314 1234; do
NCCL_NET=Socket \
DATA_DIR=./data \
CASEOPS_ENABLED=1 \
TRAIN_SHARD_LIMIT=240 \
PHASED_TTT_PREFIX_DOCS=2000 \
PHASED_TTT_NUM_PHASES=4 \
TTT_LORA_LR=0.00005 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
MLP_CLIP_SIGMAS=12.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
MIN_LR=0.1 \
FUSED_CE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
TTT_WARM_START_A=1 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
SEED=$SEED \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > train_seed${SEED}.log 2>&1
done
```

## Requirements

The official CUDA environment already includes most dependencies. If installing manually:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn-interface sentencepiece triton numpy brotli
```

## Rule Notes

- Artifact is under the decimal 16,000,000-byte cap.
- Training stops under the 600s wall-clock cap.
- SmearGate masks BOS document boundaries in both `_forward_hidden(...)` and `forward_ttt(...)`, preventing previous-document residual carry into a new document.
- TTT evaluation is score-before-update and completed in 272.1-356.7s across the three measured seeds.
- Validation BPB is computed from the original raw UTF-8 byte sidecar, not token count.
- The model uses no external network or downloads during evaluation.
- This is a clean 3-seed end-to-end reproduction with the PR #1797 SmearGate boundary issue fixed. It is suitable as a leaderboard package against the public leaderboard, but it does not beat the stronger open PR #1797 frontier claim.

## Included Files

- `train_gpt.py` — self-contained training/evaluation script.
- `README.md` — methodology, run command, and rule notes.
- `submission.json` — 3-seed metadata.
- `train_seed42.log`, `train_seed314.log`, `train_seed1234.log` — clean rerun logs.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SentencePiece model.
- `lossless_caps.py` — bijective CaseOps transform.
- `prepare_caseops_data.py` — one-time CaseOps shard and byte-sidecar generation.
