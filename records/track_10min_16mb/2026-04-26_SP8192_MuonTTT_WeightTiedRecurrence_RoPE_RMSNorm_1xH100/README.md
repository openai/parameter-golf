# SP8192 MuonTTT WeightTied Recurrence — RoPE + RMSNorm + LeakyReLU²

**val_bpb: 1.4305** (seed 2025, post-TTT int8+zlib roundtrip) | **10.46 MB** artifact | 1×H100 SXM 80GB

> Seeds 1337 and 42 pending. Table will be updated with 3-seed mean.

## Results (1×H100 80GB SXM)

| Seed | step_avg | steps | Pre-quant bpb | **Post-TTT bpb** | TTT steps | TTT time | Artifact |
|------|----------|-------|--------------|-----------------|-----------|----------|----------|
| 2025 | 489.06ms | 1,227 | 1.4115 | **1.4305** | 3 (lr=3e-4) | 160s | 10,457,841 |
| 1337 | pending | — | — | — | — | — | — |
| 42   | pending | — | — | — | — | — | — |

> **Note on step time**: 489ms/step vs ~290ms with SP1024 is expected — the SP8192 embedding/head projection (`8192×512`) is 8× larger than SP1024, dominating the forward/backward cost. Reducing `TRAIN_SEQ_LEN=512` should bring step time back to ~260-280ms.

## Architecture

Same as `2026-04-26_MuonTTT_WeightTiedRecurrence_RoPE_RMSNorm_1xH100` but with SP8192 tokenizer.

| Component | Setting |
|-----------|---------|
| **Tokenizer** | **SentencePiece BPE 8192** (kevclark/parameter-golf) |
| Model dim | 512 |
| Heads | 8 (head_dim = 64) |
| Recurrence depth | 4 (weight-tied) |
| Positional encoding | RoPE (persistent=False) |
| Normalization | RMSNorm |
| MLP activation | LeakyReLU(0.5)² |
| Logit softcap | 30.0 |
| Train seq len | 1,024 |
| Parameters | ~7.3M (unique) |

## vs SP1024 version

| Config | val_bpb (post-TTT) | Steps | Tokenizer |
|--------|-------------------|-------|-----------|
| SP1024 (seed 1337) | 1.5061 | 2,004 | SP-1024 |
| **SP8192 (seed 2025)** | **1.4305** | 1,227 | **SP-8192** |
| **Improvement** | **−0.076** | | |

SP8192 gives a solid ~0.076 bpb gain despite ~40% fewer steps due to larger embedding compute. Switching to `TRAIN_SEQ_LEN=512` is expected to recover the step count.

## Quantization gap

| Stage | val_bpb |
|-------|---------|
| Pre-quantization (fp32/bf16) | 1.4115 |
| Post int8+zlib + TTT (3 steps) | 1.4305 |
| Gap | +0.019 |

## Artifact Size

| Component | Size |
|-----------|------|
| `train_gpt.py` code | 44,575 bytes |
| Model int8+zlib payload | 10,413,266 bytes |
| **Total** | **10,457,841 bytes** |

Cap: 16,000,000 bytes. Remaining headroom: **5,542,159 bytes** (~5.5 MB).

## Run Command

```bash
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 10

export RUN_ID=r4_512_sp8192_seed2025
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export MODEL_DIM=512
export NUM_HEADS=8
export RECUR_STEPS=4
export NOISY_QAT=0
export TTT_STEPS=3
export TTT_LR=3e-4
export SEED=2025
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.txt
```
