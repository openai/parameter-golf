# SP8192 MuonTTT WeightTied Recurrence — RoPE + RMSNorm + LeakyReLU²

**val_bpb: 1.3844** (seed 1337, 2×H100, post-TTT int8+zlib) | **11.06 MB** artifact

> Seed 42 pending. Table will be updated with 3-seed mean.

## Results

| Seed | Hardware | step_avg | steps | Pre-quant bpb | **Post-TTT bpb** | TTT time | Artifact |
|------|----------|----------|-------|--------------|-----------------|----------|----------|
| 1337 | 2×H100 | 102.49ms | 5,855 | — | **1.38435** | 80s | 11,055,486 |
| 2025 | 1×H100 | 489.06ms | 1,227 | 1.4115 | 1.43052 | 160s | 10,457,841 |
| 42   | pending | — | — | — | — | — | — |

> **Note on hardware**: Seed 1337 used 2×H100 (`--nproc_per_node=2`, `grad_accum_steps=4`), giving 4.8× more steps than the 1×H100 seed 2025 run (102ms vs 489ms per step). The −0.046 bpb improvement is primarily from more gradient steps, not the extra GPU itself.

## Architecture

| Component | Setting |
|-----------|---------|
| **Tokenizer** | **SentencePiece BPE 8192** (kevclark/parameter-golf) |
| Model dim | 512 |
| Heads | 8 (head_dim = 64) |
| Recurrence depth | 4 (weight-tied, 1 shared block) |
| Positional encoding | RoPE (persistent=False — not stored in artifact) |
| Normalization | RMSNorm |
| MLP activation | LeakyReLU(0.5)² |
| MLP width | 4× hidden (512 → 2048 → 512) |
| Logit softcap | 30.0 |
| Train seq len | 1,024 |
| Parameters | ~7.3M (unique) |

## Progression vs SP1024

| Config | val_bpb | Steps | Hardware |
|--------|---------|-------|----------|
| SP1024, 1×H100 (seed 1337) | 1.5061 | 2,004 | 1×H100 |
| SP8192, 1×H100 (seed 2025) | 1.4305 | 1,227 | 1×H100 |
| **SP8192, 2×H100 (seed 1337)** | **1.3844** | **5,855** | **2×H100** |

## Quantization

| Stage | val_bpb |
|-------|---------|
| Pre-quant (seed 2025, fp32) | 1.4115 |
| Post int8+zlib + TTT 3 steps | 1.4305 |
| Quant gap | +0.019 |

## Artifact Size (seed 1337)

| Component | Size |
|-----------|------|
| `train_gpt.py` code | 44,575 bytes |
| Model int8+zlib | 11,010,911 bytes |
| **Total** | **11,055,486 bytes** |

Cap: 16,000,000 bytes. Remaining headroom: **4,944,514 bytes** (~4.9 MB).

## Run Commands

```bash
# Download SP8192 data (first time only)
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 10

# Training (2×H100)
export RUN_ID=r4_512_sp8192_seed1337
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export MODEL_DIM=512
export NUM_HEADS=8
export RECUR_STEPS=4
export NOISY_QAT=0
export TTT_STEPS=3
export TTT_LR=3e-4
export SEED=1337
torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/${RUN_ID}.txt
```
