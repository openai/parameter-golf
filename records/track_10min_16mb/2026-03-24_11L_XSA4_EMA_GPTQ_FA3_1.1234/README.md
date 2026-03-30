# 11L XSA4 + EMA + GPTQ + FlashAttention-3

val_bpb: 1.12336724 (single preserved official run) | 15,853,809 bytes | 8xH100 SXM | non-record 10-minute submission

## Summary

This folder contains the saved official log for our March 24, 2026 `v6` run on `8xH100 SXM` with a 600-second training budget. The stack combines an 11-layer 512d GQA model with 2048-token training, tied embeddings, BigramHash + SmearGate token mixing, XSA on the last 4 layers, late-layer VE, EMA before export, late QAT enablement, and full GPTQ-based int6 export compressed with `zstd`.

We are submitting this as a non-record 10-minute entry. This folder contains one fully preserved official run log, and we are not claiming the multi-seed statistical significance required for a new leaderboard record.

## Results

| Seed | step_avg | Steps in 600s | Wallclock-stop val_bpb | Post-EMA val_bpb | Final int6 roundtrip | Final sliding exact | Artifact bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 96.28 ms | 6,233 | 1.1433 | 1.1424 | 1.14704222 | 1.12336724 | 15,853,809 |

## Architecture And Training Stack

| Component | Setting |
|---|---|
| Layers / width | 11 layers, 512 model dim |
| Attention | 8 heads, 4 KV heads, GQA |
| Sequence length | 2048 train / 2048 eval |
| Train batch | 786,432 tokens |
| Parameters | 26,993,756 |
| MLP | 3x expansion |
| Token mixing | tied embeddings + BigramHash(2048, 128) + SmearGate |
| Extra attention | XSA on the last 4 layers |
| Late-layer module | VE enabled on layers 9 and 10, dim 128 |
| Optimizer split | Muon for matrix params, Adam-style groups for token/scalar params |
| Averaging | EMA applied before export |
| Quantization | late QAT trigger, then full GPTQ int6 export |
| Compression | `zstd` |
| Attention kernel | FlashAttention 3 on Hopper when available, PyTorch SDPA fallback otherwise |

## Logged Milestones

| Event | Value |
|---|---|
| Validation checkpoint | `step:4000/20000 val_bpb:1.2053` |
| Late QAT enable | `step:5709 scale:0.1498 qat_lr_mult:1.0000` |
| Wallclock stop | `train_time:600075ms step:6233/20000` |
| Post-EMA eval | `val_bpb:1.1424` |
| GPTQ Hessians | `66` layers collected in `4.3s` |
| GPTQ quantization | `66` layers, `0` fallback |
| GPTQ total time | `16.5s` |
| Quantized model bytes | `15,772,843` |
| Total submission bytes | `15,853,809` |
| Run exit | `0` |

## Run Command

Requires:

```bash
pip install -r requirements.txt
```

For the exact fast path used in the saved log on Hopper, make `flash_attn_interface` importable before launch. One example is:

```bash
export PYTHONPATH=/path/to/flash-attention/hopper/build/lib.linux-x86_64-cpython-311:$PYTHONPATH
```

If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA.

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
NUM_LAYERS=11 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
VAL_LOSS_EVERY=4000 \
TRAIN_LOG_EVERY=500 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
XSA_LAST_N=4 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
SMEAR_INIT=-3.0 \
GPTQ_ENABLED=1 \
GPTQ_CALIBRATION_BATCHES=128 \
LATE_QAT_THRESHOLD=0.15 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py`: exact script for this submission folder
- `train_seed1337.log`: preserved official `8xH100 SXM` run log
- `submission.json`: structured metadata
- `requirements.txt`: Python dependencies used by this script

## Credit And Lineage

This stack builds on public ideas already shared in this repo, and we want to credit that lineage clearly:

- `2026-03-19_smeargate_orthoinit_muonwd` by aquariouseworkman for the SmearGate + int6/QAT + sliding-window direction.
- `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` by Raahil Shah for the MLP3x + BigramHash + Muon WD framing.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` by thwu1 for stronger BigramHash scaling and the weight-decay-first compression mindset.
- The public March 22 11-layer EMA + GPTQ-lite + late-QAT submission for the 11-layer EMA/GPTQ export direction.

This exact script, hyperparameter combination, and preserved run log are our own submission artifacts.
