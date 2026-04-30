# Non-record: Pre-TTT Anchor Port (SDPA)

Date: 2026-03-28
Track: non_record_16mb
Status: Pre-run (script complete, awaiting first measured Pegasus run)

## What this is

A clean pre-TTT anchor script built from the repo-root `train_gpt.py` skeleton with selective feature transplants from the 2026-03-21 donor record (`1.1248` BPB). This is Session 03 of the campaign plan.

Target: `val_bpb 1.123-1.128` on 8xH100 in 600s.

## Architecture

| Parameter | Value | Source |
|-----------|-------|--------|
| Layers | 11 | Donor anchor |
| Model dim | 512 | Root/donor |
| Heads / KV heads | 8 / 4 | Root/donor |
| MLP multiplier | 3.0 | Donor anchor |
| Sequence length | 2048 | Donor anchor |
| Batch tokens | 786,432 | Donor anchor |

## Inherited from root train_gpt.py

- U-Net skip connections with learnable skip weights
- relu^2 MLP activation
- GQA (Grouped Query Attention)
- Compiled DDP training loop with warmup
- SDPA attention backend (`torch.nn.functional.scaled_dot_product_attention`)
- TokenStream / DistributedTokenLoader data pipeline
- SentencePiece tokenizer evaluation infrastructure

## Transplanted from 2026-03-21 donor

| Feature | Implementation note |
|---------|-------------------|
| SmearGate | Direct transplant. Sigmoid gate interpolates current/previous token. |
| BigramHashEmbedding | Direct transplant. XOR hash with 2048 buckets, 128-dim embed, scale=0.05. |
| XSA on last 4 layers | Adapted for SDPA layout. Transpose SDPA output (B,H,T,D) -> (B,T,H,D) before self-value subtraction, then reshape. |
| Partial RoPE (16/64) | Adapted Rotary cache layout to root's `[None, None, :, :]` (SDPA format). Only first 16 dims rotated, rest pass through. |
| NTK-aware RoPE | `rope_train_seq_len=1024` with training at 2048 — intentional 2x NTK scaling. |
| Layerwise LN scale | `1/sqrt(layer_idx + 1)` applied to both attn_norm and mlp_norm outputs. |
| EMA (decay=0.997) | Updated every step, applied before export. Re-initialized after warmup. |
| Muon weight decay (0.04) | Decoupled: `p.data.mul_(1 - lr * wd)` before orthogonalized update. |
| Adam weight decay (0.04) | Via AdamW for tok/scalar/head optimizers. |
| Mixed int6 export | int6 for mlp+attn params, int8 for embeddings+other. |
| Zstd compression (level 22) | With zlib fallback if zstandard not installed. |
| Stride-64 sliding eval | Using `forward_logits()` method, compiled inference. |
| Orthogonal init | With `1/sqrt(2*num_layers)` scaling for projection layers. |

## Key adaptation: SDPA vs flash_attn_3

The donor uses `flash_attn_3_func` with `(B, T, H, D)` tensor layout. This anchor uses `scaled_dot_product_attention` with `(B, H, T, D)` layout. Adaptations:

- Rotary cache: `[None, None, :, :]` (root) instead of `[None, :, None, :]` (donor)
- q_gain broadcast: `[None, :, None, None]` (root) instead of `[None, None, :, None]` (donor)
- XSA: transpose SDPA output to `(B, T, H, D)` before `_xsa_efficient`, keep v in `(B, T, Hkv, D)`

## Intentionally excluded

| Feature | Reason |
|---------|--------|
| Late QAT | Unclear correctness under `torch.compile(fullgraph=True)` |
| SWA | Conflicts with EMA; not in clean anchor |
| MTP (multi-token prediction) | Training-only auxiliary heads, not part of non-TTT lineage |
| Value Embeddings | Only in 2026-03-22 (kitchen-sink file) |
| DTG | Default-off in donor, no evidence of contribution |
| GPTQ-lite clip search | Only in 2026-03-22; deferred to post-anchor delta |
| flash_attn_interface | External dependency; SDPA used instead |
| TTT | Out of scope for pre-TTT anchor |
| Warmdown 3500 | Only tested in 2026-03-22 alongside other changes |

## Hardcoded anchor constants

Architecture and training values are fixed in code, not exposed as env vars. Only operational controls are env-configurable:

- `DATA_PATH`, `TOKENIZER_PATH`, `RUN_ID`, `SEED`
- `MAX_WALLCLOCK_SECONDS`, `ITERATIONS`
- `VAL_LOSS_EVERY`, `TRAIN_LOG_EVERY`, `VAL_BATCH_SIZE`
- `VOCAB_SIZE`, `AMP_DTYPE`

## Launcher template (8xH100 on Pegasus)

```bash
salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --time=02:00:00

srun --gpu-bind=none bash -c '
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_IB_DISABLE=1
cd /netscratch/ayach/parameter-golf
RUN_ID=pre_ttt_anchor_8xh100_600s \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
AMP_DTYPE=auto \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
python3 -u records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py
' 2>&1 | tee /netscratch/ayach/pre_ttt_anchor_8xh100_600s.log
```

## Run results

(To be filled after first measured Pegasus run)
