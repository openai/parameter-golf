# Non-Record: mHC-Lite Residual Mixing + Attention Sink Probe

**local val_bpb: 1.4246** (3-seed mean on local RTX 5080 smoke train shard + full validation) | **13.36 MB max artifact** | **non-record submission**

This is a non-record submission. It is not an official leaderboard claim: the bundled scores were produced locally on one RTX 5080 using the `sp1024` smoke setup with one train shard and the full validation split. Two 8xH100 Runpod attempts were made on the final day, but both pods exited before producing a validation score.

## Results

Local command family:

```bash
SEED=<seed> \
MAX_WALLCLOCK_SECONDS=598 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MHC_LITE_ENABLED=1 \
MHC_RESID_INIT_LOGIT=4.0 \
MHC_SKIP_MODE=none \
GRAD_CLIP_NORM=1.0625 \
LOGIT_SOFTCAP=15 \
QK_GAIN_INIT=3 \
WARMDOWN_ITERS=150 \
MUON_MOMENTUM=0.97 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
LATE_GATE_RAMP_ENABLED=0 \
MTP_AUX_ENABLED=0 \
ATTN_SINK_ENABLED=1 \
ATTN_SINK_INIT=-6.5 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

| Seed | Local val_bpb | Train time | Steps | Eval time | Artifact bytes |
|---:|---:|---:|---:|---:|---:|
| 42 | 1.42542064 | 598.341s | 516 | 40.017s | 13,332,695 |
| 1337 | 1.42551199 | 598.587s | 518 | 38.697s | 13,335,887 |
| 2025 | 1.42293649 | 598.504s | 522 | 40.256s | 13,363,475 |
| **Mean** | **1.42462304** | | | | |

For context, the same local configuration with the default 600-second cap reached a 3-seed mean of 1.42271793, but those runs slightly exceeded the strict 600-second train cap on the local machine (`600.675s` to `600.707s`), so the safer 598-second numbers above are the submitted local evidence.

## What Changed

This keeps the root `sp1024` transformer scaffold and adds two optional clean-neural mechanisms, disabled by default unless the environment variables above are set.

1. **mHC-lite residual mixing**
   - Reparameterizes each block's residual/input mixing weights through a softmax.
   - Initializes the residual lane strongly with logits `[+4, -4]`.
   - Leaves the default behavior unchanged when `MHC_LITE_ENABLED=0`.

2. **Attention sink**
   - Adds a learned per-head sigmoid scale for the first value vector in the current causal sequence.
   - This is causal: each position only receives information from position 0 of the same left-to-right context.
   - The best local initialization in the sweep was `ATTN_SINK_INIT=-6.5`.

3. **Muon schedule**
   - Uses `MUON_MOMENTUM=0.97` with warmup from `0.92`.
   - Local probes at `0.95` and `0.99` were worse on seed 42.

Negative local probes included late gate ramping, shared-head MTP auxiliary loss, nearby attention-sink initializations, and nearby mHC initialization strengths.

## Compliance Notes

- No test-time training.
- No casefolding, lossy normalization, PPM, byte sidecar, validation cache, or future-token cache.
- No network access or external binaries are used during evaluation.
- Full local validation split is evaluated.
- Artifact accounting is `train_gpt.py` bytes plus compressed int8+zlib model bytes.
- Max local counted artifact in the safe three-seed set is 13,363,475 bytes, below the decimal 16,000,000 byte cap. The bundled script is 105 bytes larger than the logged training script because the final package includes the guarded `nvidia-smi` logging fix, so the table adds 105 bytes to the logged totals.

## 8xH100 Attempts

Two Runpod 8xH100 SXM attempts were made after local triage:

- `runpod_sinkm65_cap598_seed42_20260430T062654Z`: pod exited during compile before warmup completed.
- `runpod_sinkm65_cap300_seed42_20260430T063624Z`: recovery pod reached training step 250 (`train_loss=2.7375`) and then exited before validation/export.

Both attempts are logged in `runs/flywheel_manifest.jsonl` in the working tree. They produced no official `val_bpb`, so this folder is intentionally placed under `track_non_record_16mb`.

## Reproduction

From this folder:

```bash
python -m py_compile train_gpt.py

SEED=42 \
MAX_WALLCLOCK_SECONDS=598 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MHC_LITE_ENABLED=1 \
MHC_RESID_INIT_LOGIT=4.0 \
MHC_SKIP_MODE=none \
GRAD_CLIP_NORM=1.0625 \
LOGIT_SOFTCAP=15 \
QK_GAIN_INIT=3 \
WARMDOWN_ITERS=150 \
MUON_MOMENTUM=0.97 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
LATE_GATE_RAMP_ENABLED=0 \
MTP_AUX_ENABLED=0 \
ATTN_SINK_ENABLED=1 \
ATTN_SINK_INIT=-6.5 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Use `--nproc_per_node=8` on a stable 8xH100 box for an official-style run. No accepted leaderboard score is claimed here.
