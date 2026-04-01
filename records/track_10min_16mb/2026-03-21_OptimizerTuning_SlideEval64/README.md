This record combines optimizer tuning, training at longer sequence length, and sliding window evaluation to improve on the naive baseline without changing the model architecture.

## Key Changes from Baseline

### Training Improvements
- **Sequence length 2048** (baseline: 1024): Longer context during training improves the model's ability to use positional information. Steps are ~18% slower but quality gain is worth it.
- **Warmdown 10000** (baseline: 1200): Much longer learning rate decay schedule. With the wallclock-based warmdown, this means the LR decays throughout most of training, producing a smoother convergence.
- **Muon backend steps 10** (baseline: 5): More Newton-Schulz iterations in the Muon optimizer produce better gradient orthogonalization.
- **Gradient clipping norm=1.0** (baseline: disabled): Stabilizes training, especially important with the longer warmdown.
- **Adam beta2=0.99** (baseline: 0.95): Smoother second moment estimate for embedding and scalar parameters.
- **Scalar LR=0.02** (baseline: 0.04): Lower learning rate for scale/gate parameters (attn_scale, mlp_scale, resid_mix, skip_weights) improves stability.

### Evaluation Improvement
- **Sliding window eval (stride=64)**: Instead of chopping the validation set into non-overlapping 2048-token chunks (where the first token has zero context), we use overlapping windows advancing by 64 tokens. Only the last 64 tokens of each window are scored, giving every token 1984+ tokens of context. The first window scores all tokens. This is a pure eval improvement — the model weights are identical.

### What Didn't Work (Tried and Reverted)
- SwiGLU MLP: Better per-param quality but the 3-matrix design uses more params per layer, blowing the 16MB budget at convergence.
- FP16 embedding passthrough: Reduces quantization error from ~0.007 to ~0.0003 BPB, but adds ~500KB to the artifact, pushing over 16MB.
- More layers (10-12): Better BPB but always exceeded the 16MB artifact limit at full convergence. The int8+zlib compression ratio is ~0.93 bytes/param at 8xH100 convergence.
- Higher/lower learning rates for matrix_lr, tied_embed_lr: The defaults (0.04, 0.05) are well-tuned.
- Depth recurrence, lower RoPE base, different KV head counts: All worse.

## Configuration

Same architecture as baseline:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- ReLU^2 MLP (unchanged)

Modified hyperparameters:
- `TRAIN_SEQ_LEN=2048` (was 1024)
- `WARMDOWN_ITERS=10000` (was 1200)
- `MUON_BACKEND_STEPS=10` (was 5)
- `GRAD_CLIP_NORM=1.0` (was 0.0)
- `BETA2=0.99` (was 0.95)
- `SCALAR_LR=0.02` (was 0.04)
- `EVAL_STRIDE=64` (sliding window evaluation)

## Command

```bash
RUN_ID=submission_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=2000 \
EVAL_BATCH_SEQS=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics (from `train.log`)

- Timed training stopped at `11520/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0313`, `val_bpb:1.2031`
- Post-quant sliding window eval: `val_loss:2.0032`, `val_bpb:1.1864`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.18641686`
- Train time: `600019ms` (`step_avg:52.08ms`)
- Peak memory: `10121 MiB allocated`, `10440 MiB reserved`
- Eval time: `132519ms` (sliding window, stride=64, batch_seqs=1024)
- Serialized model int8+zlib: `15808653 bytes`
- Code size: `52684 bytes`
- Total submission size int8+zlib: `15861337 bytes`

## Training Volume

- Global batch: `524288` tokens/step
- Total train tokens seen: `6,044,098,560`

## Reproducibility (3 seeds)

| Seed | Steps | val_loss | val_bpb | Artifact |
|------|-------|----------|---------|----------|
| 1337 | 11,520 | 2.00321 | 1.18642 | 15,861,337 |
| 1338 | 11,520 | 2.00428 | 1.18705 | 15,859,751 |
| 1339 | 11,523 | 2.00667 | 1.18847 | 15,867,480 |

- Sample mean val_loss: `2.00472`
- Sample std: `0.00177`
- Current SOTA val_loss: `2.01348`
- Required improvement: `0.005 nats`
- Actual improvement: `0.00876 nats`
- One-sided t-test: `t=8.57`, `df=2`, `p < 0.01`

## Methodology

Changes were discovered through 46 iterations of automated experimentation (autoresearch) on a proxy test setup (RTX 3090, 2000 steps), then validated on 4xH100 and finally 8xH100. The proxy correctly identified directional improvements but could not predict exact artifact sizes at full convergence, leading to several over-budget configurations being tested on H100.

## Included Files

- `train_gpt.py` (code snapshot used for the run)
- `train.log` (canonical run, SEED=1337)
- `train_seed1338.log` (reproducibility run, SEED=1338)
- `train_seed1339.log` (reproducibility run, SEED=1339)
- `submission.json` (leaderboard metadata)
