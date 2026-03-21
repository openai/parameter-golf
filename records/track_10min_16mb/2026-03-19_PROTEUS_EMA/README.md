# PROTEUS Combined — Parameter Golf Submission

**Built with [PROTEUS](https://lightspeedup.com) by LightSpeedUp**

## Approach

Four published techniques stacked on the baseline:

1. **EMA weight averaging** (Polyak 1992) — decay=0.999, fp32, every 10 steps. Smooths weight distributions for reduced INT8 quantization loss.
2. **seq_len=2048** — train and evaluate with longer context. Each token sees more information.
3. **FP16 embedding passthrough** — keep `tok_emb.weight` at FP16 instead of quantizing to INT8. The tied embedding pulls double duty as the output head — precision matters most here.
4. **Sliding window evaluation** (stride=64) — score each token with ~960 tokens of context instead of ~512. Pure eval-time improvement, zero training cost.

Plus hyperparameter tuning: `WARMDOWN_ITERS=3600`, `MATRIX_LR=0.06`, `SCALAR_LR=0.06`, `TIED_EMBED_LR=0.04`.

## What We Learned (and What's Next)

This submission was our v1 — built in a single session, exploring from scratch. Along the way we tested and documented several negative results that may be useful to others:

**INT4 post-training quantization fails catastrophically.** Per-row, per-group (gs=64), and QAT with STE all produce ~3.7 BPB roundtrip vs ~1.2 training BPB. Root cause: INT4 has 15 quantization levels, and error compounds through transformer layers — cosine similarity between original and quantized hidden states drops to 0.90 at 18 layers. INT8 stays above 0.995. Production INT4 (GPTQ/AWQ) uses Hessian-guided error compensation which is fundamentally different from round-to-nearest.

**Shared-weight depth recurrence (LoopFormer) loses to more training tokens at this budget.** We tested 9 layers × 2 passes (18 effective layers) vs 9 layers × 1 pass on 8×H100 for 600s. 1-pass wins: 1.2265 BPB vs 1.2450 BPB. Fewer tokens per step outweighs the depth benefit at 17M params.

**EMA reduces INT8 quantization loss** from 0.0072 BPB (baseline) to 0.0048 BPB. But the ~2ms/step overhead from the EMA update loop costs ~600 training steps, partially offsetting the gain.

**Next targets:** INT6 quantization, zstd-22 compression, 3× MLP expansion, SmearGate, and paid prefix strategies as documented in [Issue #140](https://github.com/openai/parameter-golf/issues/140).

## Configuration

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key env vars (all have defaults baked into the script):
- `TRAIN_SEQ_LEN=2048`
- `WARMDOWN_ITERS=3600`
- `MATRIX_LR=0.06`
- `SCALAR_LR=0.06`
- `TIED_EMBED_LR=0.04`
- `EMA_ENABLED=1`
- `EMA_DECAY=0.999`
- `EMA_EVERY=10`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=512`

Architecture unchanged from baseline:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288`

## Key Metrics

- Timed training stopped at `13366/20000` steps due to wallclock cap
- Pre-quant eval at stop: `val_loss:2.0643`, `val_bpb:1.2226`
- **Post-quant sliding window eval: `val_loss:2.0085`, `val_bpb:1.1896`**
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.18956858`
- Train time: `600026ms` (`step_avg:44.89ms`)
- Eval time: `72611ms` (sliding window, stride=64)
- Artifact size: `15,878,735 bytes` (99.2% of 16MB cap)

### Training volume

- Global batch: `524288` tokens/step
- Total train tokens seen: `~6.8B`

## Platform

Run on Modal 8×H100 SXM. Pending verification on RunPod 8×H100 SXM (official hardware).

## Included Files

- `train_gpt.py` — training script (baseline + EMA + sliding eval + FP16 embed)
- `train.log` — full training log from the submission run
- `submission.json` — leaderboard metadata
- `README.md` — this file
