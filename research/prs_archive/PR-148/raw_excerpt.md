# PR 148 — Depth Recurrence + Cross-Repeat Skip + Value Embeddings + Sliding Window

**Author:** Ivan Verbovoy (iverbovoy)
**Branch created:** 2026-03-20
**Claimed BPB:** 1.21958209 (sliding window, stride=256 on int8+zlib roundtrip); 1.25328684 (standard roundtrip)
**Artifact size:** 12,829,176 bytes (12.83 MB, int8+zlib)
**Seeds:** not stated (single run)

## Files retrieved
- `records__track_10min_16mb__2026-03-20_DepthRecurrence_CrossRepeatSkip__README.md`
- `records__track_10min_16mb__2026-03-20_DepthRecurrence_CrossRepeatSkip__submission.json`
- `records__track_10min_16mb__2026-03-20_DepthRecurrence_CrossRepeatSkip__train_gpt.py`

## Claimed changes (from README, verbatim)

> Replaced the baseline's 9 unique transformer blocks with 3 shared blocks repeated 4 times (12 effective layers). Trades unique parameters for effective depth.
> - Depth recurrence: 3 blocks x 4 repeats = 12 effective layers (vs 9 in baseline)
> - Cross-Repeat Skip (original): each block gets a weighted residual of its own output from the previous repeat, turning stateless recurrence into stateful. Per-repeat learned scales, ~7.5K params total.
> - Value Embeddings: 2 extra embedding tables mixed into the residual stream at each effective layer with learned scales. From snimu's modded-nanogpt record.
> - Loop Embedding: learned per-layer vector added before each block as depth-wise positional encoding.
> - Model dim 832 (vs 512), 8 heads, 4 KV heads, MLP 2x
> - Removed U-Net skip connections (Cross-Repeat Skip covers this role)
> - 17.14M params, 12.83MB artifact

Training: MATRIX_LR=0.012, SCALAR_LR=0.012, TIED_EMBED_LR=0.015, GRAD_CLIP_NORM=0.3, WARMDOWN_ITERS=3000, TRAIN_SEQ_LEN=1024, LR x0.3 from baseline. Ablations (RTX 3060, 2000 steps): Cross-Repeat Skip -0.041 bpb, Value Embeddings -0.079 bpb, LR x0.3 -0.052 bpb, Sliding window eval -0.034 bpb, WARMDOWN 3000 -0.027 bpb. 4494 steps at 133ms/step on 8xH100.
