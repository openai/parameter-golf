# Local Blackwell 8GB Run

This submission reports a local constrained-hardware run using `train_merged_gpt_flagged.py` on an 8 GB Blackwell-class GPU. It is not a leaderboard-eligible 8xH100 record attempt; the goal was to measure how far the flagged code path could be pushed on a small local card while still preserving strong compressed-model quality.

## Method

Training used `train_merged_gpt_flagged.py`.

Model/features:
- 10 transformer layers
- grouped-query attention with 8 attention heads and 4 KV heads
- tied token embeddings
- U-Net-style skip structure
- BigramHash feature path
- Muon for matrix parameters
- Adam for embedding and scalar parameters
- sliding-window evaluation support
- int6 + zstd compressed export

Compression / post-processing:
- trained a local raw checkpoint
- exported to compressed `int6 + zstd`
- the first packed artifact was only about `40 KB` over the exact `16,000,000` byte cap
- that slightly-over artifact had post-pack quality around `~1.21` BPB
- lightly pruned and repacked the checkpoint to bring the final artifact under the size limit
- selected the final artifact based on post-pack quality and size
- the smaller under-cap artifact used for this folder is `my_saved_model_prune0p144.int6.ptz`

## Hardware

- local 8 GB Blackwell-class GPU
- local single-GPU run
- non-record / constrained-hardware submission

## Result

The best local compressed/pruned result landed at roughly:

- `val_bpb: ~1.25`

This was achieved after pruning and repacking the flagged-model checkpoint into the final compressed artifact. The original packed `int6 + zstd` artifact was only about `40 KB` over the exact `16,000,000` byte limit and had post-pack quality around `~1.21` BPB. The final reported artifact is the lightly pruned and repacked version that fit under the cap, at the cost of degrading to roughly `~1.25` BPB.

The smaller under-cap packed model used here is:

- `my_saved_model_prune0p144.int6.ptz`
- size: `15,794,840` bytes

## Attribution

This local run uses `train_merged_gpt_flagged.py`, which is a merged derivative of earlier public Parameter Golf record scripts rather than a from-scratch implementation. The code path borrows and adapts ideas from prior public submissions including:

- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py`
- `records/track_10min_16mb/2026-03-19_10L_MixedPrecision/train_gpt_v5.py`
- `records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/train_gpt_v9.py`
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py`

## Takeaway

The main result is that the `train_merged_gpt_flagged.py` branch remains strong even under severe local hardware constraints. While this is not a record-track submission, it shows that reasonably good FineWeb compression performance is still possible on consumer-scale memory budgets with a compact transformer, low-bit export, and light post-training pruning.
