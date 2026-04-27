# SP4096 + depth recurrence + MuonEq-R + misc improvements

Stacking stuff that works from recent PRs. Nothing too fancy, just trying to get everything working together before adding SLOT/TTT later.

## what changed vs baseline

- switched to **sp4096** tokenizer (bigger vocab = better compression per byte)
- **11 layers with depth recurrence** on layers 3-5 (shared MLP), so effectively 14 virtual layers for 0 extra params
- **MLP 4x** (2048 hidden) instead of 2x
- **LeakyReLU(0.5)²** instead of relu²
- **MuonEq-R**: added row-normalization before newton-schulz in muon. small thing but helps
- **QK-Gain 5.0** (init was 1.5, bumped it up based on what others found works)
- **BigramHash** 3072x112 + projection to model dim
- **SmearGate** for blending adjacent token embeddings
- **EMA** (0.997 decay) applied at the end before quantization
- decoupled **weight decay** (0.04) in muon for better quantization later
- warmdown bumped to 4000 iters
- tuned LRs: matrix=0.025, scalar=0.025, embed=0.035
- muon momentum 0.99 (warmup from 0.92)
- grad clip 0.3

## quantization

still using baseline int8 + zlib for now. plan is to switch to int6 + lzma once I verify everything trains properly.

## expected results

haven't run this yet (waiting on compute). aiming for somewhere around 1.09-1.12 based on what similar setups get in other PRs.

## to run

```bash
python3 data/cached_challenge_fineweb.py --variant sp4096
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/submission_v1/train_gpt.py
```
