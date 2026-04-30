# PR 1394 — SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R + SDClip + Simplifications

**Author:** Kevin Clark (clarkkev)
**Claimed BPB:** 1.08563 (5-seed mean, std 0.0007); pre-quant 1.09021; post-quant 1.10235
**Artifact size:** ~15.99 MB (mean 15,985,678 bytes)
**Seeds:** [1, 42, 1234, 1337, 2025] — [1.08554, 1.08664, 1.08463, 1.08554, 1.08578]
**Track:** 10min_16mb, 8xH100 80GB SXM
**Date:** 2026-04-05
**Base PR:** #1218

## Files retrieved
- `records__track_10min_16mb__2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2__README.md`
- `records__track_10min_16mb__2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2__submission.json`
- `records__track_10min_16mb__2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2__train_gpt.py`
- `records__track_10min_16mb__2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2__train_gpt_human.py`

## Environment variables (from run command)
`RUN_ID=1337`, `SEED=1337`, torchrun standalone nproc_per_node=8.

## Claimed changes (from README, verbatim)
"This script builds on #1218. The main changes are:
- Increase the vocabulary size from 4096 to 8192.
- GPTQ-quantize the embedding matrix instead of using simple round-to-nearest quantization. The other matrices were already using GPTQ.
- Remove the value embeddings.
- Replace the coprime-stride data loader from #726 with a simpler ShuffledSequenceLoader.
- Loop layers 4-5 twice (while sharing params): the idea is from #1204, but this script uses a simpler implementation and loops twice rather than once.
- Use row-normalized Muon from #1217.
- Choose the quantization clip threshold based on the standard deviation of the row rather than searching for a quantile with low reconstruction error. Used b=6, k=12.85 for matrix parameters and b=8, k=20 for embeddings. c = k * std(row). Flash Attention 3 (Hopper) required, PyTorch 2.11.0+cu130."
