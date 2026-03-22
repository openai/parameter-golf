# 8xH100 Agent Briefing — Parameter Golf Competition

## Task
Fix the training speed on 8xH100 to achieve competitive step times (~85ms/step) and beat val_bpb < 1.12.

## Competition
- **Repository**: https://github.com/openai/parameter-golf
- **Goal**: Train best LM in 16MB artifact, 10 min on 8xH100, evaluated by BPB on FineWeb
- **Current SOTA**: 1.1233 (PR #414), 1.1428 (merged leaderboard #1)
- **Our best**: 1.1386 (1xH100 80min), 1.1573 (8xH100 torch 2.4 + FA3)
- **Issue #140**: https://github.com/openai/parameter-golf/issues/140 — live leaderboard tracking
- **Top PRs to study**: #414 (1.1233), #315 (1.1248), #287 (1.1280)

## Our Training Script
- **Location**: `parameter-golf/transformer/train.py` — single-file training script
- **Architecture**: 11L transformer, 512-dim, 8/4 GQA heads, 3x MLP, U-Net skips
- **Key techniques**: XSA (last 4 layers), Partial RoPE (16/64), LN Scale, EMA, SWA, Late QAT, GPTQ-lite, int6+zstd, sliding window eval
- **Runs with**: `torchrun --standalone --nproc_per_node=8 transformer/train.py`

## Known Issues on 8xH100
1. **torch 2.4 (old RunPod)**: FA3 works, 109ms/step, but `enable_gqa` not available (uses slow repeat_interleave). Still best result.
2. **torch 2.8 (new RunPod)**: Native GQA available but torch.compile takes 2+ min for warmup, and DDP optimizer has issues. fullgraph=True causes process count explosion (273 python procs). Step time 143ms even after warmup.
3. **FA3 + torch.compile**: flash_attn_func may not trace well under torch.compile. The top submissions compile around FA3 or exclude it from the graph.
4. **GQA fallback**: We use try/except resolved at import time (_HAS_NATIVE_GQA flag), but the repeat_interleave fallback on torch 2.4 adds ~23ms/step.

## What the Top Submission (#414) Does Differently
- **torch version**: Likely 2.5-2.6 (has enable_gqa + fast compile)
- **FlashAttention 3**: Direct `flash_attn_func` calls, not through torch.compile
- **Step time**: 85ms/step on 8xH100 (vs our 109-143ms)
- **Compile strategy**: May use `torch.compile` with `mode="reduce-overhead"` or exclude attention

## Target
- Get step time to ~85ms on 8xH100 in 10 min
- This alone would give ~7000 steps (vs our 4500-5300)
- Expected improvement: ~0.01 bpb from more training steps

## Environment
- **SSH config**: `gcp-single-h100` for 1xH100, RunPod for 8xH100
- **Data**: `data/datasets/fineweb10B_sp1024/` (80 shards + val)
- **Tokenizer**: `data/tokenizers/fineweb_1024_bpe.model` (vocab 1024)
- **Experiments log**: `experiments.csv`

## Key Files to Read
1. `transformer/train.py` — our training script
2. `experiments.csv` — all experiment results
3. Top submission code: `git fetch upstream 'pull/414/head:pr-414'` then `git show pr-414:records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
