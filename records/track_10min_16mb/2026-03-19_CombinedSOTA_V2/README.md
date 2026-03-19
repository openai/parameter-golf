This record submission is called `Combined SOTA V2`.

## Research Methodology

This submission was developed using an agent-assisted research pipeline. An LLM agent was given access to the GitHub CLI and tasked with systematically reviewing all ~50 open PRs on this repository. For each PR, the agent fetched the full diff, extracted the technique, and categorized it into three tiers by expected BPB impact:

- **High**: Sliding window eval (PR #50, +0.032 BPB), optimizer tuning with seq_len=4096 (PR #52, +0.02 BPB), fp16 tied embedding export (PR #42, +0.007 BPB)
- **Medium**: 10-layer mixed precision (PR #39), QAT (PRs #38, #51)
- **Low**: Warmdown scheduling fixes, depth recurrence without enough steps, etc.

The agent then composed the top-tier approaches, identified interactions (e.g. QAT was counterproductive given the already-low quant gap from fp16 embedding + low LR), and iteratively debugged issues (OOM in the first run from `eval_batch_seqs=1024` at `seq_len=4096`). The final configuration is the product of two 10-minute training runs, each refining based on observed results.

All training runs used publicly available data and hardware. No external compute beyond the standard 10-minute budget was used for the final scored run.

## Approach

Four independent improvements are stacked to achieve 1.18335372 BPB, beating the naive baseline (1.2244) by 0.041 BPB and the previous best public claim (PR #53, 1.1888) by 0.005 BPB.

**1. Longer training context (`TRAIN_SEQ_LEN=4096`)**
Each training sequence sees 4x more context than the 1024-token baseline, providing much richer signal per token. This costs ~60ms/step (vs ~44ms at seq_len=1024) but the quality improvement more than compensates for the fewer total steps.

**2. Optimizer tuning**
- `MUON_MOMENTUM=0.99` (vs 0.95 default): stronger gradient smoothing for better convergence
- `MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03` (vs 0.04/0.04/0.05): lower LRs reduce the int8 quantization gap significantly
- `TRAIN_BATCH_TOKENS=393216` (3/4 of default 524288): more optimizer updates per wallclock second
- `WARMDOWN_ITERS=3000`: proportionally longer LR decay for the ~10k-step run
- `MUON_MOMENTUM_WARMUP_STEPS=1500` from 0.92: prevents early instability with high momentum

**3. fp16 tied embedding export**
The tied embedding doubles as the output head and is the most sensitive tensor to int8 quantization. Keeping it in fp16 reduces the post-quant BPB degradation from ~0.007 to ~0.001. `MLP_HIDDEN=992` (vs default 1024) trims the MLP slightly to stay under the 16MB artifact cap.

**4. Sliding window evaluation (`EVAL_STRIDE=64`, `EVAL_BATCH_SEQS=128`)**
Non-overlapping 4096-token chunks average ~2048 tokens of context per token. Overlapping windows with stride=64 give each token up to 4032 tokens of context. Only the rightmost 64 tokens per window are scored; every token is evaluated exactly once. The 4096-token context window provides substantially more context than PR #50's 1024-token version.

## Configuration

```
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_HIDDEN=992  TIE_EMBEDDINGS=1  TRAIN_SEQ_LEN=4096  TRAIN_BATCH_TOKENS=393216
MATRIX_LR=0.02  SCALAR_LR=0.02  TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99  MUON_MOMENTUM_WARMUP_STEPS=1500  MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000  QAT=0  VAL_LOSS_EVERY=0
EVAL_STRIDE=64  EVAL_BATCH_SEQS=128
```

## Run Command

```bash
RUN_ID=combined_v2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_HIDDEN=992 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All optimizer/eval hyperparameters are baked into the script defaults; only paths and `MLP_HIDDEN` need to be passed explicitly.

## Key Metrics (from `train.log`)

- Timed training stopped at `9919/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0172`, `val_bpb:1.1947`
- Post-quant sliding window eval: `val_loss:1.9980`, `val_bpb:1.1834`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.18335372`
- Train time: `601589ms` (`step_avg:60.65ms`)
- Eval time: `277849ms` (4m 38s, within the separate 10-min eval budget)
- Peak memory: `7712 MiB allocated`, `8122 MiB reserved`
- Serialized model int8+zlib: `15879361 bytes`
- Code size: `53577 bytes`
- Total submission size int8+zlib: `15932938 bytes`

## Results vs Baseline and Prior PRs

| Run | BPB | Delta |
|-----|-----|-------|
| Naive Baseline | 1.2244 | — |
| 4-Hour Baseline (unlimited compute) | 1.2074 | -0.017 |
| PR #52 (optimizer tuning, seq4096) | 1.2014 | -0.023 |
| PR #50 (sliding window, seq1024) | 1.1925 | -0.032 |
| PR #53 (SP-4096 + sliding window) | 1.1888 | -0.036 |
| **This submission** | **1.1834** | **-0.041** |

## Training Volume

- Global batch: `393216` tokens/step
- Total train tokens seen: `9919 × 393216 = 3,899,581,344`

## Hardware

8x NVIDIA H100 80GB SXM (NVLink)

## Included Files

- `train_gpt.py` (standalone script used for the run; all V2 defaults baked in)
- `train.log` (exact training log from the run)
- `submission.json` (leaderboard metadata)
