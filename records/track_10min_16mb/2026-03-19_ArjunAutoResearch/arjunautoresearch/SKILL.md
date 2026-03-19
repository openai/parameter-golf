---
name: arjunautoresearch
description: Pulls all open PRs from a GitHub repo, ranks techniques by expected impact, composes the best combination, runs iterative experiments, and packages a compliant submission. Use when researching a competitive ML challenge, optimizing a metric across many candidate approaches, or preparing a leaderboard submission.
metadata:
  author: arjun-krishna1
  version: "1.0"
---

# Parameter Golf AutoResearch

Six-phase workflow for beating the openai/parameter-golf leaderboard.

## Phase 1: Pull and rank all competitor PRs

Fetch every PR's metadata and diff in one shot:

```bash
scripts/pull-pr-diffs.sh openai/parameter-golf pr_diffs 60
```

This saves `pr_<number>_view.txt` and `pr_<number>_diff.txt` for each PR into the output directory. Then sort each technique into three buckets:

| Tier | Criteria | Examples |
|------|----------|---------|
| **High** | Verified score on 8xH100 | Sliding window eval, seq_len tuning, fp16 embed |
| **Medium** | Promising but untested at scale | QAT, mixed-precision layers |
| **Low** | Negative or architecture-specific | Warmdown-only, depth recurrence |

Per PR, note: score delta, artifact size impact, step time change, quant gap.

## Phase 2: Compose the config

Stack High-tier techniques in BPB-impact order. Key interactions to watch:

- **fp16 embed + low LR** already brings quant gap to ~0.001 — QAT is redundant and slows steps
- **seq_len=4096 + eval_batch_seqs=1024** — OOM (4.2M tokens/fwd pass). Use `eval_batch_seqs=128`
- **val_loss_every > 0** eats ~20s of training budget at seq_len=4096 — set to 0

Best validated config (1.18335372 BPB, 8xH100 SXM):

```bash
TRAIN_SEQ_LEN=4096         # 4x richer context per step
TRAIN_BATCH_TOKENS=393216  # 3/4 batch = more steps per minute
MATRIX_LR=0.02             # halved — lower quant gap, smoother convergence
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000
MLP_HIDDEN=992             # trim MLP to fit fp16 embedding under 16MB
QAT=0                      # redundant with fp16 embed
VAL_LOSS_EVERY=0           # skip mid-run evals, saves ~300 steps
EVAL_STRIDE=64             # each token gets ~4032 tokens of context
EVAL_BATCH_SEQS=128        # safe at seq_len=4096
```

## Phase 3: Run and watch the log

```bash
RUN_ID=<name> SEED=1337 DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MLP_HIDDEN=992 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

In `logs/<RUN_ID>.txt`, look for:
- `step_avg` ~55-65ms/step is healthy
- `final_eval_mode:sliding_window stride:64 batch_seqs:128` — eval started
- `final_int8_zlib_roundtrip_exact val_bpb:<score>` — final score

If the log stops after `final_eval_mode:...` with no score, the sliding window eval OOM'd. Halve `EVAL_BATCH_SEQS` and rerun.

## Phase 4: Run 3 seeds for significance

The challenge requires p < 0.01 beating SOTA by >= 0.005 BPB. Repeat Phase 3 with SEED=1338 and SEED=1339, then:

```python
threshold = current_SOTA - 0.005
t = (threshold - mean_bpb) / (std_bpb / sqrt(3))
# Need t > 6.965 for p < 0.01 (df=2, one-tailed)
```

## Phase 5: Package the submission

```
records/track_10min_16mb/YYYY-MM-DD_<Name>/
├── README.md
├── submission.json
├── train.log
├── train_seed1338.log
├── train_seed1339.log
└── train_gpt.py
```

Run command in README must use the full path:
```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/<Name>/train_gpt.py
```

Artifact check: `bytes_model_int8_zlib + bytes_code < 16,000,000`

## Phase 6: PR checklist

- [ ] PR only adds the new `records/` folder — no root `train_gpt.py` changes
- [ ] `train_gpt.py` is under 1500 lines
- [ ] `val_bpb` in `submission.json` matches the exact log line
- [ ] README has t-statistic, df, and p-value
- [ ] Artifact is under 16MB

## Reference

- Baseline: 1.2244 BPB
- Best result with this workflow: **1.18335372** (mean 1.18418 across 3 seeds, std 0.00075)
- Sliding window eval takes ~4-5 min at seq_len=4096 on 8xH100
