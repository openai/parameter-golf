---
name: parameter-golf-arjunautoresearch
description: Systematically beat the OpenAI Parameter Golf challenge leaderboard by using the GitHub CLI to analyze all open PRs, bucket techniques by expected BPB impact, compose the best combination, run training experiments iteratively, package a compliant submission, and open a PR. Use when working on the parameter-golf challenge, trying to improve val_bpb, or packaging a leaderboard submission.
---

# Parameter Golf AutoResearch

Workflow for beating the openai/parameter-golf leaderboard using systematic PR analysis and iterative experimentation.

## Phase 1: Research — Analyze All PRs

Use the GitHub CLI to inspect every open PR on the upstream repo:

```bash
gh pr list --repo openai/parameter-golf --state all --limit 60
# For each high-potential PR:
gh pr view --repo openai/parameter-golf <number>
gh pr diff --repo openai/parameter-golf <number> > pr_diffs/pr_<number>_diff.txt
```

Bucket each PR's technique into three tiers:

| Tier | Criteria | Examples |
|------|----------|---------|
| **High** | Verified score + clean mechanism | Sliding window eval, seq_len tuning, fp16 embed |
| **Medium** | Promising but unverified on 8xH100 | QAT, mixed-precision layers |
| **Low** | Negative result or architecture-specific | Warmdown-only, depth recurrence without sufficient steps |

Key metrics to extract per PR: score delta, artifact size, step time change, quant gap before/after.

## Phase 2: Compose Changes

Stack High-tier techniques in order of expected BPB gain. Watch for interactions:

- **fp16 tied embedding + low LR** already reduces quant gap to ~0.001 → QAT becomes redundant and costs step time
- **seq_len=4096 + eval_batch_seqs=1024** → OOM at 4.2M tokens/forward pass; use `eval_batch_seqs=128` instead
- **val_loss_every > 0** wastes ~20s of training budget at seq_len=4096; set to 0

Proven configuration (1.18335372 BPB on 8xH100 SXM):

```bash
TRAIN_SEQ_LEN=4096       # 4x richer training signal per step
TRAIN_BATCH_TOKENS=393216  # 3/4 batch = more updates per wallclock second
MATRIX_LR=0.02            # half default → lower quant gap, better convergence
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99        # stronger smoothing
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000
MLP_HIDDEN=992            # trim to fit fp16 embedding under 16MB
QAT=0                     # off: fp16 embed already kills quant gap
VAL_LOSS_EVERY=0          # no intermediate evals; saves ~300 steps
EVAL_STRIDE=64            # sliding window: each token gets ~4032 context
EVAL_BATCH_SEQS=128       # safe with seq_len=4096 (524K tokens/fwd pass)
```

## Phase 3: Run and Monitor

```bash
RUN_ID=<name> SEED=1337 DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MLP_HIDDEN=992 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Watch the log (`logs/<RUN_ID>.txt`) for:
- `step_avg` — should be ~55-65ms/step at seq_len=4096 on 8xH100
- Final line before sliding window: `final_eval_mode:sliding_window stride:64 batch_seqs:128`
- Score line: `final_int8_zlib_roundtrip_exact val_bpb:<score>`

If the log ends at `final_eval_mode:...` with no score → OOM in sliding window eval. Reduce `EVAL_BATCH_SEQS`.

## Phase 4: Statistical Significance (3 seeds)

The challenge requires p < 0.01 beating SOTA by ≥ 0.005 BPB. Run seeds 1337, 1338, 1339:

```bash
# Repeat Phase 3 with SEED=1338 and SEED=1339
```

Compute one-sample t-test:
- threshold = current_SOTA - 0.005
- t = (threshold - mean) / (std / sqrt(3))
- Critical value df=2, p < 0.01 one-tailed: **t = 6.965**

## Phase 5: Package Submission

Folder structure required:

```
records/track_10min_16mb/YYYY-MM-DD_<SubmissionName>/
├── README.md       # approach, config, run command, metrics, significance stats
├── submission.json # author, github_id, name, val_bpb, bytes_total, bytes_code
├── train.log       # canonical seed 1337 log
├── train_seed1338.log
├── train_seed1339.log
└── train_gpt.py    # standalone; must compile and run from this folder
```

**submission.json required fields:** `author`, `github_id`, `name`, `val_bpb`, `bytes_total`, `bytes_code`

**Run command in README must use full path:**
```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/<FolderName>/train_gpt.py
```

**Artifact size check:** `bytes_model_int8_zlib + bytes_code < 16,000,000`

## Phase 6: PR Compliance Checklist

Before opening PR against `openai/parameter-golf`:

- [ ] PR diff touches **only** files inside the new `records/` folder (no root `train_gpt.py`)
- [ ] `train_gpt.py` inside the folder is under 1500 lines
- [ ] Run command references the full records path
- [ ] `val_bpb` in `submission.json` matches the exact `final_int8_zlib_roundtrip_exact` log line
- [ ] Statistical significance section in README with t-statistic and p-value
- [ ] All three seed logs included
- [ ] Artifact ≤ 16,000,000 bytes

```bash
git add records/track_10min_16mb/<FolderName>/
git commit -m "Add <SubmissionName> record: val_bpb <score>"
git push -u origin <branch>
gh pr create --repo openai/parameter-golf --head <user>:<branch> --base main \
  --title "Record: <SubmissionName> — val_bpb <score>" --body "..."
```

## Reference

- Baseline: 1.2244 BPB (9L 512d SP-1024 seq1024)
- Best validated result with this workflow: **1.18335372 BPB** (mean 1.18418 across 3 seeds)
- Eval budget: separate 10 min beyond training (sliding window ~4-5 min at seq_len=4096)
