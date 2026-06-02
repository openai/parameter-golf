# Submission Roadmap — Parameter Golf

> **Goal**: Submit a model to `records/track_10min_16mb/` (or `track_non_record_16mb/`) that beats or approaches SOTA (1.1147 BPB as of 2026-04-14).

---

## Phase 0 — Environment Setup (Day 1)

- [ ] Clone repo and create venv: `python3 -m venv .venv && source .venv/bin/activate`
- [ ] Install base deps: `pip install mlx numpy sentencepiece huggingface-hub datasets tqdm`
- [ ] Download FineWeb validation + 1 training shard (quick smoke test):
      `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`
- [ ] Run local MLX smoke test (Mac M-series):
      `RUN_ID=smoke ITERATIONS=200 VAL_LOSS_EVERY=0 python3 train_gpt_mlx.py`
- [ ] Verify baseline compiles and prints `val_bpb ≈ 1.22`
- [ ] Sign up on RunPod (needed for any serious run)

---

## Phase 1 — Baseline Understanding (Day 1–2)

- [ ] Read `train_gpt.py` end-to-end; map every class to the leaderboard
- [ ] Read `config.py` (this project's central config)
- [ ] Study the top 3 record READMEs in `records/track_10min_16mb/`
- [ ] Reproduce the Naive Baseline score locally (≈1.22 BPB)
- [ ] Understand the artifact size formula: `code_bytes + compressed_model_bytes ≤ 16,000,000`

---

## Phase 2 — Identify Your Ideas (Day 2–3)

Pick ONE primary idea from the techniques list (`docs/techniques-analysis.md`).
Focus areas that are still open per the README's "Requests for PRs":

| Idea | Difficulty | Expected Δ BPB | Status |
|------|-----------|----------------|--------|
| Tune WARMDOWN_ITERS to 3500–4000 | Easy | −0.003 | Not tried |
| MLP_MULT = 3 (1536 hidden) + Int6 QAT | Medium | −0.005 | In top runs |
| Sliding Window Eval (stride=64) | Medium | −0.02 | Done ✓ |
| BigramHash augmentation | Medium | −0.007 | Done ✓ |
| XSA on all layers | Medium | −0.005 | Done ✓ |
| Full GPTQ (AR self-gen calibration) | Hard | −0.008 | Done ✓ |
| JEPA / Text diffusion | Hard | Unknown | Open |
| State-space model | Hard | Unknown | Open |
| H-net tokenization | Hard | Unknown | Open |

---

## Phase 3 — Iterative Experiments (Day 3–7)

Run on a **1×H100 pod** first (cheaper), `MAX_WALLCLOCK_SECONDS=0` for full runs:

```bash
# Template: change env vars from config.py
SEED=42 NUM_LAYERS=11 MLP_MULT=3 WARMDOWN_ITERS=3500 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Experiment log discipline:
- Save each run output to `logs/<run_id>.log`
- Note: step time, pre-quant BPB, final BPB, artifact bytes
- Compare against current SOTA before scaling to 8×H100

---

## Phase 4 — Multi-Seed Validation (Day 7–9)

Required for a record submission: must prove `p < 0.01` with Welch's t-test.
- Run **3 seeds** (e.g. 42, 314, 999) on **8×H100 SXM** with 600 s wall-clock cap
- Collect mean ± std of `val_bpb`
- Your mean must beat SOTA by ≥ 0.005 BPB (≥ 0.003 nats)

```bash
# Seed 1
SEED=42  TARGET_MB=15.9 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/seed42.log
# Seed 2
SEED=314 TARGET_MB=15.9 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/seed314.log
# Seed 3
SEED=999 TARGET_MB=15.9 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/seed999.log
```

---

## Phase 5 — Prepare Submission Files

Required files in `records/track_10min_16mb/YYYY-MM-DD_<YourRunName>/`:

| File | Content |
|------|---------|
| `README.md` | Architecture table, changes, lineage, run command |
| `submission.json` | author, github_id, val_bpb, seeds, hardware, delta vs SOTA |
| `train_seed<N>.log` | Full stdout from each seed run |
| `train_gpt.py` | Your modified script (must compile & run from the records folder) |
| `requirements.txt` | Any extra packages beyond the base env |

Use the top record's README as a template (`records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md`).

---

## Phase 6 — Open a PR

1. Fork `openai/parameter-golf` on GitHub
2. Create branch: `record/<your-run-name>`
3. Add only the new records folder (no other changes)
4. PR title: `Record: <BPB> — <short technique description>`
5. PR body: paste key stats + lineage from your README
6. Check: does it beat SOTA by ≥ 0.005 BPB? Is the delta statistically significant?

---

## Non-Record Path (Faster)

If your idea is interesting but doesn't beat SOTA, submit to `records/track_non_record_16mb/`.
Requirements are lighter — just needs to run successfully and have a well-written README.
Good for: JEPA, text diffusion, SSMs, negative results, 4-hour runs.

---

## Timeline Summary

| Day | Milestone |
|-----|-----------|
| 1 | Env working, baseline runs locally |
| 2 | Top records read, idea chosen |
| 3–5 | Experiments on 1×H100, technique validated |
| 6–7 | Stack combined, full 600s run on 1×H100 |
| 8–9 | 3-seed run on 8×H100 SXM |
| 10 | PR opened |

**Challenge closes: April 30, 2026** — 16 days from today.
