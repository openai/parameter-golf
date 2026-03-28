# Parameter Golf — full map (where everything lives)

This file is the **single place** that describes what is on GitHub vs what is only on your Mac.

## 1. What counts for the challenge (on GitHub)

| Item | Location |
|------|----------|
| Your fork | `https://github.com/0xjaishy/parameter-golf` |
| PR to OpenAI | `https://github.com/openai/parameter-golf/pull/223` |
| Git branch | `submission/allinone-smeargate-int6qat-slidingwindow` |
| **Submission folder** | `records/track_10min_16mb/2026-03-20_SOTA_TTT_RoPE50K_EMA_Curriculum/` |
| Entry script | `train_gpt.py` (run with `torchrun --standalone --nproc_per_node=8` on **8×H100 SXM**) |
| Metadata | `submission.json` — set `val_loss`, `val_bpb`, `bytes_total` after a real run |
| Short write-up | `README.md` in that same folder |

**Rule:** For a clean competition PR, reviewers mainly care about that **records/...** directory. Everything else in the repo is optional.

## 2. One folder on your Mac (no duplicate repo)

Use **only this git clone** (your fork), e.g.:

**`/Users/shivashish/Desktop/parameter-golf-fork`**

Open **that** path in Cursor/VS Code. Do **not** keep a second `parameter-golf` copy on Desktop — it wastes space and drifts out of sync.

| Path | Purpose |
|------|---------|
| `HANDOFF.md` (this file) | Map of URLs, paths, commands |
| `README.md` | Upstream readme + Mac / prep notes |
| `scripts/check_submission_local.py` | CPU/MPS smoke test for a `train_gpt.py` |
| `scripts/sample_fineweb_tokens.py` | Decode shard samples |
| `scripts/validate_submission.py` | AST + sliding ref + import + forward/quant (defaults to SOTA `train_gpt.py`) |
| `data/datasets/`, `data/tokenizers/` | Downloaded data (**gitignored**, stays local) |
| `.venv/` | Python venv (**gitignored**, stays local) |

Committed to git: code, `records/`, `HANDOFF.md`, `scripts/`. **Not** committed: `data/datasets`, `.venv` (see `.gitignore`).

## 3. Commands (copy-paste)

**Download minimal data (val + 1 train shard):**

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

**Peek at real text in the corpus:**

```bash
python3 scripts/sample_fineweb_tokens.py --shard val --num-samples 5 --length 96
```

**Smoke-test your submission file (paths may differ):**

```bash
python3 scripts/check_submission_local.py \
  records/track_10min_16mb/2026-03-20_SOTA_TTT_RoPE50K_EMA_Curriculum/train_gpt.py
```

**Validate (AST + optional torch checks; same default path):**

```bash
python3 scripts/validate_submission.py
```

**MLX baseline on Apple Silicon:**

```bash
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 python3 train_gpt_mlx.py
```

**Official training (8×H100, from submission directory):**

```bash
cd records/track_10min_16mb/2026-03-20_SOTA_TTT_RoPE50K_EMA_Curriculum
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 4. Git workflow

Work inside this repo only. Edit, then:

```bash
git add -A && git status
git commit -m "your message"
git push origin submission/allinone-smeargate-int6qat-slidingwindow
```

## 5. After you get a real GPU run

1. Note `val_bpb` (and `val_loss`, artifact bytes) from the log.  
2. Update `submission.json` in the submission folder.  
3. Commit and push the fork branch; PR #223 updates automatically.  
4. Mark the PR ready for review when you meet record rules (e.g. multiple seeds if required).

---

*Last aligned with: SOTA+ submission (PR #198 base + RoPE50K + EMA + curriculum + TTT).*
