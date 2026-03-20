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

## 2. What is only on your Mac (Cursor workspace)

Path: **`/Users/shivashish/Desktop/parameter-golf`** — this folder is **not** a git repository by default. Tools and doc edits here are **local** until you copy or commit them somewhere.

| File | Purpose |
|------|---------|
| `README.md` (root) | Extra sections: Mac workflow, `sample_fineweb_tokens`, prep checklist |
| `scripts/check_submission_local.py` | Load `train_gpt.py` as a module; CPU/MPS forward + int6 roundtrip smoke test |
| `scripts/sample_fineweb_tokens.py` | Decode random windows from `.bin` shards with SentencePiece |
| `scripts/validate_submission.py` | Optional deeper checks (if present) |
| `data/datasets/fineweb10B_sp1024/*.bin` | Downloaded data (large; do **not** commit to git) |

Same `HANDOFF.md` should exist at the **fork root** after you pull/push — see section 4.

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

**MLX baseline on Apple Silicon:**

```bash
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 python3 train_gpt_mlx.py
```

**Official training (8×H100, from submission directory):**

```bash
cd records/track_10min_16mb/2026-03-20_SOTA_TTT_RoPE50K_EMA_Curriculum
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 4. Keeping GitHub in sync with this handoff

- **Fork** (`parameter-golf-fork`): should contain `HANDOFF.md` at repo root and `scripts/` with the two helper scripts so nothing is “only on one laptop” without a trace.
- **Workspace** (`Desktop/parameter-golf`): keep editing here; periodically copy `HANDOFF.md` + `scripts/*` into the fork and `git commit && git push origin <branch>`.

## 5. After you get a real GPU run

1. Note `val_bpb` (and `val_loss`, artifact bytes) from the log.  
2. Update `submission.json` in the submission folder.  
3. Commit and push the fork branch; PR #223 updates automatically.  
4. Mark the PR ready for review when you meet record rules (e.g. multiple seeds if required).

---

*Last aligned with: SOTA+ submission (PR #198 base + RoPE50K + EMA + curriculum + TTT).*
