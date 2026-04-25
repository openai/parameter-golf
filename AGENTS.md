# AGENTS.md

This repo's agent context lives in [CLAUDE.md](CLAUDE.md). Read that file for project structure, setup, env-driven config, submission workflow, and common mistakes.

Three invariants you must not violate even if you skip [CLAUDE.md](CLAUDE.md):

1. **The submission artifact is capped at 16,000,000 bytes** (decimal, not 16 MiB). Counted as code bytes + compressed model bytes.
2. **Validation data must never be accessed during training.** No "paid prefix" tricks compressing the val set into the artifact. Test-time training is only allowed on val tokens already scored.
3. **`train_gpt.py` and `train_gpt_mlx.py` must remain under 1500 lines** each — see the docstrings at the top of both files.

For the full rule set, see [CLAUDE.md](CLAUDE.md) "Hard Rules" and [README.md](README.md) "Submission Process".
