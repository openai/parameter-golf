Run exactly one Parameter Golf experiment-preparation iteration in the current repository.

Context to read before acting:
- `gpt-pro.md`
- `ideas/README.md`
- `ideas_wild/README.md`
- `results.tsv` if it exists
- the latest relevant log under `logs/`

Goal:
- Improve final `final_int8_zlib_roundtrip_exact val_bpb` on the FineWeb validation set.
- Lower `val_bpb` is better.

Protocol:
1. Inspect `git status`, recent commits, `results.tsv`, and recent logs.
2. Choose exactly one focused idea or ablation to test.
3. Make only the code changes needed for that one idea. You may edit `train_gpt.py` directly.
4. Write `controller_state/current_run.env` with exactly these shell variables:
   - `IDEA`
   - `NOTES`
   - `EXTRA_ENV`
5. `EXTRA_ENV` must be a single-line space-separated list of additional `KEY=VALUE` pairs for this run, for example `TRAIN_SEQ_LEN=512 EVAL_SEQ_LEN=1024`.
6. Make exactly one git commit for your experiment before stopping so the controller can export it as a single patch and a later reviewer can revert it cleanly if it loses.
7. Do not run training yourself. The controller will run the exact experiment remotely on the GPU box.
8. Stop after one completed experiment-preparation iteration.

Rules:
- Do not ask for confirmation.
- Do not delete or redownload the dataset.
- Do not change the tokenizer or dataset export path unless that is the explicit experiment.
- Prefer bounded changes that can be evaluated in one run.
- Keep the code change self-contained and cherry-pick friendly. The controller may apply it onto a slightly newer reviewed state.
- Keep the repo runnable after the iteration.
- Do not update `results.tsv` or `reviews.tsv` yourself.
- Do not revert the experiment commit yourself. The reviewer handles the keep/revert decision after the remote run finishes.
- The repo must be left with a clean working tree except for the committed experiment change and the untracked `controller_state/current_run.env`.
- Stop after one completed experiment-preparation iteration.
