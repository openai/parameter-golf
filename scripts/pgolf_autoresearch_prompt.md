Run exactly one Parameter Golf experiment iteration in the current repository.

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
3. Make only the code changes needed for that one idea.
4. Commit your experiment before running it so it can be reverted cleanly if it loses.
5. Run exactly one training/eval command using the environment variables already provided by the harness.
6. Parse the run log for:
   - `final_int8_zlib_roundtrip_exact`
   - `Total submission size int8+zlib`
7. Append one TSV row to `results.tsv` with:
   - `iteration`
   - `timestamp`
   - `model`
   - `run_id`
   - `status`
   - `val_bpb`
   - `val_loss`
   - `size_bytes`
   - `commit`
   - `idea`
   - `notes`
8. If the run improves the best prior kept `val_bpb`, keep the commit.
9. If it does not improve, revert only the commit you just made.

Rules:
- Do not ask for confirmation.
- Do not delete or redownload the dataset.
- Do not change the tokenizer or dataset export path unless that is the explicit experiment.
- Prefer bounded changes that can be evaluated in one run.
- Keep the repo runnable after the iteration.
- Only revert your own just-created commit, not unrelated history.
- Stop after one completed experiment iteration.
