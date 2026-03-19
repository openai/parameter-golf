Review exactly one just-finished Parameter Golf experiment in the current repository.

Context to read before acting:
- `results.tsv`
- the fetched remote log under `remote_logs/`
- `git log --oneline -n 5`
- the diff and commit for the most recent experiment commit
- `gpt-pro.md`, `ideas/README.md`, and `ideas_wild/README.md` if needed

Goal:
- Review the latest experiment from a fresh perspective.
- Decide whether to keep or revert the latest experiment commit based on exact evidence.

Protocol:
1. Identify the most recent experiment commit and the corresponding latest run log.
2. Review the code changes with a code-review mindset:
   - bugs
   - invalid comparisons
   - regressions
   - missing accounting
   - signs the claimed win is not trustworthy
3. Read the latest result row in `results.tsv` and compare it against the best prior kept result.
4. Decide one of:
   - `keep`
   - `revert`
5. If `keep`, leave the commit in place.
6. If `revert`, revert only the latest experiment commit and leave the repo clean.
7. Update the latest row in `results.tsv`:
   - set `decision`
   - preserve existing fields
   - append your short decision summary to `notes`
8. Append one TSV row to `reviews.tsv` with:
   - `iteration`
   - `timestamp`
   - `model`
   - `run_id`
   - `decision`
   - `commit`
   - `summary`
   - `findings`
9. Commit the ledger updates so the repo is clean at the end:
   - if `keep`, make one commit that records the updated `results.tsv` and `reviews.tsv`
   - if `revert`, first make the revert commit, then make one commit that records the updated `results.tsv` and `reviews.tsv`
10. Stop after one completed review.

Rules:
- Do not run another training job.
- Do not modify older history or unrelated commits.
- Only keep or revert the latest experiment commit.
- Be skeptical of tiny wins and call out when hardware noise or evaluation mismatch may explain them.
- Keep the repo runnable after the review.
