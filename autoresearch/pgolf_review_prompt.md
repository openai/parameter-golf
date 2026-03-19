Review exactly one just-finished Parameter Golf experiment after it has already passed pre-review and been run on the GPU.

Context to read before acting:
- `results.tsv`
- `reviews.tsv`
- the fetched remote log under `remote_logs/`
- `git log --oneline -n 5`
- the latest experiment commit
- the candidate manifest and run manifest
- `gpt-pro.md`, `ideas/README.md`, and `ideas_wild/README.md` if needed

Goal:
- Review the latest experiment from a fresh perspective.
- Decide whether to keep or revert the latest experiment commit based mainly on metric quality and trustworthiness.

Protocol:
1. Identify the most recent experiment commit and the corresponding latest run log.
2. Inspect the candidate hypothesis and expected signals.
3. Check whether the observed metrics and logs support the hypothesis.
4. Focus on:
   - actual metric outcome
   - trustworthiness of the comparison
   - any reason the result should not be believed
5. Read the best prior kept result and compare this run against it.
6. Decide one of:
   - `keep`
   - `revert`
7. Write the decision file as a JSON object with exactly these string fields:
   - `DECISION`
   - `SUMMARY`
   - `FINDINGS`
8. Stop after one completed post-review.

Rules:
- Do not run another training job.
- Do not edit the repository yourself.
- Only decide whether the latest experiment commit should be kept or reverted.
- Be skeptical of tiny wins and call out when noise or evaluation mismatch may explain them.
- Pre-review already handled most style and code-quality concerns. Revisit them only when they affect trustworthiness or interpretation of the result.
