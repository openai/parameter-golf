Review exactly one proposed Parameter Golf patch before it is allowed onto the remote GPU queue.

Context to read before acting:
- the candidate patch file
- the candidate rationale/env file
- `results.tsv`
- `reviews.tsv`
- `controller_state/autoresearch/history/summary.md` if it exists

Goal:
- Catch weak, buggy, untrustworthy, or invalid changes before they spend GPU time.
- Approve only candidates that are clear enough to test.

Protocol:
1. Read the patch and the candidate rationale carefully.
2. Check whether the hypothesis and expected signals actually match the code change.
3. Review with a code-review mindset:
   - correctness
   - invalid comparisons
   - hidden confounders
   - missing accounting
   - reasons the claimed win would not be trustworthy
4. Decide one of:
   - `approve`
   - `revise`
5. Write the decision file as a JSON object with exactly these string fields:
   - `DECISION`
   - `SUMMARY`
   - `FINDINGS`
   - `FEEDBACK`
6. If `revise`, `FEEDBACK` must contain concrete instructions for the next proposer round.
7. Stop after one completed pre-review.

Rules:
- Do not edit the repository yourself.
- Do not run training.
- Be strict about trustworthiness.
- Prefer `revise` over approving a vague or weakly justified patch.
