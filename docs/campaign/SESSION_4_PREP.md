# Session 4 Prep

Date: 2026-04-19

## Active objective

Run Fallback Cascade Level 1A on the preserved Gate-A checkpoint from Session 3. This is a zero-retraining export-only pass using `#1586`-style levers:
- per-layer adaptive GPTQ `clip_sigmas`
- int7 embeddings

Time-box:
- `1–2` requant-only runs
- expected cost `~$6–12`
- kill criterion: `<0.001 BPB gain` or artifact exceeds the `16,000,000 B` cap

## Preconditions

1. Terminate idle pod `utwe9wnuze72ds` via the RunPod web UI.
2. Rotate the RunPod API key that was pasted during Session 3.
3. Rotate the HF token that was pasted during Session 3.
4. Rebuild and push the RunPod image with the local Dockerfile keepalive fix, then repin the new digest in `scripts/runpod_pipeline/pod_launch.md`.
5. Read `docs/runpod_pitfalls.md` before launching any paid pod.

## Required follow-up review

1. Do a sign-direction code review of the corrector blend (`+ alpha * logit_bias` vs `- alpha * logit_bias`).
2. If a sign error is found, the Session 3 negative result is invalidated and the ablations must be rerun.
3. If the implementation is correct, the negative result stands and the corrector lane remains closed.

## Execution scope

- Use the preserved Gate-A artifact:
  - `amay01/parameter-golf-session3-artifacts/runs/runs_20260418_2204.tar.gz`
  - MD5 `caf8adf63d8c80965f6671beba95d7aa`
- Do not retrain.
- Do not reopen the corrector lane unless the sign review proves the implementation itself was wrong.
- Do not auto-file the upstream PR; Session 3 PR material is for human filing only.
