# parGolfMPK

Personal working fork for OpenAI's [Parameter Golf](https://github.com/openai/parameter-golf) challenge.

This repo is for public-facing code and general tooling while I explore parameter-efficient language modeling ideas. It is not intended to mirror the full upstream README.

## What This Repo Contains

- a sanitized training script based on the challenge baseline
- local quality-of-life improvements for iteration
- public-safe repo structure for later submissions

Current public-safe additions include:

- progress bars for training and validation
- validation-size limiting for faster local proxy runs
- automatic SDP backend fallback for local CUDA setups
- compile toggles for easier debugging

## Why This Is Worth Compute

This fork exists because local experiments produced a credible positive signal on a novel architecture direction for Parameter Golf.

Public-safe summary of current evidence:

- local matched-budget comparisons beat a parameter-matched baseline on post-quantization `val_bpb`
- tuned larger variants improved further once the optimizer recipe was adjusted
- a longer local proxy run reached `final_int8_zlib_roundtrip val_bpb 2.1952`
- that same local proxy run stayed under the 16 MB compressed artifact budget at `8,856,992` bytes

The point of requesting compute is not to test a vague idea. It is to determine whether an already-promising local result survives under official challenge-like training and evaluation conditions.

## Public vs Private

I am intentionally keeping some materials out of the public branch until I am ready to submit or disclose them more broadly.

Ignored local/private files include:

- `final_model.pt`
- `final_model.int8.ptz`
- `MPK_EXPERIMENTS.md`
- `PORTFOLIO_RUNNING_NOTES.md`
- `PRIVATE_*`

## Upstream Challenge

The official challenge, rules, leaderboard, and submission instructions live in the upstream repo:

- [openai/parameter-golf](https://github.com/openai/parameter-golf)

If you want the canonical challenge setup, use the upstream repository.

## Notes

This fork is primarily a working repository and backup point, not a polished public release.
