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
