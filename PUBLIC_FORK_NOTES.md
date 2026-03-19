# Public Fork Notes

## Purpose

This branch is a public-safe working branch for OpenAI Parameter Golf compute-request visibility and general project backup.

It is meant to show that this is a real technical project with concrete model/training work, while keeping some private experimental details local until I am ready to disclose them more broadly.

## Project Summary

I am exploring an MPK-inspired approach to parameter-efficient causal language modeling. The core idea is to use parallel token-processing pathways operating at different temporal granularities, with an intermediate control pathway modulating the others before fusion.

The full experimental method is being kept local for now, but the public branch reflects the surrounding engineering work used to support the project.

## Public-Safe Code Included Here

This branch includes a sanitized training-script variant with general-purpose iteration improvements:

- progress bars for training and validation
- validation-size limiting for fast local proxy runs
- automatic scaled-dot-product-attention backend fallback for local CUDA setups
- compile toggles for easier debugging and portability
- ignore rules for generated artifacts and local/private notes

These changes are visible in `train_gpt.py`, `.gitignore`, and the README updates in this branch.

## What Has Been Tried So Far

At a high level, I have:

- integrated a new model path into the baseline training workflow locally
- run matched-baseline comparisons
- explored depth vs width tradeoffs
- tested optimizer retuning for larger variants
- run a longer local proxy experiment to check whether improvements persist with more training

## Public-Safe Results Summary

Without exposing the full local notes, the key result is:

- tuned MPK-style variants beat a parameter-matched baseline in local proxy experiments on post-quantization `val_bpb`
- the leading longer local proxy run reached `final_int8_zlib_roundtrip val_bpb 2.1952`
- the corresponding compressed artifact remained under the 16 MB challenge limit in that local proxy setup

These results are being used to justify requesting larger-scale compute for official-condition testing.

## What Is Intentionally Not Public Here

The following are intentionally kept local/private for now:

- detailed experiment logs
- generated model artifacts
- private portfolio/research notes
- some unreleased method details and tuned configurations

Ignored local/private files include:

- `final_model.pt`
- `final_model.int8.ptz`
- `MPK_EXPERIMENTS.md`
- `PORTFOLIO_RUNNING_NOTES.md`
- `PRIVATE_*`
