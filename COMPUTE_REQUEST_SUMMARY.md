# Why This Project Merits Compute

## Short Version

This project already has strong local evidence and is no longer in the "untested idea" stage.

## What Was Built

- a new MPK-inspired causal language-model path was developed locally against the Parameter Golf baseline workflow
- controlled local experiments were run to compare it against baseline configurations
- additional tooling was added to support honest iteration on consumer/local GPUs:
  - validation slicing
  - backend fallback
  - progress reporting
  - compile toggles

## Why More Compute Is Justified

The strongest local proxy result so far reached:

- after a `500`-step local proxy run
- `final_int8_zlib_roundtrip val_bpb 2.1952`
- compressed artifact size `8,856,992` bytes

Run context:

- `500` training iterations
- `1,048,576` validation tokens
- tuned `8 x 384` MPK configuration

That means the current leading local configuration:

- improves substantially beyond short smoke tests
- remains under the 16 MB artifact budget
- appears competitive enough to justify longer official-condition testing

## What Has Already Been Shown

- the approach beat a parameter-matched baseline in local proxy testing
- naive scaling was not enough by itself
- tuned optimization materially improved the larger model
- the best tuned configuration kept improving through a longer 500-step run

## Why The Compute Ask Is Reasonable

The remaining uncertainty is not whether the project is real. The remaining uncertainty is whether the local advantage survives under official challenge-style runs with larger compute budgets and fuller evaluation.

That is exactly the stage where external compute is most useful.
