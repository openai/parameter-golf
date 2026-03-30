# Shared Agent Entry Point

Start here for both Claude Code and Codex.

## Read First

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`

## Purpose

`docs/campaign/AGENT_SYNC.md` is the mutable source of truth for:

- current objective
- current scope
- latest measured results
- next commands to run

`CLAUDE.md` contains the standing coordination rules for sessions, updates, and disagreement handling.

## Current Working Mode

- Active goal: commit, push, and launch Session 05c-plus training bundle on `8xH100`
- Code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py` (implemented, validated)
- Bundle: XSA-all + VE128 + warmdown 3500 + LeakyReLU(0.5)² on Session 03 anchor
- Next phase: evaluate 05c-plus naive int6 results, then consider GPTQ port if quality improves
- Out of scope: FA3, TTT, SWA, Session 05b GPTQ debugging
