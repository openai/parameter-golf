# Agent Sync

Date: 2026-03-30

## Current Objective

Implement and run Session 05c-plus training bundle on 8xH100.

GPTQ is **parked** for the current model stack after 7 conclusive ablations.
The next compute allocation is a training-quality run, not more GPTQ debugging.

## Challenge Reality

- Official leaderboard entry is **record-gated**, not top-5-open-entry.
- A record submission must beat the current official SOTA by at least `0.005` nats and show `p < 0.01`.
- Current official merged #1 is PR #1019 at `1.1147` BPB (3-seed mean `1.88218` nats).
- Record threshold: `<= 1.87718` nats.
- Current open frontier is lower:
  - PR #1089: `1.1086` BPB, 3-seed mean
  - PR #1060: `1.1122` BPB, 3-seed mean

## Current Mainline Plan

### Phase 1: Session 05c-plus training bundle (ACTIVE)

Plan: `docs/superpowers/plans/2026-03-30-session-05c-plus.md`
Code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

Four changes on the Session 03 anchor:
1. **XSA 4 → 11** — XSA on all layers (trivial constant)
2. **VE128 on layers 9-10** — shared ValueEmbedding (new module)
3. **Warmdown 3000 → 3500** — trivial constant
4. **LeakyReLU(0.5)²** — replaces relu in MLP (one line, aligns with PR #1019)

SWA is **not included** — dead code in both PR #1019 and #634 (collected but only EMA applied).

Target: sliding s64 val_bpb < 1.126 (vs anchor 1.129)

### Phase 2: GPTQ test on new checkpoint (NOT YET EXECUTABLE)

After 05c-plus training completes:
1. Evaluate with naive int6 export first (already in 05c-plus script)
2. To run GPTQ replay: must first port VE128 + LeakyReLU² into the GPTQ export script
   - The parked Session 05b script (`2026-03-29_full_hessian_gptq/train_gpt.py`) has the old architecture (no VE, relu²)
   - A merge step is required before GPTQ replay can load a 05c-plus checkpoint
3. If GPTQ is sane → continue from there
4. If GPTQ is still bad → park permanently, keep naive-int6 result

### Parked

- Session 05b GPTQ on current anchor (7 ablations, all failed, code proven correct)
- Saved-container FA3 throughput path
- TTT
- Broad novelty probes

## Session 05b GPTQ: Conclusive Parking Summary

Seven ablations on the same Session 03 checkpoint:

| # | Variant | gptq_diag | Roundtrip gap | Outcome |
|---|---------|-----------|---------------|---------|
| 1 | Initial smoke (1xH100) | 66/66 | +0.212 | Bug found |
| 2 | Loop fix + percentile search | 66/66 | +0.335 | Still bad |
| 3 | actorder=False | 66/66 | +0.395 | Worse |
| 4 | block_size=full | 66/66 | +0.395 | No change |
| 5 | Hessian normalize+damp | 66/66 | +0.337 | Identical |
| 6 | PR #1019 verbatim transplant | 66/66 | +0.337 | **Byte-identical MSE** |
| 7 | AR self-gen calibration | crash | N/A | Non-PD Hessian |

Key conclusion: ablation #6 proves the GPTQ code is functionally correct. The failure is model-specific, not a code bug. PR #1019 uses `leaky_relu(0.5)` while our anchor uses `relu`.

## Fixed Reference Results

- Session 03 anchor (`8xH100`, `serv-3342`)
  - sliding s64 `val_bpb=1.12904446`
  - pre-quant EMA `val_bpb=1.14472403`
  - int6 roundtrip `val_bpb=1.15247273`
  - steps `6564`
  - step_avg `91.37 ms`
  - artifact `15751324` bytes

## Canonical Files

- Shared mutable state: `docs/campaign/AGENT_SYNC.md`
- Stable rules: `CLAUDE.md`
- 05c-plus plan: `docs/superpowers/plans/2026-03-30-session-05c-plus.md`
- 05c-plus code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
- GPTQ experiment (parked): `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/`
- Codex memory:
  - `docs/codex-memory/decisions.md`
  - `docs/codex-memory/project-state.md`
  - `docs/codex-memory/next-session.md`

## Workspace

- Local repo: `/home/amay/Work/parameter-golf`
- Remote repo: `/netscratch/$USER/parameter-golf`

Use `git clone` and `git pull` by default.
