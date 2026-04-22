# Spec 005 — weight-delta analysis — execution notes

Pod: `cigivrlyk6j5aa`, 1×H100 SXM US-NE-1, ~3 min wall, ~$0.15 cost. Terminated after pull.

## Mechanical

- Loaded 3 checkpoints from `/workspace/runs/000-sota-replication/checkpoints/` on NA-1 volume; all float params processed (113 entries, ~10 skipped as non-floating).
- Param naming observed: `blocks.{N}.attn.c_q.weight`, `c_k.weight`, `c_v.weight`, `c_proj.weight`, `mlp.fc.weight`, `mlp.proj.weight`, `mlp.fc_s.weight` (scalar), `mlp.c_s.weight`. Classifier mapped `c_q/c_k/c_v` → `attn_other` (not `attn_qkv`) because names aren't "qkv".
- Loop layers: 3, 4, 5 (per CLAUDE.md 3-layer depth recurrence).

## Quick read on the numbers

**Per-step weight movement was ~2.5× FASTER during the flat zone (step 1500→2275) than after (step 2275→3412)**, uniformly across all layers and param types. Loop-layer avg 1203 vs 464 per-step relative delta (×1e-6); non-loop 1112 vs 457.

That's consistent with **cause A** (weights reorganizing while loss is stuck), not cause B (model genuinely stuck). During the flat zone the model is changing a lot; after the flat zone it's settling into a more stable regime with slower movement per step.

Loop layers (3,4,5) don't move *disproportionately* compared to non-loop layers in either interval — the ratio loop/non-loop is ~1.08× in both periods. So the flat zone isn't specifically a "recurrence adaptation event" localized to loop layers.

Handback is research's to interpret further.

## Artifacts in this dir

- `delta_table.md` — readable per-(layer, ptype) table + loop/non-loop aggregate + top-15 movers
- `delta_layers.json` — raw per-param numbers for any follow-up analysis
- `weight_delta.py` — the script
- `run.log` — execution output
