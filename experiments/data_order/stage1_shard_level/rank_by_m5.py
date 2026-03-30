"""Re-rank shards by M5 (val-trained bigram LM cross-entropy).
Validated as most reliable method: Spearman rho=0.984 across contiguous splits."""

import json

with open("experiments/data_order/stage1_shard_level/shard_analysis.json") as f:
    data = json.load(f)

# Lower CE under val LM = more val-like. m5_neg_val_ce is negated, so higher = more similar.
ranked = sorted(data, key=lambda x: x["m5_neg_val_ce"], reverse=True)

print("All 80 shards ranked by M5 (val-trained LM CE):")
print(f"  #  {'shard':>28s}    CE")
for i, r in enumerate(ranked):
    ce = -r["m5_neg_val_ce"]
    tag = " SELECTED" if i < 38 else ""
    print(f"  {i+1:2d}  {r['shard']:>28s}  {ce:.4f}{tag}")

selected = [r["shard"] for r in ranked[:38]]
excluded = [r["shard"] for r in ranked[38:]]

# Training order: least val-similar of selected first, most val-similar last
training_order = list(reversed(selected))

print(f"\nTraining order ({len(training_order)} shards, least -> most val-similar):")
for i, name in enumerate(training_order):
    ce = -next(r["m5_neg_val_ce"] for r in data if r["shard"] == name)
    print(f"  Step {i+1:2d}: {name}  (CE={ce:.4f})")

print(f"\nExcluded {len(excluded)} shards:")
for name in excluded:
    ce = -next(r["m5_neg_val_ce"] for r in data if r["shard"] == name)
    print(f"  {name}  (CE={ce:.4f})")

result = {
    "n_selected": 38,
    "training_order": training_order,
    "excluded": excluded,
    "method": "M5 val-trained bigram LM cross-entropy (validated Spearman rho=0.984)",
}
with open("experiments/data_order/stage1_shard_level/selected_shards.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved to experiments/data_order/stage1_shard_level/selected_shards.json")
