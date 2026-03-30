"""Create a symlinked dataset directory with shards ordered by M5 (val CE), best first."""
import os
import json

src_dir = "data/datasets/fineweb10B_sp1024"
dst_dir = "data/datasets/fineweb10B_sp1024_m5_ordered"
os.makedirs(dst_dir, exist_ok=True)

# Symlink val
val_src = os.path.abspath(os.path.join(src_dir, "fineweb_val_000000.bin"))
val_dst = os.path.join(dst_dir, "fineweb_val_000000.bin")
if not os.path.exists(val_dst):
    os.symlink(val_src, val_dst)

# Load M5 rankings
with open("experiments/data_order/shard_analysis.json") as f:
    data = json.load(f)

# Sort: highest m5_neg_val_ce = lowest CE = most val-like = first
ranked = sorted(data, key=lambda x: x["m5_neg_val_ce"], reverse=True)

for new_idx, r in enumerate(ranked):
    old_name = r["shard"] + ".bin"
    new_name = f"fineweb_train_{new_idx:06d}.bin"
    src = os.path.abspath(os.path.join(src_dir, old_name))
    dst = os.path.join(dst_dir, new_name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
    ce = -r["m5_neg_val_ce"]
    if new_idx < 10:
        print(f"  {new_name} -> {old_name}  (CE={ce:.4f})")

total = len([f for f in os.listdir(dst_dir) if f.startswith("fineweb_train_")])
print(f"  ...")
print(f"Total: {total} train shards symlinked (best val-match first)")
