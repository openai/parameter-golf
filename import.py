import os
path = "./data/datasets/fineweb10B_sp1024"

# Rename the files to match the script's expected pattern
if os.path.exists(os.path.join(path, "train_000000.bin")):
    os.rename(os.path.join(path, "train_000000.bin"), os.path.join(path, "fineweb_train_000000.bin"))

if os.path.exists(os.path.join(path, "val_000000.bin")):
    os.rename(os.path.join(path, "val_000000.bin"), os.path.join(path, "fineweb_val_000000.bin"))

print("Filenames updated to match fineweb_train_*.bin and fineweb_val_*.bin")