import numpy as np
import os

path = "./data/datasets/fineweb10B_sp1024"
os.makedirs(path, exist_ok=True)

# 256 int32s for the header
header = np.zeros(256, dtype=np.int32)
header[0] = 20240520 
header[1] = 1        
header[2] = 32000    # Tell the script we have 32k tokens

# Generate 32,000 tokens (well above the 1024 seq_len requirement)
tokens = np.random.randint(0, 198, size=(32000,), dtype=np.uint16)

def save_shard(filename):
    with open(os.path.join(path, filename), "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())

save_shard("fineweb_train_000000.bin")
save_shard("fineweb_val_000000.bin")

print("Large mock shards (32k tokens) created.")