import os
from huggingface_hub import hf_hub_download

repo_id = "sproos/parameter-golf-tokenizers"

# Define your custom paths (matching train_gpt.py expectations)
tok_dir = "./data/tokenizers/"
data_dir = "./data/datasets/fineweb10B_sp8192/"

# 1. Download Tokenizer files (8k version)
tokenizer_files = ["tokenizers/fineweb_8192_bpe.model", "tokenizers/fineweb_8192_bpe.vocab"]
for file in tokenizer_files:
    print(f"Downloading {file}...")
    hf_hub_download(repo_id=repo_id, filename=file, local_dir=tok_dir)

# 2. Download Dataset shards (Train 00-79 + Val) — 8192 vocab version
dataset_subdir = "datasets/fineweb10B_sp8192"
dataset_files = [f"{dataset_subdir}/fineweb_train_{i:06d}.bin" for i in range(80)]
dataset_files.append(f"{dataset_subdir}/fineweb_val_000000.bin")

for file in dataset_files:
    print(f"Downloading {file}...")
    hf_hub_download(repo_id=repo_id, filename=file, local_dir=data_dir)

