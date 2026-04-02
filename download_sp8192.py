import huggingface_hub

print("Downloading fineweb10B_sp8192 metadata...")
huggingface_hub.snapshot_download(
    repo_id="sproos/parameter-golf-tokenizers",
    repo_type="dataset",
    allow_patterns=[
        "datasets/fineweb10B_sp8192/fineweb_val_*.bin", 
        "datasets/fineweb10B_sp8192/fineweb_train_000000.bin"
    ],
    local_dir="./data"
)
print("Download complete.")
