import os
import sys
from huggingface_hub import hf_hub_download

# Download SentencePiece-4k token shards for fineweb10B from huggingface.
# This matches the style of cached_fineweb10B.py.

REPO_ID = "cocohearts/4096-bpe"
SHARD_PREFIX = "fineweb10B_sp4k"
TOKENIZER_PREFIX = "tokenizers"


def get(relative_path: str) -> None:
    local_dir = os.path.dirname(__file__)
    full_local_path = os.path.join(local_dir, relative_path)
    if not os.path.exists(full_local_path):
        hf_hub_download(
            repo_id=REPO_ID,
            filename=relative_path,
            repo_type="dataset",
            local_dir=local_dir,
        )


get(f"{SHARD_PREFIX}/fineweb_val_%06d.bin" % 0)

# Full cocohearts/4096-bpe fineweb10B_sp4k as currently published.
num_chunks = 9
if len(sys.argv) >= 2:
    num_chunks = int(sys.argv[1])

for i in range(1, num_chunks + 1):
    get(f"{SHARD_PREFIX}/fineweb_train_%06d.bin" % i)

# Keep tokenizer artifacts in sync with downloaded shards.
get(f"{TOKENIZER_PREFIX}/fineweb_4k_bpe.model")
get(f"{TOKENIZER_PREFIX}/fineweb_4k_bpe.vocab")
