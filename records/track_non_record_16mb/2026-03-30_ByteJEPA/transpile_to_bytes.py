import os
import glob
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

"""
Token Transpiler for ByteJEPA
Run this script to transpile the Parameter Golf fineweb10B_sp1024 
shards back into mathematically pure byte-level shards (fineweb10B_bytes).
"""

# Adjust path based on where you run this script (assumes running from PG root or ByteJEPA dir)
BASE_DATA_DIR = os.environ.get("DATA_PATH_BASE", "../../data") 
TOKENIZER_PATH = os.path.join(BASE_DATA_DIR, "tokenizers/fineweb_1024_bpe.model")
INPUT_DIR = os.path.join(BASE_DATA_DIR, "datasets/fineweb10B_sp1024")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "datasets/fineweb10B_bytes")

def main():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}. Please download the sp1024 dataset first.")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)

    input_shards = sorted(glob.glob(f"{INPUT_DIR}/*.bin"))
    print(f"Found {len(input_shards)} SentencePiece shards to transpile...")

    for filepath in tqdm(input_shards):
        filename = os.path.basename(filepath)
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        # skip if already done
        if os.path.exists(out_path):
            continue
            
        with open(filepath, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            num_tokens = header[2]
            sp_tokens = np.frombuffer(f.read(), dtype=np.uint16)[:num_tokens]
        
        # Decode and immediately re-encode to raw bytes
        text = sp.decode(sp_tokens.tolist())
        bytes_arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        bytes_uint16 = bytes_arr.astype(np.uint16)
        
        # Save exact Parameter Golf bin format
        new_header = np.zeros(256, dtype=np.int32)
        new_header[0] = 20240520
        new_header[1] = 1
        new_header[2] = len(bytes_uint16)
        
        with open(out_path, "wb") as f:
            f.write(new_header.tobytes())
            f.write(bytes_uint16.tobytes())
            
    print("Transpilation to byte-level shards complete! You can now run ByteJEPA.")

if __name__ == "__main__":
    main()
