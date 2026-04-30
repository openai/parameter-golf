import lzma, base64
from pathlib import Path

def golf(input_path, output_path):
    code = Path(input_path).read_text(encoding='utf-8')
    # Simple minification: remove extra whitespace and comments
    # (Though the input is already relatively compact)
    compressed = lzma.compress(code.encode('utf-8'))
    encoded = base64.b85encode(compressed).decode('utf-8')
    final_payload = f"import lzma as L,base64 as B;exec(L.decompress(B.b85decode('{encoded}')))"
    Path(output_path).write_text(final_payload, encoding='utf-8')
    print(f"Golfed {len(code)} bytes -> {len(final_payload)} bytes")

if __name__ == "__main__":
    import os
    target_dir = r"d:\projects\parameter-golf\records\track_10min_16mb\2026-04-20_Geodesic_Bigram_3LayerRecur_ParResid_LegalTTT"
    input_file = os.path.join(target_dir, "train_gpt_full.py")
    output_file = os.path.join(target_dir, "train_gpt.py")
    golf(input_file, output_file)
