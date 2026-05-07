import ast
import subprocess
import lzma
import base64
import os

class DeadCodeRemover(ast.NodeTransformer):
    def visit_If(self, node):
        source = ast.unparse(node.test)
        
        # Dead config branches based on defaults:
        dead_tests = [
            "A.load_snapshot",
            "A.snapshot_post_hessian"
        ]
        
        for dt in dead_tests:
            if dt in source:
                return None
        
        self.generic_visit(node)
        return node

def shrink_pipeline(input_file, output_file):
    print(f"[*] Starting shrinking pipeline for: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"[!] Error: Input file '{input_file}' not found.")
        return

    # 1. Read Original Source
    with open(input_file, 'r', encoding='utf-8') as f:
        source = f.read()
    print(f"[*] Original size: {len(source)} bytes")

    # 2. Prune AST (Dead Code Elimination)
    print(f"[*] Pruning dead code and evaluation logic from AST...")
    tree = ast.parse(source)
    transformer = DeadCodeRemover()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    
    pruned_source = ast.unparse(tree)
    temp_pruned_file = input_file + ".pruned.tmp.py"
    with open(temp_pruned_file, 'w', encoding='utf-8') as f:
        f.write(pruned_source)

    # 3. Minify using pyminify
    print(f"[*] Running pyminify to minimize identifiers and strip whitespace/comments/hints...")
    temp_minified_file = input_file + ".min.tmp.py"
    try:
        subprocess.run([
            "uvx", "--from", "python-minifier", "pyminify", temp_pruned_file,
            "--output", temp_minified_file,
            "--remove-literal-statements", 
            "--remove-asserts", 
            "--remove-debug", 
            "--remove-class-attribute-annotations"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[!] PyMinify failed: {e}")
        if os.path.exists(temp_pruned_file): os.remove(temp_pruned_file)
        return

    with open(temp_minified_file, 'rb') as f:
        minified_bytes = f.read()
    print(f"[*] Minified size: {len(minified_bytes)} bytes")

    # 4. LZMA + Base85 Self-Extracting Compression
    print(f"[*] Compressing into LZMA Base85 executable wrap...")
    compressed = lzma.compress(minified_bytes, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
    b85_encoded = base64.b85encode(compressed).decode('ascii')

    # Chunk the b85 string to avoid overly long single lines
    chunk_size = 100
    chunks = [b85_encoded[i:i+chunk_size] for i in range(0, len(b85_encoded), chunk_size)]
    formatted_b85 = '"\n"'.join(chunks)

    header = f"""import lzma as L,base64 as B\nexec(L.decompress(B.b85decode(("{formatted_b85}")),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))\n"""

    with open(output_file, 'w', encoding='ascii') as f:
        f.write(header)

    final_size = os.path.getsize(output_file)
    print(f"[*] Packed output size: {final_size} bytes")
    print(f"[*] Total Reduction: {((len(source) - final_size) / len(source) * 100):.1f}%")

    # 5. Clean up temporary files
    os.remove(temp_pruned_file)
    os.remove(temp_minified_file)
    print(f"[*] Success! Optimized submission saved to: {output_file}")

if __name__ == '__main__':
    human_file = "train_gpt_human.py"
    output_file = "train_gpt.py"

    if not os.path.exists(output_file):
        print(f"[!] Error: Input file '{output_file}' not found.")
        exit(1)

    # Rename the original file to train_gpt_human.py
    if os.path.exists(human_file):
        os.remove(human_file)
        
    os.rename(output_file, human_file)
    print(f"[*] Renamed '{output_file}' to '{human_file}'")
    
    # Shrink the human readable file and output back to train_gpt.py
    shrink_pipeline(human_file, output_file)
