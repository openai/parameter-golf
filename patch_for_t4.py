import sys, re

def patch_file(filepath):
    data = open(filepath).read()
    changes = 0
    
    # 1. Fix enable_gqa - all variants
    for old in [
        'enable_gqa=(self.num_kv_heads != self.num_heads)',
        'enable_gqa=self.num_heads != self.num_kv_heads',
        'enable_gqa=(self.num_heads != self.num_kv_heads)',
    ]:
        if old in data:
            data = data.replace(old, 'enable_gqa=False')
            changes += 1
    
    # 2. Add KV head repeat before scaled_dot_product_attention (if GQA with different head counts)
    # Look for the attention call pattern and add repeat logic before it
    if 'enable_gqa=False' in data and 'n_rep = self.num_heads // self.num_kv_heads' not in data:
        # Find the attention call and add repeat before it
        attn_pattern = r'(\s+)(y = F\.scaled_dot_product_attention\(\s*\n\s*q,\s*\n\s*k,\s*\n\s*v,)'
        replacement = r'''\1# Repeat KV heads for T4 compat (no native GQA)
\1if self.num_kv_heads != self.num_heads:
\1    n_rep = self.num_heads // self.num_kv_heads
\1    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(k.shape[0], self.num_heads, k.shape[2], k.shape[3])
\1    v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(v.shape[0], self.num_heads, v.shape[2], v.shape[3])
\1\2'''
        new_data = re.sub(attn_pattern, replacement, data)
        if new_data != data:
            data = new_data
            changes += 1
    
    # 3. Disable torch.compile
    data = re.sub(
        r'(\w+)\s*=\s*torch\.compile\((\w+(?:\.\w+)*),\s*[^)]*\)',
        r'\1 = \2  # torch.compile disabled for T4',
        data
    )
    # Also handle single-arg compile
    data = re.sub(
        r'(\w+)\s*=\s*torch\.compile\((\w+(?:\.\w+)*)\)',
        r'# \1 = torch.compile(\2)  # disabled for T4',
        data
    )
    
    # 4. Fix SDP backends
    data = data.replace('enable_flash_sdp(True)', 'enable_flash_sdp(False)  # T4 no flash')
    data = data.replace('enable_mem_efficient_sdp(False)', 'enable_mem_efficient_sdp(True)')
    data = data.replace('enable_math_sdp(False)', 'enable_math_sdp(True)')
    
    open(filepath, 'w').write(data)
    
    # Verify syntax
    import ast
    try:
        ast.parse(data)
        print(f'{filepath}: patched ({changes} changes), syntax OK')
    except SyntaxError as e:
        print(f'{filepath}: SYNTAX ERROR after patching: {e}')

if __name__ == '__main__':
    for f in sys.argv[1:]:
        patch_file(f)
