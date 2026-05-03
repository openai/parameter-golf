import os
import glob

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Rename functions
    content = content.replace('matmul_bf16', 'matmul_f32')
    content = content.replace('batched_matmul_bf16', 'batched_matmul_f32')
    content = content.replace('matmul_bf16_bt', 'matmul_f32_bt')
    content = content.replace('batched_matmul_f32_bt', 'batched_matmul_f32_bt') # safety
    
    # Replace CUDA types in gemm.rs
    if filepath.endswith('gemm.rs'):
        content = content.replace('CUDA_R_16BF', 'CUDA_R_32F')

    with open(filepath, 'w') as f:
        f.write(content)

for root, dirs, files in os.walk('crates'):
    for file in files:
        if file.endswith('.rs'):
            process_file(os.path.join(root, file))

