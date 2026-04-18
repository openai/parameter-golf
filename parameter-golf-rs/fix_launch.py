import re

with open("crates/pg-kernels/src/gpu_kernels.rs", "r") as f:
    code = f.read()

# Replace .grid_dim(X) .block_dim(Y) .shared_mem_bytes(Z) .launch()
pattern1 = re.compile(r'\.grid_dim\(([^)]+)\)\s*\.block_dim\(([^)]+)\)\s*\.shared_mem_bytes\(([^)]+)\)\s*\.launch\(\)')
code = pattern1.sub(r'.launch(cudarc::driver::LaunchConfig { grid_dim: (\1, 1, 1).into(), block_dim: (\2, 1, 1).into(), shared_mem_bytes: \3 })', code)

# Replace .grid_dim(X) .block_dim(Y) .launch()
pattern2 = re.compile(r'\.grid_dim\(([^)]+)\)\s*\.block_dim\(([^)]+)\)\s*\.launch\(\)')
code = pattern2.sub(r'.launch(cudarc::driver::LaunchConfig { grid_dim: (\1, 1, 1).into(), block_dim: (\2, 1, 1).into(), shared_mem_bytes: 0 })', code)

with open("crates/pg-kernels/src/gpu_kernels.rs", "w") as f:
    f.write(code)
