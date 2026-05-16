with open("crates/pg-kernels/src/gpu_kernels.rs", "r") as f:
    text = f.read()

import re

# We will just replace load_fn("NAME")? with module.load_function("NAME").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel: {:?}", e)))?
text = re.sub(r'load_fn\("([^"]+)"\)\?', r'module.load_function("\1").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel: {:?}", e)))?', text)

# Remove let load_fn = ...
text = re.sub(r'let load_fn\s*=\s*\|name: &str\|[^}]+};\n\n', '', text)

with open("crates/pg-kernels/src/gpu_kernels.rs", "w") as f:
    f.write(text)
