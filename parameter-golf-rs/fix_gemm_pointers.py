import re

with open("crates/pg-kernels/src/gemm.rs", "r") as f:
    text = f.read()

# Replace &CudaSlice<bf16> with u64
text = text.replace("&CudaSlice<bf16>", "u64")
text = text.replace("&mut CudaSlice<bf16>", "u64")

# Now change the `blas.gemm` calls to `gemm_ex` calls.
# Notice that `blas.gemm` expects references, but now they are primitive values.
# The user doesn't even need `gemm.rs` to do anything complex. I'll rewrite it manually since it's only 160 lines.
