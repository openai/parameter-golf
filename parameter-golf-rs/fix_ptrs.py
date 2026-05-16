import re

with open("crates/pg-kernels/src/gpu_kernels.rs", "r") as f:
    text = f.read()

# Add CudaPtr definition if missing
if "struct CudaPtr" not in text:
    struct_idx = text.find("pub struct GpuKernels {")
    text = text[:struct_idx] + "#[repr(transparent)]\n#[derive(Clone, Copy, Debug)]\npub struct CudaPtr(pub u64);\nunsafe impl cudarc::driver::DeviceRepr for CudaPtr {}\n\n" + text[struct_idx:]

# Replace arguments
text = re.sub(r'&mut CudaSlice<[a-zA-Z0-9]+>', 'CudaPtr', text)
text = re.sub(r'&CudaSlice<[a-zA-Z0-9]+>', 'CudaPtr', text)

# Now, we need to change .arg(x) to .arg(&x).
# But wait, .arg(x) for pointers vs .arg(&(dim as i32)) for values.
# Actually, the user already had `.arg(&(dim as i32))`. So non-pointer args already had `&`.
# The only things that lacked `&` were the slice pointers which were already references!
# E.g. `arg(x)` where `x` is `&CudaSlice`. If we change `x` to `CudaPtr`, it becomes a value.
# So we must change `.arg(x)` to `.arg(&x)` IF it doesn't already have `&`.
# We can just match `.arg([a-zA-Z0-9_]+)` and replace with `.arg(&\1)`.
text = re.sub(r'\.arg\(([a-zA-Z0-9_]+)\)', r'.arg(&\1)', text)

with open("crates/pg-kernels/src/gpu_kernels.rs", "w") as f:
    f.write(text)
