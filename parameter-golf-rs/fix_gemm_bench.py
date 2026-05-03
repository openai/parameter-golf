import re
with open("crates/pg-bench/src/bin/gemm_bench.rs", "r") as f:
    text = f.read()

# Replace matmul arguments
text = re.sub(r'\.matmul_bf16\(&a, &b, &mut c,',
              '.matmul_bf16(a.cu_device_ptr, b.cu_device_ptr, c.cu_device_ptr,', text)

text = re.sub(r'\.batched_matmul_bf16\(&a, &b, &mut c,',
              '.batched_matmul_bf16(a.cu_device_ptr, b.cu_device_ptr, c.cu_device_ptr,', text)

# For CudaSlice, device_ptr requires a stream. Wait, in CudaSlice, cu_device_ptr is a private field.
text = text.replace('a.cu_device_ptr', 'cudarc::driver::DevicePtr::device_ptr(&a, engine.stream()).0')
text = text.replace('b.cu_device_ptr', 'cudarc::driver::DevicePtr::device_ptr(&b, engine.stream()).0')
text = text.replace('c.cu_device_ptr', 'cudarc::driver::DevicePtr::device_ptr(&c, engine.stream()).0')

with open("crates/pg-bench/src/bin/gemm_bench.rs", "w") as f:
    f.write(text)
