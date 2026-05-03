import re

with open("crates/pg-kernels/src/gpu_kernels.rs", "r") as f:
    text = f.read()

# 1. Add PushKernelArg
text = text.replace("use cudarc::driver::{CudaContext, CudaStream, CudaSlice, CudaModule, CudaFunction};",
                    "use cudarc::driver::{CudaContext, CudaStream, CudaSlice, CudaModule, CudaFunction, PushKernelArg};")

# 2. Fix sm_90
text = text.replace('arch: Some("sm_90".to_string()),', 'arch: Some("sm_90"),')

# 3. Fix load_function and module lifetime
text = re.sub(r'load_fn\("([^"]+)"\)\?',
              r'module.load_function("\1").map_err(|e| PgError::InvalidOp(format!("Failed to load kernel {}", e)))?',
              text)
text = text.replace('_module: module,', '_module: module.clone(),')
text = re.sub(r'let load_fn = \|name: &str\| -> PgResult<CudaFunction> {[\s\S]*?};\n', '', text)

with open("crates/pg-kernels/src/gpu_kernels.rs", "w") as f:
    f.write(text)
