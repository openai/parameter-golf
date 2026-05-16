import re

with open("crates/pg-kernels/src/gemm.rs", "r") as f:
    text = f.read()

# Replace &CudaSlice<bf16> with u64
text = text.replace("&CudaSlice<bf16>", "u64")
text = text.replace("&mut CudaSlice<bf16>", "u64")

# Update `matmul_bf16` body
matmul_body = """{
        let stream_ptr = unsafe {
            let mut ptr: cudarc::driver::sys::CUstream = std::ptr::null_mut();
            // Since we can't extract the internal stream handle easily without unsafe casting, we can just let cublas use its default stream or set it via CudaBlas 
            // Wait, CudaBlas ALREADY sets the stream internally!
        };
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                a as *const _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| PgError::CuBlas(format!("gemm_ex failed: {:?}", e)))?;
        }
        Ok(())
    }"""
text = re.sub(r'pub unsafe fn matmul_bf16.*?Ok\(\(\)\)\n    \}',
              'pub unsafe fn matmul_bf16(&self, a: u64, b: u64, c: u64, m: usize, n: usize, k: usize, alpha: f32, beta: f32) -> PgResult<()> ' + matmul_body,
              text, flags=re.DOTALL)

# Update `batched_matmul_bf16` body
batch_body = """{
        let stride_a = (m * k) as i64;
        let stride_b = (k * n) as i64;
        let stride_c = (m * n) as i64;
        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                stride_b,
                a as *const _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                stride_a,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                stride_c,
                batch as i32,
                cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| PgError::CuBlas(format!("gemm_strided_batched_ex failed: {:?}", e)))?;
        }
        Ok(())
    }"""
text = re.sub(r'pub unsafe fn batched_matmul_bf16.*?Ok\(\(\)\)\n    \}',
              'pub unsafe fn batched_matmul_bf16(&self, a: u64, b: u64, c: u64, batch: usize, m: usize, n: usize, k: usize, alpha: f32, beta: f32) -> PgResult<()> ' + batch_body,
              text, flags=re.DOTALL)

# Update `matmul_bf16_bt` body
matmul_bt_body = """{
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                a as *const _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudarc::driver::sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| PgError::CuBlas(format!("gemm_ex failed: {:?}", e)))?;
        }
        Ok(())
    }"""
text = re.sub(r'pub unsafe fn matmul_bf16_bt.*?Ok\(\(\)\)\n    \}',
              'pub unsafe fn matmul_bf16_bt(&self, a: u64, b: u64, c: u64, m: usize, n: usize, k: usize, alpha: f32, beta: f32) -> PgResult<()> ' + matmul_bt_body,
              text, flags=re.DOTALL)

with open("crates/pg-kernels/src/gemm.rs", "w") as f:
    f.write(text)
