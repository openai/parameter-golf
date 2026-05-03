import re

with open("crates/pg-model/src/gpu.rs", "r") as f:
    text = f.read()

clean_forward = """    pub fn forward(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
    ) -> PgResult<()> {
        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let hd = self.config.head_dim;
        let hkv = self.config.num_kv_heads;
        let kv = self.config.kv_dim();
        
        use pg_kernels::gpu_kernels::CudaPtr;
        let stream = self.gemm.stream();

        let x = CudaPtr(buf.x.cu_ptr(stream)?);
        let x_in = CudaPtr(buf.x_in.cu_ptr(stream)?);
        let x0 = CudaPtr(buf.x0.cu_ptr(stream)?);

        // 1. Token Embeddings
        self.kernels.embedding_gather_fwd(
            CudaPtr(input_ids.cu_ptr(stream)?),
            CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
            x,
            d as u32,
            t as u32,
        )?;

        // 2. Initial RMSNorm
        self.kernels.rms_norm_forward(
            x, x,
            d as u32,
            t as u32,
            1e-6,
            1.0,
        )?;

        // 3. SmearGate
        self.kernels.smear_gate_fwd(
            x,
            CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
            x,
            (t * d) as u32,
            d as u32,
        )?;

        // 4. Save x0
        self.kernels.copy_fwd(x, x0, (t * d) as u32)?;

        for layer in 0..self.config.num_layers {
            // Pre-attention RMSNorm (temporarily passing 1.0 for scale until fusion supports vector scaling)
            self.kernels.copy_fwd(x, x_in, (t * d) as u32)?;
            let attn_norm = CudaPtr(buf.attn_norm.cu_ptr(stream)?);
            self.kernels.rms_norm_forward(
                x_in, attn_norm,
                d as u32, t as u32, 1e-6, 1.0, 
            )?;

            // QKV Gemms
            let q_w = self.weights.qo_bank.slice_first(2 * layer)?.cu_ptr(stream)?;
            let k_w = self.weights.kv_bank.slice_first(2 * layer)?.cu_ptr(stream)?;
            let v_w = self.weights.kv_bank.slice_first(2 * layer + 1)?.cu_ptr(stream)?;

            let p_norm = buf.attn_norm.cu_ptr(stream)?;
            unsafe {
                self.gemm.matmul_f32(p_norm, q_w, buf.q.cu_ptr(stream)?, t, d, d, 1.0, 0.0)?;
                self.gemm.matmul_f32(p_norm, k_w, buf.k.cu_ptr(stream)?, t, kv, d, 1.0, 0.0)?;
                self.gemm.matmul_f32(p_norm, v_w, buf.v.cu_ptr(stream)?, t, kv, d, 1.0, 0.0)?;
            }

            // ... FlashAttention missing here ...
            self.kernels.copy_fwd(CudaPtr(buf.v.cu_ptr(stream)?), CudaPtr(buf.attn_out.cu_ptr(stream)?), (t * d) as u32)?;
            
            // Output Projection
            let o_w = self.weights.qo_bank.slice_first(2 * layer + 1)?.cu_ptr(stream)?;
            unsafe {
                self.gemm.matmul_f32(buf.attn_out.cu_ptr(stream)?, o_w, buf.proj_out.cu_ptr(stream)?, t, d, d, 1.0, 0.0)?;
            }

            // Residual
            self.kernels.add_scaled_fwd(
                CudaPtr(buf.proj_out.cu_ptr(stream)?),
                x, 
                1.0, 
                (t * d) as u32
            )?;
        }

        Ok(())
    }"""

text = re.sub(r'    pub fn forward\([\s\S]*?Ok\(\(\)\)\n    \}', clean_forward, text)

with open("crates/pg-model/src/gpu.rs", "w") as f:
    f.write(text)
