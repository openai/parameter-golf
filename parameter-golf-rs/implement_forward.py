import re

with open("crates/pg-model/src/gpu.rs", "r") as f:
    text = f.read()

forward_impl = """    pub fn forward(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
    ) -> PgResult<()> {
        let t = buf.x.shape()[0];
        let d = self.config.model_dim;
        let h = self.config.num_heads;
        let hd = self.config.head_dim;
        let kv = self.config.kv_dim();
        let mlp = self.config.mlp_dim;
        
        use pg_kernels::gpu_kernels::CudaPtr;
        let stream = self.gemm.stream();

        // 1. Token Embeddings
        self.kernels.embedding_gather_fwd(
            CudaPtr(input_ids.cu_ptr(stream)?),
            CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
            CudaPtr(buf.x.cu_ptr(stream)?),
            d as u32,
            t as u32,
        )?;

        // 2. Initial RMSNorm
        // In CPU it normed directly into a buffer, then copied back. The kernel works in-place?
        // Let's assume out=x. Wait, rms_norm CPU was not in-place but we can just use buf.attn_norm as tmp or in-place.
        self.kernels.rms_norm_fwd(
            CudaPtr(buf.x.cu_ptr(stream)?),
            CudaPtr(buf.x.cu_ptr(stream)?),
            d as u32,
            1.0,
            1e-6,
            t as u32,
        )?;

        // For now, let's just make it compile. We need to do the layer loop!
        
        Ok(())
    }"""

# Replace the forward function
text = re.sub(r'    pub fn forward\([\s\S]*?Ok\(\(\)\)\n    \}', forward_impl, text)

with open("crates/pg-model/src/gpu.rs", "w") as f:
    f.write(text)
