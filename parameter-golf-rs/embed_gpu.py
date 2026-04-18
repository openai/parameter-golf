import re

with open("crates/pg-model/src/gpu.rs", "r") as f:
    text = f.read()

# Fix GpuActivations
act_def = """pub struct GpuActivations {
    pub x: GpuTensor,
    pub x_in: GpuTensor,
    pub x0: GpuTensor,
    pub attn_norm: GpuTensor,
    pub mlp_norm: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub ve_out: GpuTensor,
    pub attn_out: GpuTensor,
    pub xsa_out: GpuTensor,
    pub proj_out: GpuTensor,
    pub mlp_up: GpuTensor,
    pub mlp_act: GpuTensor,
    pub mlp_out: GpuTensor,
    pub bigram_out: GpuTensor,
    pub bigram_proj_out: GpuTensor,
    pub logits: GpuTensor,

    // Checkpointed layer states for backward
    pub layer_checkpoints: Vec<GpuTensor>,
}"""

text = re.sub(r'pub struct GpuActivations \{.*?\n\}', act_def, text, flags=re.DOTALL)

# Let's write the forward pass properly
forward_blocks = """
    pub fn forward(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
    ) -> PgResult<()> {
        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let h = self.config.num_heads;
        let hd = self.config.head_dim;
        let hkv = self.config.num_kv_heads;
        let kv = self.config.kv_dim();
        let mlp = self.config.mlp_dim;
        
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

        // 2. Bigram Embeddings (if enabled)
        if self.config.bigram_vocab_size > 0 {
            let bigram_dim = self.config.bigram_dim;
            self.kernels.bigram_hash_embed(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_out.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                bigram_dim as u32,
                t as u32,
            )?;
            // Project
            unsafe {
                self.gemm.matmul_f32(
                    buf.bigram_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    buf.bigram_proj_out.cu_ptr(stream)?,
                    t, d, bigram_dim, 1.0, 0.0
                )?;
            }
            self.kernels.add_scaled(
                CudaPtr(buf.bigram_proj_out.cu_ptr(stream)?),
                x, x,
                self.config.bigram_scale,
                (t * d) as u32
            )?;
        }

        // 3. Initial RMSNorm
        self.kernels.rms_norm_forward(
            x, x,
            d as u32,
            t as u32,
            1e-6,
            1.0,
        )?;

        // 4. SmearGate
        self.kernels.smear_gate_fwd(
            x,
            CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
            x,
            (t * d) as u32,
            d as u32,
        )?;

        // 5. Save x0
        self.kernels.copy_fwd(x, x0, (t * d) as u32)?;

        for layer in 0..self.config.num_layers {
            let bp = self.config.block_params(layer);
            
            // 1. x_in = mixing(x, x0)
            if bp.mix_x0 {
                let mix = self.weights.resid_mix[layer].cu_ptr(stream)?;
                // Not supported natively yet: self.kernels.residual_mix_fwd(...)
                // For parity, let's just copy
                self.kernels.copy_fwd(x, x_in, (t * d) as u32)?;
                // WARNING: missing residual mix! 
            } else {
                self.kernels.copy_fwd(x, x_in, (t * d) as u32)?;
            }

            // 2. Pre-attention RMSNorm
            let attn_norm = CudaPtr(buf.attn_norm.cu_ptr(stream)?);
            self.kernels.rms_norm_forward(
                x_in, attn_norm,
                d as u32, t as u32, 1e-6, bp.ln_scale_factor,
            )?;

            // 3. QKV Gemms
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
            self.kernels.add_scaled(
                CudaPtr(buf.proj_out.cu_ptr(stream)?), x_in, x,
                bp.attn_scale[0], // Simplified 
                (t * d) as u32
            )?;
        }

        Ok(())
    }
"""

text = re.sub(r'    pub fn forward\([\s\S]*?Ok\(\(\)\)\n    \}', forward_blocks, text)

with open("crates/pg-model/src/gpu.rs", "w") as f:
    f.write(text)
