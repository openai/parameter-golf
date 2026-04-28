use std::sync::Arc;

use bytemuck::cast_slice;
use cudarc::driver::CudaContext;
use half::bf16;
use pg_core::{DType, GpuTensor, PgResult};
use pg_kernels::gpu_kernels::{CudaPtr, GpuKernels};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(real_main);
    std::panic::set_hook(hook);
    match result {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            println!("status=kernel_parity_not_ready");
            println!("error={err}");
        }
        Err(_) => {
            println!("status=kernel_parity_not_ready");
            println!("error=panic_while_loading_cuda_runtime");
        }
    }
    Ok(())
}

fn real_main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let kernels = GpuKernels::new(ctx.clone(), stream.clone())?;
    let gemm = pg_kernels::gemm::GemmEngine::new(stream.clone())?;

    run_embedding_gather(&kernels, &stream)?;
    run_embedding_gather_bwd(&kernels, &stream)?;
    run_bigram_hash(&kernels, &stream)?;
    run_rms_norm(&kernels, &stream)?;
    run_rms_norm_bwd(&kernels, &stream)?;
    run_leaky_relu_sq(&kernels, &stream)?;
    run_leaky_relu_sq_bwd(&kernels, &stream)?;
    run_smear_gate(&kernels, &stream)?;
    run_smear_gate_bwd(&kernels, &stream)?;
    run_residual_mix(&kernels, &stream)?;
    run_residual_mix_bwd(&kernels, &stream)?;
    run_residual_add_scale(&kernels, &stream)?;
    run_residual_add_scale_bwd(&kernels, &stream)?;
    run_qk_norm(&kernels, &stream)?;
    run_qk_norm_bwd(&kernels, &stream)?;
    run_partial_rope(&kernels, &stream)?;
    run_partial_rope_bwd(&kernels, &stream)?;
    run_q_gain(&kernels, &stream)?;
    run_q_gain_bwd(&kernels, &stream)?;
    run_sparse_attn_gate(&kernels, &stream)?;
    run_sparse_attn_gate_bwd(&kernels, &stream)?;
    run_dot_accumulate(&kernels, &stream)?;
    run_cross_entropy_bwd(&kernels, &stream)?;
    run_attention(&kernels, &stream)?;
    run_attention_bwd(&kernels, &stream)?;
    run_attention_block_causal_batch(&kernels, &stream)?;
    run_xsa(&kernels, &stream)?;
    run_xsa_bwd(&kernels, &stream)?;
    run_bigram_hash_bwd(&kernels, &stream)?;
    run_gemm(&gemm, &stream)?;
    run_bf16_gemm(&kernels, &gemm, &stream)?;
    Ok(())
}

fn upload_f32(
    stream: &Arc<cudarc::driver::CudaStream>,
    data: &[f32],
    shape: &[usize],
) -> PgResult<GpuTensor> {
    GpuTensor::from_host_data_gpu(stream.clone(), cast_slice(data), shape, DType::F32)
}

fn upload_u32(
    stream: &Arc<cudarc::driver::CudaStream>,
    data: &[u32],
    shape: &[usize],
) -> PgResult<GpuTensor> {
    GpuTensor::from_host_data_gpu(stream.clone(), cast_slice(data), shape, DType::U32)
}

fn zeros(stream: &Arc<cudarc::driver::CudaStream>, shape: &[usize]) -> PgResult<GpuTensor> {
    GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32)
}

fn zeros_dtype(
    stream: &Arc<cudarc::driver::CudaStream>,
    shape: &[usize],
    dtype: DType,
) -> PgResult<GpuTensor> {
    GpuTensor::zeros_gpu(stream.clone(), shape, dtype)
}

fn download_f32(
    stream: &Arc<cudarc::driver::CudaStream>,
    tensor: &GpuTensor,
) -> PgResult<Vec<f32>> {
    stream.synchronize()?;
    let bytes = tensor.to_host_bytes()?;
    Ok(cast_slice::<u8, f32>(&bytes).to_vec())
}

fn upload_bf16_from_f32(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
    data: &[f32],
    shape: &[usize],
) -> PgResult<GpuTensor> {
    let f32_gpu = upload_f32(stream, data, shape)?;
    let bf16_gpu = zeros_dtype(stream, shape, DType::BF16)?;
    kernels.f32_to_bf16(
        CudaPtr(f32_gpu.cu_ptr(stream)?),
        CudaPtr(bf16_gpu.cu_ptr(stream)?),
        data.len() as u32,
    )?;
    stream.synchronize()?;
    Ok(bf16_gpu)
}

fn download_bf16_as_f32(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
    tensor: &GpuTensor,
) -> PgResult<Vec<f32>> {
    let out = zeros_dtype(stream, tensor.shape(), DType::F32)?;
    kernels.bf16_to_f32(
        CudaPtr(tensor.cu_ptr(stream)?),
        CudaPtr(out.cu_ptr(stream)?),
        tensor.numel() as u32,
    )?;
    download_f32(stream, &out)
}

fn round_bf16(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

fn rounded_bf16_vec(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| round_bf16(v)).collect()
}

fn linear_forward_cpu(
    x: &[f32],
    w: &[f32],
    tokens: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * out_dim];
    for t in 0..tokens {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += x[t * in_dim + i] * w[o * in_dim + i];
            }
            out[t * out_dim + o] = acc;
        }
    }
    out
}

fn linear_backward_input_cpu(
    dy: &[f32],
    w: &[f32],
    tokens: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * in_dim];
    for t in 0..tokens {
        for i in 0..in_dim {
            let mut acc = 0.0f32;
            for o in 0..out_dim {
                acc += dy[t * out_dim + o] * w[o * in_dim + i];
            }
            out[t * in_dim + i] = acc;
        }
    }
    out
}

fn linear_backward_weight_cpu(
    dy: &[f32],
    x: &[f32],
    tokens: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            let mut acc = 0.0f32;
            for t in 0..tokens {
                acc += dy[t * out_dim + o] * x[t * in_dim + i];
            }
            out[o * in_dim + i] = acc;
        }
    }
    out
}

fn report(name: &str, cpu: &[f32], gpu: &[f32]) {
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
    }
    let mean_abs = sum_abs / cpu.len() as f32;
    println!("{name}: max_abs_diff={max_abs:.6} mean_abs_diff={mean_abs:.6}");
}

fn qk_norm_backward_cpu(
    x: &[f32],
    grad_output: &[f32],
    grad_input: &mut [f32],
    head_dim: usize,
    eps: f32,
) {
    let total_heads = x.len() / head_dim;
    for head in 0..total_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        let x_head = &x[start..end];
        let go_head = &grad_output[start..end];
        let sq_sum: f32 = x_head.iter().map(|&v| v * v).sum();
        let rms = (sq_sum / head_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let x_dot_go: f32 = x_head
            .iter()
            .zip(go_head.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let coeff = x_dot_go / (rms * rms * head_dim as f32);
        for i in 0..head_dim {
            grad_input[start + i] = inv_rms * (go_head[i] - x_head[i] * coeff);
        }
    }
}

fn run_embedding_gather(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let ids = vec![3u32, 1, 0];
    let emb: Vec<f32> = (0..20).map(|i| i as f32 * 0.1 - 0.3).collect();
    let mut cpu = vec![0.0; ids.len() * 5];
    for (t, id) in ids.iter().enumerate() {
        cpu[t * 5..(t + 1) * 5].copy_from_slice(&emb[*id as usize * 5..(*id as usize + 1) * 5]);
    }
    let ids_gpu = upload_u32(stream, &ids, &[ids.len()])?;
    let emb_gpu = upload_f32(stream, &emb, &[4, 5])?;
    let out_gpu = zeros(stream, &[ids.len(), 5])?;
    kernels.embedding_gather_fwd(
        CudaPtr(ids_gpu.cu_ptr(stream)?),
        CudaPtr(emb_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        5,
        ids.len() as u32,
    )?;
    report("embedding_gather", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_embedding_gather_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let ids = vec![3u32, 1, 3];
    let grad_out: Vec<f32> = (0..15).map(|i| i as f32 * 0.07 - 0.3).collect();
    let mut cpu = vec![0.0; 4 * 5];
    for (t, &tok) in ids.iter().enumerate() {
        for d in 0..5 {
            cpu[tok as usize * 5 + d] += grad_out[t * 5 + d];
        }
    }
    let ids_gpu = upload_u32(stream, &ids, &[ids.len()])?;
    let go_gpu = upload_f32(stream, &grad_out, &[ids.len(), 5])?;
    let grad_gpu = zeros(stream, &[4, 5])?;
    kernels.embedding_gather_bwd(
        CudaPtr(ids_gpu.cu_ptr(stream)?),
        CudaPtr(go_gpu.cu_ptr(stream)?),
        CudaPtr(grad_gpu.cu_ptr(stream)?),
        5,
        ids.len() as u32,
    )?;
    report(
        "embedding_gather_bwd",
        &cpu,
        &download_f32(stream, &grad_gpu)?,
    );
    Ok(())
}

fn run_bigram_hash(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let ids = vec![5u32, 3, 7, 1];
    let embed: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 - 0.4).collect();
    let mut cpu = vec![0.0; ids.len() * 4];
    pg_kernels::bigram_hash::bigram_hash_forward(&ids, &embed, &mut cpu, 8, 4);
    let ids_gpu = upload_u32(stream, &ids, &[ids.len()])?;
    let embed_gpu = upload_f32(stream, &embed, &[8, 4])?;
    let out_gpu = zeros(stream, &[ids.len(), 4])?;
    kernels.bigram_hash_embed_fwd(
        CudaPtr(ids_gpu.cu_ptr(stream)?),
        CudaPtr(embed_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        8,
        4,
        ids.len() as u32,
        ids.len() as u32,
    )?;
    report("bigram_hash", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_rms_norm(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let x: Vec<f32> = (0..24).map(|i| i as f32 * 0.07 - 0.6).collect();
    let mut cpu = vec![0.0; x.len()];
    pg_kernels::rms_norm::rms_norm_forward_cpu(&x, &mut cpu, 6, 0.5, 1e-6);
    let x_gpu = upload_f32(stream, &x, &[4, 6])?;
    let out_gpu = zeros(stream, &[4, 6])?;
    kernels.rms_norm_forward(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        4,
        6,
        0.5,
        1e-6,
    )?;
    report("rms_norm", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_rms_norm_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let x: Vec<f32> = (0..24).map(|i| i as f32 * 0.07 - 0.6).collect();
    let dy: Vec<f32> = (0..24).map(|i| i as f32 * 0.03 - 0.2).collect();
    let mut cpu = vec![0.0; x.len()];
    pg_kernels::rms_norm::rms_norm_backward_cpu(&x, &dy, &mut cpu, 6, 0.5, 1e-6);
    let x_gpu = upload_f32(stream, &x, &[4, 6])?;
    let dy_gpu = upload_f32(stream, &dy, &[4, 6])?;
    let dx_gpu = zeros(stream, &[4, 6])?;
    kernels.rms_norm_backward(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(dy_gpu.cu_ptr(stream)?),
        CudaPtr(dx_gpu.cu_ptr(stream)?),
        4,
        6,
        0.5,
        1e-6,
    )?;
    report("rms_norm_bwd", &cpu, &download_f32(stream, &dx_gpu)?);
    Ok(())
}

fn run_leaky_relu_sq(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.09 - 0.7).collect();
    let mut cpu = vec![0.0; x.len()];
    pg_kernels::activations::leaky_relu_sq_forward(&x, &mut cpu);
    let x_gpu = upload_f32(stream, &x, &[4, 4])?;
    let out_gpu = zeros(stream, &[4, 4])?;
    kernels.leaky_relu_sq_forward(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        x.len() as u32,
    )?;
    report("leaky_relu_sq", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_leaky_relu_sq_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.09 - 0.7).collect();
    let dy: Vec<f32> = (0..16).map(|i| i as f32 * 0.04 - 0.2).collect();
    let mut cpu = vec![0.0; x.len()];
    pg_kernels::activations::leaky_relu_sq_backward(&x, &dy, &mut cpu);
    let x_gpu = upload_f32(stream, &x, &[4, 4])?;
    let dy_gpu = upload_f32(stream, &dy, &[4, 4])?;
    let dx_gpu = zeros(stream, &[4, 4])?;
    kernels.leaky_relu_sq_backward(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(dy_gpu.cu_ptr(stream)?),
        CudaPtr(dx_gpu.cu_ptr(stream)?),
        x.len() as u32,
    )?;
    report("leaky_relu_sq_bwd", &cpu, &download_f32(stream, &dx_gpu)?);
    Ok(())
}

fn run_smear_gate(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.11 - 0.5).collect();
    let gate = vec![-0.8, -0.2, 0.4, 0.9];
    let mut x_prev = vec![0.0; x.len()];
    for t in 1..3 {
        x_prev[t * 4..(t + 1) * 4].copy_from_slice(&x[(t - 1) * 4..t * 4]);
    }
    let mut cpu = vec![0.0; x.len()];
    pg_kernels::smear_gate::smear_gate_forward(&x, &x_prev, &gate, &mut cpu, 3, 4);
    let x_gpu = upload_f32(stream, &x, &[3, 4])?;
    let gate_gpu = upload_f32(stream, &gate, &[4])?;
    let out_gpu = zeros(stream, &[3, 4])?;
    kernels.smear_gate_fwd(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(gate_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        3,
        3,
        4,
    )?;
    report("smear_gate", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_smear_gate_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.11 - 0.5).collect();
    let gate = vec![-0.8, -0.2, 0.4, 0.9];
    let grad_out: Vec<f32> = (0..12).map(|i| 0.2 - i as f32 * 0.017).collect();
    let mut x_prev = vec![0.0; x.len()];
    for t in 1..3 {
        x_prev[t * 4..(t + 1) * 4].copy_from_slice(&x[(t - 1) * 4..t * 4]);
    }
    let mut cpu_dx = vec![0.0; x.len()];
    let mut cpu_dx_prev = vec![0.0; x.len()];
    let mut cpu_dgate = vec![0.0; gate.len()];
    pg_kernels::smear_gate::smear_gate_backward(
        &x,
        &x_prev,
        &gate,
        &grad_out,
        &mut cpu_dx,
        &mut cpu_dx_prev,
        &mut cpu_dgate,
        3,
        4,
    );
    // GPU backward returns the shift-ready previous-token gradient used by
    // model backward, so sequence starts are zeroed to avoid boundary leakage.
    cpu_dx_prev[0..4].fill(0.0);
    let x_gpu = upload_f32(stream, &x, &[3, 4])?;
    let gate_gpu = upload_f32(stream, &gate, &[4])?;
    let go_gpu = upload_f32(stream, &grad_out, &[3, 4])?;
    let dx_gpu = zeros(stream, &[3, 4])?;
    let dx_prev_gpu = zeros(stream, &[3, 4])?;
    let dgate_gpu = zeros(stream, &[4])?;
    kernels.smear_gate_bwd(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(gate_gpu.cu_ptr(stream)?),
        CudaPtr(go_gpu.cu_ptr(stream)?),
        CudaPtr(dx_gpu.cu_ptr(stream)?),
        CudaPtr(dx_prev_gpu.cu_ptr(stream)?),
        CudaPtr(dgate_gpu.cu_ptr(stream)?),
        3,
        3,
        4,
    )?;
    report("smear_gate_bwd_x", &cpu_dx, &download_f32(stream, &dx_gpu)?);
    report(
        "smear_gate_bwd_x_prev",
        &cpu_dx_prev,
        &download_f32(stream, &dx_prev_gpu)?,
    );
    report(
        "smear_gate_bwd_gate",
        &cpu_dgate,
        &download_f32(stream, &dgate_gpu)?,
    );
    Ok(())
}

fn run_residual_mix(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.09 - 0.4).collect();
    let x0: Vec<f32> = (0..12).map(|i| i as f32 * 0.05 - 0.2).collect();
    let mix = vec![1.0, 0.8, 0.6, 0.4, -0.1, 0.0, 0.1, 0.2];
    let mut cpu = vec![0.0; x.len()];
    for i in 0..x.len() {
        let d = i % 4;
        cpu[i] = mix[d] * x[i] + mix[4 + d] * x0[i];
    }
    let x_gpu = upload_f32(stream, &x, &[3, 4])?;
    let x0_gpu = upload_f32(stream, &x0, &[3, 4])?;
    let mix_gpu = upload_f32(stream, &mix, &[2, 4])?;
    let out_gpu = zeros(stream, &[3, 4])?;
    kernels.residual_mix_fwd(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(x0_gpu.cu_ptr(stream)?),
        CudaPtr(mix_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        4,
        x.len() as u32,
    )?;
    report("residual_mix", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_residual_mix_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.09 - 0.4).collect();
    let x0: Vec<f32> = (0..12).map(|i| i as f32 * 0.05 - 0.2).collect();
    let grad_out: Vec<f32> = (0..12).map(|i| i as f32 * 0.02 - 0.1).collect();
    let mix = vec![1.0, 0.8, 0.6, 0.4, -0.1, 0.0, 0.1, 0.2];
    let mut cpu_dx = vec![0.0; x.len()];
    let mut cpu_dx0 = vec![0.0; x.len()];
    let mut cpu_dmix = vec![0.0; mix.len()];
    for i in 0..x.len() {
        let d = i % 4;
        cpu_dx[i] = grad_out[i] * mix[d];
        cpu_dx0[i] += grad_out[i] * mix[4 + d];
        cpu_dmix[d] += grad_out[i] * x[i];
        cpu_dmix[4 + d] += grad_out[i] * x0[i];
    }
    let x_gpu = upload_f32(stream, &x, &[3, 4])?;
    let x0_gpu = upload_f32(stream, &x0, &[3, 4])?;
    let go_gpu = upload_f32(stream, &grad_out, &[3, 4])?;
    let mix_gpu = upload_f32(stream, &mix, &[2, 4])?;
    let dx_gpu = zeros(stream, &[3, 4])?;
    let dx0_gpu = zeros(stream, &[3, 4])?;
    let dmix_gpu = zeros(stream, &[2, 4])?;
    kernels.residual_mix_bwd(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(x0_gpu.cu_ptr(stream)?),
        CudaPtr(go_gpu.cu_ptr(stream)?),
        CudaPtr(mix_gpu.cu_ptr(stream)?),
        CudaPtr(dx_gpu.cu_ptr(stream)?),
        CudaPtr(dx0_gpu.cu_ptr(stream)?),
        CudaPtr(dmix_gpu.cu_ptr(stream)?),
        4,
        x.len() as u32,
    )?;
    report(
        "residual_mix_bwd_x",
        &cpu_dx,
        &download_f32(stream, &dx_gpu)?,
    );
    report(
        "residual_mix_bwd_x0",
        &cpu_dx0,
        &download_f32(stream, &dx0_gpu)?,
    );
    report(
        "residual_mix_bwd_mix",
        &cpu_dmix,
        &download_f32(stream, &dmix_gpu)?,
    );
    Ok(())
}

fn run_residual_add_scale(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let mut cpu: Vec<f32> = (0..12).map(|i| i as f32 * 0.03 - 0.1).collect();
    let proj: Vec<f32> = (0..12).map(|i| i as f32 * 0.07 - 0.2).collect();
    let scale = vec![1.0, 0.5, -0.25, 0.75];
    for i in 0..cpu.len() {
        cpu[i] += scale[i % 4] * proj[i];
    }
    let x0: Vec<f32> = (0..12).map(|i| i as f32 * 0.03 - 0.1).collect();
    let x_gpu = upload_f32(stream, &x0, &[3, 4])?;
    let proj_gpu = upload_f32(stream, &proj, &[3, 4])?;
    let scale_gpu = upload_f32(stream, &scale, &[4])?;
    kernels.residual_add_scale_fwd(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(proj_gpu.cu_ptr(stream)?),
        CudaPtr(scale_gpu.cu_ptr(stream)?),
        4,
        x0.len() as u32,
    )?;
    report("residual_add_scale", &cpu, &download_f32(stream, &x_gpu)?);
    Ok(())
}

fn run_residual_add_scale_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let proj: Vec<f32> = (0..12).map(|i| i as f32 * 0.07 - 0.2).collect();
    let grad_out: Vec<f32> = (0..12).map(|i| i as f32 * 0.02 - 0.1).collect();
    let scale = vec![1.0, 0.5, -0.25, 0.75];
    let cpu_dx = grad_out.clone();
    let mut cpu_dproj = vec![0.0; proj.len()];
    let mut cpu_dscale = vec![0.0; scale.len()];
    for i in 0..proj.len() {
        let d = i % 4;
        cpu_dproj[i] = grad_out[i] * scale[d];
        cpu_dscale[d] += grad_out[i] * proj[i];
    }
    let proj_gpu = upload_f32(stream, &proj, &[3, 4])?;
    let go_gpu = upload_f32(stream, &grad_out, &[3, 4])?;
    let scale_gpu = upload_f32(stream, &scale, &[4])?;
    let dx_gpu = zeros(stream, &[3, 4])?;
    let dproj_gpu = zeros(stream, &[3, 4])?;
    let dscale_gpu = zeros(stream, &[4])?;
    kernels.residual_add_scale_bwd(
        CudaPtr(proj_gpu.cu_ptr(stream)?),
        CudaPtr(go_gpu.cu_ptr(stream)?),
        CudaPtr(scale_gpu.cu_ptr(stream)?),
        CudaPtr(dx_gpu.cu_ptr(stream)?),
        CudaPtr(dproj_gpu.cu_ptr(stream)?),
        CudaPtr(dscale_gpu.cu_ptr(stream)?),
        4,
        proj.len() as u32,
    )?;
    report(
        "residual_add_scale_bwd_x",
        &cpu_dx,
        &download_f32(stream, &dx_gpu)?,
    );
    report(
        "residual_add_scale_bwd_proj",
        &cpu_dproj,
        &download_f32(stream, &dproj_gpu)?,
    );
    report(
        "residual_add_scale_bwd_scale",
        &cpu_dscale,
        &download_f32(stream, &dscale_gpu)?,
    );
    Ok(())
}

fn run_qk_norm(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let mut cpu: Vec<f32> = (0..24).map(|i| i as f32 * 0.04 - 0.3).collect();
    for head in 0..3 {
        let off = head * 8;
        let slice = &cpu[off..off + 8];
        let rms = (slice.iter().map(|v| v * v).sum::<f32>() / 8.0 + 1e-6).sqrt();
        for i in 0..8 {
            cpu[off + i] /= rms;
        }
    }
    let x0: Vec<f32> = (0..24).map(|i| i as f32 * 0.04 - 0.3).collect();
    let qk_gpu = upload_f32(stream, &x0, &[3, 8])?;
    kernels.qk_norm_fwd(CudaPtr(qk_gpu.cu_ptr(stream)?), 8, 3, 1e-6)?;
    report("qk_norm", &cpu, &download_f32(stream, &qk_gpu)?);
    Ok(())
}

fn run_qk_norm_bwd(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let x: Vec<f32> = (0..24).map(|i| i as f32 * 0.04 - 0.3).collect();
    let dy: Vec<f32> = (0..24).map(|i| i as f32 * 0.025 - 0.12).collect();
    let mut cpu = vec![0.0; x.len()];
    qk_norm_backward_cpu(&x, &dy, &mut cpu, 8, 1e-6);
    let x_gpu = upload_f32(stream, &x, &[3, 8])?;
    let dy_gpu = upload_f32(stream, &dy, &[3, 8])?;
    let dx_gpu = zeros(stream, &[3, 8])?;
    kernels.qk_norm_bwd(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(dy_gpu.cu_ptr(stream)?),
        CudaPtr(dx_gpu.cu_ptr(stream)?),
        8,
        3,
        1e-6,
    )?;
    report("qk_norm_bwd", &cpu, &download_f32(stream, &dx_gpu)?);
    Ok(())
}

fn run_partial_rope(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let (cos, sin) = pg_kernels::rope::precompute_rope_tables(4, 4, 10_000.0);
    let mut cpu: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 - 0.25).collect();
    pg_kernels::rope::apply_partial_rope(&mut cpu, &cos, &sin, 1, 4, 2, 4, 4);
    let x0: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 - 0.25).collect();
    let x_gpu = upload_f32(stream, &x0, &[4, 2, 4])?;
    let cos_gpu = upload_f32(stream, &cos, &[4, 2])?;
    let sin_gpu = upload_f32(stream, &sin, &[4, 2])?;
    kernels.partial_rope_fwd(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(cos_gpu.cu_ptr(stream)?),
        CudaPtr(sin_gpu.cu_ptr(stream)?),
        4,
        2,
        4,
        4,
        8,
    )?;
    report("partial_rope", &cpu, &download_f32(stream, &x_gpu)?);
    Ok(())
}

fn run_partial_rope_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let (cos, sin) = pg_kernels::rope::precompute_rope_tables(4, 4, 10_000.0);
    let mut cpu: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 - 0.25).collect();
    pg_kernels::rope::apply_partial_rope_backward(&mut cpu, &cos, &sin, 1, 4, 2, 4, 4);
    let x0: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 - 0.25).collect();
    let x_gpu = upload_f32(stream, &x0, &[4, 2, 4])?;
    let cos_gpu = upload_f32(stream, &cos, &[4, 2])?;
    let sin_gpu = upload_f32(stream, &sin, &[4, 2])?;
    kernels.partial_rope_bwd(
        CudaPtr(x_gpu.cu_ptr(stream)?),
        CudaPtr(cos_gpu.cu_ptr(stream)?),
        CudaPtr(sin_gpu.cu_ptr(stream)?),
        4,
        2,
        4,
        4,
        8,
    )?;
    report("partial_rope_bwd", &cpu, &download_f32(stream, &x_gpu)?);
    Ok(())
}

fn run_q_gain(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let gains = vec![1.5, 0.75];
    let mut cpu: Vec<f32> = (0..32).map(|i| i as f32 * 0.03 - 0.2).collect();
    for head in 0..8 {
        let g = gains[head % 2];
        let off = head * 4;
        for i in 0..4 {
            cpu[off + i] *= g;
        }
    }
    let x0: Vec<f32> = (0..32).map(|i| i as f32 * 0.03 - 0.2).collect();
    let q_gpu = upload_f32(stream, &x0, &[8, 4])?;
    let gains_gpu = upload_f32(stream, &gains, &[2])?;
    kernels.q_gain_fwd(
        CudaPtr(q_gpu.cu_ptr(stream)?),
        CudaPtr(gains_gpu.cu_ptr(stream)?),
        2,
        4,
        8,
    )?;
    report("q_gain", &cpu, &download_f32(stream, &q_gpu)?);
    Ok(())
}

fn run_q_gain_bwd(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let gains = vec![1.5, 0.75];
    let q: Vec<f32> = (0..32).map(|i| i as f32 * 0.03 - 0.2).collect();
    let dy: Vec<f32> = (0..32).map(|i| i as f32 * 0.02 - 0.1).collect();
    let mut cpu_dx = vec![0.0; q.len()];
    let mut cpu_dgain = vec![0.0; gains.len()];
    for head in 0..8 {
        let h = head % 2;
        let off = head * 4;
        for i in 0..4 {
            cpu_dx[off + i] = dy[off + i] * gains[h];
            cpu_dgain[h] += dy[off + i] * q[off + i];
        }
    }
    let q_gpu = upload_f32(stream, &q, &[8, 4])?;
    let dy_gpu = upload_f32(stream, &dy, &[8, 4])?;
    let gains_gpu = upload_f32(stream, &gains, &[2])?;
    let dx_gpu = zeros(stream, &[8, 4])?;
    let dgain_gpu = zeros(stream, &[2])?;
    kernels.q_gain_bwd(
        CudaPtr(q_gpu.cu_ptr(stream)?),
        CudaPtr(dy_gpu.cu_ptr(stream)?),
        CudaPtr(gains_gpu.cu_ptr(stream)?),
        CudaPtr(dx_gpu.cu_ptr(stream)?),
        CudaPtr(dgain_gpu.cu_ptr(stream)?),
        2,
        4,
        8,
    )?;
    report("q_gain_bwd_x", &cpu_dx, &download_f32(stream, &dx_gpu)?);
    report(
        "q_gain_bwd_gain",
        &cpu_dgain,
        &download_f32(stream, &dgain_gpu)?,
    );
    Ok(())
}

fn run_sparse_attn_gate(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let tokens = 3usize;
    let h = 2usize;
    let hd = 4usize;
    let d = 8usize;
    let width = 3usize;
    let scale = 1.2f32;
    let attn: Vec<f32> = (0..tokens * h * hd)
        .map(|i| i as f32 * 0.031 - 0.25)
        .collect();
    let gate_input: Vec<f32> = (0..tokens * d).map(|i| i as f32 * 0.019 - 0.11).collect();
    let weight: Vec<f32> = (0..h * width).map(|i| i as f32 * 0.013 - 0.04).collect();
    let mut cpu_out = vec![0.0f32; attn.len()];
    let mut cpu_gate = vec![0.0f32; tokens * h];
    for tok in 0..tokens {
        for head in 0..h {
            let mut score = 0.0f32;
            for j in 0..width {
                score += weight[head * width + j] * gate_input[tok * d + j];
            }
            let gate = 1.0 / (1.0 + (-(scale * score)).exp());
            cpu_gate[tok * h + head] = gate;
            let base = (tok * h + head) * hd;
            for j in 0..hd {
                cpu_out[base + j] = attn[base + j] * gate;
            }
        }
    }
    let attn_gpu = upload_f32(stream, &attn, &[tokens, h, hd])?;
    let input_gpu = upload_f32(stream, &gate_input, &[tokens, d])?;
    let weight_gpu = upload_f32(stream, &weight, &[h, width])?;
    let out_gpu = zeros(stream, &[tokens, h, hd])?;
    let gate_gpu = zeros(stream, &[tokens, h])?;
    kernels.sparse_attn_gate_fwd(
        CudaPtr(attn_gpu.cu_ptr(stream)?),
        CudaPtr(input_gpu.cu_ptr(stream)?),
        CudaPtr(weight_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        CudaPtr(gate_gpu.cu_ptr(stream)?),
        tokens as u32,
        h as u32,
        hd as u32,
        d as u32,
        width as u32,
        scale,
    )?;
    report(
        "sparse_attn_gate",
        &cpu_out,
        &download_f32(stream, &out_gpu)?,
    );
    report(
        "sparse_attn_gate_values",
        &cpu_gate,
        &download_f32(stream, &gate_gpu)?,
    );
    Ok(())
}

fn run_sparse_attn_gate_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let tokens = 3usize;
    let h = 2usize;
    let hd = 4usize;
    let d = 8usize;
    let width = 3usize;
    let scale = 1.2f32;
    let attn: Vec<f32> = (0..tokens * h * hd)
        .map(|i| i as f32 * 0.031 - 0.25)
        .collect();
    let gate_input: Vec<f32> = (0..tokens * d).map(|i| i as f32 * 0.019 - 0.11).collect();
    let weight: Vec<f32> = (0..h * width).map(|i| i as f32 * 0.013 - 0.04).collect();
    let grad_out: Vec<f32> = (0..tokens * h * hd)
        .map(|i| i as f32 * -0.017 + 0.33)
        .collect();
    let mut gate_values = vec![0.0f32; tokens * h];
    for tok in 0..tokens {
        for head in 0..h {
            let mut score = 0.0f32;
            for j in 0..width {
                score += weight[head * width + j] * gate_input[tok * d + j];
            }
            gate_values[tok * h + head] = 1.0 / (1.0 + (-(scale * score)).exp());
        }
    }
    let mut cpu_grad_attn = vec![0.0f32; attn.len()];
    let mut cpu_grad_input = vec![0.0f32; gate_input.len()];
    let mut cpu_grad_weight = vec![0.0f32; weight.len()];
    for tok in 0..tokens {
        for head in 0..h {
            let gate = gate_values[tok * h + head];
            let base = (tok * h + head) * hd;
            let mut grad_gate = 0.0f32;
            for j in 0..hd {
                let go = grad_out[base + j];
                cpu_grad_attn[base + j] = go * gate;
                grad_gate += go * attn[base + j];
            }
            let grad_score = grad_gate * scale * gate * (1.0 - gate);
            for j in 0..width {
                cpu_grad_weight[head * width + j] += grad_score * gate_input[tok * d + j];
                cpu_grad_input[tok * d + j] += grad_score * weight[head * width + j];
            }
        }
    }
    let attn_gpu = upload_f32(stream, &attn, &[tokens, h, hd])?;
    let input_gpu = upload_f32(stream, &gate_input, &[tokens, d])?;
    let weight_gpu = upload_f32(stream, &weight, &[h, width])?;
    let gate_gpu = upload_f32(stream, &gate_values, &[tokens, h])?;
    let grad_out_gpu = upload_f32(stream, &grad_out, &[tokens, h, hd])?;
    let grad_attn_gpu = zeros(stream, &[tokens, h, hd])?;
    let grad_input_gpu = zeros(stream, &[tokens, d])?;
    let grad_weight_gpu = zeros(stream, &[h, width])?;
    kernels.sparse_attn_gate_bwd(
        CudaPtr(attn_gpu.cu_ptr(stream)?),
        CudaPtr(input_gpu.cu_ptr(stream)?),
        CudaPtr(gate_gpu.cu_ptr(stream)?),
        CudaPtr(grad_out_gpu.cu_ptr(stream)?),
        CudaPtr(weight_gpu.cu_ptr(stream)?),
        CudaPtr(grad_attn_gpu.cu_ptr(stream)?),
        CudaPtr(grad_input_gpu.cu_ptr(stream)?),
        CudaPtr(grad_weight_gpu.cu_ptr(stream)?),
        tokens as u32,
        h as u32,
        hd as u32,
        d as u32,
        width as u32,
        scale,
    )?;
    report(
        "sparse_attn_gate_bwd_attn",
        &cpu_grad_attn,
        &download_f32(stream, &grad_attn_gpu)?,
    );
    report(
        "sparse_attn_gate_bwd_input",
        &cpu_grad_input,
        &download_f32(stream, &grad_input_gpu)?,
    );
    report(
        "sparse_attn_gate_bwd_weight",
        &cpu_grad_weight,
        &download_f32(stream, &grad_weight_gpu)?,
    );
    Ok(())
}

fn run_dot_accumulate(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let a: Vec<f32> = (0..32).map(|i| i as f32 * 0.03 - 0.2).collect();
    let b: Vec<f32> = (0..32).map(|i| i as f32 * -0.02 + 0.4).collect();
    let alpha = 1.7f32;
    let cpu = vec![alpha * a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>()];
    let a_gpu = upload_f32(stream, &a, &[32])?;
    let b_gpu = upload_f32(stream, &b, &[32])?;
    let out_gpu = zeros(stream, &[1])?;
    kernels.dot_accumulate(
        CudaPtr(a_gpu.cu_ptr(stream)?),
        CudaPtr(b_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        alpha,
        a.len() as u32,
    )?;
    report("dot_accumulate", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_cross_entropy_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let logits: Vec<f32> = (0..30).map(|i| i as f32 * 0.07 - 0.8).collect();
    let targets = vec![2u32, 4, 1];
    let mut cpu = vec![0.0; logits.len()];
    pg_kernels::cross_entropy::cross_entropy_backward(
        &logits,
        &targets,
        &mut cpu,
        10,
        30.0,
        1.0 / targets.len() as f32,
    );
    let logits_gpu = upload_f32(stream, &logits, &[3, 10])?;
    let targets_gpu = upload_u32(stream, &targets, &[3])?;
    let grad_gpu = zeros(stream, &[3, 10])?;
    kernels.cross_entropy_bwd(
        CudaPtr(logits_gpu.cu_ptr(stream)?),
        CudaPtr(targets_gpu.cu_ptr(stream)?),
        CudaPtr(grad_gpu.cu_ptr(stream)?),
        10,
        30.0,
        1.0 / targets.len() as f32,
        targets.len() as u32,
    )?;
    report("cross_entropy_bwd", &cpu, &download_f32(stream, &grad_gpu)?);
    Ok(())
}

fn run_attention(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let q: Vec<f32> = (0..24).map(|i| i as f32 * 0.02 - 0.15).collect();
    let k: Vec<f32> = (0..24).map(|i| i as f32 * 0.03 - 0.2).collect();
    let v: Vec<f32> = (0..24).map(|i| i as f32 * 0.04 - 0.25).collect();
    let mut cpu = vec![0.0; q.len()];
    pg_kernels::attention::causal_attention_forward(&q, &k, &v, &mut cpu, 3, 2, 2, 4);
    let q_gpu = upload_f32(stream, &q, &[3, 2, 4])?;
    let k_gpu = upload_f32(stream, &k, &[3, 2, 4])?;
    let v_gpu = upload_f32(stream, &v, &[3, 2, 4])?;
    let out_gpu = zeros(stream, &[3, 2, 4])?;
    kernels.causal_attention_online_fwd(
        CudaPtr(q_gpu.cu_ptr(stream)?),
        CudaPtr(k_gpu.cu_ptr(stream)?),
        CudaPtr(v_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        3,
        3,
        2,
        2,
        4,
    )?;
    report("attention_online", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_attention_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let q: Vec<f32> = (0..24).map(|i| i as f32 * 0.02 - 0.15).collect();
    let k: Vec<f32> = (0..24).map(|i| i as f32 * 0.03 - 0.2).collect();
    let v: Vec<f32> = (0..24).map(|i| i as f32 * 0.04 - 0.25).collect();
    let grad_out: Vec<f32> = (0..24).map(|i| i as f32 * -0.015 + 0.18).collect();
    let mut out = vec![0.0; q.len()];
    let mut cpu_dq = vec![0.0; q.len()];
    let mut cpu_dk = vec![0.0; k.len()];
    let mut cpu_dv = vec![0.0; v.len()];
    pg_kernels::attention::causal_attention_forward(&q, &k, &v, &mut out, 3, 2, 2, 4);
    pg_kernels::attention::causal_attention_backward(
        &q,
        &k,
        &v,
        &out,
        &grad_out,
        &mut cpu_dq,
        &mut cpu_dk,
        &mut cpu_dv,
        3,
        2,
        2,
        4,
    );
    let q_gpu = upload_f32(stream, &q, &[3, 2, 4])?;
    let k_gpu = upload_f32(stream, &k, &[3, 2, 4])?;
    let v_gpu = upload_f32(stream, &v, &[3, 2, 4])?;
    let go_gpu = upload_f32(stream, &grad_out, &[3, 2, 4])?;
    let dq_gpu = zeros(stream, &[3, 2, 4])?;
    let dk_gpu = zeros(stream, &[3, 2, 4])?;
    let dv_gpu = zeros(stream, &[3, 2, 4])?;
    kernels.causal_attention_online_bwd(
        CudaPtr(q_gpu.cu_ptr(stream)?),
        CudaPtr(k_gpu.cu_ptr(stream)?),
        CudaPtr(v_gpu.cu_ptr(stream)?),
        CudaPtr(go_gpu.cu_ptr(stream)?),
        CudaPtr(dq_gpu.cu_ptr(stream)?),
        CudaPtr(dk_gpu.cu_ptr(stream)?),
        CudaPtr(dv_gpu.cu_ptr(stream)?),
        3,
        3,
        2,
        2,
        4,
    )?;
    report(
        "attention_online_bwd_q",
        &cpu_dq,
        &download_f32(stream, &dq_gpu)?,
    );
    report(
        "attention_online_bwd_k",
        &cpu_dk,
        &download_f32(stream, &dk_gpu)?,
    );
    report(
        "attention_online_bwd_v",
        &cpu_dv,
        &download_f32(stream, &dv_gpu)?,
    );
    Ok(())
}

fn run_attention_block_causal_batch(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let batches = 2usize;
    let seq_len = 3usize;
    let tokens = batches * seq_len;
    let num_heads = 2usize;
    let num_kv_heads = 2usize;
    let head_dim = 4usize;
    let q: Vec<f32> = (0..tokens * num_heads * head_dim)
        .map(|i| i as f32 * 0.017 - 0.31)
        .collect();
    let k: Vec<f32> = (0..tokens * num_kv_heads * head_dim)
        .map(|i| i as f32 * -0.013 + 0.22)
        .collect();
    let v: Vec<f32> = (0..tokens * num_kv_heads * head_dim)
        .map(|i| i as f32 * 0.019 - 0.18)
        .collect();
    let grad_out: Vec<f32> = (0..tokens * num_heads * head_dim)
        .map(|i| i as f32 * -0.011 + 0.27)
        .collect();

    let mut cpu_out = vec![0.0; q.len()];
    let mut cpu_dq = vec![0.0; q.len()];
    let mut cpu_dk = vec![0.0; k.len()];
    let mut cpu_dv = vec![0.0; v.len()];
    for b in 0..batches {
        let q_base = b * seq_len * num_heads * head_dim;
        let kv_base = b * seq_len * num_kv_heads * head_dim;
        let q_end = q_base + seq_len * num_heads * head_dim;
        let kv_end = kv_base + seq_len * num_kv_heads * head_dim;
        pg_kernels::attention::causal_attention_forward(
            &q[q_base..q_end],
            &k[kv_base..kv_end],
            &v[kv_base..kv_end],
            &mut cpu_out[q_base..q_end],
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        pg_kernels::attention::causal_attention_backward(
            &q[q_base..q_end],
            &k[kv_base..kv_end],
            &v[kv_base..kv_end],
            &cpu_out[q_base..q_end],
            &grad_out[q_base..q_end],
            &mut cpu_dq[q_base..q_end],
            &mut cpu_dk[kv_base..kv_end],
            &mut cpu_dv[kv_base..kv_end],
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );
    }

    let q_gpu = upload_f32(stream, &q, &[tokens, num_heads, head_dim])?;
    let k_gpu = upload_f32(stream, &k, &[tokens, num_kv_heads, head_dim])?;
    let v_gpu = upload_f32(stream, &v, &[tokens, num_kv_heads, head_dim])?;
    let out_gpu = zeros(stream, &[tokens, num_heads, head_dim])?;
    kernels.causal_attention_online_fwd(
        CudaPtr(q_gpu.cu_ptr(stream)?),
        CudaPtr(k_gpu.cu_ptr(stream)?),
        CudaPtr(v_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        tokens as u32,
        seq_len as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
    )?;
    report(
        "attention_online_block_causal_batch",
        &cpu_out,
        &download_f32(stream, &out_gpu)?,
    );

    let go_gpu = upload_f32(stream, &grad_out, &[tokens, num_heads, head_dim])?;
    let dq_gpu = zeros(stream, &[tokens, num_heads, head_dim])?;
    let dk_gpu = zeros(stream, &[tokens, num_kv_heads, head_dim])?;
    let dv_gpu = zeros(stream, &[tokens, num_kv_heads, head_dim])?;
    kernels.causal_attention_online_bwd(
        CudaPtr(q_gpu.cu_ptr(stream)?),
        CudaPtr(k_gpu.cu_ptr(stream)?),
        CudaPtr(v_gpu.cu_ptr(stream)?),
        CudaPtr(go_gpu.cu_ptr(stream)?),
        CudaPtr(dq_gpu.cu_ptr(stream)?),
        CudaPtr(dk_gpu.cu_ptr(stream)?),
        CudaPtr(dv_gpu.cu_ptr(stream)?),
        tokens as u32,
        seq_len as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
    )?;
    report(
        "attention_online_block_causal_batch_bwd_q",
        &cpu_dq,
        &download_f32(stream, &dq_gpu)?,
    );
    report(
        "attention_online_block_causal_batch_bwd_k",
        &cpu_dk,
        &download_f32(stream, &dk_gpu)?,
    );
    report(
        "attention_online_block_causal_batch_bwd_v",
        &cpu_dv,
        &download_f32(stream, &dv_gpu)?,
    );
    Ok(())
}

fn run_xsa(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let y: Vec<f32> = (0..24).map(|i| i as f32 * 0.05 - 0.3).collect();
    let v: Vec<f32> = (0..12).map(|i| i as f32 * 0.04 - 0.2).collect();
    let mut cpu = vec![0.0; y.len()];
    pg_kernels::xsa::xsa_forward(&y, &v, &mut cpu, 3, 2, 1, 4);
    let y_gpu = upload_f32(stream, &y, &[3, 2, 4])?;
    let v_gpu = upload_f32(stream, &v, &[3, 1, 4])?;
    let out_gpu = zeros(stream, &[3, 2, 4])?;
    kernels.xsa_fwd(
        CudaPtr(y_gpu.cu_ptr(stream)?),
        CudaPtr(v_gpu.cu_ptr(stream)?),
        CudaPtr(out_gpu.cu_ptr(stream)?),
        3,
        2,
        1,
        4,
    )?;
    report("xsa", &cpu, &download_f32(stream, &out_gpu)?);
    Ok(())
}

fn run_xsa_bwd(kernels: &GpuKernels, stream: &Arc<cudarc::driver::CudaStream>) -> PgResult<()> {
    let y: Vec<f32> = (0..24).map(|i| i as f32 * 0.05 - 0.3).collect();
    let v: Vec<f32> = (0..12).map(|i| i as f32 * 0.04 - 0.2).collect();
    let grad_out: Vec<f32> = (0..24).map(|i| i as f32 * 0.03 - 0.15).collect();
    let mut cpu_dy = vec![0.0; y.len()];
    let mut cpu_dv = vec![0.0; v.len()];
    pg_kernels::xsa::xsa_backward(&y, &v, &grad_out, &mut cpu_dy, &mut cpu_dv, 3, 2, 1, 4);
    let y_gpu = upload_f32(stream, &y, &[3, 2, 4])?;
    let v_gpu = upload_f32(stream, &v, &[3, 1, 4])?;
    let go_gpu = upload_f32(stream, &grad_out, &[3, 2, 4])?;
    let dy_gpu = zeros(stream, &[3, 2, 4])?;
    let dv_gpu = zeros(stream, &[3, 1, 4])?;
    kernels.xsa_bwd(
        CudaPtr(y_gpu.cu_ptr(stream)?),
        CudaPtr(v_gpu.cu_ptr(stream)?),
        CudaPtr(go_gpu.cu_ptr(stream)?),
        CudaPtr(dy_gpu.cu_ptr(stream)?),
        CudaPtr(dv_gpu.cu_ptr(stream)?),
        3,
        2,
        1,
        4,
    )?;
    report("xsa_bwd_y", &cpu_dy, &download_f32(stream, &dy_gpu)?);
    report("xsa_bwd_v", &cpu_dv, &download_f32(stream, &dv_gpu)?);
    Ok(())
}

fn run_bigram_hash_bwd(
    kernels: &GpuKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let ids = vec![5u32, 3, 7, 1];
    let grad_out: Vec<f32> = (0..16).map(|i| i as f32 * 0.03 - 0.2).collect();
    let mut cpu = vec![0.0; 8 * 4];
    pg_kernels::bigram_hash::bigram_hash_backward(&ids, &grad_out, &mut cpu, 8, 4);
    let ids_gpu = upload_u32(stream, &ids, &[ids.len()])?;
    let go_gpu = upload_f32(stream, &grad_out, &[ids.len(), 4])?;
    let grad_gpu = zeros(stream, &[8, 4])?;
    kernels.bigram_hash_embed_bwd(
        CudaPtr(ids_gpu.cu_ptr(stream)?),
        CudaPtr(go_gpu.cu_ptr(stream)?),
        CudaPtr(grad_gpu.cu_ptr(stream)?),
        8,
        4,
        ids.len() as u32,
        ids.len() as u32,
    )?;
    report("bigram_hash_bwd", &cpu, &download_f32(stream, &grad_gpu)?);
    Ok(())
}

fn run_gemm(
    gemm: &pg_kernels::gemm::GemmEngine,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let x: Vec<f32> = (0..15).map(|i| i as f32 * 0.03 - 0.2).collect();
    let w: Vec<f32> = (0..20).map(|i| i as f32 * 0.02 - 0.1).collect();
    let dy: Vec<f32> = (0..12).map(|i| i as f32 * 0.04 - 0.15).collect();
    let mut cpu = vec![0.0; 12];
    pg_kernels::linear::linear_forward(&x, &w, &mut cpu, 3, 4, 5);
    let x_gpu = upload_f32(stream, &x, &[3, 5])?;
    let w_gpu = upload_f32(stream, &w, &[4, 5])?;
    let out_gpu = zeros(stream, &[3, 4])?;
    unsafe {
        gemm.matmul_f32(
            x_gpu.cu_ptr(stream)?,
            w_gpu.cu_ptr(stream)?,
            out_gpu.cu_ptr(stream)?,
            3,
            4,
            5,
            1.0,
            0.0,
        )?;
    }
    report("gemm_linear", &cpu, &download_f32(stream, &out_gpu)?);

    let mut cpu_dx = vec![0.0; 15];
    let mut cpu_dw = vec![0.0; 20];
    pg_kernels::linear::linear_backward_input(&dy, &w, &mut cpu_dx, 3, 4, 5);
    pg_kernels::linear::linear_backward_weight(&dy, &x, &mut cpu_dw, 3, 4, 5);
    let dy_gpu = upload_f32(stream, &dy, &[3, 4])?;
    let dx_gpu = zeros(stream, &[3, 5])?;
    let dw_gpu = zeros(stream, &[4, 5])?;
    unsafe {
        gemm.linear_backward_input_f32(
            dy_gpu.cu_ptr(stream)?,
            w_gpu.cu_ptr(stream)?,
            dx_gpu.cu_ptr(stream)?,
            3,
            4,
            5,
            1.0,
            0.0,
        )?;
        gemm.linear_backward_weight_f32(
            dy_gpu.cu_ptr(stream)?,
            x_gpu.cu_ptr(stream)?,
            dw_gpu.cu_ptr(stream)?,
            3,
            4,
            5,
            1.0,
            0.0,
        )?;
    }
    report(
        "gemm_linear_bwd_input",
        &cpu_dx,
        &download_f32(stream, &dx_gpu)?,
    );
    report(
        "gemm_linear_bwd_weight",
        &cpu_dw,
        &download_f32(stream, &dw_gpu)?,
    );
    Ok(())
}

fn run_bf16_gemm(
    kernels: &GpuKernels,
    gemm: &pg_kernels::gemm::GemmEngine,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> PgResult<()> {
    let tokens = 6usize;
    let in_dim = 9usize;
    let out_dim = 7usize;
    let x: Vec<f32> = (0..tokens * in_dim)
        .map(|i| (i as f32 * 0.017 - 0.41).sin() * 0.7)
        .collect();
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32 * -0.013 + 0.29).cos() * 0.4)
        .collect();
    let dy: Vec<f32> = (0..tokens * out_dim)
        .map(|i| (i as f32 * 0.021 - 0.17).sin() * 0.5)
        .collect();

    let x_bf16_cpu = rounded_bf16_vec(&x);
    let w_bf16_cpu = rounded_bf16_vec(&w);
    let dy_bf16_cpu = rounded_bf16_vec(&dy);

    let x_gpu = upload_bf16_from_f32(kernels, stream, &x, &[tokens, in_dim])?;
    let w_gpu = upload_bf16_from_f32(kernels, stream, &w, &[out_dim, in_dim])?;
    let dy_gpu = upload_bf16_from_f32(kernels, stream, &dy, &[tokens, out_dim])?;

    let out_gpu = zeros_dtype(stream, &[tokens, out_dim], DType::BF16)?;
    unsafe {
        gemm.matmul_bf16_bt(
            x_gpu.cu_ptr(stream)?,
            w_gpu.cu_ptr(stream)?,
            out_gpu.cu_ptr(stream)?,
            tokens,
            out_dim,
            in_dim,
            1.0,
            0.0,
        )?;
    }
    let cpu_forward: Vec<f32> =
        linear_forward_cpu(&x_bf16_cpu, &w_bf16_cpu, tokens, out_dim, in_dim)
            .into_iter()
            .map(round_bf16)
            .collect();
    report(
        "gemm_bf16_linear",
        &cpu_forward,
        &download_bf16_as_f32(kernels, stream, &out_gpu)?,
    );

    let out_f32_gpu = zeros_dtype(stream, &[tokens, out_dim], DType::F32)?;
    unsafe {
        gemm.matmul_bf16_bt_to_f32(
            x_gpu.cu_ptr(stream)?,
            w_gpu.cu_ptr(stream)?,
            out_f32_gpu.cu_ptr(stream)?,
            tokens,
            out_dim,
            in_dim,
            1.0,
            0.0,
        )?;
    }
    let cpu_forward_f32 = linear_forward_cpu(&x_bf16_cpu, &w_bf16_cpu, tokens, out_dim, in_dim);
    report(
        "gemm_bf16_linear_to_f32",
        &cpu_forward_f32,
        &download_f32(stream, &out_f32_gpu)?,
    );

    let dx_gpu = zeros_dtype(stream, &[tokens, in_dim], DType::BF16)?;
    unsafe {
        gemm.linear_backward_input_bf16(
            dy_gpu.cu_ptr(stream)?,
            w_gpu.cu_ptr(stream)?,
            dx_gpu.cu_ptr(stream)?,
            tokens,
            out_dim,
            in_dim,
            1.0,
            0.0,
        )?;
    }
    let cpu_dx: Vec<f32> =
        linear_backward_input_cpu(&dy_bf16_cpu, &w_bf16_cpu, tokens, out_dim, in_dim)
            .into_iter()
            .map(round_bf16)
            .collect();
    report(
        "gemm_bf16_bwd_input",
        &cpu_dx,
        &download_bf16_as_f32(kernels, stream, &dx_gpu)?,
    );

    let dx_f32_gpu = zeros_dtype(stream, &[tokens, in_dim], DType::F32)?;
    unsafe {
        gemm.linear_backward_input_bf16_to_f32(
            dy_gpu.cu_ptr(stream)?,
            w_gpu.cu_ptr(stream)?,
            dx_f32_gpu.cu_ptr(stream)?,
            tokens,
            out_dim,
            in_dim,
            1.0,
            0.0,
        )?;
    }
    let cpu_dx_f32 = linear_backward_input_cpu(&dy_bf16_cpu, &w_bf16_cpu, tokens, out_dim, in_dim);
    report(
        "gemm_bf16_bwd_input_to_f32",
        &cpu_dx_f32,
        &download_f32(stream, &dx_f32_gpu)?,
    );

    let dw_seed: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| i as f32 * 0.003 - 0.07)
        .collect();
    let dw_gpu = upload_f32(stream, &dw_seed, &[out_dim, in_dim])?;
    unsafe {
        gemm.linear_backward_weight_bf16_to_f32(
            dy_gpu.cu_ptr(stream)?,
            x_gpu.cu_ptr(stream)?,
            dw_gpu.cu_ptr(stream)?,
            tokens,
            out_dim,
            in_dim,
            1.0,
            1.0,
        )?;
    }
    let mut cpu_dw = dw_seed;
    let add_dw = linear_backward_weight_cpu(&dy_bf16_cpu, &x_bf16_cpu, tokens, out_dim, in_dim);
    for (dst, add) in cpu_dw.iter_mut().zip(add_dw.iter()) {
        *dst += *add;
    }
    report(
        "gemm_bf16_bwd_weight_beta1",
        &cpu_dw,
        &download_f32(stream, &dw_gpu)?,
    );
    Ok(())
}
