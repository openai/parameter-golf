/// CUDA-side optimizer helpers.
///
/// This module provides reusable device-side optimizer primitives:
/// - grad norm clipping on device
/// - fused AdamW for scalar / embedding tensors
/// - device-resident Muon / NS5 for bank updates
///
/// Multi-rank orchestration is owned by `pg-train`, where NCCL communicators,
/// per-rank model replicas, and sharded bank buffers are available together.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use pg_core::{DType, GpuTensor, PgError, PgResult};
#[cfg(feature = "cuda")]
use pg_kernels::gemm::GemmEngine;
#[cfg(feature = "cuda")]
use pg_kernels::gpu_kernels::{CudaPtr, GpuKernels};

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct AdamWHyper {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

#[cfg(feature = "cuda")]
pub struct GpuAdamWState {
    pub m: GpuTensor,
    pub v: GpuTensor,
    pub step: usize,
}

#[cfg(feature = "cuda")]
impl GpuAdamWState {
    pub fn new_like(param: &GpuTensor, stream: Arc<CudaStream>) -> PgResult<Self> {
        if param.dtype() != DType::F32 {
            return Err(PgError::InvalidOp(format!(
                "GpuAdamWState requires F32 params, got {:?}",
                param.dtype()
            )));
        }
        Ok(Self {
            m: GpuTensor::zeros_gpu(stream.clone(), param.shape(), DType::F32)?,
            v: GpuTensor::zeros_gpu(stream, param.shape(), DType::F32)?,
            step: 0,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct GpuMuonBankState {
    pub momentum_buffer: GpuTensor,
    pub ns_update: GpuTensor,
    pub ns_x: GpuTensor,
    pub ns_a: GpuTensor,
    pub ns_aa: GpuTensor,
    pub ns_b: GpuTensor,
    pub ns_tmp: GpuTensor,
    pub scale: f32,
    pub batch: usize,
    pub rows: usize,
    pub cols: usize,
    pub gram_dim: usize,
}

#[cfg(feature = "cuda")]
impl GpuMuonBankState {
    pub fn new(shape: [usize; 3], stream: Arc<CudaStream>) -> PgResult<Self> {
        let [batch, rows, cols] = shape;
        let gram_dim = rows.min(cols);
        Ok(Self {
            momentum_buffer: GpuTensor::zeros_gpu(stream.clone(), &shape, DType::F32)?,
            ns_update: GpuTensor::zeros_gpu(stream.clone(), &shape, DType::F32)?,
            ns_x: GpuTensor::zeros_gpu(stream.clone(), &shape, DType::F32)?,
            ns_a: GpuTensor::zeros_gpu(stream.clone(), &[batch, gram_dim, gram_dim], DType::F32)?,
            ns_aa: GpuTensor::zeros_gpu(stream.clone(), &[batch, gram_dim, gram_dim], DType::F32)?,
            ns_b: GpuTensor::zeros_gpu(stream.clone(), &[batch, gram_dim, gram_dim], DType::F32)?,
            ns_tmp: GpuTensor::zeros_gpu(stream, &shape, DType::F32)?,
            scale: (rows as f32 / cols as f32).max(1.0).sqrt(),
            batch,
            rows,
            cols,
            gram_dim,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct GpuMuon {
    gemm: GemmEngine,
    pub lr: f32,
    pub momentum: f32,
    pub nesterov: bool,
    pub weight_decay: f32,
    pub ns_steps: usize,
    pub bank_states: Vec<GpuMuonBankState>,
}

#[cfg(feature = "cuda")]
impl GpuMuon {
    pub fn new(
        stream: Arc<CudaStream>,
        lr: f32,
        momentum: f32,
        ns_steps: usize,
        nesterov: bool,
        weight_decay: f32,
        bank_shapes: &[[usize; 3]],
    ) -> PgResult<Self> {
        let gemm = GemmEngine::new(stream.clone())?;
        let bank_states = bank_shapes
            .iter()
            .copied()
            .map(|shape| GpuMuonBankState::new(shape, stream.clone()))
            .collect::<PgResult<Vec<_>>>()?;
        Ok(Self {
            gemm,
            lr,
            momentum,
            nesterov,
            weight_decay,
            ns_steps,
            bank_states,
        })
    }

    pub fn step_bank(
        &mut self,
        kernels: &GpuKernels,
        bank_idx: usize,
        param: &GpuTensor,
        grad: &GpuTensor,
    ) -> PgResult<()> {
        const NS_A: f32 = 3.4445;
        const NS_B: f32 = -4.7750;
        const NS_C: f32 = 2.0315;

        let state = self
            .bank_states
            .get_mut(bank_idx)
            .ok_or_else(|| PgError::InvalidOp(format!("invalid Muon bank index {bank_idx}")))?;
        if param.shape() != grad.shape()
            || param.shape() != state.momentum_buffer.shape()
            || param.shape().len() != 3
        {
            return Err(PgError::InvalidOp(
                "GpuMuon::step_bank expects matching rank-3 bank tensors".into(),
            ));
        }

        let bank_numel = param.numel() as u32;
        kernels.scale_inplace(
            CudaPtr(state.momentum_buffer.cu_ptr(kernels.stream())?),
            self.momentum,
            bank_numel,
        )?;
        kernels.add_scaled_fwd(
            CudaPtr(state.momentum_buffer.cu_ptr(kernels.stream())?),
            CudaPtr(grad.cu_ptr(kernels.stream())?),
            1.0,
            bank_numel,
        )?;

        if self.nesterov {
            kernels.copy_fwd(
                CudaPtr(grad.cu_ptr(kernels.stream())?),
                CudaPtr(state.ns_update.cu_ptr(kernels.stream())?),
                bank_numel,
            )?;
            kernels.add_scaled_fwd(
                CudaPtr(state.ns_update.cu_ptr(kernels.stream())?),
                CudaPtr(state.momentum_buffer.cu_ptr(kernels.stream())?),
                self.momentum,
                bank_numel,
            )?;
        } else {
            kernels.copy_fwd(
                CudaPtr(state.momentum_buffer.cu_ptr(kernels.stream())?),
                CudaPtr(state.ns_update.cu_ptr(kernels.stream())?),
                bank_numel,
            )?;
        }

        kernels.normalize_matrices(
            CudaPtr(state.ns_update.cu_ptr(kernels.stream())?),
            CudaPtr(state.ns_x.cu_ptr(kernels.stream())?),
            (state.rows * state.cols) as u32,
            state.batch as u32,
            1e-7,
        )?;

        for _ in 0..self.ns_steps {
            for bi in 0..state.batch {
                let x_i = state.ns_x.slice_first(bi)?;
                let a_i = state.ns_a.slice_first(bi)?;
                let aa_i = state.ns_aa.slice_first(bi)?;
                let b_i = state.ns_b.slice_first(bi)?;
                let tmp_i = state.ns_tmp.slice_first(bi)?;

                if state.rows <= state.cols {
                    unsafe {
                        self.gemm.matmul_f32(
                            x_i.cu_ptr(self.gemm.stream())?,
                            x_i.cu_ptr(self.gemm.stream())?,
                            a_i.cu_ptr(self.gemm.stream())?,
                            state.rows,
                            state.rows,
                            state.cols,
                            1.0,
                            0.0,
                        )?;
                        self.gemm.matmul_f32_nn(
                            a_i.cu_ptr(self.gemm.stream())?,
                            a_i.cu_ptr(self.gemm.stream())?,
                            aa_i.cu_ptr(self.gemm.stream())?,
                            state.gram_dim,
                            state.gram_dim,
                            state.gram_dim,
                            1.0,
                            0.0,
                        )?;
                    }
                    kernels.copy_fwd(
                        CudaPtr(a_i.cu_ptr(kernels.stream())?),
                        CudaPtr(b_i.cu_ptr(kernels.stream())?),
                        a_i.numel() as u32,
                    )?;
                    kernels.scale_inplace(
                        CudaPtr(b_i.cu_ptr(kernels.stream())?),
                        NS_B,
                        b_i.numel() as u32,
                    )?;
                    kernels.add_scaled_fwd(
                        CudaPtr(b_i.cu_ptr(kernels.stream())?),
                        CudaPtr(aa_i.cu_ptr(kernels.stream())?),
                        NS_C,
                        b_i.numel() as u32,
                    )?;
                    unsafe {
                        self.gemm.matmul_f32_nn(
                            b_i.cu_ptr(self.gemm.stream())?,
                            x_i.cu_ptr(self.gemm.stream())?,
                            tmp_i.cu_ptr(self.gemm.stream())?,
                            state.rows,
                            state.cols,
                            state.rows,
                            1.0,
                            0.0,
                        )?;
                    }
                } else {
                    unsafe {
                        self.gemm.matmul_f32_tn(
                            x_i.cu_ptr(self.gemm.stream())?,
                            x_i.cu_ptr(self.gemm.stream())?,
                            a_i.cu_ptr(self.gemm.stream())?,
                            state.cols,
                            state.cols,
                            state.rows,
                            1.0,
                            0.0,
                        )?;
                        self.gemm.matmul_f32_nn(
                            a_i.cu_ptr(self.gemm.stream())?,
                            a_i.cu_ptr(self.gemm.stream())?,
                            aa_i.cu_ptr(self.gemm.stream())?,
                            state.gram_dim,
                            state.gram_dim,
                            state.gram_dim,
                            1.0,
                            0.0,
                        )?;
                    }
                    kernels.copy_fwd(
                        CudaPtr(a_i.cu_ptr(kernels.stream())?),
                        CudaPtr(b_i.cu_ptr(kernels.stream())?),
                        a_i.numel() as u32,
                    )?;
                    kernels.scale_inplace(
                        CudaPtr(b_i.cu_ptr(kernels.stream())?),
                        NS_B,
                        b_i.numel() as u32,
                    )?;
                    kernels.add_scaled_fwd(
                        CudaPtr(b_i.cu_ptr(kernels.stream())?),
                        CudaPtr(aa_i.cu_ptr(kernels.stream())?),
                        NS_C,
                        b_i.numel() as u32,
                    )?;
                    unsafe {
                        self.gemm.matmul_f32_nn(
                            x_i.cu_ptr(self.gemm.stream())?,
                            b_i.cu_ptr(self.gemm.stream())?,
                            tmp_i.cu_ptr(self.gemm.stream())?,
                            state.rows,
                            state.cols,
                            state.cols,
                            1.0,
                            0.0,
                        )?;
                    }
                }

                kernels.scale_inplace(
                    CudaPtr(x_i.cu_ptr(kernels.stream())?),
                    NS_A,
                    x_i.numel() as u32,
                )?;
                kernels.add_scaled_fwd(
                    CudaPtr(x_i.cu_ptr(kernels.stream())?),
                    CudaPtr(tmp_i.cu_ptr(kernels.stream())?),
                    1.0,
                    x_i.numel() as u32,
                )?;
            }
        }

        kernels.scale_inplace(
            CudaPtr(state.ns_x.cu_ptr(kernels.stream())?),
            state.scale,
            bank_numel,
        )?;
        kernels.decay_sgd_step(
            CudaPtr(param.cu_ptr(kernels.stream())?),
            CudaPtr(state.ns_x.cu_ptr(kernels.stream())?),
            self.lr,
            self.weight_decay,
            bank_numel,
        )?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct GpuOptimizer;

#[cfg(feature = "cuda")]
impl GpuOptimizer {
    pub fn new() -> Self {
        Self
    }

    pub fn clip_grad_norm(
        &mut self,
        kernels: &GpuKernels,
        grads: &[&GpuTensor],
        max_norm: f32,
        scratch_sum_sq: &mut GpuTensor,
    ) -> PgResult<()> {
        if scratch_sum_sq.dtype() != DType::F32 || scratch_sum_sq.numel() != 1 {
            return Err(PgError::InvalidOp(
                "scratch_sum_sq must be an F32 scalar tensor".into(),
            ));
        }
        scratch_sum_sq.copy_from_host_bytes(bytemuck::bytes_of(&0.0f32))?;
        for grad in grads {
            if grad.dtype() != DType::F32 {
                return Err(PgError::InvalidOp(format!(
                    "clip_grad_norm requires F32 grads, got {:?}",
                    grad.dtype()
                )));
            }
            kernels.dot_accumulate(
                CudaPtr(grad.cu_ptr(kernels.stream())?),
                CudaPtr(grad.cu_ptr(kernels.stream())?),
                CudaPtr(scratch_sum_sq.cu_ptr(kernels.stream())?),
                1.0,
                grad.numel() as u32,
            )?;
        }
        for grad in grads {
            kernels.clip_by_global_norm(
                CudaPtr(grad.cu_ptr(kernels.stream())?),
                CudaPtr(scratch_sum_sq.cu_ptr(kernels.stream())?),
                max_norm,
                grad.numel() as u32,
            )?;
        }
        Ok(())
    }

    pub fn adamw_step(
        &mut self,
        kernels: &GpuKernels,
        param: &GpuTensor,
        grad: &GpuTensor,
        state: &mut GpuAdamWState,
        hyper: AdamWHyper,
    ) -> PgResult<()> {
        if param.shape() != grad.shape()
            || param.shape() != state.m.shape()
            || param.shape() != state.v.shape()
        {
            return Err(PgError::InvalidOp(
                "adamw_step param / grad / state shape mismatch".into(),
            ));
        }
        state.step += 1;
        let bc1 = 1.0 - hyper.beta1.powi(state.step as i32);
        let bc2 = 1.0 - hyper.beta2.powi(state.step as i32);
        kernels.adamw_step(
            CudaPtr(param.cu_ptr(kernels.stream())?),
            CudaPtr(grad.cu_ptr(kernels.stream())?),
            CudaPtr(state.m.cu_ptr(kernels.stream())?),
            CudaPtr(state.v.cu_ptr(kernels.stream())?),
            hyper.lr,
            hyper.beta1,
            hyper.beta2,
            bc1,
            bc2,
            hyper.eps,
            hyper.weight_decay,
            param.numel() as u32,
        )
    }

    pub fn decay_sgd_step(
        &mut self,
        kernels: &GpuKernels,
        param: &GpuTensor,
        grad: &GpuTensor,
        lr: f32,
        weight_decay: f32,
    ) -> PgResult<()> {
        if param.shape() != grad.shape() {
            return Err(PgError::InvalidOp(
                "decay_sgd_step param / grad shape mismatch".into(),
            ));
        }
        kernels.decay_sgd_step(
            CudaPtr(param.cu_ptr(kernels.stream())?),
            CudaPtr(grad.cu_ptr(kernels.stream())?),
            lr,
            weight_decay,
            param.numel() as u32,
        )
    }
}
