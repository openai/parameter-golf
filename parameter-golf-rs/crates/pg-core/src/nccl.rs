/// NCCL multi-GPU communication wrapper.
///
/// This module intentionally exposes correctness-first collectives only. The
/// train loop must still gate distributed record runs until GPU backward and
/// distributed optimizer code consume these methods end to end.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
#[cfg(feature = "cuda")]
use cudarc::nccl::{Comm, ReduceOp};
#[cfg(feature = "cuda")]
use half::bf16;

#[cfg(feature = "cuda")]
use crate::{GpuTensor, PgError, PgResult};

#[cfg(feature = "cuda")]
struct RawF32Buffer {
    ptr: u64,
    len: usize,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
struct RawBf16Buffer {
    ptr: u64,
    len: usize,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl DeviceSlice<f32> for RawF32Buffer {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

#[cfg(feature = "cuda")]
impl DeviceSlice<bf16> for RawBf16Buffer {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

#[cfg(feature = "cuda")]
impl DevicePtr<f32> for RawF32Buffer {
    fn device_ptr<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        // These wrappers are only used on the communicator's owning stream and
        // for tensors already synchronized through the same training stream.
        (self.ptr, SyncOnDrop::Record(None))
    }
}

#[cfg(feature = "cuda")]
impl DevicePtr<bf16> for RawBf16Buffer {
    fn device_ptr<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

#[cfg(feature = "cuda")]
impl DevicePtrMut<f32> for RawF32Buffer {
    fn device_ptr_mut<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

#[cfg(feature = "cuda")]
impl DevicePtrMut<bf16> for RawBf16Buffer {
    fn device_ptr_mut<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

#[cfg(feature = "cuda")]
pub struct NcclComm {
    pub rank: usize,
    pub world_size: usize,
    comm: Option<Comm>,
}

// The wrapper is shared across host worker threads by the distributed training
// runtime so each rank can enqueue its own communicator's collectives while
// that rank's backward pass is still running. The runtime invariant is one
// host worker per communicator during the bucketed-backward phase; the main
// thread does not reuse those communicators until the scoped workers have
// joined. The underlying NCCL operations are still ordered by their bound CUDA
// streams/events. Do not call methods concurrently on the same `NcclComm`.
#[cfg(feature = "cuda")]
unsafe impl Send for NcclComm {}
#[cfg(feature = "cuda")]
unsafe impl Sync for NcclComm {}

#[cfg(not(feature = "cuda"))]
pub struct NcclComm {
    pub rank: usize,
    pub world_size: usize,
}

#[cfg(feature = "cuda")]
impl NcclComm {
    /// Metadata-only constructor retained for CPU/data-shard preflight code.
    /// Collective methods return an error unless the communicator came from
    /// `from_local_devices`.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size,
            comm: None,
        }
    }

    /// Build one NCCL communicator per local CUDA stream/device.
    pub fn from_local_devices(streams: Vec<Arc<CudaStream>>) -> PgResult<Vec<Self>> {
        let world_size = streams.len();
        let comms = Comm::from_devices(streams)
            .map_err(|e| PgError::Nccl(format!("Comm::from_devices failed: {e:?}")))?;
        Ok(comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| Self {
                rank,
                world_size,
                comm: Some(comm),
            })
            .collect())
    }

    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }

    pub fn all_reduce_sum_f32(
        &self,
        send: &CudaSlice<f32>,
        recv: &mut CudaSlice<f32>,
    ) -> PgResult<()> {
        self.comm()?
            .all_reduce::<_, _, f32>(send, recv, &ReduceOp::Sum)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("all_reduce_sum_f32 failed: {e:?}")))
    }

    pub fn reduce_scatter_sum_f32(
        &self,
        send: &CudaSlice<f32>,
        recv: &mut CudaSlice<f32>,
    ) -> PgResult<()> {
        self.comm()?
            .reduce_scatter::<_, _, f32>(send, recv, &ReduceOp::Sum)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("reduce_scatter_sum_f32 failed: {e:?}")))
    }

    pub fn all_gather_f32(&self, send: &CudaSlice<f32>, recv: &mut CudaSlice<f32>) -> PgResult<()> {
        self.comm()?
            .all_gather::<_, _, f32>(send, recv)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("all_gather_f32 failed: {e:?}")))
    }

    pub fn all_reduce_sum_tensor_f32_in_place(&self, tensor: &mut GpuTensor) -> PgResult<()> {
        if tensor.dtype() != crate::DType::F32 {
            return Err(PgError::Nccl(format!(
                "all_reduce_sum_tensor_f32_in_place requires F32 tensor, got {:?}",
                tensor.dtype()
            )));
        }
        let stream = self.comm()?.stream();
        let ptr = tensor.cu_ptr(&stream)?;
        let mut raw = RawF32Buffer {
            ptr,
            len: tensor.numel(),
            stream,
        };
        self.comm()?
            .all_reduce_in_place::<_, f32>(&mut raw, &ReduceOp::Sum)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("all_reduce_sum_tensor_f32_in_place failed: {e:?}")))
    }

    pub fn reduce_scatter_sum_tensor_f32(
        &self,
        send: &GpuTensor,
        recv: &mut GpuTensor,
    ) -> PgResult<()> {
        if send.dtype() != crate::DType::F32 || recv.dtype() != crate::DType::F32 {
            return Err(PgError::Nccl(format!(
                "reduce_scatter_sum_tensor_f32 requires F32 tensors, got {:?} -> {:?}",
                send.dtype(),
                recv.dtype()
            )));
        }
        if send.numel() != recv.numel() * self.world_size {
            return Err(PgError::Nccl(format!(
                "reduce_scatter_sum_tensor_f32 shape mismatch: send elements {} must equal recv elements {} * world_size {}",
                send.numel(),
                recv.numel(),
                self.world_size
            )));
        }
        let stream = self.comm()?.stream();
        let send_ptr = send.cu_ptr(&stream)?;
        let recv_ptr = recv.cu_ptr(&stream)?;
        let raw_send = RawF32Buffer {
            ptr: send_ptr,
            len: send.numel(),
            stream: stream.clone(),
        };
        let mut raw_recv = RawF32Buffer {
            ptr: recv_ptr,
            len: recv.numel(),
            stream,
        };
        self.comm()?
            .reduce_scatter::<_, _, f32>(&raw_send, &mut raw_recv, &ReduceOp::Sum)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("reduce_scatter_sum_tensor_f32 failed: {e:?}")))
    }

    pub fn reduce_sum_tensor_f32_to_rank(
        &self,
        send: &GpuTensor,
        recv: Option<&mut GpuTensor>,
        root_rank: usize,
    ) -> PgResult<()> {
        if send.dtype() != crate::DType::F32 {
            return Err(PgError::Nccl(format!(
                "reduce_sum_tensor_f32_to_rank requires F32 send tensor, got {:?}",
                send.dtype()
            )));
        }
        if root_rank >= self.world_size {
            return Err(PgError::Nccl(format!(
                "reduce_sum_tensor_f32_to_rank root rank {root_rank} >= world_size {}",
                self.world_size
            )));
        }
        if self.rank == root_rank && recv.is_none() {
            return Err(PgError::Nccl(format!(
                "reduce_sum_tensor_f32_to_rank root rank {} must provide recv tensor",
                self.rank
            )));
        }
        if let Some(recv) = recv.as_ref() {
            if recv.dtype() != crate::DType::F32 {
                return Err(PgError::Nccl(format!(
                    "reduce_sum_tensor_f32_to_rank requires F32 recv tensor, got {:?}",
                    recv.dtype()
                )));
            }
            if recv.numel() != send.numel() {
                return Err(PgError::Nccl(format!(
                    "reduce_sum_tensor_f32_to_rank shape mismatch: send elements {} recv elements {}",
                    send.numel(),
                    recv.numel()
                )));
            }
        }
        let stream = self.comm()?.stream();
        let send_ptr = send.cu_ptr(&stream)?;
        let raw_send = RawF32Buffer {
            ptr: send_ptr,
            len: send.numel(),
            stream: stream.clone(),
        };
        let mut raw_recv = if let Some(recv) = recv {
            Some(RawF32Buffer {
                ptr: recv.cu_ptr(&stream)?,
                len: recv.numel(),
                stream,
            })
        } else {
            None
        };
        self.comm()?
            .reduce::<_, _, f32>(
                &raw_send,
                raw_recv.as_mut(),
                &ReduceOp::Sum,
                root_rank as i32,
            )
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("reduce_sum_tensor_f32_to_rank failed: {e:?}")))
    }

    pub fn reduce_sum_tensor_bf16_to_rank(
        &self,
        send: &GpuTensor,
        recv: Option<&mut GpuTensor>,
        root_rank: usize,
    ) -> PgResult<()> {
        if send.dtype() != crate::DType::BF16 {
            return Err(PgError::Nccl(format!(
                "reduce_sum_tensor_bf16_to_rank requires BF16 send tensor, got {:?}",
                send.dtype()
            )));
        }
        if root_rank >= self.world_size {
            return Err(PgError::Nccl(format!(
                "reduce_sum_tensor_bf16_to_rank root rank {root_rank} >= world_size {}",
                self.world_size
            )));
        }
        if self.rank == root_rank && recv.is_none() {
            return Err(PgError::Nccl(format!(
                "reduce_sum_tensor_bf16_to_rank root rank {} must provide recv tensor",
                self.rank
            )));
        }
        if let Some(recv) = recv.as_ref() {
            if recv.dtype() != crate::DType::BF16 {
                return Err(PgError::Nccl(format!(
                    "reduce_sum_tensor_bf16_to_rank requires BF16 recv tensor, got {:?}",
                    recv.dtype()
                )));
            }
            if recv.numel() != send.numel() {
                return Err(PgError::Nccl(format!(
                    "reduce_sum_tensor_bf16_to_rank shape mismatch: send elements {} recv elements {}",
                    send.numel(),
                    recv.numel()
                )));
            }
        }
        let stream = self.comm()?.stream();
        let send_ptr = send.cu_ptr(&stream)?;
        let raw_send = RawBf16Buffer {
            ptr: send_ptr,
            len: send.numel(),
            stream: stream.clone(),
        };
        let mut raw_recv = if let Some(recv) = recv {
            Some(RawBf16Buffer {
                ptr: recv.cu_ptr(&stream)?,
                len: recv.numel(),
                stream,
            })
        } else {
            None
        };
        self.comm()?
            .reduce::<_, _, bf16>(
                &raw_send,
                raw_recv.as_mut(),
                &ReduceOp::Sum,
                root_rank as i32,
            )
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("reduce_sum_tensor_bf16_to_rank failed: {e:?}")))
    }

    pub fn reduce_scatter_sum_tensor_bf16(
        &self,
        send: &GpuTensor,
        recv: &mut GpuTensor,
    ) -> PgResult<()> {
        if send.dtype() != crate::DType::BF16 || recv.dtype() != crate::DType::BF16 {
            return Err(PgError::Nccl(format!(
                "reduce_scatter_sum_tensor_bf16 requires BF16 tensors, got {:?} -> {:?}",
                send.dtype(),
                recv.dtype()
            )));
        }
        if send.numel() != recv.numel() * self.world_size {
            return Err(PgError::Nccl(format!(
                "reduce_scatter_sum_tensor_bf16 shape mismatch: send elements {} must equal recv elements {} * world_size {}",
                send.numel(),
                recv.numel(),
                self.world_size
            )));
        }
        let stream = self.comm()?.stream();
        let send_ptr = send.cu_ptr(&stream)?;
        let recv_ptr = recv.cu_ptr(&stream)?;
        let raw_send = RawBf16Buffer {
            ptr: send_ptr,
            len: send.numel(),
            stream: stream.clone(),
        };
        let mut raw_recv = RawBf16Buffer {
            ptr: recv_ptr,
            len: recv.numel(),
            stream,
        };
        self.comm()?
            .reduce_scatter::<_, _, bf16>(&raw_send, &mut raw_recv, &ReduceOp::Sum)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("reduce_scatter_sum_tensor_bf16 failed: {e:?}")))
    }

    pub fn all_gather_tensor_f32(&self, send: &GpuTensor, recv: &mut GpuTensor) -> PgResult<()> {
        if send.dtype() != crate::DType::F32 || recv.dtype() != crate::DType::F32 {
            return Err(PgError::Nccl(format!(
                "all_gather_tensor_f32 requires F32 tensors, got {:?} -> {:?}",
                send.dtype(),
                recv.dtype()
            )));
        }
        if recv.numel() != send.numel() * self.world_size {
            return Err(PgError::Nccl(format!(
                "all_gather_tensor_f32 shape mismatch: recv elements {} must equal send elements {} * world_size {}",
                recv.numel(),
                send.numel(),
                self.world_size
            )));
        }
        let stream = self.comm()?.stream();
        let send_ptr = send.cu_ptr(&stream)?;
        let recv_ptr = recv.cu_ptr(&stream)?;
        let raw_send = RawF32Buffer {
            ptr: send_ptr,
            len: send.numel(),
            stream: stream.clone(),
        };
        let mut raw_recv = RawF32Buffer {
            ptr: recv_ptr,
            len: recv.numel(),
            stream,
        };
        self.comm()?
            .all_gather::<_, _, f32>(&raw_send, &mut raw_recv)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("all_gather_tensor_f32 failed: {e:?}")))
    }

    pub fn all_gather_tensor_bf16(&self, send: &GpuTensor, recv: &mut GpuTensor) -> PgResult<()> {
        if send.dtype() != crate::DType::BF16 || recv.dtype() != crate::DType::BF16 {
            return Err(PgError::Nccl(format!(
                "all_gather_tensor_bf16 requires BF16 tensors, got {:?} -> {:?}",
                send.dtype(),
                recv.dtype()
            )));
        }
        if recv.numel() != send.numel() * self.world_size {
            return Err(PgError::Nccl(format!(
                "all_gather_tensor_bf16 shape mismatch: recv elements {} must equal send elements {} * world_size {}",
                recv.numel(),
                send.numel(),
                self.world_size
            )));
        }
        let stream = self.comm()?.stream();
        let send_ptr = send.cu_ptr(&stream)?;
        let recv_ptr = recv.cu_ptr(&stream)?;
        let raw_send = RawBf16Buffer {
            ptr: send_ptr,
            len: send.numel(),
            stream: stream.clone(),
        };
        let mut raw_recv = RawBf16Buffer {
            ptr: recv_ptr,
            len: recv.numel(),
            stream,
        };
        self.comm()?
            .all_gather::<_, _, bf16>(&raw_send, &mut raw_recv)
            .map(|_| ())
            .map_err(|e| PgError::Nccl(format!("all_gather_tensor_bf16 failed: {e:?}")))
    }

    fn comm(&self) -> PgResult<&Comm> {
        self.comm.as_ref().ok_or_else(|| {
            PgError::Nccl(
                "NCCL communicator is metadata-only; construct with NcclComm::from_local_devices"
                    .into(),
            )
        })
    }
}

#[cfg(not(feature = "cuda"))]
impl NcclComm {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }
}
