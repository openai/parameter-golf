/// Manages the three CUDA streams per device:
/// 1. Compute — GEMM, kernels, forward/backward
/// 2. NCCL — all collective communication
/// 3. Memcpy — host↔device transfers (data loading, checkpoint I/O)
///
/// Synchronization between streams uses CUDA events, not stream synchronize
/// (which would block the CPU thread).

pub struct StreamManager {
    // Will hold cudarc stream handles when CUDA is available.
    // On CPU-only builds, this is a no-op placeholder.
    _private: (),
}

impl StreamManager {
    pub fn new_cpu() -> Self {
        Self { _private: () }
    }
}
