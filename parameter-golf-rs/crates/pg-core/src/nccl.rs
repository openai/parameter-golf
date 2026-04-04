/// NCCL multi-GPU communication wrapper.
///
/// Wraps cudarc::nccl for the three collective operations used by
/// the Parallel Muon optimizer pipeline:
/// 1. reduce_scatter — distribute gradient shards across GPUs
/// 2. all_gather — reconstruct full update after Newton-Schulz
/// 3. all_reduce — synchronize scalar/embed gradients + TTT gradients
///
/// All operations are async, returning NcclWork handles.
/// The 3-phase Muon pipeline overlaps:
///   Phase 1: launch async RS for all banks (largest first)
///   Phase 2: all_reduce small params + AdamW (overlaps with RS)
///   Phase 3: wait RS → NS5 → async AG (each AG overlaps next NS5)

pub struct NcclComm {
    pub rank: usize,
    pub world_size: usize,
    // comm: cudarc::nccl::NcclComm — initialized when CUDA device available
}

impl NcclComm {
    /// Initialize NCCL communicator. Call once per process.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }
}
