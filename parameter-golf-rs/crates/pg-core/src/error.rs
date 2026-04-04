use thiserror::Error;

use crate::DType;

#[derive(Error, Debug)]
pub enum PgError {
    #[cfg(feature = "cuda")]
    #[error("CUDA driver error: {0}")]
    CudaDriver(#[from] cudarc::driver::DriverError),

    #[error("cuBLAS error: {0}")]
    CuBlas(String),

    #[error("NCCL error: {0}")]
    Nccl(String),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("DType mismatch: expected {expected}, got {got}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("Invalid operation: {0}")]
    InvalidOp(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Data format error: {0}")]
    DataFormat(String),
}

pub type PgResult<T> = Result<T, PgError>;
