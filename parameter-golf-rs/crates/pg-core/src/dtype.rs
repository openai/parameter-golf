use std::fmt;

/// Data types supported by the tensor system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    I8,
    I32,
    U16,
    U32,
}

impl DType {
    /// Size in bytes of a single element.
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::BF16 | DType::F16 | DType::U16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::I8 => 1,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, DType::BF16 | DType::F16 | DType::F32)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::BF16 => write!(f, "bf16"),
            DType::F16 => write!(f, "f16"),
            DType::F32 => write!(f, "f32"),
            DType::I8 => write!(f, "i8"),
            DType::I32 => write!(f, "i32"),
            DType::U16 => write!(f, "u16"),
            DType::U32 => write!(f, "u32"),
        }
    }
}
