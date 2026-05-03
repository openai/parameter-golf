/// Minimal in-memory safetensors writer.
///
/// We do not depend on the `safetensors` crate to keep the workspace lean.
/// This writer produces files identical in layout to what `safetensors.save()`
/// would emit from PyTorch:
///   [u64 LE: header_size] [JSON header] [tensor data]
///
/// Used by:
///   - tests round-tripping `pg_compat::loader::load_safetensors`
///   - the per-step equivalence harness, which loads the same dump from both
///     PyTorch and Rust to validate forward/backward parity.
use std::collections::BTreeMap;

use half::bf16;

/// A single tensor entry in a safetensors file.
pub struct OutTensor<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: TensorBytes<'a>,
}

/// Dtype tag — matches the strings safetensors uses.
#[derive(Debug, Clone, Copy)]
pub enum Dtype {
    F32,
    BF16,
    F16,
}

impl Dtype {
    fn as_str(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::BF16 => "BF16",
            Dtype::F16 => "F16",
        }
    }

    fn elem_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
        }
    }
}

/// Owned or borrowed raw bytes for a tensor.
pub enum TensorBytes<'a> {
    F32Slice(&'a [f32]),
    Bytes(Vec<u8>),
}

impl<'a> TensorBytes<'a> {
    fn write_into(&self, out: &mut Vec<u8>, dtype: Dtype) {
        match self {
            TensorBytes::F32Slice(slice) => match dtype {
                Dtype::F32 => {
                    for &v in *slice {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                Dtype::BF16 => {
                    for &v in *slice {
                        let b = bf16::from_f32(v).to_bits();
                        out.extend_from_slice(&b.to_le_bytes());
                    }
                }
                Dtype::F16 => {
                    for &v in *slice {
                        let b = half::f16::from_f32(v).to_bits();
                        out.extend_from_slice(&b.to_le_bytes());
                    }
                }
            },
            TensorBytes::Bytes(bytes) => out.extend_from_slice(bytes),
        }
    }

    fn byte_len(&self, numel: usize, dtype: Dtype) -> usize {
        match self {
            TensorBytes::Bytes(b) => b.len(),
            TensorBytes::F32Slice(_) => numel * dtype.elem_bytes(),
        }
    }
}

/// Build a safetensors file in memory and return its bytes.
pub fn write_safetensors(tensors: &[OutTensor]) -> Vec<u8> {
    // Compute byte ranges first using a sorted map to give the header
    // a deterministic key order (matches how Python torch.save emits dicts).
    let mut sorted: BTreeMap<&str, &OutTensor> = BTreeMap::new();
    for t in tensors {
        sorted.insert(t.name, t);
    }

    let mut header = String::from("{");
    let mut offset = 0usize;
    let mut first = true;
    let mut data_layout: Vec<(&OutTensor, usize, usize)> = Vec::new();

    for (name, t) in &sorted {
        let numel: usize = t.shape.iter().product();
        let nbytes = t.data.byte_len(numel, t.dtype);
        let start = offset;
        let end = offset + nbytes;
        offset = end;

        if !first {
            header.push(',');
        }
        first = false;

        let shape_str: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            t.dtype.as_str(),
            shape_str.join(","),
            start,
            end
        ));
        data_layout.push((t, start, end));
    }
    header.push('}');

    // Pad header to 8-byte alignment for tensor data
    while header.len() % 8 != 0 {
        header.push(' ');
    }

    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset);
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);

    // Write data in the same order we recorded
    for (t, _start, _end) in &data_layout {
        t.data.write_into(&mut out, t.dtype);
    }

    out
}

/// Convenience: build a single F32-stored tensor entry from a slice.
pub fn f32_tensor<'a>(name: &'a str, shape: Vec<usize>, data: &'a [f32]) -> OutTensor<'a> {
    OutTensor {
        name,
        dtype: Dtype::F32,
        shape,
        data: TensorBytes::F32Slice(data),
    }
}

/// Convenience: build a single BF16-stored tensor entry from an f32 slice
/// (the slice will be converted to bf16 during write).
pub fn bf16_tensor<'a>(name: &'a str, shape: Vec<usize>, data: &'a [f32]) -> OutTensor<'a> {
    OutTensor {
        name,
        dtype: Dtype::BF16,
        shape,
        data: TensorBytes::F32Slice(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::SafeTensorsFile;

    #[test]
    fn test_write_then_read_f32() {
        let weight = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![0.5f32, -0.5];

        let tensors = vec![
            f32_tensor("layer.weight", vec![2, 3], &weight),
            f32_tensor("layer.bias", vec![2], &bias),
        ];

        let bytes = write_safetensors(&tensors);
        let st = SafeTensorsFile::from_bytes(&bytes).unwrap();

        let w = st.get_tensor_f32("layer.weight").unwrap();
        let b = st.get_tensor_f32("layer.bias").unwrap();
        assert_eq!(w, weight);
        assert_eq!(b, bias);

        // Shapes round-tripped
        assert_eq!(st.tensors["layer.weight"].shape, vec![2, 3]);
        assert_eq!(st.tensors["layer.bias"].shape, vec![2]);
    }

    #[test]
    fn test_write_then_read_bf16() {
        // Use values with exact bf16 representation
        let weight: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensors = vec![bf16_tensor("w", vec![4, 4], &weight)];
        let bytes = write_safetensors(&tensors);
        let st = SafeTensorsFile::from_bytes(&bytes).unwrap();
        let w = st.get_tensor_f32("w").unwrap();
        // bf16 has 8 mantissa bits — small ints round-trip exactly
        assert_eq!(w, weight);
    }
}
