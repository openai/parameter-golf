use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use pg_core::error::{PgError, PgResult};
use std::io::{Read, Write};

/// Custom binary serialization format for model artifacts.
///
/// Format:
///   [4B magic "PGRS"]
///   [4B version = 1]
///   [4B num_tensors]
///   For each tensor:
///     [4B name_len][name_bytes]
///     [4B ndim][shape: ndim × 4B]
///     [1B dtype_code]
///     [4B data_len][data_bytes]
///     [4B CRC32 of data_bytes]
///   [4B metadata_json_len][metadata_json]
///
/// CRC32 per tensor catches silent bit corruption in compressed data.

const MAGIC: &[u8; 4] = b"PGRS";
const VERSION: u32 = 1;

/// DType codes for serialization.
fn dtype_to_code(dtype: pg_core::DType) -> u8 {
    match dtype {
        pg_core::DType::BF16 => 0,
        pg_core::DType::F16 => 1,
        pg_core::DType::F32 => 2,
        pg_core::DType::I8 => 3,
        pg_core::DType::I32 => 4,
        pg_core::DType::U16 => 5,
        pg_core::DType::U32 => 6,
    }
}

fn code_to_dtype(code: u8) -> PgResult<pg_core::DType> {
    match code {
        0 => Ok(pg_core::DType::BF16),
        1 => Ok(pg_core::DType::F16),
        2 => Ok(pg_core::DType::F32),
        3 => Ok(pg_core::DType::I8),
        4 => Ok(pg_core::DType::I32),
        5 => Ok(pg_core::DType::U16),
        6 => Ok(pg_core::DType::U32),
        _ => Err(PgError::DataFormat(format!("unknown dtype code: {}", code))),
    }
}

/// A named tensor for serialization.
pub struct SerializedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: pg_core::DType,
    pub data: Vec<u8>,
}

/// Write a model artifact to a writer.
pub fn write_artifact<W: Write>(
    writer: &mut W,
    tensors: &[SerializedTensor],
    metadata_json: &str,
) -> PgResult<()> {
    writer.write_all(MAGIC)?;
    writer.write_u32::<LittleEndian>(VERSION)?;
    writer.write_u32::<LittleEndian>(tensors.len() as u32)?;

    for tensor in tensors {
        // Name
        let name_bytes = tensor.name.as_bytes();
        writer.write_u32::<LittleEndian>(name_bytes.len() as u32)?;
        writer.write_all(name_bytes)?;

        // Shape
        writer.write_u32::<LittleEndian>(tensor.shape.len() as u32)?;
        for &dim in &tensor.shape {
            writer.write_u32::<LittleEndian>(dim as u32)?;
        }

        // DType
        writer.write_u8(dtype_to_code(tensor.dtype))?;

        // Data + CRC32
        writer.write_u32::<LittleEndian>(tensor.data.len() as u32)?;
        writer.write_all(&tensor.data)?;

        let mut hasher = Hasher::new();
        hasher.update(&tensor.data);
        writer.write_u32::<LittleEndian>(hasher.finalize())?;
    }

    // Metadata JSON
    let meta_bytes = metadata_json.as_bytes();
    writer.write_u32::<LittleEndian>(meta_bytes.len() as u32)?;
    writer.write_all(meta_bytes)?;

    Ok(())
}

/// Read a model artifact from a reader.
pub fn read_artifact<R: Read>(reader: &mut R) -> PgResult<(Vec<SerializedTensor>, String)> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(PgError::DataFormat("wrong magic bytes".into()));
    }

    let version = reader.read_u32::<LittleEndian>()?;
    if version != VERSION {
        return Err(PgError::DataFormat(format!(
            "unsupported version: {}",
            version
        )));
    }

    let num_tensors = reader.read_u32::<LittleEndian>()? as usize;
    let mut tensors = Vec::with_capacity(num_tensors);

    for _ in 0..num_tensors {
        // Name
        let name_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| PgError::DataFormat(format!("invalid tensor name: {}", e)))?;

        // Shape
        let ndim = reader.read_u32::<LittleEndian>()? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(reader.read_u32::<LittleEndian>()? as usize);
        }

        // DType
        let dtype = code_to_dtype(reader.read_u8()?)?;

        // Data
        let data_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        // CRC32 verification
        let expected_crc = reader.read_u32::<LittleEndian>()?;
        let mut hasher = Hasher::new();
        hasher.update(&data);
        let actual_crc = hasher.finalize();
        if actual_crc != expected_crc {
            return Err(PgError::DataFormat(format!(
                "CRC32 mismatch for tensor '{}': expected {:08x}, got {:08x}",
                name, expected_crc, actual_crc
            )));
        }

        tensors.push(SerializedTensor {
            name,
            shape,
            dtype,
            data,
        });
    }

    // Metadata
    let meta_len = reader.read_u32::<LittleEndian>()? as usize;
    let mut meta_bytes = vec![0u8; meta_len];
    reader.read_exact(&mut meta_bytes)?;
    let metadata = String::from_utf8(meta_bytes)
        .map_err(|e| PgError::DataFormat(format!("invalid metadata: {}", e)))?;

    Ok((tensors, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let tensors = vec![
            SerializedTensor {
                name: "weight.0".into(),
                shape: vec![4, 8],
                dtype: pg_core::DType::I8,
                data: vec![
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                ],
            },
            SerializedTensor {
                name: "scale.0".into(),
                shape: vec![4],
                dtype: pg_core::DType::F16,
                data: vec![0; 8], // 4 × f16
            },
        ];
        let metadata = r#"{"format":"int6","version":1}"#;

        let mut buf = Vec::new();
        write_artifact(&mut buf, &tensors, metadata).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let (loaded, loaded_meta) = read_artifact(&mut cursor).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "weight.0");
        assert_eq!(loaded[0].shape, vec![4, 8]);
        assert_eq!(loaded[0].data, tensors[0].data);
        assert_eq!(loaded[1].name, "scale.0");
        assert_eq!(loaded_meta, metadata);
    }

    #[test]
    fn test_crc_corruption_detected() {
        let tensors = vec![SerializedTensor {
            name: "w".into(),
            shape: vec![2, 2],
            dtype: pg_core::DType::I8,
            data: vec![1, 2, 3, 4],
        }];

        let mut buf = Vec::new();
        write_artifact(&mut buf, &tensors, "{}").unwrap();

        // Corrupt one data byte
        let data_start = 4 + 4 + 4 + 4 + 1 + 4 + 2 * 4 + 1 + 4; // offset to data
        buf[data_start] ^= 0xFF;

        let mut cursor = std::io::Cursor::new(buf);
        let result = read_artifact(&mut cursor);
        assert!(result.is_err(), "should detect CRC corruption");
    }
}
